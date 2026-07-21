"""
Block-Coordinate Descent (BCD).

Implements:
    Mairal, Bach, Ponce, Sapiro. "Online Dictionary Learning for
    Sparse Coding". ICML 2009.

Algorithm 1 (outer loop, sparse coding via LARS-Lasso, A/B statistics
accumulation) and Algorithm 2 (block-coordinate dictionary update) are
implemented as described. Since this library targets fixed training
sets rather than a genuine data stream, the outer loop follows Sec 3.4
("Handling Fixed-Size Datasets"): each epoch cycles through a randomly
permuted partition of X into mini-batches to simulate i.i.d. sampling
of p(x), and A_t / B_t are accumulated across the whole run (not reset
per epoch) using the mini-batch forgetting-factor update of Eq. 11.

Frozen-atom support (``D_frozen``) mirrors "Frozen K-SVD" as
implemented in ``KSVD`` (Carroll et al. 2017, Sec. III-A): sparse
coding is always performed jointly over ``[D_frozen | D_active]``, but
only the non-frozen columns are ever updated by Algorithm 2 or
replaced by atom-purging.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from reppi.base import BaseDictionaryLearner
from reppi.dictionary.bcd.utils import bcd_dictionary_update, update_forgetting_factor
from reppi.dictionary.ksvd.utils import _clear_dict
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.utils import _check_dict_normalized
from reppi.sparse.lars.lasso import LARSLasso
from reppi.sparse.utils import col_norms_squared, normalize_columns, rep_error_squared

_CHECKPOINT_FILENAME = "bcd_checkpoint.npz"


class BCD(BaseDictionaryLearner):
    """
    Online dictionary learning via Block-Coordinate Descent.

    Alternates, per mini-batch:
      1. Sparse coding — encode the mini-batch over the current D via
         LARS-Lasso.
      2. Statistics accumulation — fold the mini-batch's coefficients
         into running A, B matrices (Eq. 11).
      3. Dictionary update — warm-started BCD sweep(s) over A, B (Eq. 10).

    Parameters
    ----------
    n_components : int
        Number of dictionary atoms this instance learns. When
        ``D_frozen`` is supplied to ``fit()``, this is the count of
        *new* atoms only.
    lambda_ : float or None
        L1 regularization weight for the Lasso sparse-coding
        sub-problem (Eq. 2). If None (default), uses the paper's
        convention ``1.2 / sqrt(n_features)`` (Sec 5.1).
    batch_size : int
        Mini-batch size eta (default 256, per the paper's empirical
        finding for natural image patches).
    n_iter : int
        Number of epochs (full passes over the training set).
    max_dict_iter : int
        Maximum BCD sweeps per dictionary-update call. Default 1,
        matching the paper's finding that a single sweep suffices
        given warm restart (Sec 3.3).
    dict_tol : float
        Early-stop tolerance for BCD sweeps (max column change).
    mu_thresh : float
        Mutual-incoherence threshold in (0, 1] for atom purging.
        Set to 1.0 to disable (default 0.99).
    random_state : int or None
        Seed for reproducible atom initialisation and mini-batch order.
    verbose : bool
        Print per-epoch progress (default False).

    Attributes
    ----------
    D_ : np.ndarray, shape (n_features, n_frozen + n_components)
        Learned dictionary (set after fit()). Includes any ``D_frozen``
        columns, unchanged, as its leading columns.
    Gamma_ : np.ndarray, shape (n_frozen + n_components, n_samples)
        Sparse codes for the training data from the final epoch.
    errors_ : list of float
        Per-epoch RMSE on the training data.
    lambda_used_ : float
        The lambda value actually used (resolved default included).
    """

    def __init__(
        self,
        n_components: int,
        lambda_: float | None = None,
        batch_size: int = 256,
        n_iter: int = 10,
        max_dict_iter: int = 1,
        dict_tol: float = 1e-6,
        mu_thresh: float = 0.99,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if lambda_ is not None and lambda_ < 0:
            raise ValueError("lambda_ must be >= 0.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if max_dict_iter < 1:
            raise ValueError("max_dict_iter must be >= 1.")

        self.n_components = n_components
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.max_dict_iter = max_dict_iter
        self.dict_tol = dict_tol
        self.mu_thresh = mu_thresh
        self.random_state = random_state
        self.verbose = verbose

        # Set after fit
        self.D_: np.ndarray | None = None
        self.Gamma_: np.ndarray | None = None
        self.errors_: list[float] = []
        self.lambda_used_: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        D_init: np.ndarray | None = None,
        D_frozen: np.ndarray | None = None,
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> "BCD":
        """
        Learn a dictionary from training signals.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        D_init : np.ndarray or None, shape (n_features, n_components)
            Optional initial dictionary for the *new* (non-frozen) atoms
            only. If None, random training signals are chosen. Ignored
            when resuming from an existing checkpoint.
        D_frozen : np.ndarray or None, shape (n_features, n_frozen_atoms)
            Optional pre-trained atoms to prepend to the dictionary and
            hold constant through every epoch. Must already have
            unit-norm columns (validated). Signals are still
            sparse-coded jointly over the full ``[D_frozen | D_active]``
            dictionary; only the ``n_components`` new atoms are ever
            touched by the BCD update or by incoherence-based
            replacement in ``_clear_dict``.
        checkpoint_dir : str or None
            If given, a checkpoint is written to
            ``<checkpoint_dir>/bcd_checkpoint.npz`` after every epoch.
        resume : bool
            If True (default) and a checkpoint is found, training
            resumes from it.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        n_features, n_samples = X.shape
        rng = np.random.RandomState(self.random_state)

        lambda_used = (
            self.lambda_ if self.lambda_ is not None else 1.2 / np.sqrt(n_features)
        )
        self.lambda_used_ = lambda_used

        if D_frozen is not None:
            D_frozen = np.asarray(D_frozen, dtype=np.float32)
            if D_frozen.shape[0] != n_features:
                raise DictionaryLearningError(
                    f"D_frozen has {D_frozen.shape[0]} features, but X has "
                    f"{n_features} features."
                )
            _check_dict_normalized(D_frozen)
        n_frozen = 0 if D_frozen is None else D_frozen.shape[1]

        checkpoint_path = None
        start_epoch = 0
        D = None
        A = None
        B = None
        theta = 0.0
        self.errors_ = []

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, _CHECKPOINT_FILENAME)

            if resume and os.path.exists(checkpoint_path):
                (D, A, B, theta, self.errors_, start_epoch) = self._load_checkpoint(
                    checkpoint_path, X, n_frozen, D_frozen, lambda_used
                )
                if self.verbose:
                    print(
                        f"Resuming from checkpoint at epoch {start_epoch}/"
                        f"{self.n_iter} ({checkpoint_path})"
                    )

        if D is None:
            D_active = self._init_dict(X, D_init, rng)
            D = np.hstack([D_frozen, D_active]) if D_frozen is not None else D_active

        n_total = D.shape[1]
        if A is None:
            A = np.zeros((n_total, n_total), dtype=np.float32)
            B = np.zeros((n_features, n_total), dtype=np.float32)

        coder = LARSLasso(alpha=lambda_used, check_dict=False)
        Gamma = None

        for epoch in range(start_epoch, self.n_iter):
            perm = rng.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                Xb = X[:, idx]
                eta = Xb.shape[1]

                G = D.T @ D
                Alpha = coder.encode(Xb, D, G=G)

                theta, beta = update_forgetting_factor(theta, eta)
                A = beta * A + Alpha @ Alpha.T
                B = beta * B + Xb @ Alpha.T

                D = bcd_dictionary_update(
                    D, A, B, n_frozen, self.max_dict_iter, self.dict_tol
                )

            # --- epoch-end bookkeeping: error, purging ---
            G = D.T @ D
            Gamma = coder.encode(X, D, G=G)
            err = np.float32(np.sqrt(rep_error_squared(X, D, Gamma).sum() / X.size))
            self.errors_.append(err)

            sample_usage = np.count_nonzero(Gamma, axis=0)
            unused = np.where(sample_usage == 0)[0]
            if unused.size == 0:
                unused = np.arange(n_samples)
            replaced = np.zeros(n_total, dtype=bool)
            D, _ = _clear_dict(
                D, Gamma, X, self.mu_thresh, unused, replaced, frozen_atoms=n_frozen
            )

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.n_iter}  RMSE={err:.6f}")

            if checkpoint_path is not None:
                self._save_checkpoint(
                    checkpoint_path, X, D, A, B, theta, epoch + 1, n_frozen, lambda_used
                )

        self.D_ = D
        self.Gamma_ = Gamma
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode X using the learned dictionary."""
        if self.D_ is None:
            raise DictionaryLearningError("Call fit() before transform().")
        coder = LARSLasso(alpha=self.lambda_used_, check_dict=False)
        return coder.encode(X, self.D_)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_dict(
        self,
        X: np.ndarray,
        D_init: np.ndarray | None,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        Build the initial dictionary for this instance's *own* atoms.

        Only ever constructs ``self.n_components`` columns — any frozen
        atoms are handled separately by ``fit()`` and concatenated
        afterward, never passed through this method.
        """
        n_features, n_samples = X.shape
        k = self.n_components

        if D_init is not None:
            D = np.asarray(D_init, dtype=np.float32)
            if D.shape != (n_features, k):
                raise DictionaryLearningError(
                    f"D_init shape {D.shape} does not match "
                    f"(n_features={n_features}, n_components={k})."
                )
        else:
            valid = np.where(col_norms_squared(X) > 1e-6)[0]
            if len(valid) < k:
                raise DictionaryLearningError(
                    "Not enough non-zero training signals to initialise the dictionary."
                )
            chosen = rng.choice(valid, size=k, replace=False)
            D = X[:, chosen].copy()

        return normalize_columns(D)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
            self,
            path: str,
            X: np.ndarray,
            D: np.ndarray,
            A: np.ndarray,
            B: np.ndarray,
            theta: float,
            completed_epoch: int,
            n_frozen: int,
            lambda_used: float,
        ) -> None:
            """
            Atomically write the training state to ``path`` (temp file in the
            same directory, then os.replace into place).
            """
            directory = os.path.dirname(path) or "."
            fd, tmp_path = tempfile.mkstemp(
                dir=directory, prefix=".bcd_checkpoint_", suffix=".npz.tmp"
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    np.savez(
                        f,
                        D=D,
                        A=A,
                        B=B,
                        theta=theta,
                        errors_=np.asarray(self.errors_, dtype=np.float32),
                        completed_epoch=completed_epoch,
                        n_iter=self.n_iter,
                        n_components=self.n_components,
                        lambda_used=lambda_used,
                        batch_size=self.batch_size,
                        n_features=X.shape[0],
                        n_samples=X.shape[1],
                        n_frozen=n_frozen,
                    )
                os.replace(tmp_path, path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def _load_checkpoint(
        self,
        path: str,
        X: np.ndarray,
        n_frozen: int,
        D_frozen: np.ndarray | None,
        lambda_used: float,
    ):
        """
        Load and validate a checkpoint against the current config, X,
        and the frozen-dictionary configuration for this call. Raises
        DictionaryLearningError on any mismatch.
        """
        with np.load(path) as data:
            n_features = int(data["n_features"])
            n_samples = int(data["n_samples"])
            n_components = int(data["n_components"])
            n_iter = int(data["n_iter"])
            batch_size = int(data["batch_size"])
            lambda_ckpt = np.float32(data["lambda_used"])
            completed_epoch = int(data["completed_epoch"])
            n_frozen_ckpt = int(data["n_frozen"]) if "n_frozen" in data else 0

            if (n_features, n_samples) != X.shape:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} was computed on data of shape "
                    f"{(n_features, n_samples)}, but X has shape {X.shape}."
                )
            if n_components != self.n_components:
                raise DictionaryLearningError(
                    f"Checkpoint n_components={n_components} does not match "
                    f"BCD.n_components={self.n_components}."
                )
            if n_iter != self.n_iter:
                raise DictionaryLearningError(
                    f"Checkpoint was created with n_iter={n_iter}, but this "
                    f"BCD instance has n_iter={self.n_iter}."
                )
            if batch_size != self.batch_size:
                raise DictionaryLearningError(
                    f"Checkpoint batch_size={batch_size} does not match "
                    f"BCD.batch_size={self.batch_size}."
                )
            if not np.isclose(lambda_ckpt, lambda_used, rtol=1e-8, atol=1e-12):
                raise DictionaryLearningError(
                    f"Checkpoint lambda={lambda_ckpt} does not match the "
                    f"resolved lambda={lambda_used} for this fit() call."
                )
            if completed_epoch >= self.n_iter:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} already completed all "
                    f"{self.n_iter} epochs; nothing to resume."
                )
            if n_frozen_ckpt != n_frozen:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} was trained with {n_frozen_ckpt} "
                    f"frozen atoms, but this fit() call has n_frozen={n_frozen}."
                )

            D = data["D"].copy()
            A = data["A"].copy()
            B = data["B"].copy()
            theta = np.float32(data["theta"])
            errors_ = list(data["errors_"])

        if n_frozen > 0:
            if D_frozen is None:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} expects {n_frozen} frozen atoms, "
                    "but D_frozen=None was passed to this fit() call."
                )
            if not np.allclose(D[:, :n_frozen], D_frozen, atol=1e-6):
                raise DictionaryLearningError(
                    f"Checkpoint at {path}'s frozen atoms do not match the "
                    "D_frozen passed to this fit() call. Resuming would "
                    "silently mix training states from different frozen "
                    "dictionaries."
                )

        return D, A, B, theta, errors_, completed_epoch