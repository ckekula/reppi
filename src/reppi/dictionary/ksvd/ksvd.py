"""
K-SVD dictionary learning.

Implements the K-SVD algorithm described in:
    Aharon, Elad, Bruckstein. "The K-SVD: An Algorithm for Designing
    Overcomplete Dictionaries for Sparse Representation".
    IEEE Trans. Signal Processing, 54(11), 2006.

Batch-OMP integration follows:
    Elad, Rubinstein, Zibulevsky. "Efficient Implementation of the K-SVD
    Algorithm using Batch Orthogonal Matching Pursuit". Technion TR, 2008.

Frozen-atom support (``D_frozen``) implements "Frozen K-SVD" as described
in:
    Carroll, Whitaker, Dayley, Anderson. "Outlier Learning via Augmented
    Frozen Dictionaries". IEEE/ACM TASLP, 25(6), 2017, Sec. III-A: the
    atom-update loop (Algorithm 1, line 3) is restricted to non-frozen
    columns; everything else — sparse coding, error computation — still
    operates over the full, joint dictionary every iteration.
"""

from __future__ import annotations

import os
import tempfile
import logging
import numpy as np

from reppi.base import BaseDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp.omp import OMP, batch_omp
from reppi.sparse.utils import _check_dict_normalized
from reppi.sparse.utils import col_norms_squared, normalize_columns, rep_error_squared

from reppi.dictionary.ksvd.utils import _optimize_atom, _clear_dict

_CHECKPOINT_FILENAME = "ksvd_checkpoint.npz"

logger = logging.getLogger(__name__)

class KSVD(BaseDictionaryLearner):
    """
    K-SVD dictionary learner.

    Alternates between:
      1. Sparse coding — encode each training signal over the current D.
      2. Dictionary update — update each atom (and its coefficients) via a
         rank-1 approximation of the residual matrix.

    Parameters
    ----------
    n_components : int
        Number of dictionary atoms this instance learns. When ``D_frozen``
        is supplied to ``fit()``, this is the count of *new* atoms only —
        the frozen atoms are additional, held-constant columns and are not
        counted here.
    n_nonzero_coefs : int
        Sparsity target T: each signal is represented with at most T atoms.
    n_iter : int
        Number of K-SVD iterations (default 10).
    exact_svd : bool
        If True, use full SVD for the atom update (exact K-SVD).
        If False (default), use the faster approximate update.
    mu_thresh : float
        Mutual-incoherence threshold in (0, 1].  Atoms whose pairwise
        correlation exceeds this value are replaced.  Set to 1.0 to
        disable (default 0.99).
    mem_usage : str
        One of 'high', 'normal' (default), 'low'.
        Controls whether G = D'D (and DtX = D'X) are precomputed.
    random_state : int or None
        Seed for reproducible atom initialisation.
    verbose : bool
        Print iteration progress (default False).

    Attributes
    ----------
    D_ : np.ndarray, shape (n_features, n_frozen + n_components)
        Learned dictionary (set after fit()). Includes any ``D_frozen``
        columns passed to ``fit()``, unchanged, as its leading columns.
    Gamma_ : np.ndarray, shape (n_frozen + n_components, n_samples)
        Sparse codes for the training data from the final iteration
        (set after fit()). Exposed so callers that need the training
        codes (e.g. LC-KSVD, which reuses this class on an augmented
        system) don't have to re-run sparse coding.
    errors_ : list of float
        Per-iteration RMSE on the training data.
    """

    def __init__(
        self,
        n_components: int,
        n_nonzero_coefs: int,
        n_iter: int = 10,
        exact_svd: bool = False,
        mu_thresh: float = 0.99,
        mem_usage: str = "normal",
        random_state: int | None = None,
        verbose: bool = True,
    ) -> None:
        if mem_usage not in ("high", "normal", "low"):
            raise ValueError("mem_usage must be 'high', 'normal', or 'low'.")
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs
        self.n_iter = n_iter
        self.exact_svd = exact_svd
        self.mu_thresh = mu_thresh
        self.mem_usage = mem_usage
        self.random_state = random_state
        self.verbose = verbose

        # Set after fit
        self.D_: np.ndarray | None = None
        self.Gamma_: np.ndarray | None = None
        self.errors_: list[float] = []

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
    ) -> "KSVD":
        """
        Learn a dictionary from training signals.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        D_init : np.ndarray or None, shape (n_features, n_components)
            Optional initial dictionary for the *new* (non-frozen) atoms
            only. If None, random training signals are chosen as initial
            atoms. Ignored when resuming from an existing checkpoint.
        D_frozen : np.ndarray or None, shape (n_features, n_frozen_atoms)
            Optional pre-trained atoms to prepend to the dictionary and
            hold constant through every iteration. Must already have
            unit-norm columns (validated). Signals are still sparse-coded
            jointly over the full ``[D_frozen | D_active]`` dictionary at
            every iteration; only the ``n_components`` new atoms are ever
            updated by the atom-update step or by incoherence-based
            replacement in ``_clear_dict`` — ``D_frozen`` is never
            modified, not even at initialisation (it is concatenated
            as-is, never passed through ``normalize_columns``). See
            Carroll et al. 2017, Sec. III-A ("Frozen K-SVD").
        checkpoint_dir : str or None
            If given, a checkpoint is written to
            ``<checkpoint_dir>/ksvd_checkpoint.npz`` after every iteration,
            overwriting the previous one. The directory is created if it
            does not exist.
        resume : bool
            If True (default) and a checkpoint is found in
            ``checkpoint_dir``, training resumes from it. If False, any
            existing checkpoint in ``checkpoint_dir`` is ignored and
            overwritten.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)

        if D_frozen is not None:
            D_frozen = np.asarray(D_frozen, dtype=float)
            if D_frozen.shape[0] != X.shape[0]:
                raise DictionaryLearningError(
                    f"D_frozen has {D_frozen.shape[0]} features, but X has "
                    f"{X.shape[0]} features."
                )
            _check_dict_normalized(D_frozen)
        n_frozen = 0 if D_frozen is None else D_frozen.shape[1]

        if X.shape[1] < self.n_components:
            raise DictionaryLearningError(
                f"n_samples={X.shape[1]} is less than n_components={self.n_components}. "
            )

        checkpoint_path = None
        start_iter = 0
        D = None
        Gamma = None
        self.errors_ = []

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, _CHECKPOINT_FILENAME)

            if resume and os.path.exists(checkpoint_path):
                (
                    D,
                    Gamma,
                    unused,
                    replaced,
                    self.errors_,
                    start_iter,
                ) = self._load_checkpoint(checkpoint_path, X, n_frozen, D_frozen)
                if self.verbose:
                    logger.info(
                        f"Resuming from checkpoint at iteration {start_iter}/"
                        f"{self.n_iter} ({checkpoint_path})"
                    )

        if D is None:
            D_active = self._init_dict(X, D_init, rng)
            D = np.hstack([D_frozen, D_active]) if D_frozen is not None else D_active

        n_total = D.shape[1]
        unused = np.arange(X.shape[1])
        replaced = np.zeros(n_total, dtype=bool)

        for it in range(start_iter, self.n_iter):
            G = D.T @ D if self.mem_usage in ("high", "normal") else None
            Gamma = self._sparse_code(X, D, G)

            unused = np.arange(X.shape[1])
            replaced = np.zeros(n_total, dtype=bool)

            for j in range(n_frozen, n_total):
                D[:, j], gamma_j, idx, unused, replaced = _optimize_atom(
                    X, D, j, Gamma, unused, replaced, self.exact_svd
                )
                Gamma[j, idx] = gamma_j

            err = float(np.sqrt(rep_error_squared(X, D, Gamma).sum() / X.size))
            self.errors_.append(err)

            D, _ = _clear_dict(
                D, Gamma, X, self.mu_thresh, unused, replaced, frozen_atoms=n_frozen
            )

            if self.verbose:
                logger.info(f"Iter {it + 1}/{self.n_iter}  RMSE={err:.6f}")

            if checkpoint_path is not None:
                self._save_checkpoint(
                    checkpoint_path, X, D, Gamma, unused, replaced, it + 1, n_frozen
                )

        self.D_ = D
        self.Gamma_ = Gamma
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode X using the learned dictionary."""
        if self.D_ is None:
            raise DictionaryLearningError("Call fit() before transform().")
        logger.debug("Sparse Coding %d signals with learned dictionary of shape %s", X.shape[1], self.D_.shape)
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
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

        Note: this only ever constructs ``self.n_components`` columns —
        any frozen atoms are handled separately by ``fit()`` and
        concatenated afterward, never passed through this method (so they
        are never renormalised or otherwise touched at initialisation).
        """
        n_features, n_samples = X.shape
        k = self.n_components

        if D_init is not None:
            D = np.asarray(D_init, dtype=float)
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

    def _sparse_code(
        self,
        X: np.ndarray,
        D: np.ndarray,
        G: np.ndarray | None,
    ) -> np.ndarray:
        if self.mem_usage == "high" and G is not None:
            return batch_omp(D.T @ X, G, self.n_nonzero_coefs)
        logger.debug("Sparse coding %d signals with dictionary of shape %s", X.shape[1], D.shape)
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, D, G=G)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        path: str,
        X: np.ndarray,
        D: np.ndarray,
        Gamma: np.ndarray,
        unused: np.ndarray,
        replaced: np.ndarray,
        completed_iter: int,
        n_frozen: int,
    ) -> None:
        """
        Atomically write the training state to ``path``.

        Written to a temp file in the same directory first, then moved
        into place with os.replace, so an abrupt stop mid-write can never
        leave a corrupt/truncated checkpoint at ``path``.
        """
        directory = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=".ksvd_checkpoint_", suffix=".npz.tmp"
        )
        try:
            # Pass the open file descriptor itself (not the path string) to
            # np.savez: given a string, np.savez silently appends '.npz' if
            # the name doesn't already end with exactly that extension —
            # since tmp_path ends in '.npz.tmp', that would write the real
            # data to an orphaned "<tmp_path>.npz" file while leaving the
            # empty file mkstemp created at tmp_path (and hence, after
            # os.replace, at `path`) untouched. Writing through the fd
            # avoids the extension-mangling entirely.
            with os.fdopen(fd, "wb") as f:
                np.savez(
                    f,
                    D=D,
                    Gamma=Gamma,
                    unused=unused,
                    replaced=replaced,
                    errors_=np.asarray(self.errors_, dtype=float),
                    completed_iter=completed_iter,
                    n_iter=self.n_iter,
                    n_components=self.n_components,
                    n_nonzero_coefs=self.n_nonzero_coefs,
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
    ):
        """
        Load and validate a checkpoint against the current config, X, and
        the frozen-dictionary configuration for this call.

        Raises DictionaryLearningError on any mismatch, rather than
        silently resuming with an incompatible state.
        """
        with np.load(path) as data:
            n_features = int(data["n_features"])
            n_samples = int(data["n_samples"])
            n_components = int(data["n_components"])
            n_nonzero_coefs = int(data["n_nonzero_coefs"])
            n_iter = int(data["n_iter"])
            completed_iter = int(data["completed_iter"])
            n_frozen_ckpt = int(data["n_frozen"]) if "n_frozen" in data else 0

            if (n_features, n_samples) != X.shape:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} was computed on data of shape "
                    f"{(n_features, n_samples)}, but X has shape {X.shape}."
                )
            if n_components != self.n_components:
                raise DictionaryLearningError(
                    f"Checkpoint n_components={n_components} does not match "
                    f"KSVD.n_components={self.n_components}."
                )
            if n_nonzero_coefs != self.n_nonzero_coefs:
                raise DictionaryLearningError(
                    f"Checkpoint n_nonzero_coefs={n_nonzero_coefs} does not "
                    f"match KSVD.n_nonzero_coefs={self.n_nonzero_coefs}."
                )
            if n_iter != self.n_iter:
                raise DictionaryLearningError(
                    f"Checkpoint was created with n_iter={n_iter}, but this "
                    f"KSVD instance has n_iter={self.n_iter}."
                )
            if completed_iter >= self.n_iter:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} already completed all "
                    f"{self.n_iter} iterations; nothing to resume."
                )
            if n_frozen_ckpt != n_frozen:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} was trained with {n_frozen_ckpt} "
                    f"frozen atoms, but this fit() call has n_frozen={n_frozen}."
                )

            D = data["D"].copy()
            Gamma = data["Gamma"].copy()
            unused = data["unused"].copy()
            replaced = data["replaced"].copy()
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

        return D, Gamma, unused, replaced, errors_, completed_iter