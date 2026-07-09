"""
Label Consistent K-SVD (LC-KSVD) dictionary learning.

Implements LC-KSVD1 and LC-KSVD2 as described in:
    Zhuolin Jiang, Zhe Lin, Larry S. Davis.
    "Learning A Discriminative Dictionary for Sparse Coding via Label
     Consistent K-SVD", CVPR 2011.

LC-KSVD augments the standard K-SVD objective with:
  - A label-consistency term (LC-KSVD1) that encourages atoms associated
    with the same class to produce similar sparse codes.
  - An additional linear classifier term (LC-KSVD2) that jointly trains a
    classifier W alongside the dictionary.

Optimization problems
---------------------
LC-KSVD1:
    min_{D, A, X}  ||Y - DX||_F^2 + alpha * ||Q - AX||_F^2
    s.t.  ||x_i||_0 <= T

LC-KSVD2:
    min_{D, A, W, X}  ||Y - DX||_F^2 + alpha * ||Q - AX||_F^2
                      + beta * ||H - WX||_F^2
    s.t.  ||x_i||_0 <= T

where:
  Y = training signals
  D = dictionary
  X = sparse codes
  Q = label-consistent sparse code targets (binary atom-class assignments)
  A = linear mapping for Q-consistency
  W = linear classifier
  H = class label matrix (one-hot per column)
  alpha, beta = trade-off weights

Implementation note
--------------------
Both variants reduce to running ordinary K-SVD on an augmented system:

    Y_aug = [Y ; sqrt(alpha)*Q ; sqrt(beta)*H]     (rows stacked)
    D_aug = [D ; sqrt(alpha)*A ; sqrt(beta)*W]

Each outer iteration here delegates to `KSVD` (with n_iter=1) on the
augmented system and then splits the result back into D, A, W. This
mirrors the construction in the original paper and keeps a single
source of truth for the K-SVD atom-update step.

Frozen-atom support
--------------------
When ``D_frozen`` is supplied to ``fit()``, this class generalises
Carroll et al. 2017's frozen-dictionary idea to LC-KSVD: ``D_frozen``'s
atoms (and, since they live in the same augmented matrix, their
corresponding A/W columns) are held constant while ``n_components`` new
atoms are learned jointly alongside them every outer iteration. This is
achieved by passing ``D_frozen`` straight through to the inner KSVD
atom-updater on every call — freezing a column of the augmented
dictionary automatically freezes the corresponding column of A and W too,
since the atom-update loop simply never touches that column, regardless
of which row-block (D, A, or W) it belongs to.

Usage
-----
For LC-KSVD1::

    model = LCKSVD(
        n_components=570,
        n_nonzero_coefs=30,
        alpha=4.0,
        variant="lcksvd1",
    )
    model.fit(X_train, H_train)
    predictions = model.predict(X_test)

For LC-KSVD2::

    model = LCKSVD(
        n_components=570,
        n_nonzero_coefs=30,
        alpha=4.0,
        beta=2.0,
        variant="lcksvd2",
    )
    model.fit(X_train, H_train)
    predictions = model.predict(X_test)
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from reppi.base import BaseDiscriminativeDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp.omp import OMP
from reppi.sparse.omp.utils import _check_dict_normalized
from reppi.sparse.utils import normalize_columns, rep_error_squared
from reppi.dictionary.ksvd.ksvd import KSVD

from reppi.dictionary.lc_ksvd.utils import initialization4lcksvd, _augment_data

_CHECKPOINT_FILENAME = "lc_ksvd_checkpoint.npz"


class LCKSVD(BaseDiscriminativeDictionaryLearner):
    """
    Label Consistent K-SVD dictionary learner (LC-KSVD1 and LC-KSVD2).

    Parameters
    ----------
    n_components : int
        Number of dictionary atoms this instance learns. When ``D_frozen``
        is supplied to ``fit()``, this is the count of *new* atoms only.
    n_nonzero_coefs : int
        Sparsity level T.
    alpha : float
        Weight for the label-consistency term (sqrt_alpha in the paper).
    beta : float
        Weight for the classifier term (sqrt_beta; LC-KSVD2 only).
    variant : {'lcksvd1', 'lcksvd2'}
        Which variant to train.
    n_iter : int
        Number of LC-KSVD iterations (default 50).
    n_iter_init : int
        K-SVD iterations for the initialisation phase (default 20).
    exact_svd : bool
        Use exact SVD in the atom-update step (slower but slightly better).
    mu_thresh : float
        Mutual-incoherence threshold (default 0.99).
    random_state : int or None
    verbose : bool

    Attributes
    ----------
    D_ : np.ndarray, shape (n_features, n_frozen + n_components)
        Learned dictionary. Includes any frozen atoms passed to ``fit()``
        as its leading, unchanged columns.
    W_ : np.ndarray, shape (n_classes, n_frozen + n_components)
        Learned linear classifier weights.
    A_ : np.ndarray, shape (n_frozen + n_components, n_frozen + n_components)
        Learned label-consistency transform.
    errors_ : list of float
        Per-iteration RMSE on training data (measured on the original,
        un-augmented X, against the full combined dictionary).
    class_boundaries_ : dict[int, tuple[int, int]]
        Atom ranges per class in the full combined ``D_``. Includes any
        ``frozen_class_boundaries`` passed to ``fit()`` unchanged, plus
        this call's classes offset by the frozen atom count.
    """

    def __init__(
        self,
        n_components: int,
        n_nonzero_coefs: int,
        alpha: float = 4.0,
        beta: float = 2.0,
        variant: str = "lcksvd2",
        n_iter: int = 50,
        n_iter_init: int = 20,
        exact_svd: bool = False,
        mu_thresh: float = 0.99,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if variant not in ("lcksvd1", "lcksvd2"):
            raise ValueError("variant must be 'lcksvd1' or 'lcksvd2'.")
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.n_iter = n_iter
        self.n_iter_init = n_iter_init
        self.exact_svd = exact_svd
        self.mu_thresh = mu_thresh
        self.random_state = random_state
        self.verbose = verbose

        self.D_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.A_: np.ndarray | None = None
        self.errors_: list[float] = []
        self.class_boundaries_: dict[int, tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        H: np.ndarray,
        D_frozen: np.ndarray | None = None,
        frozen_class_boundaries: dict[int, tuple[int, int]] | None = None,
        D_init: np.ndarray | None = None,
        A_init: np.ndarray | None = None,
        W_init: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> "LCKSVD":
        """
        Learn a discriminative dictionary from labelled training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
            Training signals.
        H : np.ndarray, shape (n_classes, n_samples)
            One-hot label matrix.
        D_frozen : np.ndarray or None, shape (n_features, n_frozen_atoms)
            Pre-trained atoms from earlier incremental stages, held
            constant throughout fitting. Must already have unit-norm
            columns (validated). Only the ``n_components`` new atoms
            declared in ``__init__`` are learned; they are trained
            jointly alongside the frozen ones every outer iteration (not
            on a one-shot residual — see module docstring). ``self.D_``
            after fitting is the full ``[D_frozen | D_new]`` dictionary.
        frozen_class_boundaries : dict or None
            ``class_boundaries_`` from earlier frozen stages. Merged
            unchanged into ``self.class_boundaries_``, alongside this
            call's own classes offset by ``D_frozen.shape[1]``. Ignored
            if ``D_frozen`` is None.
        D_init : np.ndarray or None
            Initial dictionary for the *new* atoms only (never includes
            D_frozen). If None, a K-SVD initialisation is run. Ignored
            when resuming from an existing checkpoint.
        A_init : np.ndarray or None
            Initial label-consistency transform, sized to the full
            combined dictionary. Ignored when resuming.
        W_init : np.ndarray or None
            Initial classifier weights (required / used for LC-KSVD2),
            sized to the full combined dictionary. Ignored when resuming.
        Q : np.ndarray or None
            Label-consistent target matrix, sized to the full combined
            dictionary. Computed from H (and D_frozen) if None. Ignored
            when resuming.
        checkpoint_dir : str or None
            If given, a checkpoint of the *outer* LC-KSVD loop is written
            to ``<checkpoint_dir>/lc_ksvd_checkpoint.npz`` after every
            outer iteration, overwriting the previous one. The directory
            is created if it does not exist. This is independent of, and
            not passed down to, the inner per-iteration KSVD instance.
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
        H = np.asarray(H, dtype=float)
        n_features, n_samples = X.shape
        n_classes = H.shape[0]

        if D_frozen is not None:
            D_frozen = np.asarray(D_frozen, dtype=float)
            if D_frozen.shape[0] != n_features:
                raise DictionaryLearningError(
                    f"D_frozen has {D_frozen.shape[0]} features but X has "
                    f"{n_features} features."
                )
            _check_dict_normalized(D_frozen)
        n_frozen = 0 if D_frozen is None else D_frozen.shape[1]

        checkpoint_path = None
        start_iter = 0
        D = A = W = Q_loaded = None
        self.errors_ = []

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, _CHECKPOINT_FILENAME)

            if resume and os.path.exists(checkpoint_path):
                (
                    D, A, W, Q_loaded, self.errors_, start_iter,
                ) = self._load_checkpoint(checkpoint_path, X, H, n_frozen, D_frozen)
                if self.verbose:
                    print(
                        f"Resuming from checkpoint at outer iteration "
                        f"{start_iter}/{self.n_iter} ({checkpoint_path})"
                    )

        # ---- Initialisation (skipped if resuming from a checkpoint) ----
        if D is None:
            if D_init is None or A_init is None or W_init is None or Q is None:
                if self.verbose:
                    print("Running initialisation K-SVD...")
                D_init, A_init, W_init, Q = initialization4lcksvd(
                    X, H,
                    self.n_components,
                    self.n_iter_init,
                    self.n_nonzero_coefs,
                    D_frozen=D_frozen,
                    random_state=self.random_state,
                    verbose=self.verbose,
                )

            D_active = normalize_columns(D_init.copy())
            D = np.hstack([D_frozen, D_active]) if D_frozen is not None else D_active
            A = A_init.copy()
            W = W_init.copy()
        else:
            Q = Q_loaded

        sqrt_alpha = np.sqrt(self.alpha)
        sqrt_beta = np.sqrt(self.beta)

        use_classifier_term = (self.variant == "lcksvd2")

        # ---- Build augmented training data ----
        # Y_aug = [X ; sqrt_alpha*Q ; sqrt_beta*H]  (LC-KSVD2)
        # Y_aug = [X ; sqrt_alpha*Q]                 (LC-KSVD1)
        H_aug = H if use_classifier_term else None
        X_aug, _, _ = _augment_data(X, Q, H_aug, sqrt_alpha, sqrt_beta)

        # A single K-SVD instance, reused every outer iteration, drives the
        # atom update on the augmented system. Only n_components new atoms
        # are ever updated — D_frozen (and hence its corresponding rows in
        # the A/W blocks of D_aug) is passed through unchanged every call,
        # so it is trained jointly with the new atoms rather than on a
        # one-shot residual.
        atom_updater = KSVD(
            n_components=self.n_components,
            n_nonzero_coefs=self.n_nonzero_coefs,
            n_iter=1,
            exact_svd=self.exact_svd,
            mu_thresh=self.mu_thresh,
            mem_usage="normal",
            random_state=self.random_state,
            verbose=False,
        )

        for it in range(start_iter, self.n_iter):

            # ---- Build augmented dictionary ----
            # D_aug = [D ; sqrt_alpha*A ; sqrt_beta*W]
            D_aug = self._build_aug_dict(D, A, W, sqrt_alpha, sqrt_beta, use_classifier_term)
            D_aug_norm = normalize_columns(D_aug)

            D_aug_frozen = D_aug_norm[:, :n_frozen] if n_frozen > 0 else None
            D_aug_active_init = D_aug_norm[:, n_frozen:]

            # ---- Sparse code + single atom-update pass on the augmented
            #      system, delegated to KSVD. Frozen columns (if any) are
            #      passed through so they participate in the joint coding
            #      but are never updated. ----
            atom_updater.fit(
                X_aug,
                D_init=D_aug_active_init,
                D_frozen=D_aug_frozen,
                checkpoint_dir=None,
            )
            D_aug_updated = atom_updater.D_
            Gamma = atom_updater.Gamma_

            # De-augment: extract D, A, W from the updated augmented dict
            D, A, W = self._split_aug_dict(
                D_aug_updated, n_features, n_classes, sqrt_alpha, sqrt_beta, use_classifier_term
            )

            # _split_aug_dict already re-normalises D's columns using the
            # combined D/A/W column norm, which is mathematically
            # self-cancelling for frozen columns (their D sub-block always
            # comes back out exactly as D_frozen, given D_frozen itself is
            # unit-norm) — but relying on that cancellation to hold exactly
            # over many outer iterations invites floating-point drift.
            # Hard-pin the frozen block to eliminate that risk entirely.
            if n_frozen > 0:
                D[:, :n_frozen] = D_frozen

            # ---- Track RMSE on original X ----
            err = float(np.sqrt(rep_error_squared(X, D, Gamma).sum() / X.size))
            self.errors_.append(err)

            if self.verbose:
                print(f"[{self.variant.upper()}] Iter {it + 1}/{self.n_iter}  RMSE={err:.6f}")

            if checkpoint_path is not None:
                self._save_checkpoint(
                    checkpoint_path, X, H, D, A, W, Q, it + 1, n_frozen
                )

        self.D_ = D
        self.A_ = A

        if use_classifier_term:
            self.W_ = W
        else:
            # LC-KSVD1 has no classification-error term in its objective,
            # so W is not learned jointly with D/A. _split_aug_dict
            # only ever returns zeros for W when use_classifier_term is
            # False, so it must be fit here instead.
            coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
            Gamma_final = coder.encode(X, D)
            self.W_ = H @ np.linalg.pinv(Gamma_final)

        # Record per-class atom ranges. Only classes with samples present
        # in this call ("active classes") receive a share of this call's
        # n_components new atoms — matching initialization4lcksvd /
        # _build_label_consistent_target exactly, so Q's atom-class
        # assignment and class_boundaries_ never disagree. Frozen classes'
        # boundaries (from an earlier stage) are carried over unchanged.
        active_classes = [c for c in range(n_classes) if np.any(H[c, :] > 0)]
        n_active = len(active_classes)
        atoms_per_class = self.n_components // n_active

        boundaries: dict[int, tuple[int, int]] = (
            dict(frozen_class_boundaries) if frozen_class_boundaries else {}
        )
        for i, c in enumerate(active_classes):
            start_local = i * atoms_per_class
            end_local = (
                start_local + atoms_per_class if i < n_active - 1
                else self.n_components
            )
            boundaries[c] = (n_frozen + start_local, n_frozen + end_local)
        self.class_boundaries_ = boundaries

        return self

    # ------------------------------------------------------------------
    # Checkpointing (outer LC-KSVD loop)
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        path: str,
        X: np.ndarray,
        H: np.ndarray,
        D: np.ndarray,
        A: np.ndarray,
        W: np.ndarray,
        Q: np.ndarray,
        completed_iter: int,
        n_frozen: int,
    ) -> None:
        """
        Atomically write the outer LC-KSVD training state to ``path``.

        Written to a temp file in the same directory first, then moved
        into place with os.replace, so an abrupt stop mid-write can never
        leave a corrupt/truncated checkpoint at ``path``.
        """
        directory = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=".lc_ksvd_checkpoint_", suffix=".npz.tmp"
        )
        try:
            # See KSVD._save_checkpoint for why this writes through the fd
            # rather than passing tmp_path as a string to np.savez.
            with os.fdopen(fd, "wb") as f:
                np.savez(
                    f,
                    D=D,
                    A=A,
                    W=W,
                    Q=Q,
                    errors_=np.asarray(self.errors_, dtype=float),
                    completed_iter=completed_iter,
                    n_iter=self.n_iter,
                    n_components=self.n_components,
                    n_nonzero_coefs=self.n_nonzero_coefs,
                    alpha=self.alpha,
                    beta=self.beta,
                    variant=self.variant,
                    n_features=X.shape[0],
                    n_samples=X.shape[1],
                    n_classes=H.shape[0],
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
        H: np.ndarray,
        n_frozen: int,
        D_frozen: np.ndarray | None,
    ):
        """
        Load and validate a checkpoint against the current config, X, H,
        and the frozen-dictionary configuration for this call.

        Raises DictionaryLearningError on any mismatch, rather than
        silently resuming with an incompatible state.
        """
        with np.load(path, allow_pickle=True) as data:
            n_features = int(data["n_features"])
            n_samples = int(data["n_samples"])
            n_classes = int(data["n_classes"])
            n_components = int(data["n_components"])
            n_nonzero_coefs = int(data["n_nonzero_coefs"])
            n_iter = int(data["n_iter"])
            alpha = float(data["alpha"])
            beta = float(data["beta"])
            variant = str(data["variant"])
            completed_iter = int(data["completed_iter"])
            n_frozen_ckpt = int(data["n_frozen"]) if "n_frozen" in data else 0

            if (n_features, n_samples) != X.shape:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} was computed on data of shape "
                    f"{(n_features, n_samples)}, but X has shape {X.shape}."
                )
            if n_classes != H.shape[0]:
                raise DictionaryLearningError(
                    f"Checkpoint n_classes={n_classes} does not match "
                    f"H.shape[0]={H.shape[0]}."
                )
            if n_components != self.n_components:
                raise DictionaryLearningError(
                    f"Checkpoint n_components={n_components} does not match "
                    f"LCKSVD.n_components={self.n_components}."
                )
            if n_nonzero_coefs != self.n_nonzero_coefs:
                raise DictionaryLearningError(
                    f"Checkpoint n_nonzero_coefs={n_nonzero_coefs} does not "
                    f"match LCKSVD.n_nonzero_coefs={self.n_nonzero_coefs}."
                )
            if variant != self.variant:
                raise DictionaryLearningError(
                    f"Checkpoint was created with variant='{variant}', but "
                    f"this LCKSVD instance has variant='{self.variant}'."
                )
            if not np.isclose(alpha, self.alpha) or not np.isclose(beta, self.beta):
                raise DictionaryLearningError(
                    f"Checkpoint was created with alpha={alpha}, beta={beta}, "
                    f"but this LCKSVD instance has alpha={self.alpha}, "
                    f"beta={self.beta}."
                )
            if n_iter != self.n_iter:
                raise DictionaryLearningError(
                    f"Checkpoint was created with n_iter={n_iter}, but this "
                    f"LCKSVD instance has n_iter={self.n_iter}."
                )
            if completed_iter >= self.n_iter:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} already completed all "
                    f"{self.n_iter} outer iterations; nothing to resume."
                )
            if n_frozen_ckpt != n_frozen:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} was trained with {n_frozen_ckpt} "
                    f"frozen atoms, but this fit() call has n_frozen={n_frozen}."
                )

            D = data["D"].copy()
            A = data["A"].copy()
            W = data["W"].copy()
            Q = data["Q"].copy()
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

        return D, A, W, Q, errors_, completed_iter

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode X using the learned dictionary D.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)

        Returns
        -------
        Gamma : np.ndarray, shape (n_frozen + n_components, n_samples)
        """
        self._check_fitted()
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, self.D_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classify test signals using the learned classifier W.

        The predicted class for each signal is the argmax of W @ gamma.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)  integer class indices
        """
        self._check_fitted()
        if self.W_ is None:
            raise DictionaryLearningError(
                "Classifier W is not available. "
                "Use variant='lcksvd2' or access sparse codes via transform()."
            )
        Gamma = self.transform(X)
        scores = self.W_ @ Gamma          # (n_classes, n_samples)
        return np.argmax(scores, axis=0)

    def score(self, X: np.ndarray, H: np.ndarray) -> float:
        """
        Classification accuracy on (X, H).

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        H : np.ndarray, shape (n_classes, n_samples) — one-hot labels

        Returns
        -------
        accuracy : float in [0, 1]
        """
        true_labels = np.argmax(H, axis=0)
        pred_labels = self.predict(X)
        return float(np.mean(pred_labels == true_labels))

    @staticmethod
    def _build_aug_dict(
        D: np.ndarray,
        A: np.ndarray,
        W: np.ndarray,
        sqrt_alpha: float,
        sqrt_beta: float,
        use_classifier: bool,
    ) -> np.ndarray:
        """Stack [D ; sqrt_alpha*A ; (sqrt_beta*W)]."""
        parts = [D, sqrt_alpha * A]
        if use_classifier:
            parts.append(sqrt_beta * W)
        return np.vstack(parts)

    @staticmethod
    def _split_aug_dict(
        D_aug: np.ndarray,
        n_features: int,
        n_classes: int,
        sqrt_alpha: float,
        sqrt_beta: float,
        use_classifier: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recover (D, A, W) from the augmented dictionary D_aug.

        D_aug rows are: n_features | n_components | (n_classes if lcksvd2).

        D_aug's columns are unit-norm as a whole (across all stacked
        blocks), not block-by-block. To recover a properly normalised D
        together with A/W on a consistent per-atom scale, each atom's D
        sub-block norm is computed and used to rescale D, A, and W
        (in addition to removing the sqrt_alpha / sqrt_beta weighting
        from A / W).
        """
        n_components = D_aug.shape[1]
        D = D_aug[:n_features, :]
        A_rows = n_components
        A = D_aug[n_features: n_features + A_rows, :]
        if use_classifier:
            W = D_aug[n_features + A_rows:, :]
        else:
            W = np.zeros((n_classes, n_components))

        l2norms = np.linalg.norm(D, axis=0)
        l2norms = np.where(l2norms > 1e-14, l2norms, 1e-14)

        D = D / l2norms
        A = A / l2norms / max(sqrt_alpha, 1e-14)
        if use_classifier:
            W = W / l2norms / max(sqrt_beta, 1e-14)

        return D, A, W