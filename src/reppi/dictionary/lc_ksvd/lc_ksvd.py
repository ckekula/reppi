"""
Label Consistent K-SVD (LC-KSVD) dictionary learning.

LC-KSVD augments the standard K-SVD objective with:
  - A label-consistency term (LC-KSVD1) that encourages atoms associated
    with the same class to produce similar sparse codes.
  - An additional linear classifier term (LC-KSVD2) that jointly trains a
    classifier W alongside the dictionary.

Implementation note
--------------------
Both variants reduce to running ordinary K-SVD on an augmented system
(with n_iter=1) and then splits the result back into D, A, W.

Known limitation (fixed atom-class labels, Sec. 3.3.1 footnote 2)
-------------------------------------------------------------------
The paper assumes each atom's class label is fixed for the entire
training run once initialised. In this implementation the inner `KSVD`
instance still runs its own dead-atom / mutual-incoherence-based atom
replacement heuristics (`_optimize_atom`'s dead-atom fallback and
`_clear_dict`) every outer iteration, on the *augmented* system, without
any notion of class. Those heuristics can therefore replace an atom
with a signal from a different class than the one Q currently assigns
it to, silently violating the paper's fixed-labelling assumption for
that atom (Q is never rebuilt afterward, so its label-consistency target
would then be stale for the rest of training). Reducing `mu_thresh`
increases how often the incoherence-triggered path fires; the
dead-atom fallback triggers regardless of `mu_thresh` whenever an
atom's row in Gamma is entirely zero, and is not user-configurable at
the KSVD level. This is a structural limitation of composing the
generic KSVD atom-update/clearing logic with LC-KSVD's assumption.
"""
from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm

from reppi.base import BaseDiscriminativeDictionaryLearner
from reppi.dictionary.ksvd.ksvd import KSVD
from reppi.dictionary.lc_ksvd.checkpoint import load_checkpoint, save_checkpoint
from reppi.dictionary.lc_ksvd.ridge import RidgeClassifier
from reppi.dictionary.lc_ksvd.utils import _augment_data, initialization4lcksvd
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp.omp import OMP
from reppi.sparse.utils import normalize_columns, rep_error_squared

_CHECKPOINT_FILENAME = "lc_ksvd_checkpoint.npz"


class LCKSVD(BaseDiscriminativeDictionaryLearner):
    """
    Label Consistent K-SVD dictionary learner (LC-KSVD1 and LC-KSVD2).

    Parameters
    ----------
    n_components : int
        Number of dictionary atoms.
    n_nonzero_coefs : int
        Sparsity level T.
    alpha : float
        Weight for the label-consistency term (this is sqrt_alpha in the
        paper's Eq. (11) — see the module docstring for how it relates to
        the paper's reported alpha values).
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
        Mutual-incoherence threshold (default 0.99). See the module
        docstring's "Known limitation" section for how this interacts
        with LC-KSVD's fixed atom-class labelling assumption.
    lambda1 : float
        Ridge weight for the classifier, Eq. (17) (default 1e-5). Used
        both for W^(0) (LC-KSVD2's warm start) and, for LC-KSVD1, as the
        default `RidgeClassifier`'s regularisation unless a `classifier`
        override supplies its own.
    lambda2 : float
        Ridge weight for A^(0), Eq. (16) (default 1e-5).
    classifier : object or None
        Only valid when ``variant="lcksvd1"``. Any object exposing
        ``fit(Gamma, H)`` / ``predict(Gamma)`` (see ``RidgeClassifier``),
        trained once, after the dictionary has converged, on sparse codes
        from the final D — per the paper's Sec. 3.2 ("the classifier W
        for LC-KSVD1 is trained separately ... after D, A, and X are
        computed"). If None, defaults to ``RidgeClassifier(lambda1)``.
        Passing this for ``variant="lcksvd2"`` raises ``ValueError``,
        since LC-KSVD2's classifier is trained jointly with the
        dictionary as part of the augmented system and cannot be
        substituted independently.
    random_state : int or None
    verbose : bool

    Attributes
    ----------
    D_ : np.ndarray, shape (n_features, n_components)
        Learned dictionary.
    W_ : np.ndarray or None, shape (n_classes, n_components)
        Linear classifier weights. For LC-KSVD2, these are learned
        jointly with the dictionary. For LC-KSVD1, these mirror
        ``classifier_.W_`` when the fitted classifier exposes a linear
        ``W_`` attribute (e.g. the default ``RidgeClassifier``); ``None``
        if a custom classifier without a ``W_`` attribute was supplied —
        use ``classifier_.predict()`` / ``predict()`` in that case.
    classifier_ : object or None
        The fitted classifier instance for LC-KSVD1 (``None`` for
        LC-KSVD2, which uses ``W_`` directly).
    A_ : np.ndarray, shape (n_components, n_components)
        Learned label-consistency transform.
    errors_ : list of float
        Per-iteration RMSE on training data (measured on the original,
        un-augmented X).
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
        lambda1: float = 1e-5,
        lambda2: float = 1e-5,
        classifier: object | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if variant not in ("lcksvd1", "lcksvd2"):
            raise ValueError("variant must be 'lcksvd1' or 'lcksvd2'.")
        if classifier is not None and variant != "lcksvd1":
            raise ValueError(
                "classifier= is only valid for variant='lcksvd1'. "
                "LC-KSVD2's classifier is trained jointly with the "
                "dictionary and cannot be substituted independently."
            )
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.n_iter = n_iter
        self.n_iter_init = n_iter_init
        self.exact_svd = exact_svd
        self.mu_thresh = mu_thresh
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.classifier = classifier
        self.random_state = random_state
        self.verbose = verbose

        self.D_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.A_: np.ndarray | None = None
        self.classifier_: object | None = None
        self.errors_: list[float] = []
        self.class_boundaries_: dict[int, tuple[int, int]] | None = None

    def fit(
        self,
        X: np.ndarray,
        H: np.ndarray,
        D_init: np.ndarray | None = None,
        A_init: np.ndarray | None = None,
        W_init: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> LCKSVD:
        """
        Learn a discriminative dictionary from labelled training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
            Training signals.
        H : np.ndarray, shape (n_classes, n_samples)
            One-hot label matrix.
        D_init : np.ndarray or None
            Initial dictionary. If None, a K-SVD initialisation is run.
            Ignored when resuming from an existing checkpoint.
        A_init : np.ndarray or None
            Initial label-consistency transform. Ignored when resuming.
        W_init : np.ndarray or None
            Initial classifier weights (used to warm-start LC-KSVD2's
            joint optimisation only; LC-KSVD1 trains its classifier
            separately at the end regardless of this argument). Ignored
            when resuming.
        Q : np.ndarray or None
            Label-consistent target matrix. Computed from H if None.
            Ignored when resuming.
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
        """
        X = np.asarray(X, dtype=np.float32)
        H = np.asarray(H, dtype=np.float32)
        n_features, n_samples = X.shape
        n_classes = H.shape[0]

        if n_samples < self.n_components:
            raise DictionaryLearningError(
                f"n_samples={X.shape[1]} is less than n_components={self.n_components}. "
            )

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
                ) = load_checkpoint(self, checkpoint_path, X, H)
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
                    random_state=self.random_state,
                    verbose=self.verbose,
                    lambda1=self.lambda1,
                    lambda2=self.lambda2,
                )

            D = normalize_columns(D_init.copy())
            del D_init
            A = A_init.copy()
            del A_init
            W = W_init.copy()
            del W_init
        else:
            Q = Q_loaded

        sqrt_alpha = self.alpha
        sqrt_beta = self.beta

        use_classifier_term = (self.variant == "lcksvd2")

        # Build augmented training data #
        # Y_aug = [X ; sqrt_alpha*Q ; sqrt_beta*H]  (LC-KSVD2)
        # Y_aug = [X ; sqrt_alpha*Q]                 (LC-KSVD1)
        H_aug = H if use_classifier_term else None
        X_aug, _, _ = _augment_data(X, Q, H_aug, sqrt_alpha, sqrt_beta)

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

        Gamma = None
        for it in tqdm(range(start_iter, self.n_iter), desc="LC-KSVD iterations"):

            # Build augmented dictionary #
            D_aug = self._build_aug_dict(D, A, W, sqrt_alpha, sqrt_beta, use_classifier_term)
            D_aug_norm = normalize_columns(D_aug)
            del D_aug

            # Sparse code + single atom-update pass on the augmented system
            atom_updater.fit(X_aug, D_init=D_aug_norm, checkpoint_dir=None)
            del D_aug_norm
            D_aug_updated = atom_updater.D_
            Gamma = atom_updater.Gamma_
            atom_updater.D_ = None
            atom_updater.Gamma_ = None

            # De-augment: extract D, A, W from the updated augmented dict
            D, A, W = self._split_aug_dict(
                D_aug_updated, n_features, n_classes, sqrt_alpha, sqrt_beta, use_classifier_term
            )
            del D_aug_updated
            D = normalize_columns(D)

            # Track RMSE on original X
            err = np.float32(np.sqrt(rep_error_squared(X, D, Gamma).sum() / X.size))
            self.errors_.append(err)

            if self.verbose:
                print(f"[{self.variant.upper()}] Iter {it + 1}/{self.n_iter}  RMSE={err:.6f}")

            if checkpoint_path is not None:
                save_checkpoint(
                    self, checkpoint_path, X, H, D, A, W, Q, it + 1
                )

        self.D_ = D
        self.A_ = A
        del X_aug, Gamma, Q

        # ---- Classifier ----
        if self.variant == "lcksvd1":
            # Paper Sec. 3.2: LC-KSVD1's classifier is trained separately,
            # after D, A, X are computed. Re-encode with the final, converged D.
            clf = self.classifier if self.classifier is not None else RidgeClassifier(
                lambda1=self.lambda1
            )
            Gamma_final = self.transform(X)
            clf.fit(Gamma_final, H)
            self.classifier_ = clf
            # Mirror a linear W_ for convenience/back-compat when the
            # fitted classifier exposes one. None for classifiers that don't.
            self.W_ = getattr(clf, "W_", None)
        else:
            self.W_ = W
            self.classifier_ = None

        # Record per-class atom ranges matching _build_label_consistent_target
        atoms_per_class = self.n_components // n_classes
        boundaries: dict[int, tuple[int, int]] = {}
        for c in range(n_classes):
            start = c * atoms_per_class
            end = start + atoms_per_class if c < n_classes - 1 else self.n_components
            boundaries[c] = (start, end)
        self.class_boundaries_ = boundaries

        return self


    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode X using the learned dictionary D.
        """
        self._check_fitted()
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, self.D_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classify test signals using the learned classifier.

        For LC-KSVD2, this is the argmax of ``W_ @ gamma``. For LC-KSVD1,
        this delegates to the fitted ``classifier_`` (default
        ``RidgeClassifier``, or a user-supplied ``classifier``).
        """
        self._check_fitted()
        Gamma = self.transform(X)

        if self.variant == "lcksvd1":
            if self.classifier_ is None:
                raise DictionaryLearningError(
                    "No classifier is available. This should not happen "
                    "for a successfully fit LC-KSVD1 model."
                )
            return self.classifier_.predict(Gamma)

        if self.W_ is None:
            raise DictionaryLearningError(
                "Classifier W is not available. "
                "Access sparse codes via transform() instead."
            )
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
        return np.float32(np.mean(pred_labels == true_labels), dtype=np.float32)

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
            W = np.zeros((n_classes, n_components), dtype=np.float32)

        l2norms = np.linalg.norm(D, axis=0)
        l2norms = np.where(l2norms > 1e-14, l2norms, 1e-14)

        D = D / l2norms
        A = A / l2norms / max(sqrt_alpha, 1e-14)
        if use_classifier:
            W = W / l2norms / max(sqrt_beta, 1e-14)

        return D, A, W