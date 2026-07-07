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

import numpy as np

from reppi.base import BaseDiscriminativeDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp import OMP, batch_omp
from reppi.sparse.utils import normalize_columns, rep_error_squared
from reppi.dictionary.ksvd import _optimize_atom, _clear_dict

from reppi.dictionary.lc_ksvd.utils import initialization4lcksvd, _augment_data


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
    D_ : np.ndarray, shape (n_features, n_components)
        Learned dictionary.
    W_ : np.ndarray, shape (n_classes, n_components)
        Learned linear classifier weights.
    A_ : np.ndarray, shape (n_components, n_components)
        Learned label-consistency transform.
    errors_ : list of float
        Per-iteration RMSE on training data.
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
        D_init: np.ndarray | None = None,
        A_init: np.ndarray | None = None,
        W_init: np.ndarray | None = None,
        Q: np.ndarray | None = None,
    ) -> "LCKSVD":
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
        A_init : np.ndarray or None
            Initial label-consistency transform.
        W_init : np.ndarray or None
            Initial classifier weights (required / used for LC-KSVD2).
        Q : np.ndarray or None
            Label-consistent target matrix. Computed from H if None.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        H = np.asarray(H, dtype=float)
        n_features, n_samples = X.shape
        n_classes = H.shape[0]

        # ---- Initialisation ----
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
            )

        D = normalize_columns(D_init.copy())
        A = A_init.copy()
        W = W_init.copy()

        sqrt_alpha = self.alpha
        sqrt_beta = self.beta

        use_classifier_term = (self.variant == "lcksvd2")

        # ---- Build augmented training data ----
        # Y_aug = [X ; sqrt_alpha*Q ; sqrt_beta*H]  (LC-KSVD2)
        # Y_aug = [X ; sqrt_alpha*Q]                 (LC-KSVD1)
        H_aug = H if use_classifier_term else None
        X_aug, _, _ = _augment_data(X, Q, H_aug, sqrt_alpha, sqrt_beta)

        self.errors_ = []

        for it in range(self.n_iter):

            # ---- Build augmented dictionary ----
            # D_aug = [D ; sqrt_alpha*A ; sqrt_beta*W]
            D_aug = self._build_aug_dict(D, A, W, sqrt_alpha, sqrt_beta, use_classifier_term)
            D_aug_norm = normalize_columns(D_aug)

            # ---- Sparse coding on augmented system ----
            G_aug = D_aug_norm.T @ D_aug_norm
            Gamma = batch_omp(D_aug_norm.T @ X_aug, G_aug, self.n_nonzero_coefs)

            # ---- Dictionary update (on original data only) ----
            # We update D, A (and W for LC-KSVD2) jointly via the
            # augmented residual, but evaluate coherence/usage on original X.
            unused = np.arange(n_samples)
            replaced = np.zeros(self.n_components, dtype=bool)

            for j in range(self.n_components):
                D_aug_norm[:, j], gamma_j, idx, unused, replaced = _optimize_atom(
                    X_aug, D_aug_norm, j, Gamma, unused, replaced, self.exact_svd
                )
                Gamma[j, idx] = gamma_j

            # De-augment: extract D, A, W from D_aug_norm
            D, A, W = self._split_aug_dict(
                D_aug_norm, n_features, n_classes, sqrt_alpha, sqrt_beta, use_classifier_term
            )
            D = normalize_columns(D)

            # ---- Update classifier W (LC-KSVD2) via least squares ----
            if use_classifier_term:
                W = H @ np.linalg.pinv(Gamma)

            # ---- Update A via least squares ----
            A = Q @ np.linalg.pinv(Gamma)

            # ---- Clear incoherent / rarely-used atoms ----
            # Rebuild normalised augmented dict for coherence checking
            D_aug_rebuilt = self._build_aug_dict(D, A, W, sqrt_alpha, sqrt_beta, use_classifier_term)
            D_aug_rebuilt = normalize_columns(D_aug_rebuilt)
            D_aug_rebuilt, _ = _clear_dict(
                D_aug_rebuilt, Gamma, X_aug, self.mu_thresh,
                unused, replaced
            )
            D, A, W = self._split_aug_dict(
                D_aug_rebuilt, n_features, n_classes, sqrt_alpha, sqrt_beta, use_classifier_term
            )
            D = normalize_columns(D)

            # ---- Track RMSE on original X ----
            err = float(np.sqrt(rep_error_squared(X, D, Gamma).sum() / X.size))
            self.errors_.append(err)

            if self.verbose:
                print(f"[{self.variant.upper()}] Iter {it + 1}/{self.n_iter}  RMSE={err:.6f}")

        self.D_ = D
        self.A_ = A
        self.W_ = W

        # Record per-class atom ranges matching _build_label_consistent_target
        n_classes = H.shape[0]
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

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)

        Returns
        -------
        Gamma : np.ndarray, shape (n_components, n_samples)
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
        """
        n_components = D_aug.shape[1]
        D = D_aug[:n_features, :]
        A_rows = n_components
        A = D_aug[n_features: n_features + A_rows, :] / max(sqrt_alpha, 1e-14)
        if use_classifier:
            W = D_aug[n_features + A_rows:, :] / max(sqrt_beta, 1e-14)
        else:
            W = np.zeros((n_classes, n_components))
        return D, A, W