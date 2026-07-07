"""
K-SVD dictionary learning.

Implements the K-SVD algorithm described in:
    Aharon, Elad, Bruckstein. "The K-SVD: An Algorithm for Designing
    Overcomplete Dictionaries for Sparse Representation".
    IEEE Trans. Signal Processing, 54(11), 2006.

Batch-OMP integration follows:
    Elad, Rubinstein, Zibulevsky. "Efficient Implementation of the K-SVD
    Algorithm using Batch Orthogonal Matching Pursuit". Technion TR, 2008.
"""

from __future__ import annotations

import numpy as np

from reppi.base import BaseDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp import OMP, batch_omp
from reppi.sparse.utils import col_norms_squared, normalize_columns, rep_error_squared

from reppi.dictionary.ksvd.utils import _optimize_atom, _clear_dict

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
        Number of dictionary atoms to learn.
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
        verbose: bool = False,
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
        self.errors_: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, D_init: np.ndarray | None = None) -> "KSVD":
        """
        Learn a dictionary from training signals.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        D_init : np.ndarray or None, shape (n_features, n_components)
            Optional initial dictionary.  If None, random training signals
            are chosen as initial atoms.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)

        D = self._init_dict(X, D_init, rng)
        self.errors_ = []

        for it in range(self.n_iter):
            G = D.T @ D if self.mem_usage in ("high", "normal") else None
            Gamma = self._sparse_code(X, D, G)

            unused = np.arange(X.shape[1])
            replaced = np.zeros(self.n_components, dtype=bool)

            for j in range(self.n_components):
                D[:, j], gamma_j, idx, unused, replaced = _optimize_atom(
                    X, D, j, Gamma, unused, replaced, self.exact_svd
                )
                Gamma[j, idx] = gamma_j

            err = float(np.sqrt(rep_error_squared(X, D, Gamma).sum() / X.size))
            self.errors_.append(err)

            D, _ = _clear_dict(D, Gamma, X, self.mu_thresh, unused, replaced)

            if self.verbose:
                print(f"Iter {it + 1}/{self.n_iter}  RMSE={err:.6f}")

        self.D_ = D
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode X using the learned dictionary."""
        if self.D_ is None:
            raise DictionaryLearningError("Call fit() before transform().")
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
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, D, G=G)

