"""
LARS-Lasso sparse coding.

Implements the LASSO-modified Least Angle Regression path as described in:
    Efron, Hastie, Johnstone, Tibshirani. "Least Angle Regression".
    Annals of Statistics, 2004.
"""

from __future__ import annotations

import numpy as np

from reppi.base import BaseSparseCoder
from reppi.sparse.utils import _check_dict_normalized
from reppi.sparse.lars.utils import lars_lasso_cholesky


class LARSLasso(BaseSparseCoder):
    """
    LARS-Lasso sparse coder.

    Parameters
    ----------
    n_nonzero_coefs : int or None
        Target sparsity — stop once this many atoms are active.
        At least one of ``n_nonzero_coefs`` / ``alpha`` must be given.
    alpha : float or None
        Correlation / L1-penalty stopping threshold: the path stops once
        the maximum absolute correlation between the residual and the
        dictionary drops to ``alpha``. At least one of ``n_nonzero_coefs``
        / ``alpha`` must be given.
    max_iter : int or None
        Safety cap on total LARS steps (add + drop) per signal.
        Defaults to ``8 * n_atoms``.
    check_dict : bool
        Whether to verify that dictionary atoms are unit-norm (default True).
    """

    def __init__(
        self,
        n_nonzero_coefs: int | None = None,
        alpha: float | None = None,
        max_iter: int | None = None,
        check_dict: bool = True,
    ) -> None:
        if n_nonzero_coefs is None and alpha is None:
            raise ValueError(
                "At least one of n_nonzero_coefs or alpha must be provided."
            )
        if n_nonzero_coefs is not None and n_nonzero_coefs < 1:
            raise ValueError("n_nonzero_coefs must be >= 1.")
        if alpha is not None and alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if max_iter is not None and max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        self.n_nonzero_coefs = n_nonzero_coefs
        self.alpha = alpha
        self.max_iter = max_iter
        self.check_dict = check_dict

    def encode(
        self,
        X: np.ndarray,
        D: np.ndarray,
        G: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute LARS-Lasso sparse codes for each column of X.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        D : np.ndarray, shape (n_features, n_atoms)
        G : np.ndarray or None, shape (n_atoms, n_atoms)
            Precomputed Gram matrix D.T @ D. Computed internally if not
            supplied.

        Returns
        -------
        Gamma : np.ndarray, shape (n_atoms, n_samples)
        """
        X = np.asarray(X, dtype=float)
        D = np.asarray(D, dtype=float)

        if X.ndim == 1:
            X = X[:, np.newaxis]

        if self.check_dict:
            _check_dict_normalized(D)

        if G is None:
            G = D.T @ D

        n_atoms = D.shape[1]
        n_samples = X.shape[1]
        Gamma = np.zeros((n_atoms, n_samples))
        for i in range(n_samples):
            Gamma[:, i] = lars_lasso_cholesky(
                D,
                X[:, i],
                n_nonzero_coefs=self.n_nonzero_coefs,
                alpha=self.alpha,
                G=G,
                max_iter=self.max_iter,
            )
        return Gamma