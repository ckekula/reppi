"""
Orthogonal Matching Pursuit (OMP) sparse coding.

Implements Batch-OMP as described in:
    Elad, Rubinstein, Zibulevsky. "Efficient Implementation of the K-SVD
    Algorithm using Batch Orthogonal Matching Pursuit". Technion TR, 2008.
"""

from __future__ import annotations

import numpy as np

from reppi.base import BaseSparseCoder
from reppi.sparse.omp.utils import _check_dict_normalized, omp_cholesky, batch_omp

class OMP(BaseSparseCoder):
    """
    Orthogonal Matching Pursuit sparse coder.

    Parameters
    ----------
    n_nonzero_coefs : int
        Target sparsity — maximum number of non-zero coefficients per signal.
    mode : {'batch', 'cholesky'}
        Implementation variant.
        'batch'     — Batch-OMP; requires the full Gram matrix G = D'D.
                      Fastest when encoding many signals at once.
        'cholesky'  — Single-signal OMP-Cholesky; lower memory footprint.
    check_dict : bool
        Whether to verify that dictionary atoms are unit-norm (default True).
    """

    def __init__(
        self,
        n_nonzero_coefs: int,
        mode: str = "batch",
        check_dict: bool = True,
    ) -> None:
        if n_nonzero_coefs < 1:
            raise ValueError("n_nonzero_coefs must be >= 1.")
        if mode not in ("batch", "cholesky"):
            raise ValueError("mode must be 'batch' or 'cholesky'.")
        self.n_nonzero_coefs = n_nonzero_coefs
        self.mode = mode
        self.check_dict = check_dict

    def encode(
        self,
        X: np.ndarray,
        D: np.ndarray,
        G: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute sparse codes for each column of X.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        D : np.ndarray, shape (n_features, n_atoms)
        G : np.ndarray or None, shape (n_atoms, n_atoms)
            Precomputed Gram matrix D.T @ D.  Required for 'batch' mode;
            computed internally if not supplied.

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

        T = self.n_nonzero_coefs

        if self.mode == "batch":
            if G is None:
                G = D.T @ D
            DtX = D.T @ X
            return batch_omp(DtX, G, T)

        # cholesky mode — signal by signal
        n_atoms = D.shape[1]
        n_samples = X.shape[1]
        Gamma = np.zeros((n_atoms, n_samples))
        for i in range(n_samples):
            Gamma[:, i] = omp_cholesky(D, X[:, i], T)
        return Gamma