"""
Orthogonal Matching Pursuit (OMP) sparse coding.

Implements Batch-OMP as described in:
    Elad, Rubinstein, Zibulevsky. "Efficient Implementation of the K-SVD
    Algorithm using Batch Orthogonal Matching Pursuit". Technion TR, 2008.
"""

from __future__ import annotations

import logging

import numpy as np

from reppi.base import BaseSparseCoder
from reppi.sparse.omp.utils import batch_omp, omp_cholesky
from reppi.sparse.utils import _check_dict_normalized

logger = logging.getLogger(__name__)

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
        DtX: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute sparse codes for each column of X.
        """
        X = np.asarray(X, dtype=np.float32)
        D = np.asarray(D, dtype=np.float32)

        if X.ndim == 1:
            X = X[:, np.newaxis]

        if self.check_dict:
            _check_dict_normalized(D)

        T = self.n_nonzero_coefs

        if self.mode == "batch":
            if G is None:
                G = D.T @ D
            if DtX is None:
                DtX = D.T @ X
            logger.info("Encoding %d signals with dictionary of shape %s using Batch-OMP", X.shape[1], D.shape)
            return batch_omp(DtX, G, T)

        # cholesky mode — signal by signal
        n_atoms = D.shape[1]
        n_samples = X.shape[1]
        Gamma = np.zeros((n_atoms, n_samples), dtype=np.float32)
        for i in range(n_samples):
            logger.info("Encoding signal %d/%d with dictionary of shape %s using OMP-Cholesky", i + 1, n_samples, D.shape)
            Gamma[:, i] = omp_cholesky(D, X[:, i], T)    

        return Gamma
