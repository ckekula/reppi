"""
Utility functions shared across sparse coding algorithms.
"""

from __future__ import annotations

from reppi.backend import xp

def normalize_columns(D: xp.ndarray) -> xp.ndarray:
    """Return D with each column scaled to unit L2-norm.

    Columns whose norm is below 1e-10 are left unchanged to avoid division
    by zero.
    """
    norms = xp.sqrt((D * D).sum(axis=0))
    norms = xp.where(norms < 1e-10, 1.0, norms)
    return D / norms


def col_norms_squared(X: xp.ndarray, block_size: int = 2000) -> xp.ndarray:
    """Compute squared L2-norm of each column of X in blocks (memory-safe).

    Parameters
    ----------
    X : xp.ndarray, shape (n_features, n_samples)
    block_size : int
        Number of columns to process at a time.

    Returns
    -------
    norms2 : xp.ndarray, shape (n_samples,)
    """
    n_samples = X.shape[1]
    norms2 = xp.zeros(n_samples)
    for start in range(0, n_samples, block_size):
        end = min(start + block_size, n_samples)
        norms2[start:end] = (X[:, start:end] ** 2).sum(axis=0)
    return norms2


def rep_error_squared(
    X: xp.ndarray,
    D: xp.ndarray,
    Gamma: xp.ndarray,
    block_size: int = 2000,
) -> xp.ndarray:
    """Per-signal squared reconstruction error |x_i - D gamma_i|^2.

    Parameters
    ----------
    X : xp.ndarray, shape (n_features, n_samples)
    D : xp.ndarray, shape (n_features, n_atoms)
    Gamma : xp.ndarray, shape (n_atoms, n_samples)
    block_size : int

    Returns
    -------
    err2 : xp.ndarray, shape (n_samples,)
    """
    n_samples = X.shape[1]
    err2 = xp.zeros(n_samples)
    for start in range(0, n_samples, block_size):
        end = min(start + block_size, n_samples)
        diff = X[:, start:end] - D @ Gamma[:, start:end]
        err2[start:end] = (diff ** 2).sum(axis=0)
    return err2