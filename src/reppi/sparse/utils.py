"""
Utility functions shared across sparse coding algorithms.
"""

from __future__ import annotations

import numpy as np
import torch
from reppi.exceptions import DictionaryNormalizationError


def _check_dict_normalized(D: torch.Tensor, tol: float = 1e-2) -> None:
    """Raise if any atom of D deviates from unit L2-norm by more than tol."""
    norms = torch.sqrt((D * D).sum(dim=0))
    if torch.any(torch.abs(norms - 1.0) > tol):
        raise DictionaryNormalizationError(
            "Dictionary columns must be normalized to unit L2-norm. "
            f"Got norms in range [{float(norms.min()):.4f}, {float(norms.max()):.4f}]."
        )
 
 
def normalize_columns(D: torch.Tensor) -> torch.Tensor:
    """Return D with each column scaled to unit L2-norm.
 
    Columns whose norm is below 1e-10 are left unchanged to avoid division
    by zero.
    """
    norms = torch.sqrt((D * D).sum(dim=0))
    norms = torch.where(norms < 1e-10, torch.ones_like(norms), norms)
    return D / norms


def col_norms_squared(X: np.ndarray, block_size: int = 2000) -> np.ndarray:
    """Compute squared L2-norm of each column of X in blocks (memory-safe).

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_samples)
    block_size : int
        Number of columns to process at a time.

    Returns
    -------
    norms2 : np.ndarray, shape (n_samples,)
    """
    n_samples = X.shape[1]
    norms2 = np.zeros(n_samples, dtype=np.float32)
    for start in range(0, n_samples, block_size):
        end = min(start + block_size, n_samples)
        norms2[start:end] = (X[:, start:end] ** 2).sum(axis=0)
    return norms2


def rep_error_squared(
    X: np.ndarray,
    D: np.ndarray,
    Gamma: np.ndarray,
    block_size: int = 2000,
) -> np.ndarray:
    """Per-signal squared reconstruction error |x_i - D gamma_i|^2.

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_samples)
    D : np.ndarray, shape (n_features, n_atoms)
    Gamma : np.ndarray, shape (n_atoms, n_samples)
    block_size : int

    Returns
    -------
    err2 : np.ndarray, shape (n_samples,)
    """
    n_samples = X.shape[1]
    err2 = np.zeros(n_samples, dtype=np.float32)
    for start in range(0, n_samples, block_size):
        end = min(start + block_size, n_samples)
        diff = X[:, start:end] - D @ Gamma[:, start:end]
        err2[start:end] = (diff ** 2).sum(axis=0)
    return err2
