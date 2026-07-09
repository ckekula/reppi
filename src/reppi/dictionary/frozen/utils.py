"""
frozen.utils
Utility functions for Frozen Dictionary Learning.
"""

import numpy as np

def _fit_classifier(
    Gamma: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Fit a linear classifier W via least squares: W @ Gamma ≈ H.

    Parameters
    ----------
    Gamma : np.ndarray, shape (n_atoms, n_samples)
    H     : np.ndarray, shape (n_classes, n_samples)  one-hot

    Returns
    -------
    W : np.ndarray, shape (n_classes, n_atoms)
    """
    return H @ np.linalg.pinv(Gamma)

