import numpy as np
from reppi.sparse.omp.omp import OMP

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


def _encode_residual(
    X: np.ndarray,
    D_frozen: np.ndarray,
    n_nonzero_coefs: int,
) -> np.ndarray:
    """
    Compute the reconstruction residual after encoding X over D_frozen.

    R = X - D_frozen @ Gamma_frozen

    Parameters
    ----------
    X            : np.ndarray, shape (n_features, n_samples)
    D_frozen     : np.ndarray, shape (n_features, n_frozen_atoms)
    n_nonzero_coefs : int

    Returns
    -------
    R : np.ndarray, shape (n_features, n_samples)  residual signals
    """
    coder = OMP(n_nonzero_coefs, mode="batch", check_dict=False)
    Gamma_frozen = coder.encode(X, D_frozen)
    return X - D_frozen @ Gamma_frozen
