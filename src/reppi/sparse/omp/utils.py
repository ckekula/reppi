import numpy as np
import logging

from scipy import linalg

logger = logging.getLogger(__name__)

def omp_cholesky(
    D: np.ndarray,
    x: np.ndarray,
    n_nonzero: int,
) -> np.ndarray:
    """
    Single-signal OMP via Cholesky updates (OMP-Cholesky).

    Parameters
    ----------
    D : np.ndarray, shape (n_features, n_atoms)
        Normalized dictionary.
    x : np.ndarray, shape (n_features,)
        Single input signal.
    n_nonzero : int
        Maximum number of non-zero coefficients (sparsity).

    Returns
    -------
    gamma : np.ndarray, shape (n_atoms,)
        Sparse representation of x.
    """
    n_atoms = D.shape[1]
    residual = x.copy().astype(float)
    support: list[int] = []
    gamma = np.zeros(n_atoms)

    # Cholesky factor of D[:,support].T @ D[:,support]
    L = np.zeros((n_nonzero, n_nonzero))

    for k in range(n_nonzero):
        correlations = D.T @ residual
        j = int(np.argmax(np.abs(correlations)))
        support.append(j)

        # --- Cholesky update ---
        Ds = D[:, support]
        if k == 0:
            L[0, 0] = 1.0
        else:
            w = Ds[:, :-1].T @ D[:, j]  # (k,)
            # Solve L[:k,:k] * v = w
            v = linalg.solve_triangular(L[:k, :k], w, lower=True)
            l_new = np.sqrt(max(1.0 - float(v @ v), 1e-14))
            L[k, :k] = v
            L[k, k] = l_new

        # Solve (L L.T) c = Ds.T x
        rhs = Ds.T @ x

        Lt = L[: k + 1, : k + 1]
        y = linalg.solve_triangular(Lt, rhs, lower=True)
        c = linalg.solve_triangular(Lt.T, y, lower=False)
        residual = x - Ds @ c

    gamma[support] = c
    return gamma


def batch_omp(
    DtX: np.ndarray,
    G: np.ndarray,
    n_nonzero: int,
) -> np.ndarray:
    """
    Batch OMP — fastest variant; requires precomputed G = D'D and DtX = D'X.

    Parameters
    ----------
    DtX : np.ndarray, shape (n_atoms, n_samples)
        Precomputed projections D.T @ X.
    G : np.ndarray, shape (n_atoms, n_atoms)
        Precomputed Gram matrix D.T @ D.
    n_nonzero : int
        Sparsity level.

    Returns
    -------
    Gamma : np.ndarray, shape (n_atoms, n_samples)
        Sparse representations (dense).
    """
    n_atoms, n_samples = DtX.shape
    Gamma = np.zeros((n_atoms, n_samples))

    for i in range(n_samples):
        logger.info("Encoding signal %d/%d with Batch-OMP", i + 1, n_samples)
        dtx = DtX[:, i]
        residual_proj = dtx.copy()
        support: list[int] = []
        L = np.zeros((n_nonzero, n_nonzero))

        for k in range(n_nonzero):
            logger.debug("  Iteration %d/%d for signal %d", k + 1, n_nonzero, i + 1)
            j = int(np.argmax(np.abs(residual_proj)))
            support.append(j)

            # Cholesky update using Gram matrix
            if k == 0:
                L[0, 0] = 1.0
            else:
                w = G[support[:-1], j]  # (k,)
                v = linalg.solve_triangular(L[:k, :k], w, lower=True)
                l_new = np.sqrt(max(1.0 - float(v @ v), 1e-14))
                L[k, :k] = v
                L[k, k] = l_new

            # Solve (L L.T) c = DtX[support, i]
            rhs = dtx[support]
            Lt = L[: k + 1, : k + 1]
            y = linalg.solve_triangular(Lt, rhs, lower=True)
            c = linalg.solve_triangular(Lt.T, y, lower=False)

            # Update residual in projection space
            residual_proj = dtx - G[:, support] @ c

        logger.info("Finished encoding signal %d/%d", i + 1, n_samples)
        Gamma[support, i] = c

    return Gamma
