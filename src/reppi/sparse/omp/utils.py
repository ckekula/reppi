import logging

import numpy as np
from scipy import linalg
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CHOLESKY_FLOOR = 1e-14


def omp_cholesky(
    D: np.ndarray,
    x: np.ndarray,
    n_nonzero: int,
) -> np.ndarray:
    """
    Single-signal OMP via Cholesky updates.
    """
    n_atoms = D.shape[1]
    residual = x.copy().astype(np.float64)
    support: list[int] = []
    selected_mask = np.zeros(n_atoms, dtype=bool)
    gamma = np.zeros(n_atoms, dtype=np.float32)

    # Cholesky factor kept in float64 — this recursion is numerically
    # sensitive and accumulates rounding error across iterations; running
    # it in float32 causes 1 - v@v to go negative / c to diverge to Inf/NaN
    L = np.zeros((n_nonzero, n_nonzero), dtype=np.float64)

    for k in range(n_nonzero):
        correlations = D.T @ residual
        masked = np.where(selected_mask, -np.inf, np.abs(correlations))
        j = int(np.argmax(masked))
        support.append(j)
        selected_mask[j] = True

        # --- Cholesky update ---
        Ds = D[:, support]
        if k == 0:
            L[0, 0] = 1.0
        else:
            w = (Ds[:, :-1].T @ D[:, j]).astype(np.float64)
            v = linalg.solve_triangular(L[:k, :k], w, lower=True)
            proj = 1.0 - v @ v
            if proj < _CHOLESKY_FLOOR:
                logger.warning(
                    "omp_cholesky: near-singular Cholesky update at k=%d "
                    "(1 - v@v = %.3e, atom %d nearly linearly dependent on "
                    "already-selected support %s). Clamping to floor; "
                    "consider checking dictionary for near-duplicate atoms.",
                    k, proj, j, support[:-1],
                )
            l_new = np.sqrt(max(proj, _CHOLESKY_FLOOR))
            L[k, :k] = v
            L[k, k] = l_new

        rhs = (Ds.T @ x).astype(np.float64)
        Lt = L[: k + 1, : k + 1]
        y = linalg.solve_triangular(Lt, rhs, lower=True)
        c = linalg.solve_triangular(Lt.T, y, lower=False)
        residual = x.astype(np.float64) - Ds @ c

    gamma[support] = c
    return gamma


def batch_omp(
    DtX: np.ndarray,
    G: np.ndarray,
    n_nonzero: int,
) -> np.ndarray:
    """
    Batch OMP — fastest variant.
    """
    n_atoms, n_samples = DtX.shape
    Gamma = np.zeros((n_atoms, n_samples), dtype=np.float32)

    for i in tqdm(range(n_samples), desc="Processing samples with Batch-OMP"):
        dtx = DtX[:, i]
        residual_proj = dtx.astype(np.float64)
        support: list[int] = []
        selected_mask = np.zeros(n_atoms, dtype=bool)
        L = np.zeros((n_nonzero, n_nonzero), dtype=np.float64)

        for k in range(n_nonzero):
            masked = np.where(selected_mask, -np.inf, np.abs(residual_proj))
            j = int(np.argmax(masked))
            support.append(j)
            selected_mask[j] = True

            if k == 0:
                L[0, 0] = 1.0
            else:
                w = G[support[:-1], j].astype(np.float64)
                v = linalg.solve_triangular(L[:k, :k], w, lower=True)
                proj = 1.0 - v @ v
                if proj < _CHOLESKY_FLOOR:
                    logger.warning(
                        "batch_omp: near-singular Cholesky update at sample=%d "
                        "k=%d (1 - v@v = %.3e, atom %d nearly linearly "
                        "dependent on already-selected support %s). Clamping "
                        "to floor; consider checking dictionary for "
                        "near-duplicate atoms.",
                        i, k, proj, j, support[:-1],
                    )
                l_new = np.sqrt(max(proj, _CHOLESKY_FLOOR))
                L[k, :k] = v
                L[k, k] = l_new

            rhs = dtx[support].astype(np.float64)
            Lt = L[: k + 1, : k + 1]
            y = linalg.solve_triangular(Lt, rhs, lower=True)
            c = linalg.solve_triangular(Lt.T, y, lower=False)

            residual_proj = dtx.astype(np.float64) - G[:, support] @ c

        Gamma[support, i] = c

    return Gamma
