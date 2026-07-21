"""
Block-Coordinate Descent (BCD) dictionary update.

Implements Algorithm 2 of:
    Mairal, Bach, Ponce, Sapiro. "Online Dictionary Learning for
    Sparse Coding". ICML 2009.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

def bcd_dictionary_update(
    D: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    n_frozen: int,
    max_iter: int = 1,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Block-coordinate descent update of the dictionary columns (Eq. 10),
    warm-started from the current D. Frozen columns (index < n_frozen)
    are always skipped, mirroring KSVD's frozen-atom contract.

    Parameters
    ----------
    D : np.ndarray, shape (n_features, n_total)
        Current dictionary (warm start). Updated and returned in place.
    A : np.ndarray, shape (n_total, n_total)
        Accumulated Sum alpha_i @ alpha_i.T (with forgetting factor).
    B : np.ndarray, shape (n_features, n_total)
        Accumulated Sum x_i @ alpha_i.T (with forgetting factor).
    n_frozen : int
        Number of leading frozen columns, never updated.
    max_iter : int
        Maximum BCD sweeps over the non-frozen columns. The paper found
        a single sweep sufficient given warm restart from the previous
        outer-loop dictionary (Sec 3.3); this stays the default.
    tol : float
        Stop sweeping early once the largest column change (L2 norm)
        across a full sweep drops below this value.

    Returns
    -------
    D : np.ndarray
        Updated dictionary (same array, returned for convenience).
    """
    n_total = D.shape[1]
    R = B - D @ A
    
    for _ in range(max_iter):
        max_change = 0.0
        for j in tqdm(range(n_frozen, n_total), desc="BCD iterations"):
            ajj = A[j, j]
            if ajj <= 1e-12:
                continue
            u_j = R[:, j] / ajj + D[:, j]
            norm_u = np.linalg.norm(u_j)
            d_j_new = u_j / max(norm_u, 1.0)
            delta = d_j_new - D[:, j]
            change = float(np.linalg.norm(delta))
            max_change = max(max_change, change)
            R -= np.outer(delta, A[j, :])  # keep R consistent with new D[:, j]
            D[:, j] = d_j_new
        if max_change < tol:
            break
    return D


def update_forgetting_factor(theta: float, eta: int) -> tuple[float, float]:
    """
    Update theta and compute beta for the mini-batch extension (Eq. 11),
    generalized to variable mini-batch sizes.

    For a *constant* eta, this recurrence reproduces the paper's closed
    form exactly:
        theta_t = t * eta                  if t < eta   (early phase)
        theta_t = eta**2 + t - eta          if t >= eta  (steady state)
    by incrementing theta by eta per iteration during the early phase,
    then by 1 per iteration once theta has saturated to eta**2. The
    transition uses '<=' so the boundary iteration (t == eta) still
    advances by eta and lands exactly on eta**2 — using '<' here would
    leave a permanent offset of (eta - 1) in theta for every later call.

    For *variable* eta (e.g. a trailing short batch at the end of an
    epoch), the phase test is evaluated against the current call's eta.
    This is an approximation the paper doesn't cover, but it only
    affects at most one short batch per epoch rather than silently
    misestimating theta for the entire run.

    Parameters
    ----------
    theta : float
        Running theta from the previous call (0.0 at the start of training).
    eta : int
        Number of signals in the current mini-batch.

    Returns
    -------
    theta_new : float
        Updated theta, to be passed into the next call.
    beta : float
        Forgetting factor for this mini-batch's A/B update.
    """
    if theta + eta <= eta**2:
        theta_new = theta + eta
    else:
        theta_new = theta + 1
    beta = (theta_new + 1 - eta) / (theta_new + 1)
    return theta_new, beta