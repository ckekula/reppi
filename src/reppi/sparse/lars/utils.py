"""
Least Angle Regression with the LASSO modification (LARS-Lasso).

Implements the algorithm as described in:
    Efron, Hastie, Johnstone, Tibshirani. "Least Angle Regression".
    Annals of Statistics, 2004. (Sec. 3.1-3.3, LASSO modification).

Uses incremental Cholesky updates of the active-set Gram matrix, in the
same spirit as Batch-OMP / OMP-Cholesky.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg

_EPS = np.finfo(float).eps


def _cholesky_active(G: np.ndarray, active: list[int], signs: list[int]) -> np.ndarray:
    """
    Cholesky factor of the *signed* active-set Gram matrix, built from
    scratch: L L.T = diag(signs) @ G[active][:, active] @ diag(signs).

    Used after a drop step, where a rank-1 downdate is not worth the added
    complexity given that active-set sizes are bounded by n_nonzero_coefs.
    """
    s = np.asarray(signs, dtype=float)
    Gs = G[np.ix_(active, active)] * np.outer(s, s)
    try:
        L = linalg.cholesky(Gs, lower=True)
    except linalg.LinAlgError:
        # Near-singular active set (e.g. collinear/duplicate atoms).
        # Fall back to a tiny ridge to keep the update well-defined.
        jitter = 1e-10 * np.trace(Gs) / max(len(active), 1)
        L = linalg.cholesky(Gs + jitter * np.eye(len(active)), lower=True)
    return L


def _cholesky_add(
    L: np.ndarray,
    G: np.ndarray,
    active: list[int],
    signs: list[int],
    j: int,
    sign_j: int,
) -> np.ndarray:
    """Incrementally extend L to include newly-activated atom j."""
    k = len(active)
    if k == 0:
        return np.array([[1.0]])

    s = np.asarray(signs, dtype=float)
    w = sign_j * s * G[active, j]  # (k,)
    v = linalg.solve_triangular(L, w, lower=True)
    diag_sq = 1.0 - float(v @ v)
    if diag_sq <= _EPS:
        raise linalg.LinAlgError("Active set is (numerically) rank deficient.")
    L_new = np.zeros((k + 1, k + 1))
    L_new[:k, :k] = L
    L_new[k, :k] = v
    L_new[k, k] = np.sqrt(diag_sq)
    return L_new


def lars_lasso_cholesky(
    D: np.ndarray,
    x: np.ndarray,
    n_nonzero_coefs: int | None,
    alpha: float | None,
    G: np.ndarray | None = None,
    max_iter: int | None = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Single-signal LARS-Lasso via Cholesky updates.

    Parameters
    ----------
    D : np.ndarray, shape (n_features, n_atoms)
        Normalized dictionary.
    x : np.ndarray, shape (n_features,)
        Single input signal.
    n_nonzero_coefs : int or None
        Stop once this many atoms are active. None disables this
        criterion (alpha alone controls the path length).
    alpha : float or None
        Stop once the maximum absolute correlation between the residual
        and the dictionary drops to alpha (L1-penalty / correlation
        threshold, as in LassoLars). None disables this criterion.
    G : np.ndarray or None, shape (n_atoms, n_atoms)
        Precomputed Gram matrix D.T @ D. Computed internally if omitted.
    max_iter : int or None
        Safety cap on total steps (add + drop). Defaults to
        ``8 * n_atoms``.
    eps : float
        Numerical floor on the maximum correlation; the path stops once
        it is reached, treating the residual as fully explained.

    Returns
    -------
    gamma : np.ndarray, shape (n_atoms,)
    """
    n_atoms = D.shape[1]
    if G is None:
        G = D.T @ D
    if max_iter is None:
        max_iter = 8 * n_atoms

    correlation = D.T @ x
    coef = np.zeros(n_atoms)
    active: list[int] = []
    signs: list[int] = []
    L = np.zeros((0, 0))

    for _ in range(max_iter):
        C = np.max(np.abs(correlation)) if n_atoms > 0 else 0.0

        if C < eps:
            break
        if alpha is not None and C <= alpha:
            break
        if len(active) >= n_atoms:
            break

        # --- candidate entering variable (inactive only) ---
        corr_inactive = correlation.copy()
        if active:
            corr_inactive[active] = 0.0
        j = int(np.argmax(np.abs(corr_inactive)))
        sign_j = 1 if correlation[j] >= 0 else -1

        try:
            L = _cholesky_add(L, G, active, signs, j, sign_j)
        except linalg.LinAlgError:
            warnings.warn(
                "LARS-Lasso: active set became rank-deficient; "
                "stopping path early.",
                stacklevel=2,
            )
            break

        active.append(j)
        signs.append(sign_j)

        # --- equiangular direction ---
        ones_vec = np.ones(len(active))
        Ginv1 = linalg.cho_solve((L, True), ones_vec)
        A_A = 1.0 / np.sqrt(np.sum(ones_vec * Ginv1))
        w_A = A_A * Ginv1
        v = w_A * np.asarray(signs, dtype=float)  # direction of coef[active]

        a = G[:, active] @ v  # D.T @ u_A, for every atom

        # --- step size to the next equicorrelation point ---
        inactive_mask = np.ones(n_atoms, dtype=bool)
        inactive_mask[active] = False
        gamma_hat = np.inf
        if np.any(inactive_mask):
            aj = a[inactive_mask]
            cj = correlation[inactive_mask]
            denom_minus = A_A - aj
            denom_plus = A_A + aj
            with np.errstate(divide="ignore", invalid="ignore"):
                cand_minus = np.where(denom_minus > eps, (C - cj) / denom_minus, np.inf)
                cand_plus = np.where(denom_plus > eps, (C + cj) / denom_plus, np.inf)
            cand = np.concatenate([cand_minus, cand_plus])
            cand = cand[cand > eps]
            if cand.size:
                gamma_hat = float(cand.min())
        else:
            gamma_hat = C / A_A

        # --- LASSO modification: step size to next sign change ---
        coef_active = coef[active]
        with np.errstate(divide="ignore", invalid="ignore"):
            drop_cand = -coef_active / v
        drop_cand = np.where(drop_cand > eps, drop_cand, np.inf)
        gamma_tilde = float(np.min(drop_cand)) if drop_cand.size else np.inf

        gamma_step = min(gamma_hat, gamma_tilde)
        if not np.isfinite(gamma_step):
            break

        coef[active] += gamma_step * v
        correlation -= gamma_step * a

        if gamma_tilde < gamma_hat:
            # --- drop step ---
            k_drop = int(np.argmin(drop_cand))
            drop_atom = active[k_drop]
            coef[drop_atom] = 0.0
            del active[k_drop]
            del signs[k_drop]
            L = _cholesky_active(G, active, signs) if active else np.zeros((0, 0))
            continue

        if n_nonzero_coefs is not None and len(active) >= n_nonzero_coefs:
            break

    return coef