"""
Orthogonal Matching Pursuit (OMP) sparse coding.

Implements Batch-OMP as described in:
    Elad, Rubinstein, Zibulevsky. "Efficient Implementation of the K-SVD
    Algorithm using Batch Orthogonal Matching Pursuit". Technion TR, 2008.
"""

from __future__ import annotations

from reppi.backend import xp, linalg

from reppi.base import BaseSparseCoder
from reppi.exceptions import DictionaryNormalizationError


def _check_dict_normalized(D: xp.ndarray, tol: float = 1e-2) -> None:
    """Raise if any atom of D deviates from unit L2-norm by more than tol."""
    norms = xp.sqrt((D * D).sum(axis=0))
    if xp.any(xp.abs(norms - 1.0) > tol):
        raise DictionaryNormalizationError(
            "Dictionary columns must be normalized to unit L2-norm. "
            f"Got norms in range [{norms.min():.4f}, {norms.max():.4f}]."
        )


def omp_cholesky(
    D: xp.ndarray,
    x: xp.ndarray,
    n_nonzero: int,
) -> xp.ndarray:
    """
    Single-signal OMP via Cholesky updates (OMP-Cholesky).

    Parameters
    ----------
    D : xp.ndarray, shape (n_features, n_atoms)
        Normalized dictionary.
    x : xp.ndarray, shape (n_features,)
        Single input signal.
    n_nonzero : int
        Maximum number of non-zero coefficients (sparsity).

    Returns
    -------
    gamma : xp.ndarray, shape (n_atoms,)
        Sparse representation of x.
    """
    n_atoms = D.shape[1]
    residual = x.copy().astype(float)
    support: list[int] = []
    gamma = xp.zeros(n_atoms)

    # Cholesky factor of D[:,support].T @ D[:,support]
    L = xp.zeros((n_nonzero, n_nonzero))

    for k in range(n_nonzero):
        correlations = D.T @ residual
        j = int(xp.argmax(xp.abs(correlations)))
        support.append(j)

        # --- Cholesky update ---
        Ds = D[:, support]
        if k == 0:
            L[0, 0] = 1.0
        else:
            w = Ds[:, :-1].T @ D[:, j]  # (k,)
            # Solve L[:k,:k] * v = w
            v = linalg.solve_triangular(L[:k, :k], w, lower=True)
            l_new = xp.sqrt(max(1.0 - float(v @ v), 1e-14))
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
    DtX: xp.ndarray,
    G: xp.ndarray,
    n_nonzero: int,
) -> xp.ndarray:
    """
    Batch OMP — fastest variant; requires precomputed G = D'D and DtX = D'X.

    Parameters
    ----------
    DtX : xp.ndarray, shape (n_atoms, n_samples)
        Precomputed projections D.T @ X.
    G : xp.ndarray, shape (n_atoms, n_atoms)
        Precomputed Gram matrix D.T @ D.
    n_nonzero : int
        Sparsity level.

    Returns
    -------
    Gamma : xp.ndarray, shape (n_atoms, n_samples)
        Sparse representations (dense).
    """
    n_atoms, n_samples = DtX.shape
    Gamma = xp.zeros((n_atoms, n_samples))

    for i in range(n_samples):
        dtx = DtX[:, i]
        residual_proj = dtx.copy()
        support: list[int] = []
        L = xp.zeros((n_nonzero, n_nonzero))

        for k in range(n_nonzero):
            j = int(xp.argmax(xp.abs(residual_proj)))
            support.append(j)

            # Cholesky update using Gram matrix
            if k == 0:
                L[0, 0] = 1.0
            else:
                w = G[support[:-1], j]  # (k,)
                v = linalg.solve_triangular(L[:k, :k], w, lower=True)
                l_new = xp.sqrt(max(1.0 - float(v @ v), 1e-14))
                L[k, :k] = v
                L[k, k] = l_new

            # Solve (L L.T) c = DtX[support, i]
            rhs = dtx[support]
            Lt = L[: k + 1, : k + 1]
            y = linalg.solve_triangular(Lt, rhs, lower=True)
            c = linalg.solve_triangular(Lt.T, y, lower=False)

            # Update residual in projection space
            residual_proj = dtx - G[:, support] @ c

        Gamma[support, i] = c

    return Gamma


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
        X: xp.ndarray,
        D: xp.ndarray,
        G: xp.ndarray | None = None,
    ) -> xp.ndarray:
        """
        Compute sparse codes for each column of X.

        Parameters
        ----------
        X : xp.ndarray, shape (n_features, n_samples)
        D : xp.ndarray, shape (n_features, n_atoms)
        G : xp.ndarray or None, shape (n_atoms, n_atoms)
            Precomputed Gram matrix D.T @ D.  Required for 'batch' mode;
            computed internally if not supplied.

        Returns
        -------
        Gamma : xp.ndarray, shape (n_atoms, n_samples)
        """
        X = xp.asarray(X, dtype=float)
        D = xp.asarray(D, dtype=float)

        if X.ndim == 1:
            X = X[:, xp.newaxis]

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
        Gamma = xp.zeros((n_atoms, n_samples))
        for i in range(n_samples):
            Gamma[:, i] = omp_cholesky(D, X[:, i], T)
        return Gamma