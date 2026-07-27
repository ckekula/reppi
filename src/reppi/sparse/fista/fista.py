"""
FISTA sparse coder for L1-regularized least squares.

Solves, for each signal column of X,

    min_x  ||D x - b||^2 + alpha * ||x||_1

using the Fast Iterative Shrinkage-Thresholding Algorithm of:
    Beck, A. and Teboulle, M. "A Fast Iterative Shrinkage-Thresholding
    Algorithm for Linear Inverse Problems". SIAM J. Imaging Sciences,
    2(1), pp. 183-202, 2009.

grad_f and the soft-threshold prox operator are both separable across
signals, so all columns of X are solved jointly in a single vectorized
FISTA run rather than looping signal by signal: the paper's analysis
holds verbatim in R^{n_atoms x n_samples} equipped with the Frobenius
inner product (Remark 2.1), so this is a faithful, not an approximate,
generalization.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from reppi.base import BaseSparseCoder
from reppi.sparse.fista.core import fista_core
from reppi.sparse.fista.utils import lipschitz_constant_lsq, soft_threshold
from reppi.sparse.utils import _check_dict_normalized


class FISTA(BaseSparseCoder):
    """
    FISTA sparse coder for L1-regularized least squares.

    Parameters
    ----------
    alpha : float
        L1 regularization weight (lambda in the paper). Must be >= 0.
    mode : {'constant', 'backtracking'}
        Stepsize strategy (Section 4 of the paper).
        'constant'     — requires (or computes) the Lipschitz constant
                          L(f) = 2 * ||D||_2^2 up front.
        'backtracking' — no Lipschitz constant needed; adapts L via a
                          line search, useful when D is unknown/expensive
                          to bound tightly.
    max_iter : int
        Maximum number of FISTA iterations.
    tol : float or None
        Relative convergence tolerance on the iterate (practical stopping
        rule, not from the paper). None disables early stopping.
    L0 : float
        Initial Lipschitz estimate for backtracking mode.
    eta : float
        Backtracking growth factor (> 1).
    track_objective : bool
        If True, evaluate and store F(x_k) at every iteration (adds
        overhead beyond what 'backtracking' mode already requires).
    check_dict : bool
        Whether to verify that dictionary atoms are unit-norm (default True).

    Attributes
    ----------
    n_iter_ : int
        Number of iterations run in the last call to `encode`.
    objective_history_ : list of float
        F(x_k) per iteration from the last call to `encode` (populated
        whenever mode='backtracking' or track_objective=True).
    """

    def __init__(
        self,
        alpha: float,
        mode: str = "backtracking",
        max_iter: int = 500,
        tol: float | None = 1e-8,
        L0: float = 1.0,
        eta: float = 2.0,
        track_objective: bool = False,
        check_dict: bool = True,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if mode not in ("constant", "backtracking"):
            raise ValueError("mode must be 'constant' or 'backtracking'.")
        self.alpha = alpha
        self.mode = mode
        self.max_iter = max_iter
        self.tol = tol
        self.L0 = L0
        self.eta = eta
        self.track_objective = track_objective
        self.check_dict = check_dict

    def encode(
        self,
        X: np.ndarray,
        D: np.ndarray,
        L: float | None = None,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute sparse codes for each column of X.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        D : np.ndarray, shape (n_features, n_atoms)
        L : float, optional
            Lipschitz constant of grad f. If not given, computed as
            2 * ||D||_2^2 (Example 2.2). Used directly in 'constant'
            mode, and as the initial L0 estimate in 'backtracking' mode
            (overriding the constructor's L0).
        x0 : np.ndarray, optional
            Initial point, shape (n_atoms, n_samples). Defaults to zeros.

        Returns
        -------
        Gamma : np.ndarray, shape (n_atoms, n_samples)
        """
        X = np.asarray(X, dtype=np.float32)
        D = np.asarray(D, dtype=np.float32)

        if X.ndim == 1:
            X = X[:, np.newaxis]

        if self.check_dict:
            _check_dict_normalized(D)

        n_atoms = D.shape[1]
        n_samples = X.shape[1]

        if self.mode == "constant":
            L_est = lipschitz_constant_lsq(D) if L is None else np.float32(L)
            L0_init = L_est
        else:  # backtracking
            L_est = None if L is None else np.float32(L)
            L0_init = L_est if L_est is not None else self.L0

        def grad_f(Z: np.ndarray) -> np.ndarray:
            return 2.0 * (D.T @ (D @ Z - X))

        need_fg = self.mode == "backtracking" or self.track_objective
        f = (lambda Z: np.float32(np.sum((D @ Z - X) ** 2))) if need_fg else None
        g = (lambda Z: np.float32(self.alpha * np.sum(np.abs(Z)))) if need_fg else None

        def prox_g(V: np.ndarray, t: float) -> np.ndarray:
            return soft_threshold(V, self.alpha * t)

        gamma0 = (
            np.zeros((n_atoms, n_samples), dtype=np.float32) if x0 is None else np.asarray(x0, dtype=np.float32)
        )

        result = fista_core(
            grad_f=grad_f,
            prox_g=prox_g,
            x0=gamma0,
            f=f,
            g=g,
            L=L_est,
            mode=self.mode,
            L0=L0_init,
            eta=self.eta,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        self.n_iter_ = result.n_iter
        self.objective_history_ = result.objective_history
        return result.x