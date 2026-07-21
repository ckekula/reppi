"""
FISTA: Fast Iterative Shrinkage-Thresholding Algorithm.

Implements the general composite-minimization solver from:
    Beck, A. and Teboulle, M. "A Fast Iterative Shrinkage-Thresholding
    Algorithm for Linear Inverse Problems". SIAM J. Imaging Sciences,
    2(1), pp. 183-202, 2009.

Solves problems of the form

    min_x  F(x) := f(x) + g(x)

where f is smooth convex with Lipschitz-continuous gradient (constant
L(f)), and g is convex, possibly nonsmooth, with a computable proximal
operator. FISTA achieves the O(1/k^2) global rate proven in the paper
(Theorem 4.4), improving on the O(1/k) rate of plain ISTA (Theorem 3.1)
while keeping the same per-iteration cost (one gradient evaluation and
one prox evaluation).

Notation mirrors the paper directly: p_L(y) is the prox-gradient step
(eq. 2.6), Q_L(x, y) is the quadratic model (eq. 2.5), t_k / y_k are the
momentum sequence and extrapolation point (eqs. 4.2-4.3).

Backend
-------
The paper's analysis holds verbatim in any Hilbert space (Remark 2.1) —
all that's needed is an inner product and its induced norm. `x0` may
therefore be a numpy.ndarray *or* a torch.Tensor (CPU or GPU). If `x0` is
a GPU tensor and the supplied `grad_f`/`prox_g`/`f`/`g` operate on tensors
on that same device, the entire iterative loop runs on GPU with no
device transfers other than the (unavoidable) scalar convergence check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import numpy as np
import logging

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # torch is optional; numpy-only usage still works
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

Array = Union[np.ndarray, "torch.Tensor"]
ArrayFunc = Callable[[Array], Array]
ScalarFunc = Callable[[Array], float]
ProxFunc = Callable[[Array, float], Array]

logger = logging.getLogger(__name__)


@dataclass
class FISTAResult:
    """Result of a FISTA run."""

    x: Array
    n_iter: int
    converged: bool
    L: float
    objective_history: List[float] = field(default_factory=list)


def _is_torch_tensor(a: Array) -> bool:
    return _TORCH_AVAILABLE and isinstance(a, torch.Tensor)


def _inner(a: Array, b: Array) -> float:
    """Real inner product; valid for vectors or matrices (Frobenius),
    for both numpy arrays and torch tensors (CPU or GPU)."""
    if _is_torch_tensor(a):
        return float(torch.sum(a * b))
    return float(np.vdot(a, b))


def _norm_sq(a: Array) -> float:
    return _inner(a, a)


def _norm(a: Array) -> float:
    if _is_torch_tensor(a):
        return float(torch.linalg.norm(a))
    return float(np.linalg.norm(a))


def _prepare_x0(x0: Array) -> Array:
    """Cast to float32, preserving backend (numpy/torch) and, for torch,
    device (so a caller passing a CUDA tensor keeps the whole solve on GPU)."""
    if _is_torch_tensor(x0):
        return x0.detach().clone().to(dtype=torch.float32)
    return np.asarray(x0, dtype=np.float32).copy()


def _copy(a: Array) -> Array:
    return a.clone() if _is_torch_tensor(a) else a.copy()


def _q(
    x: Array,
    y: Array,
    f_y: float,
    grad_f_y: Array,
    L: float,
    g_x: float,
) -> float:
    """Quadratic model Q_L(x, y), eq. (2.5)."""
    diff = x - y
    return f_y + _inner(diff, grad_f_y) + 0.5 * L * _norm_sq(diff) + g_x


def _p_L(y: Array, grad_f_y: Array, L: float, prox_g: ProxFunc) -> Array:
    """p_L(y), eq. (2.6): prox-gradient step."""
    return prox_g(y - grad_f_y / L, 1.0 / L)


def fista(
    grad_f: ArrayFunc,
    prox_g: ProxFunc,
    x0: Array,
    f: Optional[ScalarFunc] = None,
    g: Optional[ScalarFunc] = None,
    L: Optional[float] = None,
    mode: str = "backtracking",
    L0: float = 1.0,
    eta: float = 2.0,
    max_iter: int = 500,
    tol: Optional[float] = 1e-8,
    max_backtrack_iter: int = 100,
) -> FISTAResult:
    """
    Run FISTA (Beck & Teboulle, 2009) to minimize F(x) = f(x) + g(x).

    Parameters
    ----------
    grad_f : callable
        Gradient of the smooth part, grad_f(x) -> array (same shape/type as x).
    prox_g : callable
        Proximal operator of g: prox_g(v, t) -> argmin_x g(x) + ||x-v||^2/(2t).
    x0 : np.ndarray or torch.Tensor
        Initial point. Any shape is supported (vector, matrix, ...); the
        Frobenius inner product is used internally, so the analysis holds
        verbatim in this more general Hilbert-space setting (Remark 2.1).
        Pass a torch.Tensor already on a GPU device to run the whole solve
        on GPU (grad_f/prox_g/f/g must then also operate on that device).
    f, g : callable, optional
        Value of the smooth / nonsmooth parts. Required when
        mode='backtracking' (needed to check the descent condition
        (3.2)/eq. before (4.1)). If supplied, also used to record
        objective_history.
    L : float, optional
        Lipschitz constant of grad_f. Required when mode='constant'.
    mode : {'constant', 'backtracking'}
        Stepsize strategy ("FISTA with constant stepsize" /
        "FISTA with backtracking" in Section 4 of the paper).
    L0 : float
        Initial Lipschitz estimate for backtracking mode (ignored for
        mode='constant').
    eta : float
        Backtracking growth factor, eta > 1.
    max_iter : int
        Maximum number of iterations.
    tol : float or None
        Relative stopping tolerance on ||x_k - x_{k-1}|| / max(1, ||x_{k-1}||).
        This is a practical stopping rule, not part of the original paper
        (which only bounds F(x_k) - F(x*)); pass tol=None to always run
        max_iter iterations.
    max_backtrack_iter : int
        Safety cap on the number of backtracking growth steps per outer
        iteration.

    Returns
    -------
    FISTAResult
    """
    if mode not in ("constant", "backtracking"):
        raise ValueError("mode must be 'constant' or 'backtracking'.")
    if mode == "constant" and (L is None or L <= 0):
        raise ValueError("mode='constant' requires a positive Lipschitz constant L.")
    if mode == "backtracking" and (f is None or g is None):
        raise ValueError(
            "mode='backtracking' requires both f and g to evaluate the descent condition."
        )
    if eta <= 1:
        raise ValueError("eta must be > 1.")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")

    x_prev = _prepare_x0(x0)  # x_0
    y = _copy(x_prev)  # y_1 = x_0
    t = 1.0  # t_1 = 1
    Lk = float(L) if mode == "constant" else float(L0)

    obj_history: List[float] = []
    converged = False
    n_iter = 0
    x_k = x_prev

    for k in range(1, max_iter + 1):
        n_iter = k
        grad_y = grad_f(y)
        f_xk: Optional[float] = None
        g_xk: Optional[float] = None

        if mode == "constant":
            x_k = _p_L(y, grad_y, Lk, prox_g)
        else:
            f_y = f(y)
            L_bar = Lk
            for _ in range(max_backtrack_iter):
                x_k = _p_L(y, grad_y, L_bar, prox_g)
                g_xk = g(x_k)
                f_xk = f(x_k)
                if f_xk <= _q(x_k, y, f_y, grad_y, L_bar, g_xk):
                    break
                L_bar *= eta
            else:
                raise RuntimeError(
                    "Backtracking line search failed to satisfy the descent "
                    "condition; check f/grad_f/L0/eta."
                )
            Lk = L_bar

        # eqs. (4.2)-(4.3): momentum update and extrapolation point.
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        x_diff = x_k - x_prev
        y = x_k + ((t - 1.0) / t_next) * x_diff

        if f is not None and g is not None:
            if f_xk is None or g_xk is None:  # mode == "constant"
                f_xk = f(x_k)
                g_xk = g(x_k)
            obj_history.append(f_xk + g_xk)

        if tol is not None:
            denom = max(1.0, _norm(x_prev))
            if _norm(x_diff) / denom < tol:
                x_prev = x_k
                converged = True
                break

        x_prev = x_k
        t = t_next

        if k % 50 == 0 or k == max_iter:
            logger.info("FISTA iteration %d/%d (L=%.4g)", k, max_iter, Lk)

    return FISTAResult(
        x=x_prev,
        n_iter=n_iter,
        converged=converged,
        L=Lk,
        objective_history=obj_history,
    )
