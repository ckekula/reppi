"""
Solver for the FDDL coefficient sub-problem, Eq. (7):

    min_Xi  r(Ai, D, Xi) + lambda1*||Xi||_1 + lambda2*fi(Xi)

r(Ai, D, Xi) and fi(Xi) are both smooth and convex in Xi (for
eta >= 1, per the paper's Appendix A); only ||Xi||_1 is nonsmooth.
This is exactly the shape FISTA solves: min_x g(x) + h(x) with g smooth
and h = lambda1*||.||_1 prox-friendly (Beck & Teboulle 2009).

Why the core FISTA solver, not the ``FISTA`` sparse-coder class
------------------------------------------------------------------
``FISTA.encode`` assumes grad_f and the prox are separable per sample
column, and vectorizes all columns of X in one run (see its module
docstring). fi(Xi) breaks that assumption: mi = mean(Xi) couples every
column of Xi together, so the columns of Xi cannot be solved
independently. The lower-level ``reppi.sparse.fista.core.fista``
solver underneath the class has no such restriction — it accepts
arbitrary grad_f/prox_g callables operating on the whole iterate — so
it is used directly here with a custom whole-matrix grad_f.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from reppi.sparse.fista.core import fista as fista_core
from reppi.sparse.fista.utils import soft_threshold

from reppi.dictionary.fddl.utils import (
    OtherClassStats,
    coef_grad,
    coef_value,
    fidelity_grad,
    fidelity_value,
)


def solve_class_codes(
    Xi0: np.ndarray,
    i: int,
    D_list: Sequence[np.ndarray],
    D_full: np.ndarray,
    Ai: np.ndarray,
    atom_boundaries: dict[int, tuple[int, int]],
    other_stats: OtherClassStats,
    lambda1: float,
    lambda2: float,
    eta: float,
    max_iter: int = 200,
    tol: float | None = 1e-6,
    L0: float = 1.0,
    backtrack_eta: float = 2.0,
) -> tuple[np.ndarray, int, list[float]]:
    """
    Solve Eq. (7) for one class's coefficients Xi, with D and all other
    classes' coefficients fixed.

    Parameters
    ----------
    Xi0 : np.ndarray, shape (n_atoms, n_i)
        Warm-start iterate (typically the previous outer iteration's Xi).
    i : int
        Class index being solved.
    D_list, D_full : per-class sub-dictionaries and their horizontal
        stack.
    Ai : np.ndarray, shape (n_features, n_i)
        Training signals of class i.
    atom_boundaries : dict mapping class -> (start, end) atom row-range.
    other_stats : OtherClassStats
        Precomputed (size, mean) of every class j != i, from the
        current X.
    lambda1, lambda2, eta : Eq. (6)/(7) hyperparameters.
    max_iter, tol : solver stopping controls.
    L0, backtrack_eta : FISTA backtracking controls.

    Returns
    -------
    Xi : np.ndarray
    n_iter : int
    objective_history : list of float
    """

    def grad_smooth(Xi: np.ndarray) -> np.ndarray:
        return fidelity_grad(Xi, i, D_list, D_full, Ai, atom_boundaries) + lambda2 * coef_grad(
            Xi, i, other_stats, eta
        )

    def f(Xi: np.ndarray) -> float:
        return fidelity_value(Xi, i, D_list, Ai, atom_boundaries) + lambda2 * coef_value(
            Xi, i, other_stats, eta
        )

    def g(Xi: np.ndarray) -> float:
        return np.float32(lambda1 * np.sum(np.abs(Xi)))

    def prox_g(V: np.ndarray, t: float) -> np.ndarray:
        return soft_threshold(V, lambda1 * t)

    result = fista_core(
        grad_f=grad_smooth,
        prox_g=prox_g,
        x0=Xi0,
        f=f,
        g=g,
        L=None,
        mode="backtracking",
        L0=L0,
        eta=backtrack_eta,
        max_iter=max_iter,
        tol=tol,
    )
    return result.x, result.n_iter, result.objective_history