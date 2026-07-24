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
solver underneath the class has no such restriction -- it accepts
arbitrary grad_f/prox_g callables operating on the whole iterate -- so
it is used directly here with a custom whole-matrix grad_f.

``grad_smooth`` accumulates ``fidelity_grad`` and ``lambda2 *
coef_grad`` into a single buffer (via ``coef_grad``'s ``out``/``scale``
parameters) rather than allocating both full-size gradients separately
and summing them -- this is evaluated on every FISTA iteration, so
avoiding the extra same-size allocation matters for large classes.
"""

from __future__ import annotations

from typing import Sequence

import torch

from reppi.sparse.fista.core import fista as fista_core
from reppi.sparse.fista.utils import soft_threshold

from reppi.dictionary.fddl.utils import (
    OtherClassStats,
    coef_grad,
    coef_grad_affine,
    coef_value,
    coef_value_from_stats,
    fidelity_grad,
    fidelity_value,
    streaming_column_stats,
)


def solve_class_codes(
    Xi0: torch.Tensor,
    i: int,
    D_list: Sequence[torch.Tensor],
    D_full: torch.Tensor,
    Ai: torch.Tensor,
    atom_boundaries: dict[int, tuple[int, int]],
    other_stats: OtherClassStats,
    lambda1: float,
    lambda2: float,
    eta: float,
    max_iter: int = 200,
    tol: float | None = 1e-6,
    L0: float = 1.0,
    backtrack_eta: float = 2.0,
) -> tuple[torch.Tensor, int, list[float]]:
    """
    Solve Eq. (7) for one class's coefficients Xi, with D and all other
    classes' coefficients fixed.

    Parameters
    ----------
    Xi0 : torch.Tensor, shape (n_atoms, n_i)
        Warm-start iterate (typically the previous outer iteration's Xi).
        Must be on the device the solve should run on -- callers doing
        CPU-resident storage with GPU compute (see ``FDDL.fit``) are
        responsible for moving it there first.
    i : int
        Class index being solved.
    D_list, D_full : per-class sub-dictionaries and their horizontal
        stack.
    Ai : torch.Tensor, shape (n_features, n_i)
        Training signals of class i. Same device requirement as Xi0.
    atom_boundaries : dict mapping class -> (start, end) atom row-range.
    other_stats : OtherClassStats
        Precomputed (size, mean) of every class j != i, from the
        current X. Must live on the same device as Xi0/Ai.
    lambda1, lambda2, eta : Eq. (6)/(7) hyperparameters.
    max_iter, tol : solver stopping controls.
    L0, backtrack_eta : FISTA backtracking controls.

    Returns
    -------
    Xi : torch.Tensor
    n_iter : int
    objective_history : list of float
    """

    def grad_smooth(Xi: torch.Tensor) -> torch.Tensor:
        grad = fidelity_grad(Xi, i, D_list, D_full, Ai, atom_boundaries)
        coef_grad(Xi, i, other_stats, eta, out=grad, scale=lambda2)
        return grad

    def f(Xi: torch.Tensor) -> float:
        return fidelity_value(Xi, i, D_list, Ai, atom_boundaries) + lambda2 * coef_value(
            Xi, i, other_stats, eta
        )
 
    def g(Xi: torch.Tensor) -> float:
        return float(lambda1 * torch.sum(torch.abs(Xi)))
 
    def prox_g(V: torch.Tensor, t: float) -> torch.Tensor:
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


def solve_class_codes_chunked(
    Xi0_cpu: torch.Tensor,
    i: int,
    D_list: Sequence[torch.Tensor],
    D_full: torch.Tensor,
    Ai_cpu: torch.Tensor,
    atom_boundaries: dict[int, tuple[int, int]],
    other_stats: OtherClassStats,
    lambda1: float,
    lambda2: float,
    eta: float,
    device: torch.device,
    chunk_size: int,
    max_iter: int = 200,
    tol: float | None = 1e-6,
    L0: float = 1.0,
    backtrack_eta: float = 2.0,
) -> tuple[torch.Tensor, int, list[float]]:
    """
    Chunked equivalent of ``solve_class_codes`` for classes too large to
    hold as a full GPU tensor throughout the Eq. (7) FISTA solve (i.e.
    ``n_atoms_total * n_i`` alone approaches the device's memory budget,
    before any FISTA/gradient temporaries).

    ``fista_core`` itself is unmodified and untouched: it already
    operates generically on whatever device its input tensor lives on
    (see its module docstring). Passing it a **CPU-resident** ``Xi0_cpu``
    keeps its own iterate bookkeeping (``x_prev``/``y``/``x_k``/the
    momentum extrapolation) on the CPU -- the only GPU memory used at
    any point is transient, inside this function's ``grad_f``/``f``
    closures, bounded by ``chunk_size`` columns at a time.

    This relies on two facts about the Eq. (7) objective, both
    exploited to avoid ever needing a "coupled, so it needs the whole
    matrix" fallback:

      * r(Ai, D, Xi) (Eq. 4) has *no* cross-column coupling at all --
        ``fidelity_grad``/``fidelity_value`` are called unchanged, once
        per chunk, and their outputs are exactly summed/concatenated.
      * fi(Xi) (Eq. 5) couples columns *only* through the class mean;
        once that mean is known (one cheap reduction pass via
        ``streaming_column_stats``), its gradient/value reduce to an
        affine, per-column-separable expression (``coef_grad_affine``/
        ``coef_value_from_stats``) -- so the "coupled" term costs one
        extra O(n_atoms * n_i) reduction pass, not a fundamentally
        harder chunking problem.

    Mathematically identical to ``solve_class_codes`` (verified against
    it on random/synthetic problems small enough for both to run; see
    the module's companion tests) -- strictly more computation for the
    same result, trading FLOPs (multiple passes over the class's data
    per FISTA iteration, chunked) for bounded peak GPU memory
    (independent of class size). Prefer ``solve_class_codes`` when a
    class comfortably fits whole on the GPU.

    Parameters
    ----------
    Xi0_cpu, Ai_cpu : torch.Tensor
        CPU-resident warm-start iterate / training signals for this
        class.
    device : torch.device
        Compute device chunks are streamed to.
    chunk_size : int
        Maximum number of sample columns moved to ``device`` at once.
    Other parameters as in ``solve_class_codes``.

    Returns
    -------
    Xi : torch.Tensor
        CPU-resident (matching Xi0_cpu's device).
    n_iter : int
    objective_history : list of float
    """
    ni = Xi0_cpu.shape[1]

    def grad_smooth(y_cpu: torch.Tensor) -> torch.Tensor:
        mi, _ = streaming_column_stats(y_cpu, device, chunk_size)
        scale, offset, _ = coef_grad_affine(mi, ni, other_stats, eta)
        grad_cpu = torch.empty_like(y_cpu)
        for start in range(0, ni, chunk_size):
            end = min(start + chunk_size, ni)
            y_chunk = y_cpu[:, start:end].to(device)
            Ai_chunk = Ai_cpu[:, start:end].to(device)
            g_fid = fidelity_grad(y_chunk, i, D_list, D_full, Ai_chunk, atom_boundaries)
            g_fid += lambda2 * (scale * y_chunk - offset[:, None])
            grad_cpu[:, start:end] = g_fid.to("cpu")
            del y_chunk, Ai_chunk, g_fid
        return grad_cpu

    def f(y_cpu: torch.Tensor) -> float:
        mi, sq_sum = streaming_column_stats(y_cpu, device, chunk_size)
        _, _, m = coef_grad_affine(mi, ni, other_stats, eta)
        coef_val = coef_value_from_stats(mi, sq_sum, ni, other_stats, eta, m=m)
        fid_val = 0.0
        for start in range(0, ni, chunk_size):
            end = min(start + chunk_size, ni)
            y_chunk = y_cpu[:, start:end].to(device)
            Ai_chunk = Ai_cpu[:, start:end].to(device)
            fid_val += fidelity_value(y_chunk, i, D_list, Ai_chunk, atom_boundaries)
            del y_chunk, Ai_chunk
        return fid_val + lambda2 * coef_val

    def g(y_cpu: torch.Tensor) -> float:
        return float(lambda1 * torch.sum(torch.abs(y_cpu)))

    def prox_g(V_cpu: torch.Tensor, t: float) -> torch.Tensor:
        return soft_threshold(V_cpu, lambda1 * t)

    result = fista_core(
        grad_f=grad_smooth,
        prox_g=prox_g,
        x0=Xi0_cpu,
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