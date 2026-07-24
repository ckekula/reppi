"""
Math helpers for Fisher Discrimination Dictionary Learning (FDDL).

Yang, Zhang, Feng, Zhang. "Fisher Discrimination Dictionary Learning for
Sparse Representation", ICCV 2011.

This module implements the closed-form objective values and gradients
needed to solve the two alternating sub-problems of Table 1:

  * Eq. (7) — update the coding coefficients Xi of one class, D fixed.
  * Eq. (8) — update the sub-dictionary Di of one class, X fixed.

Derivation notes
-----------------
Eq. (7)'s discriminative coefficient term fi(Xi) and Eq. (8)'s
dictionary-update objective are given in the paper in a form that
implicitly depends on the *entire* structured D/X (all classes), with
everything except the block being solved held fixed. Both are
re-derived here directly from the paper's unambiguous definitions
(Eq. 4 for the fidelity term, Eq. 5 for the Fisher coefficient term)
rather than parsed from Eq. 7/8/Appendix A as typeset, since:

  * fi(Xi) is derived by isolating, from f(X) = tr(SW(X)) - tr(SB(X))
    + eta*||X||_F^2 (Eq. 5), every term that depends on Xi when all
    Xj, j != i, are held fixed. Both mi = mean(Xi) and the global mean
    m are linear/affine functions of Xi, which is what makes fi(Xi)
    convex-quadratic (Appendix A) without needing its printed matrix
    form. Verified against finite differences.
  * The Eq. (8) dictionary-update objective, expanded from Eq. (4) by
    collecting every term containing Di, reduces to a single
    generalized least-squares fit for Di over two stacked column
    blocks of the full training set — see ``build_di_update_system``.
    Verified against the three-term form directly.
  * ``fidelity_grad`` / ``coef_grad`` now reuse buffers instead of
    building several full-size temporaries per call. ``coef_grad``
    additionally accepts ``out``/``scale`` so callers (see
    ``reppi.dictionary.fddl.coding.solve_class_codes``) can accumulate
    directly into an existing gradient tensor rather than allocating a
    second same-size tensor just to add it in.
  * ``build_di_update_system_streaming``: a chunked/streamed
    equivalent of ``build_di_update_system`` (kept below, unchanged,
    for reference/tests) that never materializes a full-dataset-width
    tensor on the compute device. Intended for callers that keep the
    per-class training signals/coefficients (``A_list``/``X_list``) on
    the CPU and stream them to the GPU in bounded-size chunks -- see
    ``FDDL.fit``.

Both are unit-tested against finite-difference / direct-computation
checks; see the module's companion tests.

Performance notes
------------------
Several formulas below are written in closed form rather than as the
direct per-class loop the paper's notation suggests, because the loop
form is either mathematically redundant (the loop total is already
available from cached running sums) or gets called from inside the
Eq. (7)/(8) inner solves many times per outer iteration.

Backend
-------
Torch-only (GPU-only operation): every array parameter here (Xi,
D_list entries, Ai, X_list entries) is a torch.Tensor, not a numpy
array. Scalar objective values are still returned as plain Python
floats (via `float(...)`, which triggers a small, unavoidable
device->host sync — the same cost paid by any convergence check).

`n_components`/sizes/boundaries stay plain Python ints/dicts — they're
bookkeeping, not compute, and are built once outside the solver loop.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------
# Partitioning helpers
# ---------------------------------------------------------------------

def block_boundaries(sizes: Sequence[int]) -> dict[int, tuple[int, int]]:
    """
    Map a list of per-class block sizes to (start, end) column/row
    ranges in the order the blocks are concatenated.

    Used both for atom ranges (Di's columns within D) and sample
    ranges (Ai's columns within the full training set A).
    """
    boundaries: dict[int, tuple[int, int]] = {}
    start = 0
    for k, size in enumerate(sizes):
        boundaries[k] = (start, start + size)
        start += size
    return boundaries


def resolve_atoms_per_class(
    n_components: int | Sequence[int], n_classes: int
) -> list[int]:
    """
    Resolve ``n_components`` into an explicit per-class atom count.

    If an int is given, atoms are split as evenly as possible across
    classes (remainder assigned to the last class), matching the
    paper's convention that all pi are usually set equal (Sec. 6.1).
    If a sequence is given, it is used as-is (one count per class).
    """
    if isinstance(n_components, (int, np.integer)):
        if n_components < n_classes:
            raise ValueError(
                f"n_components={n_components} must be >= n_classes={n_classes}."
            )
        base = n_components // n_classes
        counts = [base] * n_classes
        counts[-1] += n_components - base * n_classes
        return counts

    counts = list(n_components)
    if len(counts) != n_classes:
        raise ValueError(
            f"n_components has {len(counts)} entries but there are "
            f"{n_classes} classes."
        )
    if any(p < 1 for p in counts):
        raise ValueError("Every per-class atom count must be >= 1.")
    return counts


# ---------------------------------------------------------------------
# Eq. (4): discriminative fidelity term r(Ai, D, Xi), restricted to Xi
# ---------------------------------------------------------------------

def fidelity_value(
    Xi: torch.Tensor,
    i: int,
    D_list: Sequence[torch.Tensor],
    Ai: torch.Tensor,
    atom_boundaries: dict[int, tuple[int, int]],
) -> float:
    """r(Ai, D, Xi) of Eq. (4), as a function of Xi with D fixed. Accumulates
    as a tensor and syncs to a Python float exactly once at the end.
    """
    recon = torch.zeros_like(Ai)
    val = torch.zeros((), device=Ai.device, dtype=Ai.dtype)
    Di_recon = None
    for k, (s, e) in atom_boundaries.items():
        Pk = D_list[k] @ Xi[s:e, :]
        recon = recon + Pk
        if k == i:
            Di_recon = Pk
        else:
            val = val + (Pk ** 2).sum()
    val = val + (Ai - recon).pow(2).sum()  # D_full @ Xi
    val = val + (Ai - Di_recon).pow(2).sum()
    return float(val)


def fidelity_grad(
    Xi: torch.Tensor,
    i: int,
    D_list: Sequence[torch.Tensor],
    D_full: torch.Tensor,
    Ai: torch.Tensor,
    atom_boundaries: dict[int, tuple[int, int]],
) -> torch.Tensor:
    """Gradient of r(Ai, D, Xi) (Eq. 4) with respect to Xi.

    Written to keep at most one Xi-shaped scratch tensor alive at a
    time (in addition to the returned ``grad``), rather than the
    several same-shape temporaries an unfused expression would create
    -- this term is evaluated on every FISTA iteration, so its peak
    memory matters.
    """
    residual = D_full @ Xi
    residual -= Ai
    grad = D_full.T @ residual
    grad *= 2.0
    del residual

    for k, (s, e) in atom_boundaries.items():
        Dk = D_list[k]
        Xik = Xi[s:e, :]
        Pk = Dk @ Xik
        if k == i:
            Pk -= Ai
        contrib = Dk.T @ Pk
        grad[s:e, :] += 2.0 * contrib
        del Pk, contrib
    return grad


# ---------------------------------------------------------------------
# Eq. (5): discriminative coefficient term f(X), restricted to Xi
# ---------------------------------------------------------------------

class OtherClassStats:
    """
    Statistics of every class j != i, as needed by ``coef_value`` /
    ``coef_grad`` to see a fixed target while solving Eq. (7) for
    class i:

      * n_total              : sum of sizes over j != i
      * weighted_mean_sum    : sum_j sizes[j] * mean(Xj)      (tensor)
      * weighted_sq_norm_sum : sum_j sizes[j] * ||mean(Xj)||^2 (float)

    These three quantities are all ``coef_value``/``coef_grad`` need;
    the previous per-class ``means`` dict is not, since every place
    that consumed it summed over it in a way that reduces to one of
    these three cached totals (see the closed forms in
    ``coef_value``/``coef_grad`` below).

    Prefer constructing this via ``GlobalMeanTracker.exclude(i)``
    instead of the constructor directly when iterating over all
    classes in a sweep -- the tracker maintains these totals
    incrementally in O(1) per class instead of rebuilding them from
    scratch (O(n_classes)) for every one of the n_classes calls in a
    sweep.

    ``weighted_mean_sum`` must live on the same device as the ``Xi``
    passed to ``coef_grad``/``coef_value`` (see ``GlobalMeanTracker``'s
    ``device`` argument).
    """

    __slots__ = ("n_total", "weighted_mean_sum", "weighted_sq_norm_sum")

    def __init__(
        self, X_list: Sequence[torch.Tensor], sizes: Sequence[int], exclude: int
    ) -> None:
        n_total = 0
        weighted_sum = None
        weighted_sq_norm_sum = 0.0
        for k, Xk in enumerate(X_list):
            if k == exclude:
                continue
            mk = Xk.mean(dim=1)
            n_total += sizes[k]
            weighted_sq_norm_sum += sizes[k] * float(torch.sum(mk ** 2))
            contribution = sizes[k] * mk
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        self.n_total = n_total  # sum of sizes over j != exclude only
        self.weighted_mean_sum = (
            weighted_sum if weighted_sum is not None else None
        )  # None only when n_classes == 1
        self.weighted_sq_norm_sum = weighted_sq_norm_sum


class GlobalMeanTracker:
    """
    Maintains running per-class means, and their weighted sum /
    weighted squared-norm sum, across a full X-update sweep (one
    solve of Eq. 7 per class).

    Calling ``OtherClassStats(X_list, sizes, exclude=i)`` fresh for
    every class rebuilds every other class's mean from scratch each
    time: O(n_classes) work, done once per class, so O(n_classes^2)
    per sweep. Since exactly one class's X changes between successive
    calls within a sweep (Gauss-Seidel: class i's solve only touches
    X_list[i]), the "all classes except i" totals can instead be
    maintained incrementally:

        tracker = GlobalMeanTracker(X_list, sizes, device=gpu)
        for i in range(n_classes):
            stats = tracker.exclude(i)               # O(1)
            X_list[i] = solve_class_codes(..., stats, ...)
            tracker.update(i, X_list[i])              # O(1)

    which is O(n_classes) per sweep in total.

    ``device`` : if given, the per-class means (and therefore every
    downstream ``OtherClassStats.weighted_mean_sum``) are moved to and
    kept on that device -- independent of what device ``X_list`` lives
    on. This matters when ``X_list`` is CPU-resident storage (see
    ``FDDL.fit``'s CPU-offload path) but each class's coding solve
    runs on GPU: the means are a handful of small vectors (one
    n_atoms-length vector per class) so keeping them permanently on
    the compute device is essentially free, and lets
    ``coef_grad``/``coef_value`` operate without a device mismatch.
    """

    def __init__(
        self,
        X_list: Sequence[torch.Tensor],
        sizes: Sequence[int],
        device: torch.device | None = None,
    ) -> None:
        self.sizes = list(sizes)
        means = [Xk.mean(dim=1) for Xk in X_list]
        if device is not None:
            means = [m.to(device) for m in means]
        self.means = means
        self.n_total = sum(sizes)
        self.weighted_mean_sum = sum(
            s * m for s, m in zip(self.sizes, self.means)
        )
        self.weighted_sq_norm_sum = sum(
            s * float(torch.sum(m ** 2)) for s, m in zip(self.sizes, self.means)
        )

    def exclude(self, i: int) -> OtherClassStats:
        """O(1) view of all classes except i, as an OtherClassStats."""
        s = self.sizes[i]
        mi = self.means[i]
        stats = object.__new__(OtherClassStats)
        stats.n_total = self.n_total - s
        stats.weighted_mean_sum = self.weighted_mean_sum - s * mi
        stats.weighted_sq_norm_sum = self.weighted_sq_norm_sum - s * float(
            torch.sum(mi ** 2)
        )
        return stats

    def update(self, i: int, Xi_new: torch.Tensor) -> None:
        """O(1) update after class i's coefficients have been re-solved.

        ``Xi_new`` may live on any device (e.g. CPU, for a class whose
        solve was streamed rather than held whole on the GPU) -- the
        resulting mean is moved to match ``self.means``'s device before
        use, so the tracker itself always stays consistent regardless
        of where each class's coefficients happen to be stored.
        """
        s = self.sizes[i]
        old_mean = self.means[i]
        new_mean = Xi_new.mean(dim=1)
        if new_mean.device != old_mean.device:
            new_mean = new_mean.to(old_mean.device)
        self.weighted_mean_sum += s * (new_mean - old_mean)
        self.weighted_sq_norm_sum += s * (
            float(torch.sum(new_mean ** 2)) - float(torch.sum(old_mean ** 2))
        )
        self.means[i] = new_mean


def coef_value(Xi: torch.Tensor, i: int, stats: OtherClassStats, eta: float) -> float:
    """fi(Xi): terms of f(X) = tr(SW(X)) - tr(SB(X)) + eta*||X||_F^2 (Eq. 5)
    that depend on Xi, with all Xj, j != i, fixed (captured in ``stats``).

    Accumulates as a tensor and syncs to a Python float exactly once at the end.
    """
    ni = Xi.shape[1]
    n = stats.n_total + ni
    mi = Xi.mean(dim=1)
    c0 = 0.0 if stats.weighted_mean_sum is None else stats.weighted_mean_sum / n
    m = (Xi.sum(dim=1) / n) + c0

    within = (Xi - mi[:, None]).pow(2).sum()

    between = ni * (mi - m).pow(2).sum()
    if stats.weighted_mean_sum is not None:
        # closed form of sum_{k!=i} sizes[k]*||mk-m||^2 from the cached
        # running totals, instead of looping over every other class's
        # mean: sum_k sk*||mk-m||^2
        #     = sum_k sk*||mk||^2 - 2*m.(sum_k sk*mk) + ||m||^2*sum_k sk
        # (weighted_sq_norm_sum is a plain float -- a single O(1)-per-outer-
        # iteration sync in GlobalMeanTracker, not on this hot path.)
        between = between + (
            stats.weighted_sq_norm_sum
            - 2.0 * (m @ stats.weighted_mean_sum)
            + stats.n_total * (m ** 2).sum()
        )

    val = within - between + eta * (Xi ** 2).sum()
    return float(val)


def coef_grad(
    Xi: torch.Tensor,
    i: int,
    stats: OtherClassStats,
    eta: float,
    out: torch.Tensor | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Gradient of fi(Xi) with respect to Xi.

    Computed with at most two Xi-shaped scratch buffers alive at once
    (``result``/``between``), rather than building within_grad,
    between_grad, and the final combination as three-plus separate
    same-shape temporaries.

    Parameters
    ----------
    out : torch.Tensor, optional
        If given, ``scale * coef_grad(Xi, ...)`` is added into ``out``
        in place (``out`` is returned) instead of allocating a fresh
        tensor for the caller to add in themselves. Used by
        ``solve_class_codes`` to accumulate directly into the fidelity
        gradient, avoiding one extra full-size allocation per FISTA
        iteration.
    scale : float
        Multiplier applied when accumulating into ``out``. Ignored
        (result returned unscaled) when ``out`` is None.
    """
    ni = Xi.shape[1]
    n = stats.n_total + ni
    opts = {"device": Xi.device, "dtype": Xi.dtype}
    a = torch.full((ni,), 1.0 / n, **opts)
    mi = Xi.mean(dim=1)

    c0 = 0.0 if stats.weighted_mean_sum is None else stats.weighted_mean_sum / n
    m = Xi @ a + c0

    v = torch.full((ni,), 1.0 / ni, **opts) - a  # = u_i - a

    # result <- within_grad = 2*(Xi - mi[:,None])
    result = Xi.clone()
    result -= mi[:, None]
    result *= 2.0

    # between <- between_grad = 2*ni*outer(mi-m, v) [- 2*outer(other_sum, a)]
    between = torch.outer(mi - m, v)
    between *= 2.0 * ni
    if stats.weighted_mean_sum is not None:
        other_sum = stats.weighted_mean_sum - stats.n_total * m
        correction = torch.outer(other_sum, a)
        correction *= 2.0
        between -= correction

    result -= between
    result.add_(Xi, alpha=2.0 * eta)

    if out is None:
        return result
    out.add_(result, alpha=scale)
    return out


def streaming_column_stats(
    Xi_cpu: torch.Tensor, device: torch.device, chunk_size: int
) -> tuple[torch.Tensor, float]:
    """
    Compute (column-mean, sum-of-squares) of a possibly CPU-resident
    ``(n_atoms, ni)`` tensor by streaming column-chunks through
    ``device`` -- the only reduction ``coef_grad_affine``/
    ``coef_value_from_stats`` need over the *current* FISTA iterate
    (which, unlike the per-outer-iteration class mean tracked by
    ``GlobalMeanTracker``, changes every FISTA step and so cannot be
    cached across calls).

    Returns
    -------
    mi : torch.Tensor, shape (n_atoms,), on ``device``
    sq_sum : float
        sum(Xi ** 2) over every element.
    """
    ni = Xi_cpu.shape[1]
    col_sum = torch.zeros(Xi_cpu.shape[0], device=device, dtype=Xi_cpu.dtype)
    sq_sum = 0.0
    for start in range(0, ni, chunk_size):
        end = min(start + chunk_size, ni)
        chunk = Xi_cpu[:, start:end].to(device)
        col_sum += chunk.sum(dim=1)
        sq_sum += float(torch.sum(chunk ** 2))
        del chunk
    mi = col_sum / ni
    return mi, sq_sum


def coef_grad_affine(
    mi: torch.Tensor, ni: int, stats: OtherClassStats, eta: float
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Closed-form ``(scale, offset, m)`` such that, for *every* column j
    of Xi, ``coef_grad(Xi)[:, j] == scale * Xi[:, j] - offset``.

    Why this is possible: in the unchunked ``coef_grad``, both ``a``
    (length ``ni``, value ``1/n`` everywhere) and ``v`` (length ``ni``,
    value ``1/ni - 1/n`` everywhere) are *uniform* vectors -- every
    entry is the same constant. Consequently every column of
    ``outer(mi - m, v)`` and ``outer(other_sum, a)`` is identical
    (outer product against a constant vector just repeats the first
    argument, scaled). So ``between_grad`` -- normally the part of the
    gradient that looks like it needs the whole matrix -- is actually
    the *same* vector broadcast to every column. That leaves the
    gradient affine and fully column-separable, needing only ``mi``
    (and the handful of scalar/vector quantities derived from it, all
    ``O(n_atoms)``) rather than the full ``Xi``.

    Verified equal to ``coef_grad`` on random inputs; see the module's
    companion tests.

    Returns
    -------
    scale : float
    offset : torch.Tensor, shape (n_atoms,)
    m : torch.Tensor, shape (n_atoms,)
        Returned so callers needing ``coef_value_from_stats`` too (e.g.
        the ``f`` closure in ``solve_class_codes_chunked``) don't have
        to recompute it.
    """
    n = stats.n_total + ni
    v0 = 1.0 / ni - 1.0 / n
    c0 = 0.0 if stats.weighted_mean_sum is None else stats.weighted_mean_sum / n
    m = (ni / n) * mi + c0

    beta = (2.0 * ni * v0) * (mi - m)
    if stats.weighted_mean_sum is not None:
        other_sum = stats.weighted_mean_sum - stats.n_total * m
        beta = beta - (2.0 / n) * other_sum

    scale = 2.0 + 2.0 * eta
    offset = 2.0 * mi + beta
    return scale, offset, m


def coef_value_from_stats(
    mi: torch.Tensor,
    sq_sum: float,
    ni: int,
    stats: OtherClassStats,
    eta: float,
    m: torch.Tensor | None = None,
) -> float:
    """
    fi(Xi) (Eq. 5, restricted to Xi) computed purely from the
    precomputed mean (``mi``) and sum-of-squares (``sq_sum``) of Xi --
    see ``coef_grad_affine``'s docstring for why the value, like the
    gradient, doesn't actually need Xi itself once those are known
    (``within`` reduces to ``sq_sum - ni*||mi||^2`` via
    ``sum_j||x_j-mi||^2 = sum_j||x_j||^2 - 2*ni*||mi||^2 + ni*||mi||^2``,
    and the regularizer is exactly ``eta*sq_sum``).

    Pass ``m`` (as returned by ``coef_grad_affine``) to avoid
    recomputing it. Verified equal to ``coef_value`` on random inputs;
    see the module's companion tests.
    """
    n = stats.n_total + ni
    c0 = 0.0 if stats.weighted_mean_sum is None else stats.weighted_mean_sum / n
    if m is None:
        m = (ni / n) * mi + c0

    within = sq_sum - ni * float(mi @ mi)
    between = ni * float((mi - m) @ (mi - m))
    if stats.weighted_mean_sum is not None:
        between += (
            stats.weighted_sq_norm_sum
            - 2.0 * float(m @ stats.weighted_mean_sum)
            + stats.n_total * float(m @ m)
        )
    return within - between + eta * sq_sum


def fidelity_value_chunked(
    Xi_cpu: torch.Tensor,
    i: int,
    D_list: Sequence[torch.Tensor],
    Ai_cpu: torch.Tensor,
    atom_boundaries: dict[int, tuple[int, int]],
    device: torch.device,
    chunk_size: int,
) -> float:
    """
    Streamed equivalent of ``fidelity_value`` for a possibly
    CPU-resident class. r(Ai, D, Xi) (Eq. 4) is a sum of per-sample
    (per-column) terms with no cross-column coupling, so it is exactly
    additive over column-chunks -- this simply calls the existing,
    already-verified ``fidelity_value`` once per chunk and sums.
    """
    ni = Xi_cpu.shape[1]
    total = 0.0
    for start in range(0, ni, chunk_size):
        end = min(start + chunk_size, ni)
        Xi_chunk = Xi_cpu[:, start:end].to(device)
        Ai_chunk = Ai_cpu[:, start:end].to(device)
        total += fidelity_value(Xi_chunk, i, D_list, Ai_chunk, atom_boundaries)
        del Xi_chunk, Ai_chunk
    return total


def global_fisher_value(X_list: Sequence[torch.Tensor], eta: float) -> float:
    """f(X) = tr(SW(X)) - tr(SB(X)) + eta*||X||_F^2 (Eq. 5), computed
    directly over the full X for convergence tracking / Eq. (6) reporting.

    Runs on whatever device ``X_list`` lives on -- when ``X_list`` is
    CPU-resident storage (see ``FDDL.fit``'s CPU-offload path), this
    runs as a one-off CPU reduction rather than requiring the full
    dataset to be moved to the GPU. It is called once per outer
    iteration (for logging/convergence), not per FISTA iteration, so
    the CPU cost is not on the hot path.
    """
    sizes = [Xk.shape[1] for Xk in X_list]
    n = sum(sizes)
    means = [Xk.mean(dim=1) for Xk in X_list]
    m = sum(sizes[k] * means[k] for k in range(len(X_list))) / n

    sq_norms = [float(torch.sum(Xk ** 2)) for Xk in X_list]  # shared by sw & reg
    sw = sum(sq_norms[k] - sizes[k] * float(means[k] @ means[k]) for k in range(len(X_list)))
    sb = sum(sizes[k] * float(torch.sum((means[k] - m) ** 2)) for k in range(len(X_list)))
    reg = sum(sq_norms)
    return sw - sb + eta * reg


# ---------------------------------------------------------------------
# Eq. (8): dictionary-update objective for Di, restricted from Eq. (4)
# ---------------------------------------------------------------------

def build_di_update_system(
    i: int,
    D_i: torch.Tensor,
    D_full_sum: torch.Tensor,
    A_list: Sequence[torch.Tensor],
    atom_boundaries: dict[int, tuple[int, int]],
    X_full_stacked: torch.Tensor,
    A_full: torch.Tensor,
    sample_boundaries: dict[int, tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the sufficient statistics (A_stat, B_stat) whose solution
    minimizes Eq. (8). See ``build_di_update_system_streaming`` for a
    chunked equivalent that does not require ``A_full``/``X_full_stacked``/
    ``D_full_sum`` to be materialized on the compute device -- prefer
    that version when the full dataset does not comfortably fit in
    device memory alongside everything else. This full-materialization
    version is kept for reference and as the ground truth in the
    module's equivalence tests.

        J(Di) = ||A - Di*X^i - sum_{j!=i} Dj*X^j||_F^2
              + ||Ai - Di*Xi^i||_F^2
              + sum_{j!=i} ||Di*Xj^i||_F^2

    where X^k denotes the coefficients of the *full* training set A
    over sub-dictionary Dk (i.e. the k-th row-block, stacked across all
    classes' columns), and Xj^k is that row-block restricted to class
    j's columns.

    R = A - sum_{j!=i} Dj @ X^j                  (term 1 residual)
    T = zeros_like(A), with class-i columns = Ai  (terms 2 and 3)
    Y = [R | T],  Z = [X^i | X^i]

        Z@Z.T = 2 * (X^i @ X^i.T)
        Y@Z.T = (R + T) @ X^i.T,  where (R+T) only differs from R on
                                   class i's columns (R[:,i-cols] + Ai)

    Returns
    -------
    A_stat, B_stat : torch.Tensor
        Sufficient statistics for the dictionary update, shapes
        (p_i, p_i) and (n_features, p_i).
    """
    s, e = atom_boundaries[i]
    X_i_full = X_full_stacked[s:e, :]  # X^i, shape (p_i, n_samples)

    # Exclude class i's own contribution from the running full sum.
    R = A_full - D_full_sum + D_i @ X_i_full

    s2, e2 = sample_boundaries[i]
    A_stat = 2.0 * (X_i_full @ X_i_full.T)
    B_stat = R @ X_i_full.T + A_list[i] @ X_i_full[:, s2:e2].T
    return A_stat, B_stat


def build_di_update_system_streaming(
    i: int,
    D_list: Sequence[torch.Tensor],
    A_list_cpu: Sequence[torch.Tensor],
    X_list_cpu: Sequence[torch.Tensor],
    atom_boundaries: dict[int, tuple[int, int]],
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Streaming/chunked equivalent of ``build_di_update_system`` for the
    Eq. (8) dictionary-update sufficient statistics of class ``i``.

    Never materializes the full-dataset-width A_full / D_full_sum /
    X_full_stacked tensors on ``device``: ``A_list_cpu``/``X_list_cpu``
    are the "source of truth" storage (expected to live on the CPU),
    and are streamed to ``device`` in column-chunks of at most
    ``chunk_size`` samples, one class at a time, with A_stat/B_stat
    accumulated on ``device`` as running sums. Peak device memory for
    this function is therefore O(chunk_size * (n_features + p_i)),
    independent of the total number of training samples.

    Mathematically identical to ``build_di_update_system`` -- the key
    fact that makes streaming possible is that the residual
    R = A - sum_{j!=i} Dj@X^j does *not* depend on Di (the "+ Di@X^i"
    term added back exactly cancels the Di@X^i term inside
    D_full_sum, see ``build_di_update_system``), so R -- and therefore
    A_stat/B_stat -- can be built one sample-chunk at a time without
    ever needing class i's own dictionary or the full dataset in
    memory at once. Verified equal to ``build_di_update_system`` on
    random inputs across multiple chunk sizes and class-size splits;
    see the module's companion tests.

    Trade-off vs. the non-streaming version: this recomputes
    "sum_{j!=i} Dj@Xj_chunk" from scratch for every class's dictionary
    update (rather than reusing a single running D_full_sum patched
    incrementally across the sweep), trading extra FLOPs for a memory
    footprint bounded by ``chunk_size`` instead of the full dataset.

    Parameters
    ----------
    i : int
        Class index whose dictionary sufficient statistics are built.
    D_list : sequence of torch.Tensor
        Per-class sub-dictionaries, resident on ``device``.
    A_list_cpu, X_list_cpu : sequence of torch.Tensor
        Per-class training signals / coding coefficients. Any device
        is accepted (each chunk is moved to ``device`` via ``.to()``,
        a no-op copy if already there); CPU-resident tensors are the
        intended "large storage" case.
    atom_boundaries : dict mapping class -> (start, end) atom row-range.
    device : torch.device
        Compute device chunks are streamed to.
    chunk_size : int
        Maximum number of sample columns moved to ``device`` at once.

    Returns
    -------
    A_stat, B_stat : torch.Tensor
        Sufficient statistics for the dictionary update (on
        ``device``), shapes (p_i, p_i) and (n_features, p_i) --
        identical (up to floating-point summation order) to
        ``build_di_update_system``'s return values.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    s_i, e_i = atom_boundaries[i]
    p_i = e_i - s_i
    n_features = A_list_cpu[i].shape[0]
    dtype = D_list[i].dtype

    A_stat = torch.zeros((p_i, p_i), device=device, dtype=dtype)
    B_stat = torch.zeros((n_features, p_i), device=device, dtype=dtype)

    other_classes = [j for j in range(len(D_list)) if j != i]

    for k in range(len(D_list)):
        Ak_cpu = A_list_cpu[k]
        Xk_cpu = X_list_cpu[k]
        n_k = Ak_cpu.shape[1]
        for start in range(0, n_k, chunk_size):
            end = min(start + chunk_size, n_k)
            Ak_chunk = Ak_cpu[:, start:end].to(device)
            Xk_chunk = Xk_cpu[:, start:end].to(device)

            # R_chunk = A_chunk - sum_{j != i} Dj @ Xk_chunk[atom_boundaries[j]]
            R_chunk = Ak_chunk.clone()
            for j in other_classes:
                sj, ej = atom_boundaries[j]
                R_chunk -= D_list[j] @ Xk_chunk[sj:ej, :]

            Xi_chunk = Xk_chunk[s_i:e_i, :]
            A_stat += 2.0 * (Xi_chunk @ Xi_chunk.T)
            B_stat += R_chunk @ Xi_chunk.T
            if k == i:
                B_stat += Ak_chunk @ Xi_chunk.T

            del Ak_chunk, Xk_chunk, R_chunk, Xi_chunk

    return A_stat, B_stat