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

Both are unit-tested against finite-difference / direct-computation
checks; see the module's companion tests.

Performance notes
------------------
Several formulas below are written in closed form rather than as the
direct per-class loop the paper's notation suggests, because the loop
form is either mathematically redundant (the loop total is already
available from cached running sums) or gets called from inside the
Eq. (7)/(8) inner solves many times per outer iteration.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


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
    Xi: np.ndarray,
    i: int,
    D_list: Sequence[np.ndarray],
    Ai: np.ndarray,
    atom_boundaries: dict[int, tuple[int, int]],
) -> float:
    """r(Ai, D, Xi) of Eq. (4), as a function of Xi with D fixed."""
    recon = np.zeros_like(Ai)
    val = 0.0
    Di_recon = None
    for k, (s, e) in atom_boundaries.items():
        Pk = D_list[k] @ Xi[s:e, :]
        recon += Pk
        if k == i:
            Di_recon = Pk
        else:
            val += float(np.sum(Pk ** 2))
    val += float(np.sum((Ai - recon) ** 2))       # replaces D_full @ Xi
    val += float(np.sum((Ai - Di_recon) ** 2))
    return val


def fidelity_grad(
    Xi: np.ndarray,
    i: int,
    D_list: Sequence[np.ndarray],
    D_full: np.ndarray,
    Ai: np.ndarray,
    atom_boundaries: dict[int, tuple[int, int]],
) -> np.ndarray:
    """Gradient of r(Ai, D, Xi) (Eq. 4) with respect to Xi."""
    residual = D_full @ Xi - Ai
    grad = 2.0 * (D_full.T @ residual)
    for k, (s, e) in atom_boundaries.items():
        Dk = D_list[k]
        Xik = Xi[s:e, :]
        if k == i:
            grad[s:e, :] += 2.0 * (Dk.T @ (Dk @ Xik - Ai))
        else:
            grad[s:e, :] += 2.0 * (Dk.T @ (Dk @ Xik))
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
      * weighted_mean_sum    : sum_j sizes[j] * mean(Xj)
      * weighted_sq_norm_sum : sum_j sizes[j] * ||mean(Xj)||^2

    These three quantities are all ``coef_value``/``coef_grad`` need;
    the previous per-class ``means`` dict is not, since every place
    that consumed it summed over it in a way that reduces to one of
    these three cached totals (see the closed forms in
    ``coef_value``/``coef_grad`` below).

    Prefer constructing this via ``GlobalMeanTracker.exclude(i)``
    instead of the constructor directly when iterating over all
    classes in a sweep — the tracker maintains these totals
    incrementally in O(1) per class instead of rebuilding them from
    scratch (O(n_classes)) for every one of the n_classes calls in a
    sweep.
    """

    __slots__ = ("n_total", "weighted_mean_sum", "weighted_sq_norm_sum")

    def __init__(
        self, X_list: Sequence[np.ndarray], sizes: Sequence[int], exclude: int
    ) -> None:
        n_total = 0
        weighted_sum = None
        weighted_sq_norm_sum = 0.0
        for k, Xk in enumerate(X_list):
            if k == exclude:
                continue
            mk = Xk.mean(axis=1)
            n_total += sizes[k]
            weighted_sq_norm_sum += sizes[k] * float(np.sum(mk ** 2))
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

        tracker = GlobalMeanTracker(X_list, sizes)   # O(n_classes) once
        for i in range(n_classes):
            stats = tracker.exclude(i)               # O(1)
            X_list[i] = solve_class_codes(..., stats, ...)
            tracker.update(i, X_list[i])              # O(1)

    which is O(n_classes) per sweep in total.

    Equivalence to rebuilding ``OtherClassStats`` from scratch after
    every update is verified on random inputs; see the module's
    companion tests.
    """

    def __init__(self, X_list: Sequence[np.ndarray], sizes: Sequence[int]) -> None:
        self.sizes = list(sizes)
        self.means = [Xk.mean(axis=1) for Xk in X_list]
        self.n_total = sum(sizes)
        self.weighted_mean_sum = sum(
            s * m for s, m in zip(self.sizes, self.means)
        )
        self.weighted_sq_norm_sum = sum(
            s * float(np.sum(m ** 2)) for s, m in zip(self.sizes, self.means)
        )

    def exclude(self, i: int) -> OtherClassStats:
        """O(1) view of all classes except i, as an OtherClassStats."""
        s = self.sizes[i]
        mi = self.means[i]
        stats = object.__new__(OtherClassStats)
        stats.n_total = self.n_total - s
        stats.weighted_mean_sum = self.weighted_mean_sum - s * mi
        stats.weighted_sq_norm_sum = self.weighted_sq_norm_sum - s * float(
            np.sum(mi ** 2)
        )
        return stats

    def update(self, i: int, Xi_new: np.ndarray) -> None:
        """O(1) update after class i's coefficients have been re-solved."""
        s = self.sizes[i]
        old_mean = self.means[i]
        new_mean = Xi_new.mean(axis=1)
        self.weighted_mean_sum += s * (new_mean - old_mean)
        self.weighted_sq_norm_sum += s * (
            float(np.sum(new_mean ** 2)) - float(np.sum(old_mean ** 2))
        )
        self.means[i] = new_mean


def coef_value(Xi: np.ndarray, i: int, stats: OtherClassStats, eta: float) -> float:
    """fi(Xi): terms of f(X) = tr(SW(X)) - tr(SB(X)) + eta*||X||_F^2 (Eq. 5)
    that depend on Xi, with all Xj, j != i, fixed (captured in ``stats``)."""
    ni = Xi.shape[1]
    n = stats.n_total + ni
    mi = Xi.mean(axis=1)
    c0 = 0.0 if stats.weighted_mean_sum is None else stats.weighted_mean_sum / n
    m = (Xi.sum(axis=1) / n) + c0

    within = float(np.sum((Xi - mi[:, None]) ** 2))

    between = ni * float(np.sum((mi - m) ** 2))
    if stats.weighted_mean_sum is not None:
        # closed form of sum_{k!=i} sizes[k]*||mk-m||^2 from the cached
        # running totals, instead of looping over every other class's
        # mean: sum_k sk*||mk-m||^2
        #     = sum_k sk*||mk||^2 - 2*m.(sum_k sk*mk) + ||m||^2*sum_k sk
        between += (
            stats.weighted_sq_norm_sum
            - 2.0 * float(m @ stats.weighted_mean_sum)
            + stats.n_total * float(np.sum(m ** 2))
        )

    return within - between + eta * float(np.sum(Xi ** 2))


def coef_grad(Xi: np.ndarray, i: int, stats: OtherClassStats, eta: float) -> np.ndarray:
    """Gradient of fi(Xi) with respect to Xi."""
    ni = Xi.shape[1]
    n = stats.n_total + ni
    a = np.full(ni, 1.0 / n)
    mi = Xi.mean(axis=1)

    c0 = 0.0 if stats.weighted_mean_sum is None else stats.weighted_mean_sum / n
    m = Xi @ a + c0

    # 2*Xi @ (I - u_i@1^T), simplified: (I - u_i@1^T) is the centering
    # projector, so Xi @ (I - u_i@1^T) = Xi - (Xi@u_i)@1^T = Xi - mi[:,None].
    within_grad = 2.0 * (Xi - mi[:, None])

    v = np.full(ni, 1.0 / ni) - a  # = u_i - a

    between_grad = 2.0 * ni * np.outer(mi - m, v)
    if stats.weighted_mean_sum is not None:
        # closed form of sum_{k!=i} sizes[k]*(mk-m) from the cached
        # running total, instead of looping over every other class's
        # mean: sum_k sk*(mk-m) = (sum_k sk*mk) - m*(sum_k sk)
        #                       = weighted_mean_sum - n_total*m
        other_sum = stats.weighted_mean_sum - stats.n_total * m
        between_grad -= 2.0 * np.outer(other_sum, a)

    return within_grad - between_grad + 2.0 * eta * Xi


def global_fisher_value(X_list: Sequence[np.ndarray], eta: float) -> float:
    """f(X) = tr(SW(X)) - tr(SB(X)) + eta*||X||_F^2 (Eq. 5), computed
    directly over the full X for convergence tracking / Eq. (6) reporting."""
    sizes = [Xk.shape[1] for Xk in X_list]
    n = sum(sizes)
    means = [Xk.mean(axis=1) for Xk in X_list]
    m = sum(sizes[k] * means[k] for k in range(len(X_list))) / n

    sq_norms = [float(np.sum(Xk ** 2)) for Xk in X_list]  # shared by sw & reg
    sw = sum(sq_norms[k] - sizes[k] * float(means[k] @ means[k]) for k in range(len(X_list)))
    sb = sum(sizes[k] * float(np.sum((means[k] - m) ** 2)) for k in range(len(X_list)))
    reg = sum(sq_norms)
    return sw - sb + eta * reg


# ---------------------------------------------------------------------
# Eq. (8): dictionary-update objective for Di, restricted from Eq. (4)
# ---------------------------------------------------------------------

def build_di_update_system(
    i: int,
    D_i: np.ndarray,
    D_full_sum: np.ndarray,
    A_list: Sequence[np.ndarray],
    atom_boundaries: dict[int, tuple[int, int]],
    X_full_stacked: np.ndarray,
    A_full: np.ndarray,
    sample_boundaries: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the sufficient statistics (A_stat, B_stat) whose solution
    minimizes Eq. (8):

        J(Di) = ||A - Di*X^i - sum_{j!=i} Dj*X^j||_F^2
              + ||Ai - Di*Xi^i||_F^2
              + sum_{j!=i} ||Di*Xj^i||_F^2

    where X^k denotes the coefficients of the *full* training set A
    over sub-dictionary Dk (i.e. the k-th row-block, stacked across all
    classes' columns), and Xj^k is that row-block restricted to class
    j's columns.

    Collecting every term against a shared unknown Di gives a single
    generalized least-squares problem, J(Di) = ||Y - Di*Z||_F^2, with:

        R = A - sum_{j!=i} Dj @ X^j                  (term 1 residual)
        T = zeros_like(A), with class-i columns = Ai  (terms 2 and 3)
        Y = [R | T],  Z = [X^i | X^i]

    ``Di`` can then be updated with any generic dictionary-update
    routine that consumes sufficient statistics A_stat = Z@Z.T,
    B_stat = Y@Z.T (e.g. ``reppi.dictionary.bcd.utils.bcd_dictionary_update``).

    Since Z is the horizontal concatenation of X^i with itself, and Y
    is [R | T] with T zero outside class i's columns, both statistics
    collapse to closed forms that avoid ever materializing the
    doubled-width Y/Z or the (mostly-zero) T:

        Z@Z.T = 2 * (X^i @ X^i.T)
        Y@Z.T = (R + T) @ X^i.T,  where (R+T) only differs from R on
                                   class i's columns (R[:,i-cols] + Ai)

    This is algebraically identical to building Y, Z explicitly and
    computing Z@Z.T / Y@Z.T (verified on random inputs; see the
    module's companion tests) but does half the FLOPs and skips the
    concatenation/zero-fill entirely.

    Parameters
    ----------
    X_full_stacked : np.ndarray, optional
        ``np.hstack(X_list)``, i.e. X^k for every k stacked in the
        same row layout as ``atom_boundaries``. Callers that invoke
        this function once per class within an outer-loop sweep
        (X_list unchanged across those calls) should compute this
        once per sweep and pass it in, instead of leaving each call to
        rebuild the same hstack from scratch (this alone turns an
        O(n_classes) rebuild per call, i.e. O(n_classes^2) per sweep,
        into O(n_classes) per sweep). If omitted, it is built locally
        from ``X_list`` as before.

    Returns
    -------
    A_stat, B_stat : np.ndarray
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