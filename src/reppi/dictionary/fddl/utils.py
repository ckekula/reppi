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
    D_full: np.ndarray,
    Ai: np.ndarray,
    atom_boundaries: dict[int, tuple[int, int]],
) -> float:
    """r(Ai, D, Xi) of Eq. (4), as a function of Xi with D fixed."""
    val = float(np.sum((Ai - D_full @ Xi) ** 2))
    s, e = atom_boundaries[i]
    val += float(np.sum((Ai - D_list[i] @ Xi[s:e, :]) ** 2))
    for k, (s, e) in atom_boundaries.items():
        if k == i:
            continue
        val += float(np.sum((D_list[k] @ Xi[s:e, :]) ** 2))
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
    Cached (sample count, mean vector) of every class j != i, computed
    once from the current X before solving Xi so the Eq. (7) sub-problem
    sees a fixed target.
    """

    __slots__ = ("n_total", "weighted_mean_sum", "means", "sizes")

    def __init__(
        self, X_list: Sequence[np.ndarray], sizes: Sequence[int], exclude: int
    ) -> None:
        self.sizes = list(sizes)
        self.means: dict[int, np.ndarray] = {}
        weighted_sum = None
        n_total = 0
        for k, Xk in enumerate(X_list):
            if k == exclude:
                continue
            mk = Xk.mean(axis=1)
            self.means[k] = mk
            n_total += sizes[k]
            contribution = sizes[k] * mk
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        self.n_total = n_total  # sum of sizes over j != exclude only
        self.weighted_mean_sum = (
            weighted_sum if weighted_sum is not None else None
        )  # None only when n_classes == 1


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
    for k, mk in stats.means.items():
        between += stats.sizes[k] * float(np.sum((mk - m) ** 2))

    return within - between + eta * float(np.sum(Xi ** 2))


def coef_grad(Xi: np.ndarray, i: int, stats: OtherClassStats, eta: float) -> np.ndarray:
    """Gradient of fi(Xi) with respect to Xi."""
    ni = Xi.shape[1]
    n = stats.n_total + ni
    u_i = np.full(ni, 1.0 / ni)
    a = np.full(ni, 1.0 / n)
    v = u_i - a

    mi = Xi @ u_i
    c0 = 0.0 if stats.weighted_mean_sum is None else stats.weighted_mean_sum / n
    m = Xi @ a + c0

    within_grad = 2.0 * Xi @ (np.eye(ni) - np.outer(u_i, np.ones(ni)))

    other_sum = None
    for k, mk in stats.means.items():
        term = stats.sizes[k] * (mk - m)
        other_sum = term if other_sum is None else other_sum + term
    between_grad = 2.0 * ni * np.outer(mi - m, v)
    if other_sum is not None:
        between_grad -= 2.0 * np.outer(other_sum, a)

    return within_grad - between_grad + 2.0 * eta * Xi


def global_fisher_value(X_list: Sequence[np.ndarray], eta: float) -> float:
    """f(X) = tr(SW(X)) - tr(SB(X)) + eta*||X||_F^2 (Eq. 5), computed
    directly over the full X for convergence tracking / Eq. (6) reporting."""
    sizes = [Xk.shape[1] for Xk in X_list]
    n = sum(sizes)
    means = [Xk.mean(axis=1) for Xk in X_list]
    m = sum(sizes[k] * means[k] for k in range(len(X_list))) / n

    sw = sum(float(np.sum((Xk - mk[:, None]) ** 2)) for Xk, mk in zip(X_list, means))
    sb = sum(sizes[k] * float(np.sum((means[k] - m) ** 2)) for k in range(len(X_list)))
    reg = sum(float(np.sum(Xk ** 2)) for Xk in X_list)
    return sw - sb + eta * reg


# ---------------------------------------------------------------------
# Eq. (8): dictionary-update objective for Di, restricted from Eq. (4)
# ---------------------------------------------------------------------

def build_di_update_system(
    i: int,
    D_list: Sequence[np.ndarray],
    X_list: Sequence[np.ndarray],
    A_list: Sequence[np.ndarray],
    atom_boundaries: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the stacked (Y, Z) least-squares system whose solution
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

    Returns
    -------
    Y, Z : np.ndarray
        Stacked target and coefficient matrices, shape
        (n_features, 2 * n_samples) and (p_i, 2 * n_samples).
    """
    A_full = np.hstack(A_list)
    n_features, n_samples = A_full.shape
    sample_boundaries = block_boundaries([Ak.shape[1] for Ak in A_list])

    # X^k (full-dataset coefficients over Dk), assembled column-wise in
    # the same order as A_full, by gathering row-block k from every
    # class's own coefficient matrix.
    s_i, e_i = atom_boundaries[i]

    def row_block(k: int) -> np.ndarray:
        s, e = atom_boundaries[k]
        return np.hstack([X_list[c][s:e, :] for c in range(len(X_list))])

    Dj_sum = np.zeros((n_features, n_samples))
    for j, Dj in enumerate(D_list):
        if j == i:
            continue
        Dj_sum += Dj @ row_block(j)
    R = A_full - Dj_sum

    T = np.zeros((n_features, n_samples))
    s, e = sample_boundaries[i]
    T[:, s:e] = A_list[i]

    X_i_full = row_block(i)  # X^i, shape (p_i, n_samples)
    Y = np.hstack([R, T])
    Z = np.hstack([X_i_full, X_i_full])
    return Y, Z