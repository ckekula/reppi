"""
FDDL classification schemes, Section 5.

Both schemes code a query signal (or batch of signals) and use the
per-class reconstruction error plus a discriminative-coefficient
distance for classification (Eq. 2). Because test-time coding here is
a *fixed* target per class (the query itself, or the query plus a
fixed class-mean pull) with no cross-column mean coupling, it is fully
column-separable — so, unlike the Eq. (7) training-time solve, this
can and does reuse the ``FISTA`` sparse-coder class directly instead of
the lower-level core solver.

  * GC (Global Classifier, Eq. 9-10): for small per-class sample counts.
    Code the query over the *whole* dictionary D, then combine
    reconstruction error and coefficient-to-class-mean distance.
  * LC (Local Classifier, Eq. 11-12): for larger per-class sample
    counts. Code the query separately over each Di, pulled toward that
    class's own-block mean via an L2 term. Eq. (11)'s quadratic pull
    ``gamma2*||alpha - mi_i||^2`` is folded into a plain LASSO by
    augmenting the dictionary/target (the same trick
    ``LCKSVD``/``_augment_data`` uses), so it also reuses ``FISTA``
    unmodified.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from reppi.sparse.fista.fista import FISTA


def fit_class_means(
    X_list: Sequence[np.ndarray], atom_boundaries: dict[int, tuple[int, int]]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Compute the per-class mean coefficient vectors used at test time.

    Returns
    -------
    means_full : dict[int, np.ndarray], shape (n_atoms,)
        mi = mean(Xi) over the full stacked coefficient space, used by
        GC (Eq. 10).
    means_own : dict[int, np.ndarray], shape (p_i,)
        mi^i = mean(Xi^i), the class's own sub-dictionary block only,
        used by LC (Eq. 11-12).
    """
    means_full: dict[int, np.ndarray] = {}
    means_own: dict[int, np.ndarray] = {}
    for i, Xi in enumerate(X_list):
        means_full[i] = Xi.mean(axis=1)
        s, e = atom_boundaries[i]
        means_own[i] = Xi[s:e, :].mean(axis=1)
    return means_full, means_own


def gc_classify(
    Y: np.ndarray,
    D_full: np.ndarray,
    D_list: Sequence[np.ndarray],
    atom_boundaries: dict[int, tuple[int, int]],
    means_full: dict[int, np.ndarray],
    gamma: float,
    w: float,
    max_iter: int = 500,
    tol: float | None = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Global Classifier (Eq. 9-10).

    Parameters
    ----------
    Y : np.ndarray, shape (n_features, n_samples)
        Query signal(s).
    gamma : L1 weight for the coding step (Eq. 9's ``gamma``).
    w : weight balancing reconstruction error and coefficient distance
        (Eq. 10).

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
    scores : np.ndarray, shape (n_classes, n_samples)
        e_i(y) for every class i; ``labels = scores.argmin(axis=0)``.
    """
    coder = FISTA(alpha=gamma, mode="backtracking", max_iter=max_iter, tol=tol)
    Alpha = coder.encode(Y, D_full)  # (n_atoms, n_samples)

    n_classes = len(D_list)
    scores = np.empty((n_classes, Y.shape[1]))
    for i, Di in enumerate(D_list):
        s, e = atom_boundaries[i]
        recon_err = np.sum((Y - Di @ Alpha[s:e, :]) ** 2, axis=0)
        mean_dist = np.sum((Alpha - means_full[i][:, None]) ** 2, axis=0)
        scores[i] = recon_err + w * mean_dist

    labels = scores.argmin(axis=0)
    return labels, scores


def lc_classify(
    Y: np.ndarray,
    D_list: Sequence[np.ndarray],
    means_own: dict[int, np.ndarray],
    gamma1: float,
    gamma2: float,
    max_iter: int = 500,
    tol: float | None = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Local Classifier (Eq. 11-12).

    Parameters
    ----------
    Y : np.ndarray, shape (n_features, n_samples)
        Query signal(s).
    gamma1, gamma2 : Eq. (11)'s L1 weight and mean-pull weight, reused
        as-is in Eq. (12)'s metric (unlike GC/Eq. 10, LC's score
        includes the L1 term: e_i = ||y-Di*a||^2 + gamma1*||a||_1 +
        gamma2*||a-mi^i||^2).

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
    scores : np.ndarray, shape (n_classes, n_samples)
    """
    n_classes = len(D_list)
    n_samples = Y.shape[1]
    scores = np.empty((n_classes, n_samples))
    sqrt_gamma2 = np.sqrt(gamma2)

    for i, Di in enumerate(D_list):
        pi = Di.shape[1]
        mi = means_own[i]

        # Fold Eq. (11)'s ||alpha - mi_i||^2 pull into a plain LASSO by
        # augmenting the dictionary/target (same trick as LC-KSVD's
        # _augment_data), so FISTA can be reused unmodified.
        D_aug = np.vstack([Di, sqrt_gamma2 * np.eye(pi)])
        Y_aug = np.vstack([Y, sqrt_gamma2 * np.tile(mi[:, None], (1, n_samples))])

        coder = FISTA(
            alpha=gamma1, mode="backtracking", max_iter=max_iter, tol=tol, check_dict=False
        )
        Alpha_i = coder.encode(Y_aug, D_aug)  # (p_i, n_samples)

        recon_err = np.sum((Y - Di @ Alpha_i) ** 2, axis=0)
        l1_term = gamma1 * np.sum(np.abs(Alpha_i), axis=0)
        mean_dist = np.sum((Alpha_i - mi[:, None]) ** 2, axis=0)
        scores[i] = recon_err + l1_term + gamma2 * mean_dist

    labels = scores.argmin(axis=0)
    return labels, scores