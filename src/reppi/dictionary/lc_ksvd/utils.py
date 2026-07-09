"""
lc_ksvd.utils
Utility functions for LC-KSVD.
"""

import numpy as np

from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp.omp import OMP
from reppi.dictionary.ksvd.ksvd import KSVD

def _build_label_consistent_target(
    H: np.ndarray,
    n_components: int,
    sparse_codes: np.ndarray,
    active_classes: list[int] | None = None,
) -> np.ndarray:
    """
    Build the label-consistent target matrix Q.

    Each dictionary atom is associated with exactly one class.  Q[:,i] is a
    binary vector that is 1 in the positions of atoms belonging to the same
    class as training sample i, and 0 elsewhere.

    Parameters
    ----------
    H : np.ndarray, shape (n_classes, n_samples)
        One-hot class label matrix.
    n_components : int
        Number of dictionary atoms to assign class labels to (i.e. the
        atoms being built in this call — the *new* atoms only, when used
        in the frozen/incremental setting).
    sparse_codes : np.ndarray, shape (n_components, n_samples)
        Current sparse codes (used to determine per-class atom assignment).
        Accepted for interface symmetry with the initialisation call site;
        only its column count (n_samples) is implicitly relied upon via H.
    active_classes : list[int] or None
        Which class indices (rows of H) the ``n_components`` atoms should
        be evenly split across. Defaults to every class in H (0..n_classes-1
        in order), matching the original, non-incremental behaviour. Pass
        an explicit subset — typically just the single new class — when
        building Q for a residual/frozen learning stage, so that new atoms
        are not diluted across classes that have no training samples in
        this call.

    Returns
    -------
    Q : np.ndarray, shape (n_components, n_samples)
    """
    n_classes, n_samples = H.shape
    if active_classes is None:
        active_classes = list(range(n_classes))
    if not active_classes:
        raise DictionaryLearningError(
            "active_classes is empty; cannot assign atoms to zero classes."
        )
    n_active = len(active_classes)

    # Distribute atoms evenly across the active classes
    atoms_per_class = n_components // n_active

    # Assign atoms to classes in order
    atom_class = np.zeros(n_components, dtype=int)
    for i, c in enumerate(active_classes):
        start = i * atoms_per_class
        end = start + atoms_per_class if i < n_active - 1 else n_components
        atom_class[start:end] = c

    Q = np.zeros((n_components, n_samples))
    for i in range(n_samples):
        cls = int(np.argmax(H[:, i]))
        Q[atom_class == cls, i] = 1.0

    return Q


def initialization4lcksvd(
    X: np.ndarray,
    H: np.ndarray,
    n_components: int,
    n_iter_init: int,
    n_nonzero_coefs: int,
    D_frozen: np.ndarray | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialise D, A (label-consistency transform), W (classifier), and Q.

    This mirrors the MATLAB ``initialization4LCKSVD`` step. A separate K-SVD
    is run per class (each class contributing its allotted block of atoms,
    trained only on its own samples), the per-class dictionaries are
    concatenated, and a linear classifier W and label-consistent target Q
    are then estimated from the sparse codes of the full training set
    against this dictionary.

    Only classes with at least one sample present in ``H`` (i.e. a nonzero
    row) receive a share of the ``n_components`` atoms being initialised
    here. In the ordinary, non-incremental case this is every class; in
    the frozen/incremental case (``D_frozen`` given, ``X``/``H`` restricted
    to a single new class's data by the caller) it is just that class, so
    the new atoms are not diluted across classes with zero samples in this
    call.

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_samples)
        Training signals.
    H : np.ndarray, shape (n_classes, n_samples)
        One-hot label matrix.
    n_components : int
        Number of *new* atoms to initialise (excludes any D_frozen atoms).
    n_iter_init : int
        K-SVD iterations for the initialisation run (per class).
    n_nonzero_coefs : int
        Sparsity level T.
    D_frozen : np.ndarray or None, shape (n_features, n_frozen_atoms)
        Pre-trained frozen atoms, if any. When given, the full training
        set is sparse-coded jointly over ``[D_frozen | D_active_init]`` to
        build Q/A/W so their column count matches the eventual combined
        dictionary; Q's rows for the frozen atoms are all-zero (none of
        this call's training samples belong to a previously-frozen
        class).
    random_state : int or None
    verbose : bool

    Returns
    -------
    D_active_init : np.ndarray, shape (n_features, n_components)
        Initial dictionary for the *new* atoms only (never includes
        D_frozen — the caller concatenates).
    A_init : np.ndarray, shape (n_frozen + n_components, n_frozen + n_components)
        Initial label-consistency transform, sized to the full combined
        dictionary.
    W_init : np.ndarray, shape (n_classes, n_frozen + n_components)
        Initial linear classifier weights, sized to the full combined
        dictionary.
    Q : np.ndarray, shape (n_frozen + n_components, n_samples)
        Label-consistent sparse code targets, sized to the full combined
        dictionary (leading n_frozen rows are zero when D_frozen is given).
    """
    n_classes = H.shape[0]
    active_classes = [c for c in range(n_classes) if np.any(H[c, :] > 0)]
    if not active_classes:
        raise DictionaryLearningError(
            "H has no samples assigned to any class (all-zero one-hot matrix)."
        )
    n_active = len(active_classes)

    # Same atom allocation scheme as _build_label_consistent_target:
    # even split across active classes, remainder folded into the last one.
    atoms_per_class = n_components // n_active

    # Step 1: per-class K-SVD initialisation (only for classes with data
    # present in this call)
    D_blocks = []
    for i, c in enumerate(active_classes):
        n_atoms_c = (
            atoms_per_class if i < n_active - 1
            else n_components - atoms_per_class * (n_active - 1)
        )
        class_mask = H[c, :] > 0
        X_c = X[:, class_mask]

        class_random_state = (
            None if random_state is None else random_state + c
        )
        ksvd_c = KSVD(
            n_components=n_atoms_c,
            n_nonzero_coefs=n_nonzero_coefs,
            n_iter=n_iter_init,
            random_state=class_random_state,
            verbose=verbose,
        )
        ksvd_c.fit(X_c)
        D_blocks.append(ksvd_c.D_)

        if verbose:
            print(
                f"Class {c}: initialised {n_atoms_c} atoms from "
                f"{X_c.shape[1]} samples."
            )

    D_active_init = np.hstack(D_blocks)

    # Step 2: sparse-code the full training data jointly over
    # [D_frozen | D_active_init], so Q/A/W are sized to the full combined
    # dictionary from the start.
    D_for_coding = (
        np.hstack([D_frozen, D_active_init]) if D_frozen is not None else D_active_init
    )
    coder = OMP(n_nonzero_coefs, mode="batch", check_dict=False)
    Gamma_full = coder.encode(X, D_for_coding)

    # Step 3: build Q. Only the new atoms get real class assignments;
    # frozen atoms' rows are all-zero (no training sample here belongs to
    # a previously-frozen class).
    Q_active = _build_label_consistent_target(
        H, n_components, Gamma_full, active_classes=active_classes
    )
    if D_frozen is not None:
        Q = np.vstack([np.zeros((D_frozen.shape[1], X.shape[1])), Q_active])
    else:
        Q = Q_active

    # Step 4: fit W (classifier) via least squares: W * Gamma ≈ H
    # W = H @ Gamma.T @ pinv(Gamma @ Gamma.T)
    W_init = H @ np.linalg.pinv(Gamma_full)

    # Step 5: fit A (label-consistency map) via least squares: A * Gamma ≈ Q
    A_init = Q @ np.linalg.pinv(Gamma_full)

    return D_active_init, A_init, W_init, Q


# ---------------------------------------------------------------------------
# LC-KSVD training
# ---------------------------------------------------------------------------


def _augment_data(
    X: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray | None,
    sqrt_alpha: float,
    sqrt_beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the augmented signal/dictionary system for LC-KSVD.

    The combined objective is minimised by stacking the data matrices:

        Y_aug = [ X        ]       D_aug = [ D  ]
                [ α * Q    ]               [ α A]
                [ β * H    ]               [ β W]   (LC-KSVD2 only)

    This augmentation lets the standard K-SVD atom-update step
    simultaneously minimise reconstruction, label-consistency, and
    (optionally) classification error.

    Returns
    -------
    X_aug : np.ndarray
    alpha_scale : float  (for constructing D_aug at each iteration)
    beta_scale : float
    """
    alpha_scale = sqrt_alpha
    beta_scale = sqrt_beta

    parts = [X, sqrt_alpha * Q]
    if H is not None:
        parts.append(sqrt_beta * H)

    X_aug = np.vstack(parts)
    return X_aug, alpha_scale, beta_scale