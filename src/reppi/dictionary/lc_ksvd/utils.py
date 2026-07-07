"""
lc_ksvd.utils
Utility functions for LC-KSVD.
"""

import numpy as np

from reppi.sparse.omp import OMP
from reppi.dictionary.ksvd import KSVD

def _build_label_consistent_target(
    H: np.ndarray,
    n_components: int,
    n_nonzero_coefs: int,
    sparse_codes: np.ndarray,
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
        Total number of dictionary atoms.
    n_nonzero_coefs : int
        Sparsity level T.
    sparse_codes : np.ndarray, shape (n_components, n_samples)
        Current sparse codes (used to determine per-class atom assignment).

    Returns
    -------
    Q : np.ndarray, shape (n_components, n_samples)
    """
    n_classes, n_samples = H.shape

    # Distribute atoms evenly across classes
    atoms_per_class = n_components // n_classes

    # Assign atoms to classes in order
    atom_class = np.zeros(n_components, dtype=int)
    for c in range(n_classes):
        start = c * atoms_per_class
        end = start + atoms_per_class if c < n_classes - 1 else n_components
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

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_samples)
        Training signals.
    H : np.ndarray, shape (n_classes, n_samples)
        One-hot label matrix.
    n_components : int
        Dictionary size.
    n_iter_init : int
        K-SVD iterations for the initialisation run (per class).
    n_nonzero_coefs : int
        Sparsity level T.
    random_state : int or None
    verbose : bool

    Returns
    -------
    D_init : np.ndarray, shape (n_features, n_components)
    A_init : np.ndarray, shape (n_components, n_components)
        Initial label-consistency transform.
    W_init : np.ndarray, shape (n_classes, n_components)
        Initial linear classifier weights.
    Q : np.ndarray, shape (n_components, n_samples)
        Label-consistent sparse code targets.
    """
    n_classes = H.shape[0]

    # Same atom allocation scheme as _build_label_consistent_target:
    # even split across classes, remainder folded into the last class.
    atoms_per_class = n_components // n_classes

    # Step 1: per-class K-SVD initialisation
    D_blocks = []
    for c in range(n_classes):
        n_atoms_c = (
            atoms_per_class if c < n_classes - 1
            else n_components - atoms_per_class * (n_classes - 1)
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

    D_init = np.hstack(D_blocks)

    # Step 2: sparse-code the full training data with the initial dictionary
    coder = OMP(n_nonzero_coefs, mode="batch", check_dict=False)
    Gamma = coder.encode(X, D_init)

    # Step 3: build Q
    Q = _build_label_consistent_target(H, n_components, n_nonzero_coefs, Gamma)

    # Step 4: fit W (classifier) via least squares: W * Gamma ≈ H
    # W = H @ Gamma.T @ pinv(Gamma @ Gamma.T)
    W_init = H @ np.linalg.pinv(Gamma)

    # Step 5: fit A (label-consistency map) via least squares: A * Gamma ≈ Q
    A_init = Q @ np.linalg.pinv(Gamma)

    return D_init, A_init, W_init, Q


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

