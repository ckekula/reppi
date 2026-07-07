import numpy as np

from reppi.sparse.utils import rep_error_squared

def _optimize_atom(
    X: np.ndarray,
    D: np.ndarray,
    j: int,
    Gamma: np.ndarray,
    unused_sigs: np.ndarray,
    replaced: np.ndarray,
    exact_svd: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Update the j-th dictionary atom and the corresponding sparse codes.

    Mirrors the MATLAB ``optimize_atom`` function.

    Returns
    -------
    atom : np.ndarray, shape (n_features,)
    gamma_j : np.ndarray, non-zero coefficients for atom j
    data_indices : np.ndarray, signal indices that use atom j
    unused_sigs : np.ndarray (updated)
    replaced : np.ndarray (updated)
    """
    # Signals that actively use atom j
    data_indices = np.where(np.abs(Gamma[j, :]) > 1e-10)[0]

    # --- Dead atom: replace with the worst-reconstructed unused signal ---
    if len(data_indices) == 0:
        max_signals = 5000
        perm = np.random.permutation(len(unused_sigs))[:min(max_signals, len(unused_sigs))]
        candidates = unused_sigs[perm]
        E = rep_error_squared(X, D, Gamma, block_size=len(candidates) + 1)
        best = int(np.argmax(E[candidates]))
        atom = X[:, candidates[best]]
        atom = atom / max(np.linalg.norm(atom), 1e-14)
        gamma_j = np.zeros(len(data_indices))
        # Remove used signal from the pool
        mask = np.ones(len(unused_sigs), dtype=bool)
        mask[perm[best]] = False
        unused_sigs = unused_sigs[mask]
        replaced[j] = True
        return atom, gamma_j, data_indices, unused_sigs, replaced

    # --- Normal update ---
    small_gamma = Gamma[:, data_indices]       # (n_atoms, |support|)
    g_j = Gamma[j, data_indices]              # (|support|,)

    # Residual matrix: remove atom j's contribution then add it back
    # E = X[:,support] - D*small_gamma + d_j * g_j
    E = X[:, data_indices] - D @ small_gamma + np.outer(D[:, j], g_j)

    if exact_svd:
        # Exact update via rank-1 SVD
        U, s, Vt = np.linalg.svd(E, full_matrices=False)
        atom = U[:, 0]
        gamma_j = s[0] * Vt[0, :]
    else:
        # Approximate update (alternating optimisation)
        atom = E @ g_j
        atom_norm = np.linalg.norm(atom)
        atom = atom / max(atom_norm, 1e-14)
        gamma_j = atom @ E  # (|support|,)

    return atom, gamma_j, data_indices, unused_sigs, replaced


def _clear_dict(
    D: np.ndarray,
    Gamma: np.ndarray,
    X: np.ndarray,
    mu_thresh: float,
    unused_sigs: np.ndarray,
    replaced: np.ndarray,
    use_thresh: int = 4,
) -> tuple[np.ndarray, int]:
    """
    Replace rarely-used or highly-correlated atoms with high-error signals.

    Mirrors the MATLAB ``cleardict`` function.

    Returns
    -------
    D : np.ndarray (possibly modified)
    cleared : int  number of atoms replaced
    """
    n_atoms = D.shape[1]
    err = rep_error_squared(X, D, Gamma)
    use_count = (np.abs(Gamma) > 1e-7).sum(axis=1)  # (n_atoms,)
    cleared = 0

    for j in range(n_atoms):
        if len(unused_sigs) == 0:
            break
        Gj = D.T @ D[:, j]
        Gj[j] = 0.0
        bad_coherence = np.max(Gj ** 2) > mu_thresh ** 2
        bad_usage = use_count[j] < use_thresh

        if (bad_coherence or bad_usage) and not replaced[j]:
            best = int(np.argmax(err[unused_sigs]))
            atom = X[:, unused_sigs[best]]
            D[:, j] = atom / max(np.linalg.norm(atom), 1e-14)
            unused_sigs = np.delete(unused_sigs, best)
            cleared += 1

    return D, cleared