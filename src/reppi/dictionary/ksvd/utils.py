"""
ksvd.utils
Utility functions for KSVD.
"""


import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def _optimize_atom(
    X: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    j: int,
    Gamma: np.ndarray,
    unused_sigs: np.ndarray,
    replaced: np.ndarray,
    exact_svd: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Update the j-th dictionary atom and the corresponding sparse codes.

    Mirrors the MATLAB ``optimize_atom`` function.

    ``j`` is an absolute column index into the (possibly frozen+active)
    dictionary ``D`` and into ``Gamma``'s rows. This function has no
    frozen-atom awareness of its own — the caller is responsible for never
    invoking it with ``j`` in the frozen range (see ``KSVD.fit``'s atom
    loop, which iterates ``range(n_frozen, n_total)``). ``replaced`` must
    therefore be sized to the *full* dictionary width so that
    ``replaced[j]`` indexes correctly regardless of how many atoms are
    frozen.

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
        logger.info(f"Replacing dead atom {j} with a high-error signal")
        max_signals = 5000
        
        perm = np.random.permutation(len(unused_sigs))[:min(max_signals, len(unused_sigs))]
        candidates = unused_sigs[perm]
        err = (R[:, candidates] ** 2).sum(axis=0)
        best = int(np.argmax(err))
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
    g_j = Gamma[j, data_indices]              # (|support|,)

    # Residual matrix: remove atom j's contribution then add it back
    E = R[:, data_indices] + np.outer(D[:, j], g_j)

    if exact_svd:
        # Exact update via rank-1 SVD
        U, s, Vt = np.linalg.svd(E, full_matrices=False)
        atom = U[:, 0]
        gamma_j = atom @ E  # (|support|,)
    else:
        # Approximate update (alternating optimisation)
        atom = E @ g_j
        atom_norm = np.linalg.norm(atom)
        atom = atom / max(atom_norm, 1e-14)
        gamma_j = atom @ E  # (|support|,)

    R[:, data_indices] = E - np.outer(atom, gamma_j)
    return atom, gamma_j, data_indices, unused_sigs, replaced


def _clear_dict(
    D: np.ndarray,
    Gamma: np.ndarray,
    X: np.ndarray,
    R: np.ndarray,
    mu_thresh: float,
    unused_sigs: np.ndarray,
    replaced: np.ndarray,
    use_thresh: int = 4,
    frozen_atoms: int = 0,
) -> tuple[np.ndarray, int]:
    """
    Replace rarely-used or highly-correlated atoms with high-error signals.

    Mirrors the MATLAB ``cleardict`` function.

    Parameters
    ----------
    frozen_atoms : int
        Number of leading columns of D to treat as frozen (never
        replaced), matching the same convention as ``KSVD.fit``'s atom
        loop. Coherence (``Gj``) is still computed against the full D —
        an active atom too correlated with a *frozen* atom is still
        flagged and replaced — only the frozen columns themselves are
        exempt from being overwritten. Default 0 preserves the original,
        non-frozen behaviour exactly.

    Returns
    -------
    D : np.ndarray (possibly modified)
    cleared : int  number of atoms replaced
    """
    n_atoms = D.shape[1]
    err = (R ** 2).sum(axis=0)
    use_count = (np.abs(Gamma) > 1e-7).sum(axis=1)  # (n_atoms,)
    cleared = 0

    for j in tqdm(range(frozen_atoms, n_atoms), desc="Clearing dictionary"):
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