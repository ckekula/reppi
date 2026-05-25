"""
K-SVD dictionary learning.

Implements the K-SVD algorithm described in:
    Aharon, Elad, Bruckstein. "The K-SVD: An Algorithm for Designing
    Overcomplete Dictionaries for Sparse Representation".
    IEEE Trans. Signal Processing, 54(11), 2006.

Batch-OMP integration follows:
    Elad, Rubinstein, Zibulevsky. "Efficient Implementation of the K-SVD
    Algorithm using Batch Orthogonal Matching Pursuit". Technion TR, 2008.
"""

from __future__ import annotations

from reppi.backend import xp

from reppi.base import BaseDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp import OMP, batch_omp
from reppi.sparse.src import col_norms_squared, normalize_columns, rep_error_squared


class KSVD(BaseDictionaryLearner):
    """
    K-SVD dictionary learner.

    Alternates between:
      1. Sparse coding — encode each training signal over the current D.
      2. Dictionary update — update each atom (and its coefficients) via a
         rank-1 approximation of the residual matrix.

    Parameters
    ----------
    n_components : int
        Number of dictionary atoms to learn.
    n_nonzero_coefs : int
        Sparsity target T: each signal is represented with at most T atoms.
    n_iter : int
        Number of K-SVD iterations (default 10).
    exact_svd : bool
        If True, use full SVD for the atom update (exact K-SVD).
        If False (default), use the faster approximate update.
    mu_thresh : float
        Mutual-incoherence threshold in (0, 1].  Atoms whose pairwise
        correlation exceeds this value are replaced.  Set to 1.0 to
        disable (default 0.99).
    mem_usage : str
        One of 'high', 'normal' (default), 'low'.
        Controls whether G = D'D (and DtX = D'X) are precomputed.
    random_state : int or None
        Seed for reproducible atom initialisation.
    verbose : bool
        Print iteration progress (default False).
    """

    def __init__(
        self,
        n_components: int,
        n_nonzero_coefs: int,
        n_iter: int = 10,
        exact_svd: bool = False,
        mu_thresh: float = 0.99,
        mem_usage: str = "normal",
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if mem_usage not in ("high", "normal", "low"):
            raise ValueError("mem_usage must be 'high', 'normal', or 'low'.")
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs
        self.n_iter = n_iter
        self.exact_svd = exact_svd
        self.mu_thresh = mu_thresh
        self.mem_usage = mem_usage
        self.random_state = random_state
        self.verbose = verbose

        # Set after fit
        self.D_: xp.ndarray | None = None
        self.errors_: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: xp.ndarray, D_init: xp.ndarray | None = None) -> "KSVD":
        """
        Learn a dictionary from training signals.

        Parameters
        ----------
        X : xp.ndarray, shape (n_features, n_samples)
        D_init : xp.ndarray or None, shape (n_features, n_components)
            Optional initial dictionary.  If None, random training signals
            are chosen as initial atoms.

        Returns
        -------
        self
        """
        X = xp.asarray(X, dtype=float)
        rng = xp.random.default_rng(self.random_state)

        D = self._init_dict(X, D_init, rng)
        self.errors_ = []

        for it in range(self.n_iter):
            G = D.T @ D if self.mem_usage in ("high", "normal") else None
            Gamma = self._sparse_code(X, D, G)

            unused = xp.arange(X.shape[1])
            replaced = xp.zeros(self.n_components, dtype=bool)

            for j in range(self.n_components):
                D[:, j], gamma_j, idx, unused, replaced = _optimize_atom(
                    X, D, j, Gamma, unused, replaced, self.exact_svd
                )
                Gamma[j, idx] = gamma_j

            err = float(xp.sqrt(rep_error_squared(X, D, Gamma).sum() / X.size))
            self.errors_.append(err)

            D, _ = _clear_dict(D, Gamma, X, self.mu_thresh, unused, replaced)

            if self.verbose:
                print(f"Iter {it + 1}/{self.n_iter}  RMSE={err:.6f}")

        self.D_ = D
        return self

    def transform(self, X: xp.ndarray) -> xp.ndarray:
        """Encode X using the learned dictionary."""
        if self.D_ is None:
            raise DictionaryLearningError("Call fit() before transform().")
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, self.D_)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_dict(
        self,
        X: xp.ndarray,
        D_init: xp.ndarray | None,
        rng: xp.random.Generator,
    ) -> xp.ndarray:
        n_features, n_samples = X.shape
        k = self.n_components

        if D_init is not None:
            D = xp.asarray(D_init, dtype=float)
            if D.shape != (n_features, k):
                raise DictionaryLearningError(
                    f"D_init shape {D.shape} does not match "
                    f"(n_features={n_features}, n_components={k})."
                )
        else:
            valid = xp.where(col_norms_squared(X) > 1e-6)[0]
            if len(valid) < k:
                raise DictionaryLearningError(
                    "Not enough non-zero training signals to initialise the dictionary."
                )
            chosen = rng.choice(valid, size=k, replace=False)
            D = X[:, chosen].copy()

        return normalize_columns(D)

    def _sparse_code(
        self,
        X: xp.ndarray,
        D: xp.ndarray,
        G: xp.ndarray | None,
    ) -> xp.ndarray:
        if self.mem_usage == "high" and G is not None:
            return batch_omp(D.T @ X, G, self.n_nonzero_coefs)
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, D, G=G)


# ---------------------------------------------------------------------------
# Module-level helpers (shared with LC-KSVD)
# ---------------------------------------------------------------------------


def _optimize_atom(
    X: xp.ndarray,
    D: xp.ndarray,
    j: int,
    Gamma: xp.ndarray,
    unused_sigs: xp.ndarray,
    replaced: xp.ndarray,
    exact_svd: bool,
) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Update the j-th dictionary atom and the corresponding sparse codes.

    Mirrors the MATLAB ``optimize_atom`` function.

    Returns
    -------
    atom : xp.ndarray, shape (n_features,)
    gamma_j : xp.ndarray, non-zero coefficients for atom j
    data_indices : xp.ndarray, signal indices that use atom j
    unused_sigs : xp.ndarray (updated)
    replaced : xp.ndarray (updated)
    """
    # Signals that actively use atom j
    data_indices = xp.where(xp.abs(Gamma[j, :]) > 1e-10)[0]

    # --- Dead atom: replace with the worst-reconstructed unused signal ---
    if len(data_indices) == 0:
        max_signals = 5000
        perm = xp.random.permutation(len(unused_sigs))[:min(max_signals, len(unused_sigs))]
        candidates = unused_sigs[perm]
        E = rep_error_squared(X, D, Gamma, block_size=len(candidates) + 1)
        best = int(xp.argmax(E[candidates]))
        atom = X[:, candidates[best]]
        atom = atom / max(xp.linalg.norm(atom), 1e-14)
        gamma_j = xp.zeros(len(data_indices))
        # Remove used signal from the pool
        mask = xp.ones(len(unused_sigs), dtype=bool)
        mask[perm[best]] = False
        unused_sigs = unused_sigs[mask]
        replaced[j] = True
        return atom, gamma_j, data_indices, unused_sigs, replaced

    # --- Normal update ---
    small_gamma = Gamma[:, data_indices]       # (n_atoms, |support|)
    g_j = Gamma[j, data_indices]              # (|support|,)

    # Residual matrix: remove atom j's contribution then add it back
    # E = X[:,support] - D*small_gamma + d_j * g_j
    E = X[:, data_indices] - D @ small_gamma + xp.outer(D[:, j], g_j)

    if exact_svd:
        # Exact update via rank-1 SVD
        U, s, Vt = xp.linalg.svd(E, full_matrices=False)
        atom = U[:, 0]
        gamma_j = s[0] * Vt[0, :]
    else:
        # Approximate update (alternating optimisation)
        atom = E @ g_j
        atom_norm = xp.linalg.norm(atom)
        atom = atom / max(atom_norm, 1e-14)
        gamma_j = atom @ E  # (|support|,)

    return atom, gamma_j, data_indices, unused_sigs, replaced


def _clear_dict(
    D: xp.ndarray,
    Gamma: xp.ndarray,
    X: xp.ndarray,
    mu_thresh: float,
    unused_sigs: xp.ndarray,
    replaced: xp.ndarray,
    use_thresh: int = 4,
) -> tuple[xp.ndarray, int]:
    """
    Replace rarely-used or highly-correlated atoms with high-error signals.

    Mirrors the MATLAB ``cleardict`` function.

    Returns
    -------
    D : xp.ndarray (possibly modified)
    cleared : int  number of atoms replaced
    """
    n_atoms = D.shape[1]
    err = rep_error_squared(X, D, Gamma)
    use_count = (xp.abs(Gamma) > 1e-7).sum(axis=1)  # (n_atoms,)
    cleared = 0

    for j in range(n_atoms):
        if len(unused_sigs) == 0:
            break
        Gj = D.T @ D[:, j]
        Gj[j] = 0.0
        bad_coherence = xp.max(Gj ** 2) > mu_thresh ** 2
        bad_usage = use_count[j] < use_thresh

        if (bad_coherence or bad_usage) and not replaced[j]:
            best = int(xp.argmax(err[unused_sigs]))
            atom = X[:, unused_sigs[best]]
            D[:, j] = atom / max(xp.linalg.norm(atom), 1e-14)
            unused_sigs = xp.delete(unused_sigs, best)
            cleared += 1

    return D, cleared