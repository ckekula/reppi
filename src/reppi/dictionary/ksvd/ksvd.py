"""
K-SVD dictionary learning.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from tqdm import tqdm

from reppi.base import BaseDictionaryLearner
from reppi.dictionary.ksvd.checkpoint import load_checkpoint, save_checkpoint
from reppi.dictionary.ksvd.utils import _clear_dict, _optimize_atom
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp.omp import OMP
from reppi.sparse.utils import (
    _check_dict_normalized,
    col_norms_squared,
    normalize_columns,
)

_CHECKPOINT_FILENAME = "ksvd_checkpoint.npz"

logger = logging.getLogger(__name__)

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
        Number of dictionary atoms this instance learns. When ``D_frozen``
        is supplied to ``fit()``, this is the count of *new* atoms only —
        the frozen atoms are additional, held-constant columns and are not
        counted here.
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

    Attributes
    ----------
    D_ : np.ndarray, shape (n_features, n_frozen + n_components)
        Learned dictionary (set after fit()). Includes any ``D_frozen``
        columns passed to ``fit()``, unchanged, as its leading columns.
    Gamma_ : np.ndarray, shape (n_frozen + n_components, n_samples)
        Sparse codes for the training data from the final iteration
        (set after fit()). Exposed so callers that need the training
        codes (e.g. LC-KSVD, which reuses this class on an augmented
        system) don't have to re-run sparse coding.
    errors_ : list of float
        Per-iteration RMSE on the training data.
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
        verbose: bool = True,
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
        self.D_: np.ndarray | None = None
        self.Gamma_: np.ndarray | None = None
        self.errors_: list[float] = []


    def fit(
        self,
        X: np.ndarray,
        D_init: np.ndarray | None = None,
        D_frozen: np.ndarray | None = None,
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> KSVD:
        """
        Learn a dictionary from training signals.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        D_init : np.ndarray or None, shape (n_features, n_components)
            Optional initial dictionary for the *new* (non-frozen) atoms
            only. If None, random training signals are chosen as initial
            atoms. Ignored when resuming from an existing checkpoint.
        D_frozen : np.ndarray or None, shape (n_features, n_frozen_atoms)
            Optional pre-trained atoms to prepend to the dictionary and
            hold constant through every iteration. Must already have
            unit-norm columns (validated). Signals are still sparse-coded
            jointly over the full ``[D_frozen | D_active]`` dictionary at
            every iteration; only the ``n_components`` new atoms are ever
            updated by the atom-update step or by incoherence-based
            replacement in ``_clear_dict`` — ``D_frozen`` is never
            modified, not even at initialisation (it is concatenated
            as-is, never passed through ``normalize_columns``). See
            Carroll et al. 2017, Sec. III-A ("Frozen K-SVD").
        checkpoint_dir : str or None
            If given, a checkpoint is written to
            ``<checkpoint_dir>/ksvd_checkpoint.npz`` after every iteration,
            overwriting the previous one. The directory is created if it
            does not exist.
        resume : bool
            If True (default) and a checkpoint is found, training resumes from it.
            If False, any existing checkpoint in ``checkpoint_dir`` is ignored and
            overwritten.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.RandomState(self.random_state)

        if D_frozen is not None:
            D_frozen = np.asarray(D_frozen, dtype=np.float32)
            if D_frozen.shape[0] != X.shape[0]:
                raise DictionaryLearningError(
                    f"D_frozen has {D_frozen.shape[0]} features, but X has "
                    f"{X.shape[0]} features."
                )
            _check_dict_normalized(D_frozen)
        n_frozen = 0 if D_frozen is None else D_frozen.shape[1]

        if X.shape[1] < self.n_components:
            raise DictionaryLearningError(
                f"n_samples={X.shape[1]} is less than n_components={self.n_components}. "
            )

        checkpoint_path = None
        start_iter = 0
        D = None
        Gamma = None
        self.errors_ = []

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, _CHECKPOINT_FILENAME)

            if resume and os.path.exists(checkpoint_path):
                (
                    D,
                    Gamma,
                    unused,
                    replaced,
                    self.errors_,
                    start_iter,
                ) = load_checkpoint(self, checkpoint_path, X, n_frozen, D_frozen)
                if self.verbose:
                    logger.info(
                        f"Resuming from checkpoint at iteration {start_iter}/"
                        f"{self.n_iter} ({checkpoint_path})"
                    )

        if D is None:
            D_active = self._init_dict(X, D_init, rng)
            D = np.hstack([D_frozen, D_active]) if D_frozen is not None else D_active

        n_total = D.shape[1]
        if n_frozen > 0:
            D_frozen_cols = D[:, :n_frozen]
            G_ff = D_frozen_cols.T @ D_frozen_cols
            DtX_frozen = D_frozen_cols.T @ X
        else:
            G_ff = None
            DtX_frozen = None
        
        unused = np.arange(X.shape[1])
        replaced = np.zeros(n_total, dtype=bool)

        for it in range(start_iter, self.n_iter):
            if self.mem_usage in ("high", "normal"):
                D_active = D[:, n_frozen:]
                DtX_active = D_active.T @ X
                G_aa = D_active.T @ D_active

                G = np.empty((n_total, n_total), dtype=np.float32)
                DtX = np.empty((n_total, X.shape[1]), dtype=np.float32)
                if n_frozen > 0:
                    G_fa = D_frozen_cols.T @ D_active
                    G[:n_frozen, :n_frozen] = G_ff
                    G[:n_frozen, n_frozen:] = G_fa
                    G[n_frozen:, :n_frozen] = G_fa.T
                    DtX[:n_frozen, :] = DtX_frozen
                G[n_frozen:, n_frozen:] = G_aa
                DtX[n_frozen:, :] = DtX_active
            else:
                G = None
                DtX = None

            Gamma = self._sparse_code(X, D, G, DtX)            
            R = X - D @ Gamma

            unused = np.arange(X.shape[1])
            replaced = np.zeros(n_total, dtype=bool)

            for j in tqdm(range(n_frozen, n_total), desc="Updating atoms"):
                D[:, j], gamma_j, idx, unused, replaced = _optimize_atom(
                    X, D, R, j, Gamma, unused, replaced, self.exact_svd
                )
                Gamma[j, idx] = gamma_j

            err = np.float32(np.sqrt((R ** 2).sum() / X.size))
            self.errors_.append(err)
            logger.info(f"Iter {it + 1}/{self.n_iter}  RMSE={err:.6f}")

            D, _ = _clear_dict(
                D, Gamma, X, R, self.mu_thresh, unused, replaced, frozen_atoms=n_frozen
            )
            
            if checkpoint_path is not None:
                save_checkpoint(
                    self, checkpoint_path, X, D, Gamma, unused, replaced, it + 1, n_frozen
                )

        self.D_ = D
        self.Gamma_ = Gamma
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode X using the learned dictionary."""
        if self.D_ is None:
            raise DictionaryLearningError("Call fit() before transform().")
        logger.info("Sparse Coding %d signals with learned dictionary of shape %s", X.shape[1], self.D_.shape)
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, self.D_)


    def _init_dict(
        self,
        X: np.ndarray,
        D_init: np.ndarray | None,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        Build the initial dictionary for this instance's *own* atoms.

        Note: this only ever constructs ``self.n_components`` columns —
        any frozen atoms are handled separately by ``fit()`` and
        concatenated afterward, never passed through this method (so they
        are never renormalised or otherwise touched at initialisation).
        """
        n_features, _n_samples = X.shape
        k = self.n_components

        if D_init is not None:
            D = np.asarray(D_init, dtype=np.float32)
            if D.shape != (n_features, k):
                raise DictionaryLearningError(
                    f"D_init shape {D.shape} does not match "
                    f"(n_features={n_features}, n_components={k})."
                )
            return normalize_columns(D)

        valid = np.where(col_norms_squared(X) > 1e-6)[0]
        if len(valid) < k:
            raise DictionaryLearningError(
                "Not enough non-zero training signals to initialise the dictionary."
            )

        # Greedily pick columns, rejecting candidates too coherent with atoms
        # already chosen (same incoherence threshold used by _clear_dict).
        candidates = rng.permutation(valid)
        X_norm = normalize_columns(X[:, candidates].copy())

        chosen_idx: list[int] = []
        for pos in range(len(candidates)):
            if len(chosen_idx) == k:
                break
            cand_col = X_norm[:, pos]
            if chosen_idx:
                chosen_cols = X_norm[:, chosen_idx]
                max_coh = np.abs(chosen_cols.T @ cand_col).max()
                if max_coh > self.mu_thresh:
                    continue
            chosen_idx.append(pos)

        if len(chosen_idx) < k:
            logger.warning(
                "_init_dict: only found %d/%d mutually-incoherent candidate "
                "atoms (mu_thresh=%.3f) among %d valid signals; falling back "
                "to filling remaining atoms without the incoherence check. "
                "This suggests significant redundancy/near-duplication in the "
                "training data for this class.",
                len(chosen_idx), k, self.mu_thresh, len(valid),
            )
            remaining_needed = k - len(chosen_idx)
            remaining_pool = [p for p in range(len(candidates)) if p not in chosen_idx]
            chosen_idx.extend(remaining_pool[:remaining_needed])

        D = X[:, candidates[chosen_idx]].copy()
        return normalize_columns(D)


    def _sparse_code(
        self,
        X: np.ndarray,
        D: np.ndarray,
        G: np.ndarray | None,
        DtX: np.ndarray | None = None,
    ) -> np.ndarray:
        logger.info("Sparse coding %d signals with dictionary of shape %s", X.shape[1], D.shape)
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, D, G=G, DtX=DtX)
