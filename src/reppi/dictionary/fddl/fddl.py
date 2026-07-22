"""
Fisher Discrimination Dictionary Learning (FDDL).

Implements:
    Meng Yang, Lei Zhang, Xiangchu Feng, David Zhang.
    "Fisher Discrimination Dictionary Learning for Sparse
     Representation", ICCV 2011.

FDDL learns a structured dictionary D = [D1, ..., Dc], one
sub-dictionary per class, such that:

  * each Di reconstructs its own class well but poorly reconstructs
    other classes (the discriminative fidelity term, Eq. 4);
  * the coding coefficients X are themselves discriminative, via a
    Fisher-style within/between-class scatter penalty (Eq. 5).

Optimization problem (Eq. 6)
-----------------------------
    min_{D,X}  sum_i [ r(Ai,D,Xi) + lambda1*||Xi||_1 ]
               + lambda2*[ tr(SW(X)) - tr(SB(X)) + eta*||X||_F^2 ]

Optimization procedure (Table 1)
----------------------------------
Alternates, per outer iteration:
  1. Fix D, update X class-by-class (Eq. 7) — see
     ``reppi.dictionary.fddl.coding.solve_class_codes``.
  2. Fix X, update D class-by-class (Eq. 8) — reduced to a generic
     dictionary-fitting least-squares problem and solved via
     ``reppi.dictionary.bcd.utils.bcd_dictionary_update``, see
     ``reppi.dictionary.fddl.utils.build_di_update_system``.

Both sub-problems and their closed-form gradients are re-derived from
the paper's unambiguous definitions (Eq. 4, Eq. 5) rather than parsed
from the typeset Eq. 7/8/Appendix A; see the module docstring of
``reppi.dictionary.fddl.utils`` for the derivation and its
finite-difference / direct-computation verification.

Backend
-------
GPU-only: ``fit`` requires a CUDA or MPS device and raises otherwise
(no silent CPU fallback). ``X``/``y``/``D_init`` are accepted as numpy
arrays (the public API surface); internally everything is moved to the
GPU once and the full alternating-optimization loop runs there as
torch tensors. Checkpoints are still written as numpy .npz (so they're
portable/inspectable without torch), and the fitted attributes
(``D_``, ``D_list_``, ``X_list_``) are converted back to numpy at the
end of ``fit`` — downstream consumers are not assumed to be GPU-aware.

Usage
-----
    model = FDDL(
        n_components=8 * n_classes,   # e.g. 8 atoms/class, per Sec. 6.2
        lambda1=0.005, lambda2=0.005,
        classifier="gc", gamma=0.001, w=0.05,
    )
    model.fit(X_train, y_train)
"""

from __future__ import annotations

import os

import numpy as np
import torch
import logging

from reppi.dictionary.bcd.utils import bcd_dictionary_update
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.utils import _check_dict_normalized, normalize_columns

from reppi.dictionary.fddl.coding import solve_class_codes
from reppi.dictionary.fddl.utils import (
    GlobalMeanTracker,
    block_boundaries,
    build_di_update_system,
    fidelity_value,
    global_fisher_value,
    resolve_atoms_per_class,
)

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "fddl_checkpoint.npz"


def _require_gpu_device() -> torch.device:
    """Resolve a GPU device for GPU-only operation; raises if none exists."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError(
        "FDDL.fit() requires a GPU (CUDA or MPS) device; none was found. "
        "This implementation runs GPU-only and does not fall back to CPU."
    )


def _to_device(arr, device: torch.device) -> torch.Tensor:
    """Convert a numpy array (or array-like) to a float32 tensor on `device`."""
    return torch.as_tensor(np.asarray(arr, dtype=np.float32), device=device)


class FDDL():
    """
    Fisher Discrimination Dictionary Learning.

    Parameters
    ----------
    n_components : int or sequence of int
        Total number of dictionary atoms, split evenly across classes
        (remainder to the last class), or an explicit per-class atom
        count. The paper usually sets all pi equal (Sec. 6.1).
    lambda1 : float
        L1 sparsity weight (Eq. 6).
    lambda2 : float
        Weight of the Fisher discriminative coefficient term (Eq. 6).
    eta : float
        Elastic term weight in f(X) (Eq. 5). The paper fixes eta=1,
        which is sufficient for fi(Xi) to be strictly convex whenever
        eta > 1 - ni/n for every class (Appendix A); eta=1 satisfies
        this for any class with more than one training sample.
    n_iter : int
        Maximum number of outer alternating iterations (D/X updates).
    tol : float or None
        Outer-loop relative convergence tolerance on J(D,X) (Eq. 6)
        between consecutive iterations (Table 1, step 4: "return to
        step 2 until the values of J(D,X) in adjacent iterations are
        close enough"). None disables early stopping.
    coding_max_iter, coding_tol : controls for the Eq. (7) solve.
    dict_max_iter, dict_tol : controls for the Eq. (8) BCD atom update
        (passed through to ``bcd_dictionary_update``).
    random_state : int or None
    verbose : bool

    Attributes
    ----------
    D_list_ : list of np.ndarray
        Learned per-class sub-dictionaries, D_list_[i] has shape
        (n_features, p_i). Converted to numpy at the end of `fit`
        (computed on GPU internally).
    D_ : np.ndarray, shape (n_features, n_components)
        Learned dictionary, horizontally stacked D_list_.
    X_list_ : list of np.ndarray
        Learned per-class coding coefficients (full n_components rows).
        Converted to numpy at the end of `fit`.
    atom_boundaries_ : dict[int, tuple[int, int]]
        Atom row-range owned by each class within D_/X_list_[*].
    classes_ : np.ndarray
        Class labels seen during ``fit``, in the internal class-index
        order (index i corresponds to D_list_[i]).
    sample_order_ : np.ndarray
        Indices into the original training X that produce the
        class-grouped column order used internally (useful for
        re-aligning ``X_list_``/errors with the original sample order).
    objective_history_ : list of float
        J(D,X) (Eq. 6) after every outer iteration.
    """

    def __init__(
        self,
        n_components: int | list[int],
        lambda1: float = 0.005,
        lambda2: float = 0.005,
        eta: float = 1.0,
        n_iter: int = 15,
        tol: float | None = 1e-4,
        coding_max_iter: int = 200,
        coding_tol: float | None = 1e-6,
        dict_max_iter: int = 1,
        dict_tol: float = 1e-6,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if lambda1 < 0 or lambda2 < 0:
            raise ValueError("lambda1 and lambda2 must be >= 0.")
        if eta <= 0:
            raise ValueError("eta must be > 0.")

        self.n_components = n_components
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eta = eta
        self.n_iter = n_iter
        self.tol = tol
        self.coding_max_iter = coding_max_iter
        self.coding_tol = coding_tol
        self.dict_max_iter = dict_max_iter
        self.dict_tol = dict_tol
        self.random_state = random_state
        self.verbose = verbose

        self.D_list_: list[np.ndarray] | None = None
        self.X_list_: list[np.ndarray] | None = None
        self.atom_boundaries_: dict[int, tuple[int, int]] | None = None
        self.classes_: np.ndarray | None = None
        self.sample_order_: np.ndarray | None = None
        self.objective_history_: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        D_init: list[np.ndarray] | None = None,
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> "FDDL":
        """
        Learn a Fisher discriminative dictionary.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
            Training signals.
        y : np.ndarray, shape (n_samples,)
            Integer or hashable class labels, one per column of X.
        D_init : list of np.ndarray or None
            Optional initial per-class sub-dictionaries (unit-norm
            columns). If None, atoms are initialized as random
            unit-norm vectors (Table 1, step 1). Ignored when resuming.
        checkpoint_dir : str or None
            If given, a checkpoint of the outer loop is written to
            ``<checkpoint_dir>/fddl_checkpoint.npz`` after every outer
            iteration, overwriting the previous one.
        resume : bool
            If True (default) and a checkpoint exists, resume from it.

        Returns
        -------
        self
        """
        device = _require_gpu_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        n_features, n_samples = X.shape
        if y.shape[0] != n_samples:
            raise DictionaryLearningError(
                f"y has {y.shape[0]} labels but X has {n_samples} samples."
            )

        classes, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        if n_classes < 2:
            raise DictionaryLearningError("FDDL requires at least 2 classes.")

        # Group columns by class (Table 1 operates on per-class blocks A_i).
        sample_order = np.argsort(y_idx, kind="stable")
        y_sorted = y_idx[sample_order]
        sizes = [int(np.sum(y_sorted == i)) for i in range(n_classes)]
        if any(s < 2 for s in sizes):
            raise DictionaryLearningError(
                "Every class needs >= 2 samples (class means/scatter are undefined otherwise)."
            )
        sample_boundaries = block_boundaries(sizes)

        # Move the training signals to the GPU once
        X_t = torch.as_tensor(X, device=device)
        sample_order_t = torch.as_tensor(sample_order, dtype=torch.long, device=device)
        A_list = [
            X_t[:, sample_order_t[s:e]]
            for s, e in (sample_boundaries[i] for i in range(n_classes))
        ]
        A_full = torch.hstack(A_list)

        atoms_per_class = resolve_atoms_per_class(self.n_components, n_classes)
        atom_boundaries = block_boundaries(atoms_per_class)
        n_atoms = sum(atoms_per_class)

        rng = np.random.RandomState(self.random_state)

        checkpoint_path = None
        start_iter = 0
        D_list = None
        X_list = None
        self.objective_history_ = []

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, _CHECKPOINT_FILENAME)
            if resume and os.path.exists(checkpoint_path):
                D_list, X_list, self.objective_history_, start_iter = self._load_checkpoint(
                    checkpoint_path, n_classes
                )
                # Checkpoints are numpy on disk; move the resumed state to GPU.
                D_list = [_to_device(Di, device) for Di in D_list]
                X_list = [_to_device(Xi, device) for Xi in X_list]
                if self.verbose:
                    print(
                        f"Resuming from checkpoint at outer iteration "
                        f"{start_iter}/{self.n_iter} ({checkpoint_path})"
                    )

        if D_list is None:
            if D_init is not None:
                if len(D_init) != n_classes:
                    raise DictionaryLearningError(
                        f"D_init has {len(D_init)} sub-dictionaries but there are "
                        f"{n_classes} classes."
                    )
                D_list = [normalize_columns(_to_device(Di, device)) for Di in D_init]
                for Di in D_list:
                    _check_dict_normalized(Di)
            else:
                # Table 1, step 1: random unit-norm atoms.
                D_list = [
                    normalize_columns(_to_device(rng.randn(n_features, p), device))
                    for p in atoms_per_class
                ]
            X_list = [
                torch.zeros((n_atoms, sizes[i]), dtype=torch.float32, device=device)
                for i in range(n_classes)
            ]

        D_full = torch.hstack(D_list)

        for it in range(start_iter, self.n_iter):
            # ---- Step 2 (Eq. 7): update X class-by-class, D fixed ----
            tracker = GlobalMeanTracker(X_list, sizes)
            for i in range(n_classes):
                logger.info(f"Updating class {i} codes")
                stats = tracker.exclude(i)
                Xi_new, _, _ = solve_class_codes(
                    X_list[i],
                    i,
                    D_list,
                    D_full,
                    A_list[i],
                    atom_boundaries,
                    stats,
                    self.lambda1,
                    self.lambda2,
                    self.eta,
                    max_iter=self.coding_max_iter,
                    tol=self.coding_tol,
                )
                X_list[i] = Xi_new
                tracker.update(i, Xi_new)

            # ---- Step 3 (Eq. 8): update D class-by-class, X fixed ----
            X_full_stacked = torch.hstack(X_list)

            def row_block(k: int) -> torch.Tensor:
                s, e = atom_boundaries[k]
                return X_full_stacked[s:e, :]

            D_full_sum = sum(Dj @ row_block(j) for j, Dj in enumerate(D_list))

            for i in range(n_classes):
                logger.info(f"Updating class {i} dictionary")
                A_stat, B_stat = build_di_update_system(
                    i, D_list[i], D_full_sum, A_list, atom_boundaries,
                    X_full_stacked=X_full_stacked, A_full=A_full, sample_boundaries=sample_boundaries,
                )
                # bcd_dictionary_update mutates its D argument in place
                # and returns that same tensor object.
                old_contrib = D_list[i] @ row_block(i)
                Di_updated = bcd_dictionary_update(
                    D_list[i], A_stat, B_stat, 0, self.dict_max_iter, self.dict_tol
                )
                del A_stat, B_stat
                Di_new = normalize_columns(Di_updated)
                del Di_updated

                # Patch the running sum in place of a full O(n_classes) rebuild.
                D_full_sum = D_full_sum + Di_new @ row_block(i) - old_contrib
                D_list[i] = Di_new

            D_full = torch.hstack(D_list)

            # ---- Objective (Eq. 6) for convergence tracking ----
            logger.info("Computing global fischer value")
            obj = self.lambda2 * global_fisher_value(X_list, self.eta)
            for i in range(n_classes):
                logger.info(f"Computing fidelity value for class {i}")
                obj += fidelity_value(X_list[i], i, D_list, A_list[i], atom_boundaries)
                obj += self.lambda1 * float(torch.sum(torch.abs(X_list[i])))
            self.objective_history_.append(obj)

            if self.verbose:
                print(f"[FDDL] Iter {it + 1}/{self.n_iter}  J={obj:.6f}")

            if checkpoint_path is not None:
                self._save_checkpoint(checkpoint_path, D_list, X_list, self.objective_history_, it + 1)

            if (
                self.tol is not None
                and len(self.objective_history_) >= 2
                and abs(self.objective_history_[-2] - obj) < self.tol * abs(self.objective_history_[-2])
            ):
                if self.verbose:
                    print(f"[FDDL] Converged at iteration {it + 1}.")
                break

        # Public attributes: converted back to numpy here.
        self.D_list_ = [Di.detach().cpu().numpy() for Di in D_list]
        self.X_list_ = [Xi.detach().cpu().numpy() for Xi in X_list]
        self.atom_boundaries_ = atom_boundaries
        self.classes_ = classes
        self.sample_order_ = sample_order
        return self

    @property
    def D_(self) -> np.ndarray:
        if self.D_list_ is None:
            raise DictionaryLearningError("Call fit() before accessing D_.")
        return np.hstack(self.D_list_)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path, D_list, X_list, history, iteration) -> None:
        tmp_path = path + ".tmp.npz"
        payload = {"n_classes": len(D_list), "history": np.array(history), "iteration": iteration}
        for i, (Di, Xi) in enumerate(zip(D_list, X_list)):
            payload[f"D_{i}"] = Di.detach().cpu().numpy()
            payload[f"X_{i}"] = Xi.detach().cpu().numpy()
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, path)

    def _load_checkpoint(self, path, n_classes):
        data = np.load(path)
        if int(data["n_classes"]) != n_classes:
            raise DictionaryLearningError(
                "Checkpoint's class count does not match the current training data."
            )
        D_list = [data[f"D_{i}"] for i in range(n_classes)]
        X_list = [data[f"X_{i}"] for i in range(n_classes)]
        history = list(data["history"])
        iteration = int(data["iteration"])
        return D_list, X_list, history, iteration