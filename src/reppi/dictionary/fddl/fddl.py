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

Backend / memory model
-----------------------
GPU-only for compute: ``fit`` requires a CUDA or MPS device and raises
otherwise (no silent CPU fallback).

Training data and coding coefficients (``A_list``/``X_list``, one
tensor per class) are the dominant memory cost for large problems --
e.g. many total atoms x many total samples -- and are kept resident on
the **CPU** as the "source of truth" storage, not the GPU. Each phase
of the outer loop streams only the slice of that storage it currently
needs to the GPU, in bounded-size chunks, and moves results back:

  * Step 2 (Eq. 7, coding): ``solve_class_codes_chunked`` solves each
    class's Eq. (7) FISTA problem with the iterate itself CPU-resident,
    streaming ``coding_chunk_size``-sized column chunks to the GPU for
    the actual gradient/objective evaluations. A class's own coding
    solve is the single most memory-hungry step in this whole loop
    (Xi alone is n_atoms_total x n_i, and FISTA/gradient math needs
    several such buffers alive at once) -- this is what actually bounds
    peak GPU memory independent of class size, not just "one class at
    a time" (which alone is not enough once a single large class's
    working set approaches the device's capacity).
  * Step 3 (Eq. 8, dictionary): ``build_di_update_system_streaming``
    streams every class's data in ``dict_update_chunk_size``-sized
    column chunks to accumulate the Eq.(8) sufficient statistics,
    never materializing a full-dataset-width tensor on the GPU.
  * Objective (Eq. 6) computation for logging/convergence:
    ``fidelity_value_chunked`` streams the fidelity term the same way
    as Step 3; the L1 term and ``global_fisher_value`` run directly on
    the CPU-resident X_list (plain reductions, not a per-iteration hot
    path, no GPU needed at all for these).

Only the dictionaries (``D_list``/``D_full``, small: n_features x
n_atoms) stay permanently GPU-resident. This bounds peak GPU memory by
``coding_chunk_size`` for Step 2 and by ``dict_update_chunk_size`` for
Step 3 -- both independent of class/dataset size -- rather than
requiring even a single class's full coefficient matrix to fit on the
GPU at once. The trade-off is extra compute: Step 2 in particular now
makes several passes over a class's data per FISTA iteration (one
reduction pass for the Fisher term's mean, one pass for the gradient,
plus more during backtracking line search) instead of one fused
GPU-resident computation, so this is deliberately "memory-safe first",
not the fastest possible path for classes that would fit whole on the
GPU.

Set ``pin_memory=False`` if the CPU-resident tensors' pinned-memory
footprint (same total size as the data itself) is itself a concern;
pinning only speeds up host<->device transfers, it isn't required for
correctness.

Checkpoints are still written as numpy .npz (so they're portable /
inspectable without torch), and the fitted attributes (``D_``,
``D_list_``, ``X_list_``) are converted back to numpy at the end of
``fit``.

Usage
-----
    model = FDDL(
        n_components=8 * n_classes,   # e.g. 8 atoms/class, per Sec. 6.2
        lambda1=0.005, lambda2=0.005,
        classifier="gc", gamma=0.001, w=0.05,
        dict_update_chunk_size=8192,  # tune down if Step 3 still OOMs
        coding_chunk_size=8192,       # tune down if Step 2 still OOMs
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

from reppi.dictionary.fddl.coding import solve_class_codes_chunked
from reppi.dictionary.fddl.utils import (
    GlobalMeanTracker,
    block_boundaries,
    build_di_update_system_streaming,
    fidelity_value_chunked,
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


def _to_cpu_storage(arr, pin_memory: bool) -> torch.Tensor:
    """Convert a numpy array (or array-like) to a float32 CPU tensor, used
    as the resident storage for A_list/X_list under the CPU-offload model."""
    t = torch.as_tensor(np.asarray(arr, dtype=np.float32)).clone()
    if pin_memory:
        t = t.pin_memory()
    return t


def _empty_cache(device: torch.device) -> None:
    """Proactively return the caching allocator's freed blocks to the
    driver after a class's GPU-resident tensors are dereferenced.

    Not needed for correctness (PyTorch's allocator reuses freed blocks
    on its own), but this codebase pushes single classes very close to
    the device's total capacity (see FDDL.fit's module docstring), so
    the allocator having stale free blocks sized for one class's shape
    lying around when the *next* class's differently-shaped tensors
    need a fresh contiguous block is a real risk, not just a
    theoretical one. Called once per class boundary (a handful of
    times per outer iteration) -- not on any per-FISTA-iteration hot
    path, so the sync cost is negligible.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


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
    coding_chunk_size : int
        Maximum number of sample columns streamed to the GPU at once
        while solving each class's Eq. (7) FISTA problem (Step 2).
        Bounds Step 2's peak GPU memory independent of class size;
        lower it if Step 2 itself OOMs (see ``solve_class_codes_chunked``).
        This is typically the step that needs the smallest chunk size,
        since it needs several full-chunk-size buffers alive at once
        (FISTA's own iterate bookkeeping plus gradient temporaries).
    dict_update_chunk_size : int
        Maximum number of sample columns streamed to the GPU at once
        while accumulating the Eq. (8) sufficient statistics (Step 3).
        Bounds Step 3's peak GPU memory independent of dataset size;
        lower it if Step 3 itself OOMs (see ``build_di_update_system_streaming``).
    pin_memory : bool
        Whether the CPU-resident training data / coefficients
        (``A_list``/``X_list``) are allocated as pinned memory, for
        faster host<->device transfers. Set False to save that memory
        overhead if transfer speed isn't a concern.
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
        coding_chunk_size: int = 8192,
        dict_update_chunk_size: int = 8192,
        pin_memory: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if lambda1 < 0 or lambda2 < 0:
            raise ValueError("lambda1 and lambda2 must be >= 0.")
        if eta <= 0:
            raise ValueError("eta must be > 0.")
        if coding_chunk_size < 1:
            raise ValueError("coding_chunk_size must be >= 1.")
        if dict_update_chunk_size < 1:
            raise ValueError("dict_update_chunk_size must be >= 1.")

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
        self.coding_chunk_size = coding_chunk_size
        self.dict_update_chunk_size = dict_update_chunk_size
        self.pin_memory = pin_memory
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

        # A_list / X_list are the CPU-resident "source of truth" storage
        # (see module docstring): the compute loop below streams only
        # the slice it currently needs to `device`, so this is the only
        # place the full dataset is copied at once, and it stays on CPU.
        X_grouped = X[:, sample_order]
        A_list = [
            _to_cpu_storage(X_grouped[:, s:e], self.pin_memory)
            for _, (s, e) in sample_boundaries.items()
        ]
        del X_grouped

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
                # Checkpoints are numpy on disk. Dictionaries move to the
                # GPU (small, live there for the whole optimization);
                # coefficients stay on the CPU as resident storage.
                D_list = [_to_device(Di, device) for Di in D_list]
                X_list = [_to_cpu_storage(Xi, self.pin_memory) for Xi in X_list]
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
                _to_cpu_storage(
                    np.zeros((n_atoms, sizes[i]), dtype=np.float32), self.pin_memory
                )
                for i in range(n_classes)
            ]

        D_full = torch.hstack(D_list)

        for it in range(start_iter, self.n_iter):
            # ---- Step 2 (Eq. 7): update X class-by-class, D fixed ----
            # Coefficient means are tiny (one n_atoms-length vector per
            # class); kept GPU-resident throughout the sweep so they match
            # the device of each class's Xi while it's being solved there,
            # even though X_list itself lives on the CPU.
            tracker = GlobalMeanTracker(X_list, sizes, device=device)
            for i in range(n_classes):
                logger.info(f"Updating class {i} codes")
                stats = tracker.exclude(i)
                # Xi0/Ai stay CPU-resident throughout -- solve_class_codes_chunked
                # streams `coding_chunk_size`-sized column chunks to `device`
                # internally; no full-class GPU tensor is ever created here.
                Xi_new_cpu, _, _ = solve_class_codes_chunked(
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
                    device,
                    self.coding_chunk_size,
                    max_iter=self.coding_max_iter,
                    tol=self.coding_tol,
                )
                tracker.update(i, Xi_new_cpu)  # mean computed on CPU, moved to device internally
                X_list[i] = _to_cpu_storage(Xi_new_cpu.numpy(), self.pin_memory)
                del Xi_new_cpu
                _empty_cache(device)

            # ---- Step 3 (Eq. 8): update D class-by-class, X fixed ----
            # Sufficient statistics are streamed in `dict_update_chunk_size`
            # column chunks (see build_di_update_system_streaming) instead
            # of materializing a full-dataset-width tensor on the GPU.
            for i in range(n_classes):
                logger.info(f"Updating class {i} dictionary")
                A_stat, B_stat = build_di_update_system_streaming(
                    i,
                    D_list,
                    A_list,
                    X_list,
                    atom_boundaries,
                    device,
                    self.dict_update_chunk_size,
                )
                # bcd_dictionary_update mutates its D argument in place
                # and returns that same tensor object.
                Di_updated = bcd_dictionary_update(
                    D_list[i], A_stat, B_stat, 0, self.dict_max_iter, self.dict_tol
                )
                del A_stat, B_stat
                D_list[i] = normalize_columns(Di_updated)
                _empty_cache(device)

            D_full = torch.hstack(D_list)

            # ---- Objective (Eq. 6) for convergence tracking ----
            # global_fisher_value and the L1 term run directly on the
            # CPU-resident X_list -- both are plain reductions with no
            # matmuls, so there's no benefit to moving them to the GPU
            # and every reason not to for a class this large. Only the
            # fidelity term needs D (GPU-resident), so only it streams.
            logger.info("Computing global fischer value")
            obj = self.lambda2 * global_fisher_value(X_list, self.eta)
            for i in range(n_classes):
                logger.info(f"Computing fidelity value for class {i}")
                obj += fidelity_value_chunked(
                    X_list[i], i, D_list, A_list[i], atom_boundaries,
                    device, self.dict_update_chunk_size,
                )
                obj += self.lambda1 * float(torch.sum(torch.abs(X_list[i])))
                _empty_cache(device)
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
        self.X_list_ = [Xi.detach().cpu().numpy() for Xi in X_list]  # already CPU
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