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

Classification (Section 5)
-----------------------------
Two schemes are supported, chosen via ``classifier``:
  * 'gc' (Global Classifier, Eq. 9-10) — for small per-class sample
    counts, e.g. face recognition.
  * 'lc' (Local Classifier, Eq. 11-12) — for larger per-class sample
    counts, e.g. digit recognition.

Usage
-----
    model = FDDL(
        n_components=8 * n_classes,   # e.g. 8 atoms/class, per Sec. 6.2
        lambda1=0.005, lambda2=0.005,
        classifier="gc", gamma=0.001, w=0.05,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
"""

from __future__ import annotations

import os

import numpy as np

from reppi.base import BaseDiscriminativeDictionaryLearner
from reppi.dictionary.bcd.utils import bcd_dictionary_update
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.utils import _check_dict_normalized, normalize_columns

from reppi.dictionary.fddl.classify import fit_class_means, gc_classify, lc_classify
from reppi.dictionary.fddl.coding import solve_class_codes
from reppi.dictionary.fddl.utils import (
    OtherClassStats,
    block_boundaries,
    build_di_update_system,
    fidelity_value,
    global_fisher_value,
    resolve_atoms_per_class,
)

_CHECKPOINT_FILENAME = "fddl_checkpoint.npz"


class FDDL(BaseDiscriminativeDictionaryLearner):
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
    classifier : {'gc', 'lc'}
        Default classification scheme for ``predict`` (Section 5).
    gamma, w : float
        GC hyperparameters (Eq. 9-10): L1 weight and mean-distance
        weight.
    gamma1, gamma2 : float
        LC hyperparameters (Eq. 11-12): L1 weight and mean-pull weight.
    random_state : int or None
    verbose : bool

    Attributes
    ----------
    D_list_ : list of np.ndarray
        Learned per-class sub-dictionaries, D_list_[i] has shape
        (n_features, p_i).
    D_ : np.ndarray, shape (n_features, n_components)
        Learned dictionary, horizontally stacked D_list_.
    X_list_ : list of np.ndarray
        Learned per-class coding coefficients (full n_components rows).
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
        classifier: str = "gc",
        gamma: float = 0.001,
        w: float = 0.05,
        gamma1: float = 0.005,
        gamma2: float = 0.005,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if lambda1 < 0 or lambda2 < 0:
            raise ValueError("lambda1 and lambda2 must be >= 0.")
        if eta <= 0:
            raise ValueError("eta must be > 0.")
        if classifier not in ("gc", "lc"):
            raise ValueError("classifier must be 'gc' or 'lc'.")

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
        self.classifier = classifier
        self.gamma = gamma
        self.w = w
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.random_state = random_state
        self.verbose = verbose

        self.D_list_: list[np.ndarray] | None = None
        self.X_list_: list[np.ndarray] | None = None
        self.atom_boundaries_: dict[int, tuple[int, int]] | None = None
        self.classes_: np.ndarray | None = None
        self.sample_order_: np.ndarray | None = None
        self.objective_history_: list[float] = []
        self._means_full_: dict[int, np.ndarray] | None = None
        self._means_own_: dict[int, np.ndarray] | None = None

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
        X = np.asarray(X, dtype=float)
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
        A_list = [
            X[:, sample_order[s:e]] for s, e in (sample_boundaries[i] for i in range(n_classes))
        ]

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
                D_list = [normalize_columns(np.asarray(Di, dtype=float)) for Di in D_init]
                for Di in D_list:
                    _check_dict_normalized(Di)
            else:
                # Table 1, step 1: random unit-norm atoms.
                D_list = [
                    normalize_columns(rng.randn(n_features, p)) for p in atoms_per_class
                ]
            X_list = [np.zeros((n_atoms, sizes[i])) for i in range(n_classes)]

        D_full = np.hstack(D_list)

        for it in range(start_iter, self.n_iter):
            # ---- Step 2 (Eq. 7): update X class-by-class, D fixed ----
            for i in range(n_classes):
                stats = OtherClassStats(X_list, sizes, exclude=i)
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

            # ---- Step 3 (Eq. 8): update D class-by-class, X fixed ----
            for i in range(n_classes):
                Y_stack, Z_stack = build_di_update_system(
                    i, D_list, X_list, A_list, atom_boundaries
                )
                A_stat = Z_stack @ Z_stack.T
                B_stat = Y_stack @ Z_stack.T
                Di_updated = bcd_dictionary_update(
                    D_list[i], A_stat, B_stat, 0, self.dict_max_iter, self.dict_tol
                )
                # bcd_dictionary_update enforces ||d_j|| <= 1 (Mairal's
                # ball constraint); the paper requires exact unit norm.
                D_list[i] = normalize_columns(Di_updated)
            D_full = np.hstack(D_list)

            # ---- Objective (Eq. 6) for convergence tracking ----
            obj = self.lambda2 * global_fisher_value(X_list, self.eta)
            for i in range(n_classes):
                obj += fidelity_value(X_list[i], i, D_list, D_full, A_list[i], atom_boundaries)
                obj += self.lambda1 * float(np.sum(np.abs(X_list[i])))
            self.objective_history_.append(obj)

            if self.verbose:
                print(f"[FDDL] Iter {it + 1}/{self.n_iter}  J={obj:.6f}")

            if checkpoint_path is not None:
                self._save_checkpoint(checkpoint_path, D_list, X_list, self.objective_history_, it + 1)

            if (
                self.tol is not None
                and it > start_iter
                and abs(self.objective_history_[-2] - obj) < self.tol * abs(self.objective_history_[-2])
            ):
                if self.verbose:
                    print(f"[FDDL] Converged at iteration {it + 1}.")
                break

        self.D_list_ = D_list
        self.X_list_ = X_list
        self.atom_boundaries_ = atom_boundaries
        self.classes_ = classes
        self.sample_order_ = sample_order
        self._means_full_, self._means_own_ = fit_class_means(X_list, atom_boundaries)
        return self

    @property
    def D_(self) -> np.ndarray:
        if self.D_list_ is None:
            raise DictionaryLearningError("Call fit() before accessing D_.")
        return np.hstack(self.D_list_)

    def predict(
        self, X: np.ndarray, scheme: str | None = None, return_scores: bool = False
    ):
        """
        Classify query signals (Eq. 2, using Eq. 10 or Eq. 12's metric).

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        scheme : {'gc', 'lc'} or None
            Overrides ``self.classifier`` for this call.
        return_scores : bool
            If True, also return the (n_classes, n_samples) score
            matrix (lower is better).

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted labels, in the original label space of ``fit``.
        scores : np.ndarray, optional
        """
        if self.D_list_ is None:
            raise DictionaryLearningError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        scheme = scheme if scheme is not None else self.classifier
        if scheme == "gc":
            labels_idx, scores = gc_classify(
                X,
                self.D_,
                self.D_list_,
                self.atom_boundaries_,
                self._means_full_,
                self.gamma,
                self.w,
            )
        elif scheme == "lc":
            labels_idx, scores = lc_classify(
                X, self.D_list_, self._means_own_, self.gamma1, self.gamma2
            )
        else:
            raise ValueError("scheme must be 'gc' or 'lc'.")

        y_pred = self.classes_[labels_idx]
        return (y_pred, scores) if return_scores else y_pred

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path, D_list, X_list, history, iteration) -> None:
        tmp_path = path + ".tmp.npz"
        payload = {"n_classes": len(D_list), "history": np.array(history), "iteration": iteration}
        for i, (Di, Xi) in enumerate(zip(D_list, X_list)):
            payload[f"D_{i}"] = Di
            payload[f"X_{i}"] = Xi
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