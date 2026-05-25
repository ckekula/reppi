"""
Frozen dictionary learning.

Implements incremental dictionary learning where a background dictionary
D_n is first learned from normal/base data, then progressively extended
with class-specific residual dictionaries D_a_1, D_a_2, … learned one
class at a time.  All previously learned atoms are frozen during each
new residual learning step.

The combined dictionary after k classes is:

    D = [ D_n | D_a_1 | D_a_2 | ... | D_a_k ]

and the sparse code α = [α_n; α_a_1; ...; α_a_k] partitions accordingly.
A linear classifier W over the full α is re-learned (via least squares)
after every new class is added.

Design
------
Two classes are provided:

``FrozenDictionaryLearner``
    Single residual learning step.  Given a frozen D_frozen and a
    discriminative learner class, it learns D_active on the residual
    of X after projecting onto D_frozen.  The combined dictionary is
    exposed as ``D_combined_``.  Use this if you want fine-grained
    control over each step.

``IncrementalFrozenDictionary``
    Orchestrates the full sequential pipeline:
      1. Fit a base dictionary D_n on normal/background data.
      2. Call ``add_class(X, class_label)`` for each new abnormality.
         The base + all prior residual dicts are frozen; only the new
         residual dict is learned.
      3. ``predict`` / ``score`` classify over the full combined dict.

Both classes are typed against ``BaseDiscriminativeDictionaryLearner``,
so any conforming learner (LC-KSVD, FDDL, LEDL, …) can be plugged in.

Example
-------
::

    from reppi.dictionary.frozen import IncrementalFrozenDictionary
    from reppi import LCKSVD

    inc = IncrementalFrozenDictionary(
        base_learner_class=LCKSVD,
        base_learner_kwargs=dict(
            n_components=128, n_nonzero_coefs=10, variant="lcksvd2"
        ),
        residual_learner_class=LCKSVD,
        residual_learner_kwargs=dict(
            n_components=32, n_nonzero_coefs=10, variant="lcksvd2"
        ),
        n_nonzero_coefs=10,
    )

    inc.fit_base(X_normal, H_normal)
    inc.add_class(X_class1, class_label=1)
    inc.add_class(X_class2, class_label=2)

    predictions = inc.predict(X_test)
    accuracy    = inc.score(X_test, H_test)
"""

from __future__ import annotations

import numpy as np

from reppi.base import BaseDiscriminativeDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp import OMP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_classifier(
    Gamma: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Fit a linear classifier W via least squares: W @ Gamma ≈ H.

    Parameters
    ----------
    Gamma : np.ndarray, shape (n_atoms, n_samples)
    H     : np.ndarray, shape (n_classes, n_samples)  one-hot

    Returns
    -------
    W : np.ndarray, shape (n_classes, n_atoms)
    """
    return H @ np.linalg.pinv(Gamma)


def _encode_residual(
    X: np.ndarray,
    D_frozen: np.ndarray,
    n_nonzero_coefs: int,
) -> np.ndarray:
    """
    Compute the reconstruction residual after encoding X over D_frozen.

    R = X - D_frozen @ Gamma_frozen

    Parameters
    ----------
    X            : np.ndarray, shape (n_features, n_samples)
    D_frozen     : np.ndarray, shape (n_features, n_frozen_atoms)
    n_nonzero_coefs : int

    Returns
    -------
    R : np.ndarray, shape (n_features, n_samples)  residual signals
    """
    coder = OMP(n_nonzero_coefs, mode="batch", check_dict=False)
    Gamma_frozen = coder.encode(X, D_frozen)
    return X - D_frozen @ Gamma_frozen


# ---------------------------------------------------------------------------
# FrozenDictionaryLearner
# ---------------------------------------------------------------------------


class FrozenDictionaryLearner:
    """
    Learn a residual dictionary D_active given a frozen dictionary D_frozen.

    The learner encodes all signals over D_frozen first, then trains
    D_active on the reconstruction residual.  The combined dictionary

        D_combined = [ D_frozen | D_active ]

    is exposed as ``D_combined_`` after fitting.

    This class handles a single residual learning step.  For the full
    sequential pipeline, see ``IncrementalFrozenDictionary``.

    Parameters
    ----------
    D_frozen : np.ndarray, shape (n_features, n_frozen_atoms)
        Pre-trained frozen dictionary.  Never modified.
    learner_class : type[BaseDiscriminativeDictionaryLearner]
        Discriminative dictionary learning class to use for the residual.
    learner_kwargs : dict
        Keyword arguments forwarded to ``learner_class.__init__``.
    n_nonzero_coefs : int
        Sparsity level used when encoding over the frozen dictionary to
        compute the residual, and when encoding over the combined dictionary
        for downstream tasks.
    learn_on_residual : bool
        If True (default), train D_active on the reconstruction residual
        R = X - D_frozen @ Gamma_frozen.  If False, train on the original
        X — useful when the frozen dict is very small and you want the
        active dict to model the full signal, not just what D_frozen misses.
    refit_classifier : bool
        If True (default), re-learn W over the full combined dictionary
        after fitting D_active.

    Attributes
    ----------
    D_combined_ : np.ndarray, shape (n_features, n_frozen + n_active)
    W_          : np.ndarray, shape (n_classes, n_frozen + n_active)
    learner_    : fitted instance of ``learner_class``
    n_frozen_   : int  number of frozen atoms
    n_active_   : int  number of active (residual) atoms
    class_boundaries_ : dict[int, tuple[int, int]]
        Atom ranges for each class in the *combined* dictionary, merging
        frozen boundaries (if any) with the active learner's boundaries.
    """

    def __init__(
        self,
        D_frozen: np.ndarray,
        learner_class: type[BaseDiscriminativeDictionaryLearner],
        learner_kwargs: dict,
        n_nonzero_coefs: int,
        learn_on_residual: bool = True,
        refit_classifier: bool = True,
    ) -> None:
        self.D_frozen = np.asarray(D_frozen, dtype=float)
        self.learner_class = learner_class
        self.learner_kwargs = learner_kwargs
        self.n_nonzero_coefs = n_nonzero_coefs
        self.learn_on_residual = learn_on_residual
        self.refit_classifier = refit_classifier

        self.D_combined_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.learner_: BaseDiscriminativeDictionaryLearner | None = None
        self.n_frozen_: int = self.D_frozen.shape[1]
        self.n_active_: int = 0
        self.class_boundaries_: dict[int, tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        H: np.ndarray,
        frozen_class_boundaries: dict[int, tuple[int, int]] | None = None,
    ) -> "FrozenDictionaryLearner":
        """
        Fit the residual dictionary on (X, H).

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        H : np.ndarray, shape (n_classes, n_samples)  one-hot labels
        frozen_class_boundaries : dict or None
            ``class_boundaries_`` from earlier frozen stages, used to build
            the merged ``class_boundaries_`` on the combined dictionary.
            Pass None if D_frozen has no class structure (e.g. base stage).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        H = np.asarray(H, dtype=float)

        # --- Train active dict on residual (or full X) ---
        X_train = (
            _encode_residual(X, self.D_frozen, self.n_nonzero_coefs)
            if self.learn_on_residual
            else X
        )

        learner = self.learner_class(**self.learner_kwargs)
        learner.fit(X_train, H)
        self.learner_ = learner

        D_active = learner.D_
        self.n_active_ = D_active.shape[1]

        # --- Combine dictionaries ---
        self.D_combined_ = np.hstack([self.D_frozen, D_active])

        # --- Build merged class_boundaries_ ---
        n_frozen = self.n_frozen_
        active_boundaries = learner.class_boundaries_ or {}
        merged: dict[int, tuple[int, int]] = {}

        # Carry over frozen boundaries unchanged
        if frozen_class_boundaries:
            merged.update(frozen_class_boundaries)

        # Shift active boundaries by n_frozen columns
        for c, (s, e) in active_boundaries.items():
            merged[c] = (s + n_frozen, e + n_frozen)

        self.class_boundaries_ = merged

        # --- Re-learn classifier over full combined dict ---
        if self.refit_classifier:
            coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
            Gamma_full = coder.encode(X, self.D_combined_)
            self.W_ = _fit_classifier(Gamma_full, H)
        else:
            # Pad learner's W with zeros for the frozen columns
            W_active = learner.W_
            if W_active is not None:
                pad = np.zeros((W_active.shape[0], n_frozen))
                self.W_ = np.hstack([pad, W_active])
            else:
                self.W_ = None

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode X over the combined dictionary."""
        self._check_fitted()
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, self.D_combined_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify X using W_ and the combined dictionary."""
        self._check_fitted()
        Gamma = self.transform(X)
        return np.argmax(self.W_ @ Gamma, axis=0)

    def score(self, X: np.ndarray, H: np.ndarray) -> float:
        """Classification accuracy on (X, H)."""
        true = np.argmax(H, axis=0)
        pred = self.predict(X)
        return float(np.mean(pred == true))

    def _check_fitted(self) -> None:
        if self.D_combined_ is None:
            raise DictionaryLearningError(
                "Call fit() before transform() / predict()."
            )


# ---------------------------------------------------------------------------
# IncrementalFrozenDictionary
# ---------------------------------------------------------------------------


class IncrementalFrozenDictionary:
    """
    Incrementally learn class-specific residual dictionaries, freezing all
    previously learned atoms before training the next class.

    Pipeline
    --------
    1. ``fit_base(X, H)``
       Learn a base dictionary D_n from normal/background data using
       ``base_learner_class``.  This dictionary is frozen for all
       subsequent steps.

    2. ``add_class(X, H, class_label)``
       Learn a residual dictionary D_a for the new class on top of
       the currently frozen dictionary [ D_n | D_a_1 | … ].
       Only the new residual atoms are updated; all prior atoms are frozen.
       The combined dictionary is extended in-place.
       W is re-learned over all classes after each addition.

    3. ``predict(X)`` / ``score(X, H)``
       Classify using the full combined dictionary and the latest W.

    Parameters
    ----------
    base_learner_class : type[BaseDiscriminativeDictionaryLearner]
        Learner used for the initial base dictionary.
    base_learner_kwargs : dict
        Init kwargs for ``base_learner_class``.
    residual_learner_class : type[BaseDiscriminativeDictionaryLearner]
        Learner used for each residual dictionary.  Can be the same as or
        different from ``base_learner_class``.
    residual_learner_kwargs : dict
        Init kwargs for ``residual_learner_class``.  Applied identically
        for every ``add_class`` call; override per-call via
        ``add_class(..., learner_kwargs_override=...)``.
    n_nonzero_coefs : int
        Sparsity level for all encoding steps.
    learn_on_residual : bool
        Passed through to ``FrozenDictionaryLearner`` at each step.
        Default True.
    refit_classifier : bool
        Re-learn W over the full combined dict after each add_class.
        Default True (recommended — see module docstring).
    freeze_classifier : bool
        If True, W columns for previously seen classes are frozen when a
        new class is added; only the new class's W column is learned.
        Default False (re-learn all W columns jointly each time).

    Attributes
    ----------
    D_  : np.ndarray  full combined dictionary after all steps
    W_  : np.ndarray  current linear classifier
    class_labels_ : list[int]  class labels in insertion order
    class_boundaries_ : dict[int, tuple[int, int]]
        Per-class atom ranges in the full combined D_.
    stage_learners_ : list
        Fitted learner or FrozenDictionaryLearner from each stage,
        in order (index 0 = base stage).
    errors_ : dict[int, list[float]]
        Per-stage training RMSE curves keyed by class_label
        (key -1 for the base stage).
    """

    def __init__(
        self,
        base_learner_class: type[BaseDiscriminativeDictionaryLearner],
        base_learner_kwargs: dict,
        residual_learner_class: type[BaseDiscriminativeDictionaryLearner],
        residual_learner_kwargs: dict,
        n_nonzero_coefs: int,
        learn_on_residual: bool = True,
        refit_classifier: bool = True,
        freeze_classifier: bool = False,
    ) -> None:
        self.base_learner_class = base_learner_class
        self.base_learner_kwargs = base_learner_kwargs
        self.residual_learner_class = residual_learner_class
        self.residual_learner_kwargs = residual_learner_kwargs
        self.n_nonzero_coefs = n_nonzero_coefs
        self.learn_on_residual = learn_on_residual
        self.refit_classifier = refit_classifier
        self.freeze_classifier = freeze_classifier

        # State built incrementally
        self.D_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.class_labels_: list[int] = []
        self.class_boundaries_: dict[int, tuple[int, int]] = {}
        self.stage_learners_: list = []
        self.errors_: dict[int, list[float]] = {}

        # Internal: accumulated (X, H) across all classes for W refit
        self._X_all: list[np.ndarray] = []
        self._H_rows: int | None = None   # n_classes total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_base(
        self,
        X: np.ndarray,
        H: np.ndarray,
    ) -> "IncrementalFrozenDictionary":
        """
        Learn the base dictionary from normal / background data.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        H : np.ndarray, shape (n_classes, n_samples)
            One-hot labels for the base class(es).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        H = np.asarray(H, dtype=float)

        learner = self.base_learner_class(**self.base_learner_kwargs)
        learner.fit(X, H)

        self.D_ = learner.D_
        self.class_boundaries_ = dict(learner.class_boundaries_ or {})
        self.stage_learners_.append(learner)
        self.errors_[-1] = list(getattr(learner, "errors_", []))

        # Initialise W
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        Gamma = coder.encode(X, self.D_)
        self.W_ = _fit_classifier(Gamma, H)

        self._H_rows = H.shape[0]
        self._X_all.append(X)

        return self

    def add_class(
        self,
        X: np.ndarray,
        H: np.ndarray,
        class_label: int,
        learner_kwargs_override: dict | None = None,
    ) -> "IncrementalFrozenDictionary":
        """
        Learn a residual dictionary for a new class and extend D_.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
            Training signals for this class.
        H : np.ndarray, shape (n_classes_so_far + 1, n_samples)
            One-hot label matrix for *all* classes seen so far including
            the new one.  Used to refit W after extending the dictionary.
        class_label : int
            Integer label for this class.  Must not have been added before.
        learner_kwargs_override : dict or None
            If supplied, overrides ``residual_learner_kwargs`` for this
            call only.  Useful for adjusting n_components per class.

        Returns
        -------
        self
        """
        if self.D_ is None:
            raise DictionaryLearningError(
                "Call fit_base() before add_class()."
            )
        if class_label in self.class_labels_:
            raise ValueError(
                f"class_label {class_label} has already been added."
            )

        X = np.asarray(X, dtype=float)
        H = np.asarray(H, dtype=float)

        kwargs = {**self.residual_learner_kwargs, **(learner_kwargs_override or {})}

        # The residual learner only sees X (n_samples for this class).
        # Extract the H columns that correspond to X — the caller passes the
        # full H over all accumulated data, but the learner needs H for X only.
        n_new = X.shape[1]
        H_for_learner = H[:, -n_new:]   # last n_new columns = this class's signals

        frozen_step = FrozenDictionaryLearner(
            D_frozen=self.D_,
            learner_class=self.residual_learner_class,
            learner_kwargs=kwargs,
            n_nonzero_coefs=self.n_nonzero_coefs,
            learn_on_residual=self.learn_on_residual,
            refit_classifier=False,  # we handle W ourselves below
        )
        frozen_step.fit(X, H_for_learner, frozen_class_boundaries=dict(self.class_boundaries_))

        # --- Extend D_ and class_boundaries_ ---
        n_prev = self.D_.shape[1]
        D_active = frozen_step.learner_.D_
        n_active = D_active.shape[1]
        self.D_ = np.hstack([self.D_, D_active])

        # Map new class atoms into the combined dictionary
        self.class_boundaries_[class_label] = (n_prev, n_prev + n_active)
        self.class_labels_.append(class_label)
        self.stage_learners_.append(frozen_step)
        self.errors_[class_label] = list(
            getattr(frozen_step.learner_, "errors_", [])
        )

        # Accumulate training data for W refit
        self._X_all.append(X)

        # --- Refit W over all data and the full combined dict ---
        if self.refit_classifier:
            X_all = np.hstack(self._X_all)
            # H must cover all columns of X_all — caller is responsible
            # for passing the full H including all previous classes
            self._refit_W(X_all, H)
        elif self.freeze_classifier:
            self._extend_W_frozen(D_active.shape[1], H)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classify X using the full combined dictionary and the current W.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)
        """
        self._check_fitted()
        Gamma = self._encode(X)
        return np.argmax(self.W_ @ Gamma, axis=0)

    def score(self, X: np.ndarray, H: np.ndarray) -> float:
        """
        Classification accuracy on (X, H).

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        H : np.ndarray, shape (n_classes, n_samples)

        Returns
        -------
        accuracy : float in [0, 1]
        """
        true = np.argmax(H, axis=0)
        pred = self.predict(X)
        return float(np.mean(pred == true))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode X over the full combined dictionary D_."""
        self._check_fitted()
        return self._encode(X)

    def get_stage_dict(self, stage: int) -> np.ndarray:
        """
        Return the sub-dictionary learned at a given stage.

        Stage 0 is the base dictionary; stage k (k >= 1) is the k-th
        residual dictionary.

        Parameters
        ----------
        stage : int

        Returns
        -------
        D_stage : np.ndarray
        """
        self._check_fitted()
        if stage < 0 or stage >= len(self.stage_learners_):
            raise IndexError(
                f"stage must be in [0, {len(self.stage_learners_) - 1}], got {stage}."
            )
        learner = self.stage_learners_[stage]
        # Base stage: learner is a BaseDiscriminativeDictionaryLearner
        if isinstance(learner, FrozenDictionaryLearner):
            return learner.learner_.D_
        return learner.D_

    def get_class_dict(self, class_label: int) -> np.ndarray:
        """
        Return the sub-dictionary atoms for a given class label.

        Parameters
        ----------
        class_label : int

        Returns
        -------
        D_c : np.ndarray
        """
        self._check_fitted()
        if class_label not in self.class_boundaries_:
            raise KeyError(f"class_label {class_label} not found.")
        s, e = self.class_boundaries_[class_label]
        return self.D_[:, s:e]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.D_ is None:
            raise DictionaryLearningError(
                "Call fit_base() before using this method."
            )

    def _encode(self, X: np.ndarray) -> np.ndarray:
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, self.D_)

    def _refit_W(self, X_all: np.ndarray, H: np.ndarray) -> None:
        """Re-learn W jointly over all classes on the full combined dict."""
        Gamma = self._encode(X_all)
        self.W_ = _fit_classifier(Gamma, H)

    def _extend_W_frozen(self, n_new_atoms: int, H: np.ndarray) -> None:
        """
        Freeze existing W columns; learn only the columns for new atoms.

        This implements the ``freeze_classifier=True`` behaviour: old class
        boundaries in W stay fixed; only weights for the n_new_atoms are
        updated for the new class.
        """
        if self.W_ is None:
            return
        n_classes_new = H.shape[0]
        n_classes_old = self.W_.shape[0]
        n_atoms_old = self.W_.shape[1]

        # Extend W with zero rows for any new classes and zero cols for new atoms
        W_extended = np.zeros((n_classes_new, n_atoms_old + n_new_atoms))
        W_extended[:n_classes_old, :n_atoms_old] = self.W_

        # Learn only the new columns via least squares restricted to new atoms
        # Encode all accumulated data over full dict, extract new-atom codes
        X_all = np.hstack(self._X_all)
        Gamma_full = self._encode(X_all)
        Gamma_new = Gamma_full[n_atoms_old:, :]   # (n_new_atoms, n_samples)

        # Solve W_new @ Gamma_new ≈ H - W_old @ Gamma_old
        Gamma_old = Gamma_full[:n_atoms_old, :]
        residual_H = H - W_extended[:, :n_atoms_old] @ Gamma_old
        W_new_cols = residual_H @ np.linalg.pinv(Gamma_new)
        W_extended[:, n_atoms_old:] = W_new_cols

        self.W_ = W_extended