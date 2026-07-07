from __future__ import annotations

import os

import numpy as np

from reppi.base import BaseDiscriminativeDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp.omp import OMP

from reppi.dictionary.frozen.utils import _fit_classifier
from reppi.dictionary.frozen.residual_learner import FrozenDictionaryLearner

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

    Checkpointing
    -------------
    Each call to ``fit_base`` or ``add_class`` represents one *stage* of
    the incremental pipeline, and each stage trains its own, differently
    shaped, inner dictionary-learner instance. If a ``checkpoint_dir`` is
    given, this class therefore does NOT hand every stage the same path —
    doing so would cause the second stage's checkpoint load to fail (or,
    worse, be silently wrong) since its X/H shapes differ from the first
    stage's. Instead, each stage gets its own subdirectory:

        <checkpoint_dir>/base/           (fit_base)
        <checkpoint_dir>/class_<label>/  (add_class, per class_label)

    so that interrupting and resuming an individual stage's training does
    not collide with any other stage's saved state. This only has an
    effect if the underlying ``base_learner_class`` /
    ``residual_learner_class`` (e.g. KSVD, LCKSVD) support a
    ``checkpoint_dir`` argument on their own ``fit()``; learners that
    don't will simply ignore it being unset and train without
    checkpointing.

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
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> "IncrementalFrozenDictionary":
        """
        Learn the base dictionary from normal / background data.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        H : np.ndarray, shape (n_classes, n_samples)
            One-hot labels for the base class(es).
        checkpoint_dir : str or None
            If given, a ``base`` subdirectory under this path is passed to
            the base learner's own ``fit(..., checkpoint_dir=...)``, if it
            supports one. See the class docstring for why this is a
            dedicated subdirectory rather than shared across stages.
        resume : bool
            Forwarded to the base learner's ``fit()`` alongside the
            ``base`` checkpoint subdirectory. Default True.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        H = np.asarray(H, dtype=float)

        learner = self.base_learner_class(**self.base_learner_kwargs)
        fit_kwargs = {}
        if checkpoint_dir is not None:
            fit_kwargs["checkpoint_dir"] = os.path.join(checkpoint_dir, "base")
            fit_kwargs["resume"] = resume
        learner.fit(X, H, **fit_kwargs)

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
        checkpoint_dir: str | None = None,
        resume: bool = True,
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
        checkpoint_dir : str or None
            If given, a ``class_<class_label>`` subdirectory under this
            path is passed down to ``FrozenDictionaryLearner.fit()`` (and
            from there to the residual learner's own ``fit()``), if
            supported. Kept distinct per class_label, and distinct from
            the base stage's ``base`` subdirectory, so that resuming one
            stage never collides with another's saved state — see the
            class docstring.
        resume : bool
            Forwarded down to the residual learner's ``fit()`` alongside
            the per-class checkpoint subdirectory. Default True.

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
        stage_checkpoint_dir = (
            os.path.join(checkpoint_dir, f"class_{class_label}")
            if checkpoint_dir is not None
            else None
        )
        frozen_step.fit(
            X,
            H_for_learner,
            frozen_class_boundaries=dict(self.class_boundaries_),
            checkpoint_dir=stage_checkpoint_dir,
            resume=resume,
        )

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