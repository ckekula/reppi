from __future__ import annotations

import os

import numpy as np

from reppi.base import BaseDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp.omp import OMP

from reppi.dictionary.frozen.utils import _fit_classifier
from reppi.dictionary.frozen.residual_learner import FrozenDictionaryLearner

class IncrementalFrozenDictionary:
    """
    Incrementally learn class-specific residual dictionaries, freezing all
    previously learned atoms before training the next class.

    Faithful to Carroll et al. 2017 ("Outlier Learning via Augmented
    Frozen Dictionaries"): the underlying dictionary learner
    (``base_learner_class`` / ``residual_learner_class``, e.g. ``KSVD``)
    is unsupervised — it never sees class labels, only the training
    signals for whichever single class is currently being added. All
    per-class bookkeeping (``class_boundaries_``) and classification
    (``W_``) live entirely at this level, on top of the finished combined
    dictionary — exactly as in the paper, where the frozen dictionary
    produces sparse-code features and an SVM (a linear classifier ``W``
    here) is trained on those features as a separate step.

    Pipeline
    --------
    1. ``fit_base(X, H)``
       Learn a base dictionary D_n from normal/background data using
       ``base_learner_class``.  This dictionary is frozen for all
       subsequent steps.

    2. ``add_class(X, H, class_label)``
       Learn a residual dictionary D_a for the new class on top of
       the currently frozen dictionary [ D_n | D_a_1 | … ], via
       ``FrozenDictionaryLearner``. The underlying learner trains its new
       atoms jointly alongside the frozen ones every iteration (not on a
       one-shot residual — see ``BaseDictionaryLearner``'s frozen-
       dictionary contract and Carroll et al. 2017, Sec. III). W is
       re-learned over all classes after each addition.

    3. ``predict(X)`` / ``score(X, H)``
       Classify using the full combined dictionary and the latest W.

    Checkpointing
    -------------
    Each call to ``fit_base`` or ``add_class`` represents one *stage* of
    the incremental pipeline, and each stage trains its own, differently
    shaped, inner dictionary-learner instance. If a ``checkpoint_dir`` is
    given, this class therefore does NOT hand every stage the same path —
    doing so would cause the second stage's checkpoint load to fail (or,
    worse, be silently wrong) since its X shape differs from the first
    stage's. Instead, each stage gets its own subdirectory:

        <checkpoint_dir>/base/           (fit_base)
        <checkpoint_dir>/class_<label>/  (add_class, per class_label)

    so that interrupting and resuming an individual stage's training does
    not collide with any other stage's saved state.

    Parameters
    ----------
    base_learner_class : type[BaseDictionaryLearner]
        Unsupervised learner used for the initial base dictionary, e.g.
        ``KSVD``. Must accept ``D_frozen`` (default None) in its
        ``fit()``, per the frozen-dictionary contract.
    base_learner_kwargs : dict
        Init kwargs for ``base_learner_class``.
    residual_learner_class : type[BaseDictionaryLearner]
        Learner used for each residual dictionary, via
        ``FrozenDictionaryLearner``.  Can be the same as or different from
        ``base_learner_class``. Must support the frozen-dictionary
        contract (accept and honour ``D_frozen``).
    residual_learner_kwargs : dict
        Init kwargs for ``residual_learner_class``.  Applied identically
        for every ``add_class`` call; override per-call via
        ``add_class(..., learner_kwargs_override=...)`` — e.g. to give
        each class a different number of atoms.
    n_nonzero_coefs : int
        Sparsity level for all encoding steps at this level (W fitting /
        refitting, predict, score, transform).
    refit_classifier : bool
        Re-learn W over the full combined dict after each add_class.
        Default True.
    freeze_classifier : bool
        If True, W columns for previously seen classes are frozen when a
        new class is added; only the new class's W column is learned.
        Default False (re-learn all W columns jointly each time). Exactly
        one of ``refit_classifier`` / ``freeze_classifier`` must be True.

    Attributes
    ----------
    D_  : np.ndarray  full combined dictionary after all steps
    W_  : np.ndarray  current linear classifier
    class_labels_ : list[int]  class labels added via add_class, in order
    class_boundaries_ : dict[int, tuple[int, int]]
        Per-class atom ranges in the full combined D_ (includes the base
        stage's class label too).
    stage_learners_ : list
        Fitted learner (base stage) or FrozenDictionaryLearner (each
        add_class stage) from each stage, in order (index 0 = base stage).
    errors_ : dict[int, list[float]]
        Per-stage training RMSE curves keyed by class_label
        (key -1 for the base stage, regardless of its class_label).
    """

    def __init__(
        self,
        base_learner_class: type[BaseDictionaryLearner],
        base_learner_kwargs: dict,
        residual_learner_class: type[BaseDictionaryLearner],
        residual_learner_kwargs: dict,
        n_nonzero_coefs: int,
        refit_classifier: bool = True,
        freeze_classifier: bool = False,
    ) -> None:
        if refit_classifier == freeze_classifier:
            raise ValueError(
                "Exactly one of refit_classifier / freeze_classifier must be True."
            )
        self.base_learner_class = base_learner_class
        self.base_learner_kwargs = base_learner_kwargs
        self.residual_learner_class = residual_learner_class
        self.residual_learner_kwargs = residual_learner_kwargs
        self.n_nonzero_coefs = n_nonzero_coefs
        self.refit_classifier = refit_classifier
        self.freeze_classifier = freeze_classifier

        # State built incrementally
        self.D_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.base_class_label_: int | None = None
        self.class_labels_: list[int] = []
        self.class_boundaries_: dict[int, tuple[int, int]] = {}
        self.stage_learners_: list = []
        self.errors_: dict[int, list[float]] = {}

        # Internal: accumulated X across all classes for W refit
        self._X_all: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_base(
        self,
        X: np.ndarray,
        H: np.ndarray,
        class_label: int = 0,
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> "IncrementalFrozenDictionary":
        """
        Learn the base dictionary from normal / background data.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        H : np.ndarray, shape (n_classes, n_samples)
            One-hot labels for the base class(es). Used only to fit the
            top-level classifier W — never passed to the (unsupervised)
            base learner itself.
        class_label : int
            The class label this base dictionary's atoms are assigned to
            in ``class_boundaries_`` (default 0). Must be distinct from
            every ``class_label`` later passed to ``add_class``.
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

        if X.shape[1] < self.n_components:
            raise DictionaryLearningError(
                f"n_samples={X.shape[1]} is less than n_components={self.n_components}. "
            )

        learner = self.base_learner_class(**self.base_learner_kwargs)
        fit_kwargs = {}
        if checkpoint_dir is not None:
            fit_kwargs["checkpoint_dir"] = os.path.join(checkpoint_dir, "base")
            fit_kwargs["resume"] = resume
        # Unsupervised: no H, no D_frozen (nothing to freeze yet).
        learner.fit(X, **fit_kwargs)

        self.D_ = learner.D_
        self.base_class_label_ = class_label
        self.class_boundaries_ = {class_label: (0, learner.D_.shape[1])}
        self.stage_learners_.append(learner)
        self.errors_[-1] = list(getattr(learner, "errors_", []))

        # Initialise W over the base dictionary alone.
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        Gamma = coder.encode(X, self.D_)
        self.W_ = _fit_classifier(Gamma, H)

        if self.refit_classifier or self.freeze_classifier:
            self._X_all.append(X)

        # self.D_ already holds its own reference to the learned array;
        # dropping the learner's copies frees nothing-else-reads-them
        # state rather than holding it for the lifetime of the pipeline.
        # get_stage_dict(0) slices self.D_ via class_boundaries_, not
        # this attribute, so it stays correct after this.
        learner.D_ = None
        if hasattr(learner, "Gamma_"):
            learner.Gamma_ = None

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
            Training signals for this class only. Presented unsupervised
            to the residual learner — it never sees labels.
        H : np.ndarray, shape (n_classes_so_far + 1, n_samples_so_far)
            One-hot label matrix for *all* classes seen so far including
            the new one, covering every column of every X passed to
            ``fit_base``/``add_class`` so far (in that order). Used only
            to refit W over the full combined dictionary.
        class_label : int
            Integer label for this class.  Must not have been added
            before, and must differ from the base stage's class_label.
        learner_kwargs_override : dict or None
            If supplied, overrides ``residual_learner_kwargs`` for this
            call only.  Useful for giving this class a different number
            of atoms (e.g. ``{"n_components": 5}``).
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
        if class_label in self.class_labels_ or class_label == self.base_class_label_:
            raise ValueError(
                f"class_label {class_label} has already been used."
            )

        X = np.asarray(X, dtype=float)
        H = np.asarray(H, dtype=float)

        kwargs = {**self.residual_learner_kwargs, **(learner_kwargs_override or {})}

        frozen_step = FrozenDictionaryLearner(
            D_frozen=self.D_,
            learner_class=self.residual_learner_class,
            learner_kwargs=kwargs,
            n_nonzero_coefs=self.n_nonzero_coefs,
            refit_classifier=False,  # we handle W ourselves below
        )
        stage_checkpoint_dir = (
            os.path.join(checkpoint_dir, f"class_{class_label}")
            if checkpoint_dir is not None
            else None
        )
        frozen_step.fit(
            X,
            H=None,
            class_label=class_label,
            frozen_class_boundaries=dict(self.class_boundaries_),
            checkpoint_dir=stage_checkpoint_dir,
            resume=resume,
        )

        n_active = frozen_step.n_active_
        self.D_ = frozen_step.D_combined_
        self.class_boundaries_ = dict(frozen_step.class_boundaries_)
        self.class_labels_.append(class_label)
        self.stage_learners_.append(frozen_step)
        self.errors_[class_label] = list(
            getattr(frozen_step.learner_, "errors_", [])
        )

        # self.D_ already holds its own reference to this array — free the
        # now-superseded snapshots nothing downstream ever reads again
        # (get_stage_dict/get_class_dict always slice the top-level
        # self.D_, never these).
        frozen_step.D_frozen = None
        frozen_step.D_combined_ = None
        frozen_step.learner_.D_ = None
        if hasattr(frozen_step.learner_, "Gamma_"):
            frozen_step.learner_.Gamma_ = None

        # Accumulate training data for W refit
        if self.refit_classifier or self.freeze_classifier:
            self._X_all.append(X)

        # --- Refit W over all data and the full combined dict ---
        if self.refit_classifier:
            X_all = np.hstack(self._X_all)
            # H must cover all columns of X_all — caller is responsible
            # for passing the full H including all previous classes
            self._refit_W(X_all, H)
        else:
            self._extend_W_frozen(n_active, H)

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
        residual dictionary (the atoms added by the k-th ``add_class``
        call). Both are recovered via ``class_boundaries_`` on the final,
        accumulated ``D_`` — no per-stage dictionary snapshot is kept
        alive for this (see the memory-efficiency notes in ``fit_base`` /
        ``add_class``).

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
        if stage == 0:
            return self.get_class_dict(self.base_class_label_)
        class_label = self.class_labels_[stage - 1]
        return self.get_class_dict(class_label)

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