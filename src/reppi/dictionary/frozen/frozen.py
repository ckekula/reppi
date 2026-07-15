from __future__ import annotations

import os
import numpy as np
import logging

from reppi.base import BaseDictionaryLearner
from reppi.exceptions import DictionaryLearningError

from reppi.dictionary.frozen.utils import _completed_ksvd_checkpoint
from reppi.dictionary.frozen.residual_learner import FrozenDictionaryLearner

logger = logging.getLogger(__name__)


class IncrementalFrozenDictionary:
    """
    Incrementally learn class-specific residual dictionaries, freezing all
    previously learned atoms before training the next class.

    Faithful to Carroll et al. 2017 ("Outlier Learning via Augmented
    Frozen Dictionaries"): the underlying dictionary learner
    (``base_learner_class`` / ``residual_learner_class``, e.g. ``KSVD``)
    is unsupervised — it never sees class labels, only the training
    signals for whichever single class is currently being added. This
    class handles only that unsupervised, incremental dictionary-learning
    pipeline and its per-class bookkeeping (``class_boundaries_``).

    Classification is intentionally out of scope here, matching the
    paper: dictionary learning and classification are two fully separate
    stages there (Sec. IV-B), with an SVM trained afterwards on sparse
    codes from the *finished, frozen* dictionary.

    Pipeline
    --------
    1. ``fit_base(X)``
       Learn a base dictionary D_n from normal/background data using
       ``base_learner_class``.  This dictionary is frozen for all
       subsequent steps.

    2. ``add_class(X, class_label)``
       Learn a residual dictionary D_a for the new class on top of
       the currently frozen dictionary [ D_n | D_a_1 | … ], via
       ``FrozenDictionaryLearner``. The underlying learner trains its new
       atoms jointly alongside the frozen ones every iteration (not on a
       one-shot residual — see ``BaseDictionaryLearner``'s frozen-
       dictionary contract and Carroll et al. 2017, Sec. III).

    The end product of this pipeline is ``D_``, the full combined
    dictionary. Sparse coding and classification on top of it happen
    outside this class.

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
        Sparsity level passed down to ``FrozenDictionaryLearner`` during
        ``add_class``. Also the natural default sparsity for any sparse
        coding you do yourself downstream against ``D_``, though this
        class does not perform that coding.

    Attributes
    ----------
    D_  : np.ndarray  full combined dictionary after all steps
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
    ) -> None:
        self.base_learner_class = base_learner_class
        self.base_learner_kwargs = base_learner_kwargs
        self.residual_learner_class = residual_learner_class
        self.residual_learner_kwargs = residual_learner_kwargs
        self.n_nonzero_coefs = n_nonzero_coefs

        # State built incrementally
        self.D_: np.ndarray | None = None
        self.base_class_label_: int | None = None
        self.class_labels_: list[int] = []
        self.class_boundaries_: dict[int, tuple[int, int]] = {}
        self.stage_learners_: list = []
        self.errors_: dict[int, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_base(
        self,
        X: np.ndarray,
        class_label: int = 0,
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> "IncrementalFrozenDictionary":
        """
        Learn the base dictionary from normal / background data.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
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

        learner = self.base_learner_class(**self.base_learner_kwargs)
        logger.info("Fitting base dictionary with %d samples and %d features using %s", X.shape[1], X.shape[0], self.base_learner_class.__name__)

        fit_kwargs = {}
        base_checkpoint_dir = None
        if checkpoint_dir is not None:
            base_checkpoint_dir = os.path.join(checkpoint_dir, "base")
            fit_kwargs["checkpoint_dir"] = base_checkpoint_dir
            fit_kwargs["resume"] = resume

        loaded = (
            _completed_ksvd_checkpoint(
                base_checkpoint_dir,
                X,
                n_components=self.base_learner_kwargs.get("n_components"),
                n_nonzero_coefs=self.base_learner_kwargs.get("n_nonzero_coefs"),
                n_iter=self.base_learner_kwargs.get("n_iter"),
                n_frozen=0,
                D_frozen=None,
            )
            if resume
            else None
        )
        if loaded is not None:
            learner.D_, learner.Gamma_, learner.errors_ = loaded
        else:
            # Unsupervised: no D_frozen (nothing to freeze yet).
            learner.fit(X, **fit_kwargs)

        self.D_ = learner.D_
        self.base_class_label_ = class_label
        self.class_boundaries_ = {class_label: (0, learner.D_.shape[1])}
        self.stage_learners_.append(learner)
        self.errors_[-1] = list(getattr(learner, "errors_", []))

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

        kwargs = {**self.residual_learner_kwargs, **(learner_kwargs_override or {})}
        logger.info("Adding class %d with %d samples and %d features using %s", class_label, X.shape[1], X.shape[0], self.residual_learner_class.__name__)

        frozen_step = FrozenDictionaryLearner(
            D_frozen=self.D_,
            learner_class=self.residual_learner_class,
            learner_kwargs=kwargs,
            n_nonzero_coefs=self.n_nonzero_coefs,
        )
        stage_checkpoint_dir = (
            os.path.join(checkpoint_dir, f"class_{class_label}")
            if checkpoint_dir is not None
            else None
        )
        frozen_step.fit(
            X,
            class_label=class_label,
            frozen_class_boundaries=dict(self.class_boundaries_),
            checkpoint_dir=stage_checkpoint_dir,
            resume=resume,
        )

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

        return self
