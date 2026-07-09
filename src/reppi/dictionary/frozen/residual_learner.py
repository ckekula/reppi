from __future__ import annotations

import numpy as np

from reppi.base import BaseDiscriminativeDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp.omp import OMP

from reppi.dictionary.frozen.utils import _fit_classifier

class FrozenDictionaryLearner:
    """
    Learn a residual dictionary given a frozen dictionary D_frozen, by
    delegating directly to a discriminative learner that natively supports
    frozen atoms (see ``BaseDiscriminativeDictionaryLearner``'s frozen-
    dictionary contract).

    This class handles a single residual learning step: it instantiates
    ``learner_class``, calls its ``fit(X, H, D_frozen=..., ...)``, and
    exposes the resulting full combined dictionary as ``D_combined_``.
    Because the wrapped learner is itself responsible for jointly encoding
    over ``[D_frozen | D_active]`` at every iteration and for returning the
    full combined dictionary and merged ``class_boundaries_``, this wrapper
    does no dictionary concatenation or boundary-merging of its own — it
    only optionally refits a classifier over the combined result.

    For the full sequential pipeline, see ``IncrementalFrozenDictionary``.

    Parameters
    ----------
    D_frozen : np.ndarray, shape (n_features, n_frozen_atoms)
        Pre-trained frozen dictionary.  Never modified.
    learner_class : type[BaseDiscriminativeDictionaryLearner]
        Discriminative dictionary learning class to use for the residual.
        Must accept ``D_frozen`` / ``frozen_class_boundaries`` in its
        ``fit()`` (see the base class's frozen-dictionary contract).
    learner_kwargs : dict
        Keyword arguments forwarded to ``learner_class.__init__``. Its
        ``n_components`` (or equivalent) should be the count of *new*
        atoms only — D_frozen is additional, not counted there.
    n_nonzero_coefs : int
        Sparsity level used when encoding over the combined dictionary
        for downstream tasks and, if ``refit_classifier`` is True, for
        refitting W.
    refit_classifier : bool
        If True (default), re-learn W over the full combined dictionary
        after fitting. If False, use whatever ``W_`` the wrapped learner
        itself produced (may be None for learners without a classifier
        term, e.g. LC-KSVD1).

    Attributes
    ----------
    D_combined_ : np.ndarray, shape (n_features, n_frozen + n_active)
    W_          : np.ndarray, shape (n_classes, n_frozen + n_active)
    learner_    : fitted instance of ``learner_class``
    n_frozen_   : int  number of frozen atoms
    n_active_   : int  number of active (residual) atoms
    class_boundaries_ : dict[int, tuple[int, int]]
        Atom ranges for each class in the *combined* dictionary, as
        reported by the wrapped learner.
    """

    def __init__(
        self,
        D_frozen: np.ndarray,
        learner_class: type[BaseDiscriminativeDictionaryLearner],
        learner_kwargs: dict,
        n_nonzero_coefs: int,
        refit_classifier: bool = True,
    ) -> None:
        self.D_frozen = np.asarray(D_frozen, dtype=float)
        self.learner_class = learner_class
        self.learner_kwargs = learner_kwargs
        self.n_nonzero_coefs = n_nonzero_coefs
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
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> "FrozenDictionaryLearner":
        """
        Fit the residual dictionary on (X, H).

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        H : np.ndarray, shape (n_classes, n_samples)  one-hot labels
        frozen_class_boundaries : dict or None
            ``class_boundaries_`` from earlier frozen stages, forwarded
            to the wrapped learner so it can merge them into its own
            ``class_boundaries_``. Pass None if D_frozen has no class
            structure (e.g. an unsupervised base stage).
        checkpoint_dir : str or None
            If given, forwarded to the inner ``learner_class.fit()`` call
            so that training of this single stage can be
            checkpointed/resumed. Only meaningful if ``learner_class.fit``
            supports a ``checkpoint_dir`` argument; callers using a
            learner that doesn't should leave this as None.
        resume : bool
            Forwarded to the inner learner's ``fit()`` alongside
            ``checkpoint_dir``. Default True.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        H = np.asarray(H, dtype=float)

        learner = self.learner_class(**self.learner_kwargs)
        learner.fit(
            X,
            H,
            D_frozen=self.D_frozen,
            frozen_class_boundaries=frozen_class_boundaries,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
        )
        self.learner_ = learner

        if learner.D_.shape[1] <= self.n_frozen_:
            raise DictionaryLearningError(
                f"{self.learner_class.__name__}.fit() returned a combined "
                f"dictionary with {learner.D_.shape[1]} columns, which is "
                f"not larger than n_frozen={self.n_frozen_}. Learners must "
                "return the full [D_frozen | D_new] dictionary as D_, per "
                "the frozen-dictionary contract."
            )
        if not np.allclose(learner.D_[:, : self.n_frozen_], self.D_frozen, atol=1e-6):
            raise DictionaryLearningError(
                f"{self.learner_class.__name__}.fit() did not preserve "
                "D_frozen unchanged in its leading columns of D_."
            )

        self.D_combined_ = learner.D_
        self.n_active_ = self.D_combined_.shape[1] - self.n_frozen_
        self.class_boundaries_ = dict(learner.class_boundaries_ or {})

        if self.refit_classifier:
            coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
            Gamma_full = coder.encode(X, self.D_combined_)
            self.W_ = _fit_classifier(Gamma_full, H)
        else:
            self.W_ = getattr(learner, "W_", None)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode X over the combined dictionary."""
        self._check_fitted()
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, self.D_combined_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify X using W_ and the combined dictionary."""
        self._check_fitted()
        if self.W_ is None:
            raise DictionaryLearningError(
                "No classifier is available (refit_classifier=False and "
                "the wrapped learner did not produce a W_)."
            )
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