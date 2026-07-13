from __future__ import annotations

import numpy as np

from reppi.base import BaseDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp.omp import OMP

from reppi.dictionary.frozen.utils import _completed_ksvd_checkpoint, _fit_classifier

class FrozenDictionaryLearner:
    """
    Learn a residual dictionary given a frozen dictionary D_frozen, using
    any unsupervised ``BaseDictionaryLearner`` that supports the frozen-
    dictionary contract (``D_frozen`` / ``checkpoint_dir`` / ``resume`` on
    ``fit()``, and a ``D_`` attribute afterward) — e.g. ``KSVD``. This
    directly implements "Frozen K-SVD" (Carroll et al. 2017, Sec. III-A):
    the wrapped learner jointly sparse-codes over ``[D_frozen | D_active]``
    every iteration and only ever updates the new atoms.

    The wrapped learner is unsupervised — it never sees class labels.
    Per-class atom bookkeeping (``class_boundaries_``) is therefore
    computed directly by this wrapper from the given ``class_label`` and
    the number of new atoms the learner produced, rather than being
    reported by the learner itself (contrast with the earlier
    discriminative-learner design, where each learner had to infer its
    own per-class atom split from H — unnecessary here, since exactly one
    class's data is ever presented per residual-learning step).

    For the full sequential pipeline, see ``IncrementalFrozenDictionary``.

    Parameters
    ----------
    D_frozen : np.ndarray, shape (n_features, n_frozen_atoms)
        Pre-trained frozen dictionary.  Never modified.
    learner_class : type[BaseDictionaryLearner]
        Unsupervised dictionary learning class to use for the residual,
        e.g. ``KSVD``. Must accept ``D_frozen`` in its ``fit()`` and
        expose ``D_`` afterward (see the base class's frozen-dictionary
        contract). Its own ``n_components`` (or equivalent) should be the
        count of *new* atoms only — D_frozen is additional, not counted
        there.
    learner_kwargs : dict
        Init kwargs for ``learner_class``.
    n_nonzero_coefs : int
        Sparsity level used when encoding over the combined dictionary,
        if ``refit_classifier`` is True.
    refit_classifier : bool
        If True, fit W via least squares over the full combined
        dictionary using the given H. If False (default here, and what
        ``IncrementalFrozenDictionary`` always passes — it manages the
        classifier itself, over all accumulated classes, once per stage),
        ``W_`` is left as None.

    Attributes
    ----------
    D_combined_ : np.ndarray, shape (n_features, n_frozen + n_active)
    W_          : np.ndarray or None, shape (n_classes, n_frozen + n_active)
    learner_    : fitted instance of ``learner_class``
    n_frozen_   : int  number of frozen atoms
    n_active_   : int  number of active (residual) atoms
    class_boundaries_ : dict[int, tuple[int, int]]
        ``frozen_class_boundaries`` merged with a single new entry
        ``{class_label: (n_frozen, n_frozen + n_active)}``.
    """

    def __init__(
        self,
        D_frozen: np.ndarray,
        learner_class: type[BaseDictionaryLearner],
        learner_kwargs: dict,
        n_nonzero_coefs: int,
        refit_classifier: bool = False,
    ) -> None:
        self.D_frozen = np.asarray(D_frozen, dtype=float)
        self.learner_class = learner_class
        self.learner_kwargs = learner_kwargs
        self.n_nonzero_coefs = n_nonzero_coefs
        self.refit_classifier = refit_classifier

        self.D_combined_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.learner_: BaseDictionaryLearner | None = None
        self.n_frozen_: int = self.D_frozen.shape[1]
        self.n_active_: int = 0
        self.class_boundaries_: dict[int, tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        H: np.ndarray | None,
        class_label: int,
        frozen_class_boundaries: dict[int, tuple[int, int]] | None = None,
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> "FrozenDictionaryLearner":
        """
        Fit the residual dictionary on X.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
            Training signals for this class. Presented unsupervised to
            the wrapped learner — it never sees labels.
        H : np.ndarray or None, shape (n_classes, n_samples)
            One-hot labels. Only used (and required) if
            ``refit_classifier=True``, to fit ``W_`` over the combined
            dictionary. Ignored otherwise — pass None if not needed
            (this is what ``IncrementalFrozenDictionary`` does, since it
            manages the classifier itself).
        class_label : int
            The class these new atoms are assigned to in
            ``class_boundaries_``.
        frozen_class_boundaries : dict or None
            ``class_boundaries_`` from earlier frozen stages, merged
            unchanged into this call's own single new entry.
        checkpoint_dir : str or None
            Forwarded to the inner ``learner_class.fit()`` call, if
            supported.
        resume : bool
            Forwarded to the inner learner's ``fit()`` alongside
            ``checkpoint_dir``. Default True.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)

        learner = self.learner_class(**self.learner_kwargs)

        loaded = (
            _completed_ksvd_checkpoint(
                checkpoint_dir,
                X,
                n_components=self.learner_kwargs.get("n_components"),
                n_nonzero_coefs=self.learner_kwargs.get("n_nonzero_coefs"),
                n_iter=self.learner_kwargs.get("n_iter"),
                n_frozen=self.n_frozen_,
                D_frozen=self.D_frozen,
            )
            if resume
            else None
        )
        if loaded is not None:
            learner.D_, learner.Gamma_, learner.errors_ = loaded
        else:
            learner.fit(
                X,
                D_frozen=self.D_frozen,
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

        boundaries = dict(frozen_class_boundaries) if frozen_class_boundaries else {}
        if class_label in boundaries:
            raise ValueError(
                f"class_label {class_label} already present in "
                "frozen_class_boundaries."
            )
        boundaries[class_label] = (self.n_frozen_, self.n_frozen_ + self.n_active_)
        self.class_boundaries_ = boundaries

        if self.refit_classifier:
            if H is None:
                raise DictionaryLearningError(
                    "refit_classifier=True requires H to be given."
                )
            H = np.asarray(H, dtype=float)
            coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
            Gamma_full = coder.encode(X, self.D_combined_)
            self.W_ = _fit_classifier(Gamma_full, H)
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
        if self.W_ is None:
            raise DictionaryLearningError(
                "No classifier is available (refit_classifier=False)."
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