from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSparseCoder(ABC):
    """Abstract base class for sparse coding algorithms."""

    @abstractmethod
    def encode(self, X: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Compute sparse codes for signals X given dictionary D.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
            Input signals as columns.
        D : np.ndarray, shape (n_features, n_atoms)
            Dictionary with (approximately) unit-norm columns.

        Returns
        -------
        Gamma : np.ndarray, shape (n_atoms, n_samples)
            Sparse representation matrix.
        """
        raise NotImplementedError


class BaseDictionaryLearner(ABC):
    """Abstract base class for dictionary learning algorithms."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseDictionaryLearner":
        """
        Learn a dictionary from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
            Training signals as columns.

        Returns
        -------
        self
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode signals using the learned dictionary.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
            Signals to encode.

        Returns
        -------
        Gamma : np.ndarray, shape (n_atoms, n_samples)
            Sparse representations.
        """
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and return sparse codes on the training data."""
        return self.fit(X).transform(X)


class BaseDiscriminativeDictionaryLearner(BaseDictionaryLearner):
    """
    Abstract base class for discriminative dictionary learning algorithms.

    Extends ``BaseDictionaryLearner`` with the labelled-fit interface and the
    per-class sub-dictionary API required by the frozen dictionary framework.

    All discriminative learners (LC-KSVD, FDDL, LEDL, …) must subclass this
    and implement every abstract method.  The frozen/incremental wrappers are
    typed against this base, so any conforming implementation can be plugged
    in without modification.

    Frozen-dictionary contract
    --------------------------
    Subclasses must accept an optional ``D_frozen`` in ``fit()`` and honour
    it exactly, since this is what lets ``FrozenDictionaryLearner`` /
    ``IncrementalFrozenDictionary`` compose learners without knowing their
    internals (see Carroll et al. 2017, "Frozen K-SVD" / "Frozen Alternating
    Minimization", Sec. III, for the reference algorithm being generalised
    here to arbitrary discriminative learners):

    - Training signals must be sparse-coded *jointly* over the full
      ``[D_frozen | D_new]`` dictionary at every iteration — never encode
      against ``D_frozen`` alone and train ``D_new`` on a one-shot residual;
      the two must be able to co-adapt every pass.
    - ``D_frozen``'s columns must never be modified, not even at
      initialisation (e.g. via a blanket re-normalisation of the whole
      dictionary). They must be bit-for-bit identical in ``self.D_`` after
      ``fit()`` to the ``D_frozen`` that was passed in.
    - ``self.D_`` after ``fit()`` must be the **full combined** dictionary
      (frozen columns followed by newly-learned columns), not just the new
      atoms — downstream wrappers rely on this to avoid re-deriving the
      combined dictionary themselves.
    - ``self.class_boundaries_`` after ``fit()`` must include the
      (unchanged) entries from ``frozen_class_boundaries`` plus boundaries
      for any newly-learned classes, with the new entries' column ranges
      offset by ``D_frozen.shape[1]``.

    Sub-dictionary contract
    -----------------------
    Each learner internally partitions the dictionary D into per-class blocks.
    The mapping is recorded in ``class_boundaries_``, a dict of the form::

        {class_idx: (col_start, col_end)}   # half-open slice [start, end)

    so that ``D_[:, start:end]`` yields the sub-dictionary for that class.
    ``get_class_dict(c)`` is a convenience wrapper around this mapping.

    Required attributes after ``fit``
    ----------------------------------
    D_  : np.ndarray, shape (n_features, n_components)
    W_  : np.ndarray, shape (n_classes, n_components)  — linear classifier
    class_boundaries_ : dict[int, tuple[int, int]]
    """

    # ------------------------------------------------------------------
    # Abstract interface every discriminative learner must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(                                        # type: ignore[override]
        self,
        X: np.ndarray,
        H: np.ndarray,
        D_frozen: np.ndarray | None = None,
        frozen_class_boundaries: dict[int, tuple[int, int]] | None = None,
    ) -> "BaseDiscriminativeDictionaryLearner":
        """
        Learn a discriminative dictionary from labelled training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        H : np.ndarray, shape (n_classes, n_samples)
            One-hot label matrix.
        D_frozen : np.ndarray or None, shape (n_features, n_frozen_atoms)
            Pre-trained atoms from earlier incremental stages, held
            constant throughout fitting. See the class docstring's
            "Frozen-dictionary contract" for the requirements this
            implies. Pass None (default) for ordinary, non-incremental
            training — the frozen contract then has no effect.
        frozen_class_boundaries : dict or None
            ``class_boundaries_`` from earlier frozen stages. Must be
            merged unchanged into ``self.class_boundaries_``, alongside
            this call's own class boundaries offset by
            ``D_frozen.shape[1]``. Ignored if ``D_frozen`` is None.

        Returns
        -------
        self
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class indices for each column of X.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)  integer class indices
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, X: np.ndarray, H: np.ndarray) -> float:
        """
        Classification accuracy on (X, H).

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        H : np.ndarray, shape (n_classes, n_samples) — one-hot labels

        Returns
        -------
        accuracy : float in [0, 1]
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Per-class sub-dictionary API (used by the frozen framework)
    # ------------------------------------------------------------------

    def get_class_dict(self, class_idx: int) -> np.ndarray:
        """
        Return the sub-dictionary for ``class_idx``.

        Uses ``class_boundaries_`` set during ``fit``.

        Parameters
        ----------
        class_idx : int

        Returns
        -------
        D_c : np.ndarray, shape (n_features, n_atoms_for_class)
        """
        self._check_fitted()
        if not hasattr(self, "class_boundaries_") or self.class_boundaries_ is None:
            raise AttributeError(
                f"{type(self).__name__} did not set class_boundaries_ during fit. "
                "Ensure the subclass records per-class atom ranges."
            )
        if class_idx not in self.class_boundaries_:
            raise KeyError(f"class_idx {class_idx} not found in class_boundaries_.")
        start, end = self.class_boundaries_[class_idx]
        return self.D_[:, start:end]

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted yet."""
        if not hasattr(self, "D_") or self.D_ is None:
            raise AttributeError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call fit() before using this method."
            )