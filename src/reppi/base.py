"""
Base classes for representation learning algorithms.
"""

from abc import ABC, abstractmethod
from reppi.backend import xp

class BaseSparseCoder(ABC):
    """Abstract base class for sparse coding algorithms."""

    @abstractmethod
    def encode(self, X: xp.ndarray, D: xp.ndarray) -> xp.ndarray:
        """
        Compute sparse codes for signals X given dictionary D.

        Parameters
        ----------
        X : xp.ndarray, shape (n_features, n_samples)
            Input signals as columns.
        D : xp.ndarray, shape (n_features, n_atoms)
            Dictionary with (approximately) unit-norm columns.

        Returns
        -------
        Gamma : xp.ndarray, shape (n_atoms, n_samples)
            Sparse representation matrix.
        """
        raise NotImplementedError


class BaseDictionaryLearner(ABC):
    """Abstract base class for dictionary learning algorithms."""

    @abstractmethod
    def fit(self, X: xp.ndarray) -> "BaseDictionaryLearner":
        """
        Learn a dictionary from training data.

        Parameters
        ----------
        X : xp.ndarray, shape (n_features, n_samples)
            Training signals as columns.

        Returns
        -------
        self
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: xp.ndarray) -> xp.ndarray:
        """
        Encode signals using the learned dictionary.

        Parameters
        ----------
        X : xp.ndarray, shape (n_features, n_samples)
            Signals to encode.

        Returns
        -------
        Gamma : xp.ndarray, shape (n_atoms, n_samples)
            Sparse representations.
        """
        raise NotImplementedError

    def fit_transform(self, X: xp.ndarray) -> xp.ndarray:
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

    Sub-dictionary contract
    -----------------------
    Each learner internally partitions the dictionary D into per-class blocks.
    The mapping is recorded in ``class_boundaries_``, a dict of the form::

        {class_idx: (col_start, col_end)}   # half-open slice [start, end)

    so that ``D_[:, start:end]`` yields the sub-dictionary for that class.
    ``get_class_dict(c)`` is a convenience wrapper around this mapping.

    Required attributes after ``fit``
    ----------------------------------
    D_  : xp.ndarray, shape (n_features, n_components)
    W_  : xp.ndarray, shape (n_classes, n_components)  — linear classifier
    class_boundaries_ : dict[int, tuple[int, int]]
    """

    # ------------------------------------------------------------------
    # Abstract interface every discriminative learner must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(                                        # type: ignore[override]
        self,
        X: xp.ndarray,
        H: xp.ndarray,
    ) -> "BaseDiscriminativeDictionaryLearner":
        """
        Learn a discriminative dictionary from labelled training data.

        Parameters
        ----------
        X : xp.ndarray, shape (n_features, n_samples)
        H : xp.ndarray, shape (n_classes, n_samples)
            One-hot label matrix.

        Returns
        -------
        self
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: xp.ndarray) -> xp.ndarray:
        """
        Predict class indices for each column of X.

        Parameters
        ----------
        X : xp.ndarray, shape (n_features, n_samples)

        Returns
        -------
        labels : xp.ndarray, shape (n_samples,)  integer class indices
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, X: xp.ndarray, H: xp.ndarray) -> float:
        """
        Classification accuracy on (X, H).

        Parameters
        ----------
        X : xp.ndarray, shape (n_features, n_samples)
        H : xp.ndarray, shape (n_classes, n_samples) — one-hot labels

        Returns
        -------
        accuracy : float in [0, 1]
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Per-class sub-dictionary API (used by the frozen framework)
    # ------------------------------------------------------------------

    def get_class_dict(self, class_idx: int) -> xp.ndarray:
        """
        Return the sub-dictionary for ``class_idx``.

        Uses ``class_boundaries_`` set during ``fit``.

        Parameters
        ----------
        class_idx : int

        Returns
        -------
        D_c : xp.ndarray, shape (n_features, n_atoms_for_class)
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