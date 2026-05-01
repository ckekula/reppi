"""
Base classes for representation learning algorithms.
"""

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