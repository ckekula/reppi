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
    """
    Abstract base class for (unsupervised) dictionary learning algorithms.

    Frozen-dictionary contract
    --------------------------
    Implementations that are usable as ``base_learner_class`` /
    ``residual_learner_class`` in ``IncrementalFrozenDictionary`` (e.g.
    ``KSVD``) must accept an optional ``D_frozen`` in ``fit()`` and honour
    it exactly. This directly implements "Frozen K-SVD" / "Frozen
    Alternating Minimization" as described in Carroll et al. 2017,
    "Outlier Learning via Augmented Frozen Dictionaries", Sec. III —
    freezing is a property of the dictionary-learning algorithm itself,
    independent of any classifier or label information:

    - Training signals must be sparse-coded *jointly* over the full
      ``[D_frozen | D_new]`` dictionary at every iteration — never encode
      against ``D_frozen`` alone and train ``D_new`` on a one-shot
      residual; the two must be able to co-adapt every pass.
    - ``D_frozen``'s columns must never be modified, not even at
      initialisation (e.g. via a blanket re-normalisation of the whole
      dictionary). They must be bit-for-bit identical in ``self.D_``
      after ``fit()`` to the ``D_frozen`` that was passed in.
    - ``self.D_`` after ``fit()`` must be the **full combined** dictionary
      (frozen columns followed by newly-learned columns), not just the
      new atoms — ``FrozenDictionaryLearner`` relies on this to avoid
      re-deriving the combined dictionary itself.

    Implementations that will never be used in a frozen/incremental
    context (e.g. a one-shot, standalone dictionary learner) may omit
    ``D_frozen`` support — it is optional at the Python level (default
    None) precisely so non-frozen usage is unaffected.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        D_frozen: np.ndarray | None = None,
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> "BaseDictionaryLearner":
        """
        Learn a dictionary from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
            Training signals as columns.
        D_frozen : np.ndarray or None, shape (n_features, n_frozen_atoms)
            Pre-trained atoms to hold constant throughout fitting. See
            the class docstring's "Frozen-dictionary contract". Pass
            None (default) for ordinary, non-incremental training.
        checkpoint_dir : str or None
            Optional directory to checkpoint/resume training from, if
            the implementation supports it.
        resume : bool
            Whether to resume from an existing checkpoint in
            ``checkpoint_dir``, if one is found.

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
    Abstract base class for discriminative dictionary learning algorithms
    (e.g. LC-KSVD, FDDL, LEDL, …).

    These learners are trained with class labels and expose their own
    classifier (``W_``) and per-class sub-dictionary structure
    (``class_boundaries_``). They are standalone models — this base class
    intentionally does NOT include the frozen-dictionary contract from
    ``BaseDictionaryLearner``: composing a discriminative learner's own
    per-class atom allocation and internal classifier with the frozen/
    incremental framework's *own* per-class bookkeeping led to two
    conflicting sources of truth for "which atoms belong to which class"
    and "what is the classifier." ``IncrementalFrozenDictionary`` /
    ``FrozenDictionaryLearner`` are therefore typed against
    ``BaseDictionaryLearner`` (e.g. ``KSVD``) instead — matching the
    original paper, where "Frozen K-SVD" freezes a plain, unsupervised
    K-SVD dictionary, and classification (an SVM in the paper; a linear
    classifier ``W`` here) is a separate step performed once, on top of
    the finished combined dictionary.

    Sub-dictionary contract
    -----------------------
    Each learner internally partitions the dictionary D into per-class
    blocks, recorded in ``class_boundaries_``, a dict of the form::

        {class_idx: (col_start, col_end)}   # half-open slice [start, end)

    so that ``D_[:, start:end]`` yields the sub-dictionary for that class.
    ``get_class_dict(c)`` is a convenience wrapper around this mapping.

    Required attributes after ``fit``
    ----------------------------------
    D_  : np.ndarray, shape (n_features, n_components)
    W_  : np.ndarray, shape (n_classes, n_components)  — linear classifier
    class_boundaries_ : dict[int, tuple[int, int]]
    """

    @abstractmethod
    def fit(                                        # type: ignore[override]
        self,
        X: np.ndarray,
        H: np.ndarray,
    ) -> "BaseDiscriminativeDictionaryLearner":
        """
        Learn a discriminative dictionary from labelled training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        H : np.ndarray, shape (n_classes, n_samples)
            One-hot label matrix.

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
    # Per-class sub-dictionary API
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