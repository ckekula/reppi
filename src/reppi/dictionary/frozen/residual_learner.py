from __future__ import annotations

import numpy as np

from reppi.base import BaseDiscriminativeDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp.omp import OMP

from reppi.dictionary.frozen.utils import _encode_residual, _fit_classifier

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
            ``class_boundaries_`` from earlier frozen stages, used to build
            the merged ``class_boundaries_`` on the combined dictionary.
            Pass None if D_frozen has no class structure (e.g. base stage).
        checkpoint_dir : str or None
            If given, forwarded to the inner ``learner_class.fit()`` call
            (e.g. KSVD or LCKSVD) so that training of the *active* residual
            dictionary for this single stage can be checkpointed/resumed.
            Only meaningful if ``learner_class.fit`` accepts a
            ``checkpoint_dir`` argument; callers using a learner that
            doesn't support checkpointing should leave this as None.
            Each ``FrozenDictionaryLearner.fit()`` call represents exactly
            one stage, so this directory should be unique per stage (the
            caller — typically ``IncrementalFrozenDictionary`` — is
            responsible for handing this class a stage-specific
            subdirectory, not a shared one across stages).
        resume : bool
            Forwarded to the inner learner's ``fit()`` alongside
            ``checkpoint_dir``. Default True.

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
        fit_kwargs = {}
        if checkpoint_dir is not None:
            fit_kwargs["checkpoint_dir"] = checkpoint_dir
            fit_kwargs["resume"] = resume
        learner.fit(X_train, H, **fit_kwargs)
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