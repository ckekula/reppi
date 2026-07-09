"""
Label Consistent K-SVD (LC-KSVD) dictionary learning.

Implements LC-KSVD1 and LC-KSVD2 as described in:
    Zhuolin Jiang, Zhe Lin, Larry S. Davis.
    "Learning A Discriminative Dictionary for Sparse Coding via Label
     Consistent K-SVD", CVPR 2011.

LC-KSVD augments the standard K-SVD objective with:
  - A label-consistency term (LC-KSVD1) that encourages atoms associated
    with the same class to produce similar sparse codes.
  - An additional linear classifier term (LC-KSVD2) that jointly trains a
    classifier W alongside the dictionary.

Optimization problems
---------------------
LC-KSVD1:
    min_{D, A, X}  ||Y - DX||_F^2 + alpha * ||Q - AX||_F^2
    s.t.  ||x_i||_0 <= T

LC-KSVD2:
    min_{D, A, W, X}  ||Y - DX||_F^2 + alpha * ||Q - AX||_F^2
                      + beta * ||H - WX||_F^2
    s.t.  ||x_i||_0 <= T

where:
  Y = training signals
  D = dictionary
  X = sparse codes
  Q = label-consistent sparse code targets (binary atom-class assignments)
  A = linear mapping for Q-consistency
  W = linear classifier
  H = class label matrix (one-hot per column)
  alpha, beta = trade-off weights (these are sqrt_alpha / sqrt_beta from
      the paper's Eq. (11) — i.e. what you pass here is the value stacked
      directly onto Q and H, NOT the paper's reported alpha/beta. To
      reproduce a paper-reported setting such as alpha=4.0, beta=2.0
      (extended YaleB / AR face, Sec. 6.1-6.2), pass
      alpha=sqrt(4.0)=2.0, beta=sqrt(2.0)=~1.41 here.

Implementation note
--------------------
Both variants reduce to running ordinary K-SVD on an augmented system:

    Y_aug = [Y ; sqrt(alpha)*Q ; sqrt(beta)*H]     (rows stacked)
    D_aug = [D ; sqrt(alpha)*A ; sqrt(beta)*W]

Each outer iteration here delegates to `KSVD` (with n_iter=1) on the
augmented system and then splits the result back into D, A, W. This
mirrors the construction in the original paper and keeps a single
source of truth for the K-SVD atom-update step.

Known limitation (fixed atom-class labels, Sec. 3.3.1 footnote 2)
-------------------------------------------------------------------
The paper assumes each atom's class label is fixed for the entire
training run once initialised. In this implementation the inner `KSVD`
instance still runs its own dead-atom / mutual-incoherence-based atom
replacement heuristics (`_optimize_atom`'s dead-atom fallback and
`_clear_dict`) every outer iteration, on the *augmented* system, without
any notion of class. Those heuristics can therefore replace an atom
with a signal from a different class than the one Q currently assigns
it to, silently violating the paper's fixed-labelling assumption for
that atom (Q is never rebuilt afterward, so its label-consistency target
would then be stale for the rest of training). Reducing `mu_thresh`
increases how often the incoherence-triggered path fires; the
dead-atom fallback triggers regardless of `mu_thresh` whenever an
atom's row in Gamma is entirely zero, and is not user-configurable at
the KSVD level. This is a structural limitation of composing the
generic KSVD atom-update/clearing logic with LC-KSVD's assumption, not
something LC-KSVD's own code can fully prevent without changes to
KSVD's atom-clearing routines.

Note on frozen/incremental dictionary learning
------------------------------------------------
This class is a standalone discriminative dictionary learner and is NOT
used by ``IncrementalFrozenDictionary`` / ``FrozenDictionaryLearner`` —
those are built around plain, unsupervised ``KSVD`` instead, matching the
original "Frozen K-SVD" paper (Carroll et al. 2017) exactly: a fresh,
unsupervised dictionary is learned per class with prior classes' atoms
frozen, and classification is a separate linear-classifier step fit once
over the finished combined dictionary. LC-KSVD's per-call classifier term
has no way to see more than one class's data at a time in that setting
(each residual stage only ever presents one class), so composing it with
frozen dictionary learning doesn't add anything a plain unsupervised
learner doesn't already give more simply — see the class discussion in
``reppi.base.BaseDiscriminativeDictionaryLearner`` for the full reasoning.

Usage
-----
For LC-KSVD1, with the default ridge classifier::

    model = LCKSVD(
        n_components=570,
        n_nonzero_coefs=30,
        alpha=2.0,          # sqrt(4.0), matching the paper's alpha=4.0
        variant="lcksvd1",
    )
    model.fit(X_train, H_train)
    predictions = model.predict(X_test)

For LC-KSVD1 with a custom classifier::

    model = LCKSVD(
        n_components=570,
        n_nonzero_coefs=30,
        alpha=2.0,
        variant="lcksvd1",
        classifier=MyClassifier(),   # any fit(Gamma, H) / predict(Gamma)
    )
    model.fit(X_train, H_train)

For LC-KSVD2 (classifier is trained jointly; `classifier=` not accepted)::

    model = LCKSVD(
        n_components=570,
        n_nonzero_coefs=30,
        alpha=2.0,           # sqrt(4.0)
        beta=1.4142,         # sqrt(2.0), matching the paper's beta=2.0
        variant="lcksvd2",
    )
    model.fit(X_train, H_train)
    predictions = model.predict(X_test)
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from reppi.base import BaseDiscriminativeDictionaryLearner
from reppi.exceptions import DictionaryLearningError
from reppi.sparse.omp.omp import OMP
from reppi.sparse.utils import normalize_columns, rep_error_squared
from reppi.dictionary.ksvd.ksvd import KSVD

from reppi.dictionary.lc_ksvd.utils import initialization4lcksvd, _augment_data
from reppi.dictionary.lc_ksvd.ridge import RidgeClassifier

_CHECKPOINT_FILENAME = "lc_ksvd_checkpoint.npz"


class LCKSVD(BaseDiscriminativeDictionaryLearner):
    """
    Label Consistent K-SVD dictionary learner (LC-KSVD1 and LC-KSVD2).

    Parameters
    ----------
    n_components : int
        Number of dictionary atoms.
    n_nonzero_coefs : int
        Sparsity level T.
    alpha : float
        Weight for the label-consistency term (this is sqrt_alpha in the
        paper's Eq. (11) — see the module docstring for how it relates to
        the paper's reported alpha values).
    beta : float
        Weight for the classifier term (sqrt_beta; LC-KSVD2 only).
    variant : {'lcksvd1', 'lcksvd2'}
        Which variant to train.
    n_iter : int
        Number of LC-KSVD iterations (default 50).
    n_iter_init : int
        K-SVD iterations for the initialisation phase (default 20).
    exact_svd : bool
        Use exact SVD in the atom-update step (slower but slightly better).
    mu_thresh : float
        Mutual-incoherence threshold (default 0.99). See the module
        docstring's "Known limitation" section for how this interacts
        with LC-KSVD's fixed atom-class labelling assumption.
    lambda1 : float
        Ridge weight for the classifier, Eq. (17) (default 1e-5). Used
        both for W^(0) (LC-KSVD2's warm start) and, for LC-KSVD1, as the
        default `RidgeClassifier`'s regularisation unless a `classifier`
        override supplies its own.
    lambda2 : float
        Ridge weight for A^(0), Eq. (16) (default 1e-5).
    classifier : object or None
        Only valid when ``variant="lcksvd1"``. Any object exposing
        ``fit(Gamma, H)`` / ``predict(Gamma)`` (see ``RidgeClassifier``),
        trained once, after the dictionary has converged, on sparse codes
        from the final D — per the paper's Sec. 3.2 ("the classifier W
        for LC-KSVD1 is trained separately ... after D, A, and X are
        computed"). If None, defaults to ``RidgeClassifier(lambda1)``.
        Passing this for ``variant="lcksvd2"`` raises ``ValueError``,
        since LC-KSVD2's classifier is trained jointly with the
        dictionary as part of the augmented system and cannot be
        substituted independently.
    random_state : int or None
    verbose : bool

    Attributes
    ----------
    D_ : np.ndarray, shape (n_features, n_components)
        Learned dictionary.
    W_ : np.ndarray or None, shape (n_classes, n_components)
        Linear classifier weights. For LC-KSVD2, these are learned
        jointly with the dictionary. For LC-KSVD1, these mirror
        ``classifier_.W_`` when the fitted classifier exposes a linear
        ``W_`` attribute (e.g. the default ``RidgeClassifier``); ``None``
        if a custom classifier without a ``W_`` attribute was supplied —
        use ``classifier_.predict()`` / ``predict()`` in that case.
    classifier_ : object or None
        The fitted classifier instance for LC-KSVD1 (``None`` for
        LC-KSVD2, which uses ``W_`` directly).
    A_ : np.ndarray, shape (n_components, n_components)
        Learned label-consistency transform.
    errors_ : list of float
        Per-iteration RMSE on training data (measured on the original,
        un-augmented X).
    """

    def __init__(
        self,
        n_components: int,
        n_nonzero_coefs: int,
        alpha: float = 4.0,
        beta: float = 2.0,
        variant: str = "lcksvd2",
        n_iter: int = 50,
        n_iter_init: int = 20,
        exact_svd: bool = False,
        mu_thresh: float = 0.99,
        lambda1: float = 1e-5,
        lambda2: float = 1e-5,
        classifier: object | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if variant not in ("lcksvd1", "lcksvd2"):
            raise ValueError("variant must be 'lcksvd1' or 'lcksvd2'.")
        if classifier is not None and variant != "lcksvd1":
            raise ValueError(
                "classifier= is only valid for variant='lcksvd1'. "
                "LC-KSVD2's classifier is trained jointly with the "
                "dictionary and cannot be substituted independently."
            )
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.n_iter = n_iter
        self.n_iter_init = n_iter_init
        self.exact_svd = exact_svd
        self.mu_thresh = mu_thresh
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.classifier = classifier
        self.random_state = random_state
        self.verbose = verbose

        self.D_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.A_: np.ndarray | None = None
        self.classifier_: object | None = None
        self.errors_: list[float] = []
        self.class_boundaries_: dict[int, tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        H: np.ndarray,
        D_init: np.ndarray | None = None,
        A_init: np.ndarray | None = None,
        W_init: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> "LCKSVD":
        """
        Learn a discriminative dictionary from labelled training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
            Training signals.
        H : np.ndarray, shape (n_classes, n_samples)
            One-hot label matrix.
        D_init : np.ndarray or None
            Initial dictionary. If None, a K-SVD initialisation is run.
            Ignored when resuming from an existing checkpoint.
        A_init : np.ndarray or None
            Initial label-consistency transform. Ignored when resuming.
        W_init : np.ndarray or None
            Initial classifier weights (used to warm-start LC-KSVD2's
            joint optimisation only; LC-KSVD1 trains its classifier
            separately at the end regardless of this argument). Ignored
            when resuming.
        Q : np.ndarray or None
            Label-consistent target matrix. Computed from H if None.
            Ignored when resuming.
        checkpoint_dir : str or None
            If given, a checkpoint of the *outer* LC-KSVD loop is written
            to ``<checkpoint_dir>/lc_ksvd_checkpoint.npz`` after every
            outer iteration, overwriting the previous one. The directory
            is created if it does not exist. This is independent of, and
            not passed down to, the inner per-iteration KSVD instance.
        resume : bool
            If True (default) and a checkpoint is found in
            ``checkpoint_dir``, training resumes from it. If False, any
            existing checkpoint in ``checkpoint_dir`` is ignored and
            overwritten.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        H = np.asarray(H, dtype=float)
        n_features, n_samples = X.shape
        n_classes = H.shape[0]

        checkpoint_path = None
        start_iter = 0
        D = A = W = Q_loaded = None
        self.errors_ = []

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, _CHECKPOINT_FILENAME)

            if resume and os.path.exists(checkpoint_path):
                (
                    D, A, W, Q_loaded, self.errors_, start_iter,
                ) = self._load_checkpoint(checkpoint_path, X, H)
                if self.verbose:
                    print(
                        f"Resuming from checkpoint at outer iteration "
                        f"{start_iter}/{self.n_iter} ({checkpoint_path})"
                    )

        # ---- Initialisation (skipped if resuming from a checkpoint) ----
        if D is None:
            if D_init is None or A_init is None or W_init is None or Q is None:
                if self.verbose:
                    print("Running initialisation K-SVD...")
                D_init, A_init, W_init, Q = initialization4lcksvd(
                    X, H,
                    self.n_components,
                    self.n_iter_init,
                    self.n_nonzero_coefs,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    lambda1=self.lambda1,
                    lambda2=self.lambda2,
                )

            D = normalize_columns(D_init.copy())
            A = A_init.copy()
            W = W_init.copy()
        else:
            Q = Q_loaded

        sqrt_alpha = self.alpha
        sqrt_beta = self.beta

        use_classifier_term = (self.variant == "lcksvd2")

        # ---- Build augmented training data ----
        # Y_aug = [X ; sqrt_alpha*Q ; sqrt_beta*H]  (LC-KSVD2)
        # Y_aug = [X ; sqrt_alpha*Q]                 (LC-KSVD1)
        H_aug = H if use_classifier_term else None
        X_aug, _, _ = _augment_data(X, Q, H_aug, sqrt_alpha, sqrt_beta)

        # A single K-SVD instance, reused every outer iteration. Only
        # D_init changes between calls (n_iter=1 each time), so the atom
        # update / incoherence-clearing logic lives in exactly one place.
        atom_updater = KSVD(
            n_components=self.n_components,
            n_nonzero_coefs=self.n_nonzero_coefs,
            n_iter=1,
            exact_svd=self.exact_svd,
            mu_thresh=self.mu_thresh,
            mem_usage="normal",
            random_state=self.random_state,
            verbose=False,
        )

        Gamma = None
        for it in range(start_iter, self.n_iter):

            # ---- Build augmented dictionary ----
            # D_aug = [D ; sqrt_alpha*A ; sqrt_beta*W]
            D_aug = self._build_aug_dict(D, A, W, sqrt_alpha, sqrt_beta, use_classifier_term)
            D_aug_norm = normalize_columns(D_aug)

            # ---- Sparse code + single atom-update pass on the augmented
            #      system, delegated to KSVD ----
            atom_updater.fit(X_aug, D_init=D_aug_norm, checkpoint_dir=None)
            D_aug_updated = atom_updater.D_
            Gamma = atom_updater.Gamma_

            # De-augment: extract D, A, W from the updated augmented dict
            D, A, W = self._split_aug_dict(
                D_aug_updated, n_features, n_classes, sqrt_alpha, sqrt_beta, use_classifier_term
            )
            D = normalize_columns(D)

            # ---- Track RMSE on original X ----
            err = float(np.sqrt(rep_error_squared(X, D, Gamma).sum() / X.size))
            self.errors_.append(err)

            if self.verbose:
                print(f"[{self.variant.upper()}] Iter {it + 1}/{self.n_iter}  RMSE={err:.6f}")

            if checkpoint_path is not None:
                self._save_checkpoint(
                    checkpoint_path, X, H, D, A, W, Q, it + 1
                )

        self.D_ = D
        self.A_ = A

        # ---- Classifier ----
        if self.variant == "lcksvd1":
            # Paper Sec. 3.2: LC-KSVD1's classifier is trained separately,
            # after D, A, X are computed — never jointly with the
            # dictionary. Re-encode with the final, converged D.
            clf = self.classifier if self.classifier is not None else RidgeClassifier(
                lambda1=self.lambda1
            )
            Gamma_final = self.transform(X) if self.D_ is not None else Gamma
            clf.fit(Gamma_final, H)
            self.classifier_ = clf
            # Mirror a linear W_ for convenience/back-compat when the
            # fitted classifier exposes one (e.g. the default
            # RidgeClassifier); None for classifiers that don't.
            self.W_ = getattr(clf, "W_", None)
        else:
            self.W_ = W
            self.classifier_ = None

        # Record per-class atom ranges matching _build_label_consistent_target
        atoms_per_class = self.n_components // n_classes
        boundaries: dict[int, tuple[int, int]] = {}
        for c in range(n_classes):
            start = c * atoms_per_class
            end = start + atoms_per_class if c < n_classes - 1 else self.n_components
            boundaries[c] = (start, end)
        self.class_boundaries_ = boundaries

        return self

    # ------------------------------------------------------------------
    # Checkpointing (outer LC-KSVD loop)
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        path: str,
        X: np.ndarray,
        H: np.ndarray,
        D: np.ndarray,
        A: np.ndarray,
        W: np.ndarray,
        Q: np.ndarray,
        completed_iter: int,
    ) -> None:
        """
        Atomically write the outer LC-KSVD training state to ``path``.

        Written to a temp file in the same directory first, then moved
        into place with os.replace, so an abrupt stop mid-write can never
        leave a corrupt/truncated checkpoint at ``path``.
        """
        directory = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=".lc_ksvd_checkpoint_", suffix=".npz.tmp"
        )
        try:
            # See KSVD._save_checkpoint: np.savez silently appends '.npz'
            # to string paths not already ending in exactly that
            # extension, which would orphan the real data under a
            # mangled filename and leave an empty file at `path` after
            # os.replace. Writing through the fd avoids that entirely.
            with os.fdopen(fd, "wb") as f:
                np.savez(
                    f,
                    D=D,
                    A=A,
                    W=W,
                    Q=Q,
                    errors_=np.asarray(self.errors_, dtype=float),
                    completed_iter=completed_iter,
                    n_iter=self.n_iter,
                    n_components=self.n_components,
                    n_nonzero_coefs=self.n_nonzero_coefs,
                    alpha=self.alpha,
                    beta=self.beta,
                    variant=self.variant,
                    n_features=X.shape[0],
                    n_samples=X.shape[1],
                    n_classes=H.shape[0],
                )
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _load_checkpoint(self, path: str, X: np.ndarray, H: np.ndarray):
        """
        Load and validate a checkpoint against the current config, X, and H.

        Raises DictionaryLearningError on any mismatch, rather than
        silently resuming with an incompatible state.
        """
        with np.load(path, allow_pickle=True) as data:
            n_features = int(data["n_features"])
            n_samples = int(data["n_samples"])
            n_classes = int(data["n_classes"])
            n_components = int(data["n_components"])
            n_nonzero_coefs = int(data["n_nonzero_coefs"])
            n_iter = int(data["n_iter"])
            alpha = float(data["alpha"])
            beta = float(data["beta"])
            variant = str(data["variant"])
            completed_iter = int(data["completed_iter"])

            if (n_features, n_samples) != X.shape:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} was computed on data of shape "
                    f"{(n_features, n_samples)}, but X has shape {X.shape}."
                )
            if n_classes != H.shape[0]:
                raise DictionaryLearningError(
                    f"Checkpoint n_classes={n_classes} does not match "
                    f"H.shape[0]={H.shape[0]}."
                )
            if n_components != self.n_components:
                raise DictionaryLearningError(
                    f"Checkpoint n_components={n_components} does not match "
                    f"LCKSVD.n_components={self.n_components}."
                )
            if n_nonzero_coefs != self.n_nonzero_coefs:
                raise DictionaryLearningError(
                    f"Checkpoint n_nonzero_coefs={n_nonzero_coefs} does not "
                    f"match LCKSVD.n_nonzero_coefs={self.n_nonzero_coefs}."
                )
            if variant != self.variant:
                raise DictionaryLearningError(
                    f"Checkpoint was created with variant='{variant}', but "
                    f"this LCKSVD instance has variant='{self.variant}'."
                )
            if not np.isclose(alpha, self.alpha) or not np.isclose(beta, self.beta):
                raise DictionaryLearningError(
                    f"Checkpoint was created with alpha={alpha}, beta={beta}, "
                    f"but this LCKSVD instance has alpha={self.alpha}, "
                    f"beta={self.beta}."
                )
            if n_iter != self.n_iter:
                raise DictionaryLearningError(
                    f"Checkpoint was created with n_iter={n_iter}, but this "
                    f"LCKSVD instance has n_iter={self.n_iter}."
                )
            if completed_iter >= self.n_iter:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} already completed all "
                    f"{self.n_iter} outer iterations; nothing to resume."
                )

            D = data["D"].copy()
            A = data["A"].copy()
            W = data["W"].copy()
            Q = data["Q"].copy()
            errors_ = list(data["errors_"])

        return D, A, W, Q, errors_, completed_iter

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode X using the learned dictionary D.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)

        Returns
        -------
        Gamma : np.ndarray, shape (n_components, n_samples)
        """
        self._check_fitted()
        coder = OMP(self.n_nonzero_coefs, mode="batch", check_dict=False)
        return coder.encode(X, self.D_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classify test signals using the learned classifier.

        For LC-KSVD2, this is the argmax of ``W_ @ gamma``. For LC-KSVD1,
        this delegates to the fitted ``classifier_`` (default
        ``RidgeClassifier``, or a user-supplied ``classifier``).

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)  integer class indices
        """
        self._check_fitted()
        Gamma = self.transform(X)

        if self.variant == "lcksvd1":
            if self.classifier_ is None:
                raise DictionaryLearningError(
                    "No classifier is available. This should not happen "
                    "for a successfully fit LC-KSVD1 model."
                )
            return self.classifier_.predict(Gamma)

        if self.W_ is None:
            raise DictionaryLearningError(
                "Classifier W is not available. "
                "Access sparse codes via transform() instead."
            )
        scores = self.W_ @ Gamma          # (n_classes, n_samples)
        return np.argmax(scores, axis=0)

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
        true_labels = np.argmax(H, axis=0)
        pred_labels = self.predict(X)
        return float(np.mean(pred_labels == true_labels))

    @staticmethod
    def _build_aug_dict(
        D: np.ndarray,
        A: np.ndarray,
        W: np.ndarray,
        sqrt_alpha: float,
        sqrt_beta: float,
        use_classifier: bool,
    ) -> np.ndarray:
        """Stack [D ; sqrt_alpha*A ; (sqrt_beta*W)]."""
        parts = [D, sqrt_alpha * A]
        if use_classifier:
            parts.append(sqrt_beta * W)
        return np.vstack(parts)

    @staticmethod
    def _split_aug_dict(
        D_aug: np.ndarray,
        n_features: int,
        n_classes: int,
        sqrt_alpha: float,
        sqrt_beta: float,
        use_classifier: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recover (D, A, W) from the augmented dictionary D_aug.

        D_aug rows are: n_features | n_components | (n_classes if lcksvd2).

        D_aug's columns are unit-norm as a whole (across all stacked
        blocks), not block-by-block. To recover a properly normalised D
        together with A/W on a consistent per-atom scale, each atom's D
        sub-block norm is computed and used to rescale D, A, and W
        (in addition to removing the sqrt_alpha / sqrt_beta weighting
        from A / W).
        """
        n_components = D_aug.shape[1]
        D = D_aug[:n_features, :]
        A_rows = n_components
        A = D_aug[n_features: n_features + A_rows, :]
        if use_classifier:
            W = D_aug[n_features + A_rows:, :]
        else:
            W = np.zeros((n_classes, n_components))

        l2norms = np.linalg.norm(D, axis=0)
        l2norms = np.where(l2norms > 1e-14, l2norms, 1e-14)

        D = D / l2norms
        A = A / l2norms / max(sqrt_alpha, 1e-14)
        if use_classifier:
            W = W / l2norms / max(sqrt_beta, 1e-14)

        return D, A, W