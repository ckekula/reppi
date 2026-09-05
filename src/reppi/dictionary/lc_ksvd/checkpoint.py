import os
import tempfile

import numpy as np

from reppi.exceptions import DictionaryLearningError


def save_checkpoint(
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
                errors_=np.asarray(self.errors_, dtype=np.float32),
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

def load_checkpoint(self, path: str, X: np.ndarray, H: np.ndarray):
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
        alpha = np.float32(data["alpha"])
        beta = np.float32(data["beta"])
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
