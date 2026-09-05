import os
import tempfile

import numpy as np

from reppi.exceptions import DictionaryLearningError


def save_checkpoint(
    self,
    path: str,
    X: np.ndarray,
    D: np.ndarray,
    Gamma: np.ndarray,
    unused: np.ndarray,
    replaced: np.ndarray,
    completed_iter: int,
    n_frozen: int,
) -> None:
    """
    Atomically write the training state to ``path``.

    Written to a temp file in the same directory first, then moved
    into place with os.replace, so an abrupt stop mid-write can never
    leave a corrupt/truncated checkpoint at ``path``.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=".ksvd_checkpoint_", suffix=".npz.tmp"
    )
    try:
        # Pass the open file descriptor to np.savez: given a string,
        # np.savez silently appends '.npz' if
        # the name doesn't already end with exactly that extension
        with os.fdopen(fd, "wb") as f:
            np.savez(
                f,
                D=D,
                Gamma=Gamma,
                unused=unused,
                replaced=replaced,
                errors_=np.asarray(self.errors_, dtype=np.float32),
                completed_iter=completed_iter,
                n_iter=self.n_iter,
                n_components=self.n_components,
                n_nonzero_coefs=self.n_nonzero_coefs,
                n_features=X.shape[0],
                n_samples=X.shape[1],
                n_frozen=n_frozen,
            )
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def load_checkpoint(
    self,
    path: str,
    X: np.ndarray,
    n_frozen: int,
    D_frozen: np.ndarray | None,
):
    """
    Load and validate a checkpoint against the current config, X, and
    the frozen-dictionary configuration for this call.
    """
    with np.load(path) as data:
        n_features = int(data["n_features"])
        n_samples = int(data["n_samples"])
        n_components = int(data["n_components"])
        n_nonzero_coefs = int(data["n_nonzero_coefs"])
        n_iter = int(data["n_iter"])
        completed_iter = int(data["completed_iter"])
        n_frozen_ckpt = int(data["n_frozen"]) if "n_frozen" in data else 0

        if (n_features, n_samples) != X.shape:
            raise DictionaryLearningError(
                f"Checkpoint at {path} was computed on data of shape "
                f"{(n_features, n_samples)}, but X has shape {X.shape}."
            )
        if n_components != self.n_components:
            raise DictionaryLearningError(
                f"Checkpoint n_components={n_components} does not match "
                f"KSVD.n_components={self.n_components}."
            )
        if n_nonzero_coefs != self.n_nonzero_coefs:
            raise DictionaryLearningError(
                f"Checkpoint n_nonzero_coefs={n_nonzero_coefs} does not "
                f"match KSVD.n_nonzero_coefs={self.n_nonzero_coefs}."
            )
        if n_iter != self.n_iter:
            raise DictionaryLearningError(
                f"Checkpoint was created with n_iter={n_iter}, but this "
                f"KSVD instance has n_iter={self.n_iter}."
            )
        if n_frozen_ckpt != n_frozen:
            raise DictionaryLearningError(
                f"Checkpoint at {path} was trained with {n_frozen_ckpt} "
                f"frozen atoms, but this fit() call has n_frozen={n_frozen}."
            )

        D = data["D"].copy()
        Gamma = data["Gamma"].copy()
        unused = data["unused"].copy()
        replaced = data["replaced"].copy()
        errors_ = list(data["errors_"])

    if n_frozen > 0:
        if D_frozen is None:
            raise DictionaryLearningError(
                f"Checkpoint at {path} expects {n_frozen} frozen atoms, "
                "but D_frozen=None was passed to this fit() call."
            )
        if not np.allclose(D[:, :n_frozen], D_frozen, atol=1e-6):
            raise DictionaryLearningError(
                f"Checkpoint at {path}'s frozen atoms do not match the "
                "D_frozen passed to this fit() call. Resuming would "
                "silently mix training states from different frozen "
                "dictionaries."
            )

    return D, Gamma, unused, replaced, errors_, completed_iter