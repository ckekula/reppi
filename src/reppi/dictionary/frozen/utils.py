"""
frozen.utils
Utility functions for Frozen Dictionary Learning.
"""

import logging
import os

import numpy as np

from reppi.exceptions import DictionaryLearningError

logger = logging.getLogger(__name__)

_KSVD_CHECKPOINT_FILENAME = "ksvd_checkpoint.npz"


def _completed_ksvd_checkpoint(
    checkpoint_dir: str | None,
    X: np.ndarray,
    n_components: int | None,
    n_nonzero_coefs: int | None,
    n_iter: int | None,
    n_frozen: int,
    D_frozen: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[float]] | None:
    """
    Check whether ``<checkpoint_dir>/ksvd_checkpoint.npz`` already holds a
    *fully completed* fit for this exact configuration.
    """
    if checkpoint_dir is None:
        return None
    if n_components is None or n_nonzero_coefs is None or n_iter is None:
        return None

    path = os.path.join(checkpoint_dir, _KSVD_CHECKPOINT_FILENAME)
    if not os.path.exists(path):
        return None

    try:
        with np.load(path) as data:
            n_features = int(data["n_features"])
            n_samples = int(data["n_samples"])
            n_components_ckpt = int(data["n_components"])
            n_nonzero_coefs_ckpt = int(data["n_nonzero_coefs"])
            n_iter_ckpt = int(data["n_iter"])
            completed_iter = int(data["completed_iter"])
            n_frozen_ckpt = int(data["n_frozen"]) if "n_frozen" in data else 0

            if (n_features, n_samples) != X.shape:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} was computed on data of shape "
                    f"{(n_features, n_samples)}, but X has shape {X.shape}."
                )
            if n_components_ckpt != n_components:
                raise DictionaryLearningError(
                    f"Checkpoint n_components={n_components_ckpt} does not "
                    f"match the requested n_components={n_components}."
                )
            if n_nonzero_coefs_ckpt != n_nonzero_coefs:
                raise DictionaryLearningError(
                    f"Checkpoint n_nonzero_coefs={n_nonzero_coefs_ckpt} "
                    f"does not match the requested "
                    f"n_nonzero_coefs={n_nonzero_coefs}."
                )
            if n_iter_ckpt != n_iter:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} was created with "
                    f"n_iter={n_iter_ckpt}, but this call has n_iter={n_iter}."
                )

            if completed_iter < n_iter:
                return None

            if n_frozen_ckpt != n_frozen:
                raise DictionaryLearningError(
                    f"Checkpoint at {path} was trained with "
                    f"{n_frozen_ckpt} frozen atoms, but this call has "
                    f"n_frozen={n_frozen}."
                )

            D = data["D"].copy()
            Gamma = data["Gamma"].copy()
            errors_ = list(data["errors_"])
    except KeyError:
        return None

    if n_frozen > 0:
        if D_frozen is None:
            raise DictionaryLearningError(
                f"Checkpoint at {path} expects {n_frozen} frozen atoms, "
                "but D_frozen=None was passed to this call."
            )
        if not np.allclose(D[:, :n_frozen], D_frozen, atol=1e-6):
            raise DictionaryLearningError(
                f"Checkpoint at {path}'s frozen atoms do not match the "
                "D_frozen passed to this call. Loading it would silently "
                "mix training states from different frozen dictionaries."
            )

    logger.info("Skipping training: %s already complete (%d/%d iters).", path, completed_iter, n_iter)
    return D, Gamma, errors_
