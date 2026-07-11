"""
Utilities for FISTA-based L1-regularized least squares sparse coding.

Implements the shrinkage/soft-threshold operator (eq. 1.5) and the
Lipschitz constant for the least-squares smooth part, following
Example 2.2 of Beck & Teboulle (2009).
"""

from __future__ import annotations

import numpy as np


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    """
    Elementwise soft-thresholding (shrinkage) operator T_alpha, eq. (1.5).

        T_alpha(x)_i = sign(x_i) * max(|x_i| - alpha, 0)

    This is also prox_{alpha * ||.||_1}(x).
    """
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def lipschitz_constant_lsq(D: np.ndarray) -> float:
    """
    Lipschitz constant of grad f for f(x) = ||Dx - b||^2.

    Per Example 2.2: L(f) = 2 * lambda_max(D.T @ D) = 2 * ||D||_2^2,
    where ||D||_2 is the spectral (largest singular value) norm.
    """
    sigma_max = np.linalg.svd(D, compute_uv=False)[0]
    return 2.0 * float(sigma_max) ** 2