"""Kernel functions.

The design-space kernel is a unit-variance squared exponential on X. The
stress kernel is k_C(c, c') = 1 + c c'. The joint kernel is their product,
whose RKHS holds the functions a(x) + c b(x) with squared norm
||a||^2 + ||b||^2 (Appendix B).
"""

from __future__ import annotations

import numpy as np


def rbf(X1: np.ndarray, X2: np.ndarray, lengthscale: float) -> np.ndarray:
    """Unit-variance squared-exponential kernel matrix.

    k(x, x') = exp(-||x - x'||^2 / (2 * lengthscale^2)).

    Args:
        X1: Array of shape (n1, d).
        X2: Array of shape (n2, d).
        lengthscale: Kernel lengthscale (fixed value 0.3).

    Returns:
        Kernel matrix of shape (n1, n2).
    """
    sq = (
        np.sum(X1**2, axis=1)[:, None]
        + np.sum(X2**2, axis=1)[None, :]
        - 2.0 * (X1 @ X2.T)
    )
    np.maximum(sq, 0.0, out=sq)
    return np.exp(-sq / (2.0 * lengthscale**2))
