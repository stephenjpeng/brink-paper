"""Exact kernel ridge regression with a fixed kernel and a fixed lambda.

One independent model per constraint; no hyperparameter fitting. The
posterior comes from a Cholesky factorization of (K_n + lambda * I),
recomputed from scratch at each update. ``beta_sqrt`` gives the confidence
multiplier of Appendix E.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from brink.kernels import rbf

VARIANT_DETERMINISTIC = "deterministic"
VARIANT_NOISY = "noisy"
# Query rows per posterior block; bounds the (q x n) transients.
# ~160 MiB transient at n = 300; raise only with a measured RSS record
POSTERIOR_CHUNK = 16384


@dataclass
class GPModel:
    """Fitted exact-GP state for one constraint.

    Attributes:
        lengthscale: RBF lengthscale on X.
        lam: Regularization lambda (fixed per oracle variant).
        B: RKHS norm bound for the modeled constraint.
        R: Known noise scale of the oracle variant (0 when deterministic).
        X: Training inputs in X, shape (n, x_dim).
        c: Training coefficients, shape (n,); None for x-only slice models.
        y: Training targets, shape (n,).
        L: Lower Cholesky factor of (K_n + lam * I); None when n == 0.
        w: Weight vector (K_n + lam * I)^{-1} y, shape (n,).
        logdet_term: logdet(I + K_n / lam), from the Cholesky factor.
    """

    lengthscale: float
    lam: float
    B: float
    R: float
    X: np.ndarray
    c: np.ndarray | None
    y: np.ndarray
    L: np.ndarray | None
    w: np.ndarray
    logdet_term: float

    @property
    def n(self) -> int:
        """Number of training observations."""
        return self.X.shape[0]


def fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lengthscale: float,
    lam: float,
    B: float,
    R: float,
    c: np.ndarray | None = None,
) -> GPModel:
    """Fit an exact GP with fixed kernel and fixed lambda.

    Args:
        X: Training inputs in X, shape (n, x_dim).
        y: Training targets, shape (n,).
        lengthscale: RBF lengthscale on X.
        lam: Regularization lambda.
        B: RKHS norm bound of the target function.
        R: Known observation-noise scale (0 for the deterministic variant).
        c: Training coefficients for a joint model, shape (n,); None for an
            x-only slice model.

    Returns:
        Fitted GPModel (handles n == 0 with a prior-only model).
    """
    X = np.asarray(X, dtype=float).reshape(-1, X.shape[-1] if X.ndim > 1 else 1)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    if n == 0:
        return GPModel(
            lengthscale=lengthscale,
            lam=lam,
            B=B,
            R=R,
            X=X,
            c=c,
            y=y,
            L=None,
            w=np.zeros(0),
            logdet_term=0.0,
        )
    K = rbf(X, X, lengthscale)
    if c is not None:
        c = np.asarray(c, dtype=float).ravel()
        K = K * (1.0 + np.outer(c, c))
    K[np.diag_indices_from(K)] += lam
    L = cholesky(K, lower=True)
    w = cho_solve((L, True), y)
    logdet_term = float(2.0 * np.sum(np.log(np.diag(L))) - n * np.log(lam))
    return GPModel(
        lengthscale=lengthscale,
        lam=lam,
        B=B,
        R=R,
        X=X,
        c=c,
        y=y,
        L=L,
        w=w,
        logdet_term=logdet_term,
    )


def _cross_kernel(model: GPModel, Xq: np.ndarray, cq: np.ndarray | None) -> np.ndarray:
    """Cross-kernel matrix between query points and training points."""
    Kq = rbf(Xq, model.X, model.lengthscale)
    if model.c is not None:
        Kq = Kq * (1.0 + np.outer(np.asarray(cq, dtype=float).ravel(), model.c))
    return Kq


def prior_variance(model: GPModel, cq: np.ndarray | None) -> np.ndarray | float:
    """Prior variance k((x, c), (x, c)) under the model's kernel."""
    if model.c is None and cq is None:
        return 1.0
    cq = np.asarray(cq, dtype=float)
    return 1.0 + cq**2


def posterior(
    model: GPModel, Xq: np.ndarray, cq: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and standard deviation at query points.

    Args:
        model: Fitted GP model.
        Xq: Query inputs in X, shape (q, x_dim).
        cq: Query coefficients, shape (q,) or scalar for joint models; None
            for x-only slice models.

    Returns:
        Tuple (mu, sd), each of shape (q,).
    """
    Xq = np.asarray(Xq, dtype=float)
    q = Xq.shape[0]
    if cq is not None and np.ndim(cq) == 0:
        cq = np.full(q, float(cq))
    prior = np.broadcast_to(np.asarray(prior_variance(model, cq), dtype=float), (q,))
    if model.n == 0:
        return np.zeros(q), np.sqrt(prior)
    mu, var = np.empty(q), np.empty(q)
    for s in range(0, q, POSTERIOR_CHUNK):
        e = min(s + POSTERIOR_CHUNK, q)
        Kq = _cross_kernel(model, Xq[s:e], None if cq is None else cq[s:e])
        mu[s:e] = Kq @ model.w
        A = solve_triangular(model.L, Kq.T, lower=True)
        var[s:e] = prior[s:e] - np.sum(A**2, axis=0)
    return mu, np.sqrt(np.maximum(var, 0.0))


def beta_sqrt(model: GPModel, delta_j: float, variant: str) -> float:
    """Confidence-radius multiplier beta^{1/2} for one constraint model.

    Deterministic variant: the exact RKHS bound B_j. Noisy variant:
    B_j + (R / sqrt(lam)) * sqrt(2 * ln(2 / delta_j) + logdet(I + K_n / lam)),
    with delta_j the per-constraint budget (delta / m gives
    2 / delta_j = 2 m / delta as in the plan).

    Args:
        model: Fitted GP model.
        delta_j: Per-constraint confidence budget.
        variant: VARIANT_DETERMINISTIC or VARIANT_NOISY.

    Returns:
        Scalar beta^{1/2}.
    """
    if variant == VARIANT_DETERMINISTIC:
        return float(model.B)
    if variant == VARIANT_NOISY:
        return float(
            model.B
            + (model.R / np.sqrt(model.lam))
            * np.sqrt(2.0 * np.log(2.0 / delta_j) + model.logdet_term)
        )
    raise ValueError(f"unknown variant: {variant}")
