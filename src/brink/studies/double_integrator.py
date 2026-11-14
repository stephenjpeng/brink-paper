"""Double integrator under a bounded adversarial disturbance (exact operating limit).

Dynamics ``p'' = u + w`` with bounded control ``|u| <= u_max`` and disturbance
``|w| <= c``. The design is the initial state ``x = (p, v)``; the stress ``c``
is the disturbance bound. A state is viable when the stopping distance under
the net deceleration ``u_max - c`` fits in the remaining room:

    viable(p, v, c)  <=>  v^2 / (2 (u_max - c)) <= 1 - p    and  c < u_max,

so each design's last viable stress has the closed form

    c_off(p, v) = u_max - v^2 / (2 (1 - p)),

and ``theta = sup_x c_off(x) = u_max``, attained at ``v = 0`` (Appendix C.5).
The design box ``p in [0, 0.5]`` is part of the environment: it keeps the box
away from the pole of ``c_off`` at ``p = 1``. One oracle call costs one raw
evaluation. ``certified_mismatch_constants`` computes the norm bound and
mismatch allowance of Appendix B that hold by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from brink.kernels import rbf

ENV_KIND_DOUBLE_INTEGRATOR = "double-integrator"

U_MAX = 1.0
P_MAX = 0.5
V_MAX = 1.0
C_RANGE = (0.0, 1.5)
X_DIM = 2
M_PREFIX = 1

#: Operator-norm bound on the Hessian of v^2 / (2 (1 - p)) in unit design
#: coordinates over [0, 1]^2: its entries P_MAX^2 v^2 / q^3, P_MAX V_MAX v / q^2,
#: and V_MAX^2 / q, with q = 1 - p >= 1 - P_MAX = 0.5, are each at most 2, so the
#: Frobenius norm, and hence the operator norm, is at most 4 (checked against finite
#: differences in the tests).
PHI_HESSIAN_BOUND = 4.0
#: Exact threshold in physical coefficient units (attained on the v = 0 edge).
THETA_EXISTS_PHYSICAL = U_MAX
#: Exact threshold in unit coefficient units, which is what the arms report.
THETA_EXISTS_UNIT = (THETA_EXISTS_PHYSICAL - C_RANGE[0]) / (C_RANGE[1] - C_RANGE[0])


def to_physical_state(x_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map unit design points to physical ``(p, v)``.

    Args:
        x_unit: Unit design points, shape (n, 2) or (2,).

    Returns:
        Tuple of physical position and velocity arrays, each shape (n,).
    """
    X = np.atleast_2d(np.asarray(x_unit, dtype=float))
    return X[:, 0] * P_MAX, X[:, 1] * V_MAX


def c_off_physical(x_unit: np.ndarray) -> np.ndarray:
    """Closed-form last viable disturbance level, in physical units.

    Args:
        x_unit: Unit design points, shape (n, 2) or (2,).

    Returns:
        ``c_off`` per column, shape (n,).
    """
    p, v = to_physical_state(x_unit)
    return U_MAX - v**2 / (2.0 * (1.0 - p))


def c_off_unit(x_unit: np.ndarray) -> np.ndarray:
    """Closed-form ``c_off`` expressed on the unit coefficient range.

    Args:
        x_unit: Unit design points, shape (n, 2) or (2,).

    Returns:
        ``c_off`` per column in unit coefficient, shape (n,).
    """
    lo, hi = C_RANGE
    return (c_off_physical(x_unit) - lo) / (hi - lo)


def is_viable(x_unit: np.ndarray, c_unit: float) -> np.ndarray:
    """Exact viability predicate at a unit coefficient.

    Args:
        x_unit: Unit design points, shape (n, 2) or (2,).
        c_unit: Unit coefficient.

    Returns:
        Boolean array of shape (n,).
    """
    return np.asarray(c_unit, dtype=float) <= c_off_unit(x_unit)


@dataclass
class DoubleIntegratorEnvironment:
    """Frozen double-integrator environment behind the portfolio protocol.

    Attributes:
        scaler_scale: Per-channel standardization scale (IQR / 1.349).
        B: Frozen joint-RKHS norm bound per channel, shape (m,).
        B_slice: Frozen per-slice norm bound per channel, shape (m,).
        lengthscale: Frozen joint-kernel lengthscale.
        n_raw_evals: Cumulative raw-evaluation counter (one per call).
    """

    scaler_scale: np.ndarray
    B: np.ndarray
    B_slice: np.ndarray
    lengthscale: float
    n_raw_evals: int = field(default=0)

    x_dim = X_DIM
    m = M_PREFIX
    x_bounds = ((0.0, P_MAX), (0.0, V_MAX))
    c_range = C_RANGE
    noise_knob = None
    v_min = None
    beta_hat = None

    @property
    def c_star(self) -> float:
        """Exact unit-coefficient threshold (no numerical error)."""
        return THETA_EXISTS_UNIT

    def raw_channels(self, x_unit: np.ndarray, c_unit: float) -> np.ndarray:
        """Unstandardized prefix channel ``c - c_off``, shape (m,).

        Args:
            x_unit: Unit design point, shape (2,).
            c_unit: Unit coefficient.

        Returns:
            Raw channel values, shape (1,).
        """
        return np.array([float(c_unit) - float(c_off_unit(x_unit)[0])])

    def oracle(
        self,
        x: np.ndarray,
        c: float,
        rng: np.random.Generator | None = None,
        sigma: float = 0.0,
    ) -> np.ndarray:
        """Standardized channel at unit ``(x, c)``; deterministic.

        Args:
            x: Unit design point, shape (2,).
            c: Unit coefficient.
            rng: Unused (deterministic environment).
            sigma: Must be 0.0; this environment has no noise knob.

        Returns:
            Standardized channel, shape (1,), unclipped and exactly affine
            in ``c``.

        Raises:
            ValueError: When sigma is nonzero.
        """
        if sigma != 0.0:
            raise ValueError("double-integrator is deterministic; sigma must be 0")
        self.n_raw_evals += 1
        return self.raw_channels(x, c) / self.scaler_scale

    def window_counters(self) -> dict[str, int]:
        """Raw-evaluation counters; one raw evaluation per oracle call."""
        return {"n_raw_evals": self.n_raw_evals, "n_calls": self.n_raw_evals}

    def raw_feasible(self, x_unit: np.ndarray, c_unit: float) -> bool:
        """Exact feasibility predicate for the uncertified baseline."""
        return bool(is_viable(x_unit, c_unit)[0])

    def score_unit(self, x_unit: np.ndarray) -> float:
        """Reduced score ``c_off(x)`` in unit coefficient.

        The oracle is reducible: one call per design determines the whole
        slice answer for that design. Disclosed under gate G1, which makes a
        direct reduced-score baseline mandatory.
        """
        return float(c_off_unit(x_unit)[0])

    def true_volume(self, c_unit: float, n_per_axis: int = 401) -> float:
        """Fraction of the design box that is viable at a unit coefficient.

        Args:
            c_unit: Unit coefficient.
            n_per_axis: Midpoint-grid resolution per axis.

        Returns:
            Viable volume fraction.
        """
        a = (np.arange(n_per_axis) + 0.5) / n_per_axis
        X = np.stack([g.ravel() for g in np.meshgrid(a, a, indexing="ij")], axis=1)
        return float(np.mean(is_viable(X, c_unit)))


def theta_exists_truth(n_per_axis: int = 401) -> dict:
    """Exact threshold with the grid diagnostics the registration requires.

    The analytic value is exact. The grid figures are recorded because a
    midpoint grid never lands on the ``v = 0`` face, which is where the
    supremum is attained; evaluating the face explicitly is the S2 rule.

    Args:
        n_per_axis: Midpoint-grid resolution per axis.

    Returns:
        Dict with the analytic value, the grid values, and the face value.
    """
    a = (np.arange(n_per_axis) + 0.5) / n_per_axis
    X = np.stack([g.ravel() for g in np.meshgrid(a, a, indexing="ij")], axis=1)
    X_face = np.stack([np.linspace(0.0, 1.0, n_per_axis), np.zeros(n_per_axis)], axis=1)
    theta_grid = float(np.max(c_off_unit(X)))
    theta_face = float(np.max(c_off_unit(X_face)))
    return {
        "theta_unit_analytic": THETA_EXISTS_UNIT,
        "theta_physical_analytic": THETA_EXISTS_PHYSICAL,
        "theta_unit_midpoint_grid": theta_grid,
        "theta_unit_including_face": theta_face,
        "stated_error": 0.0,
        "grid_underestimate": THETA_EXISTS_UNIT - theta_grid,
        "columns_viable_at_c_max": int(np.sum(is_viable(X, 1.0))),
        "columns_viable_at_c_min": int(np.sum(is_viable(X, 0.0))),
        "n_columns": int(X.shape[0]),
    }


def load_double_integrator(path: str | Path) -> DoubleIntegratorEnvironment:
    """Build the environment from a frozen calibration file.

    Args:
        path: Path of the frozen calibration JSON.

    Returns:
        DoubleIntegratorEnvironment with frozen scaler and norm bounds.
    """
    cal = json.loads(Path(path).read_text())["calibration"]
    return DoubleIntegratorEnvironment(
        scaler_scale=np.asarray(cal["scaler_scale"], dtype=float),
        B=np.asarray(cal["B"], dtype=float),
        B_slice=np.asarray(cal["B_slice"], dtype=float),
        lengthscale=float(cal["lengthscale"]),
    )


def c_off_unit_grad(x_unit: np.ndarray) -> np.ndarray:
    """Gradient of ``c_off_unit`` in unit design coordinates.

    Args:
        x_unit: Unit design points, shape (n, 2) or (2,).

    Returns:
        Gradient, shape (n, 2).
    """
    p, v = to_physical_state(x_unit)
    q = 1.0 - p
    dphi = np.stack([P_MAX * v**2 / (2.0 * q**2), V_MAX * v / q], axis=1)
    return -dphi / (C_RANGE[1] - C_RANGE[0])


def mismatch_fit(
    scale: float, lengthscale: float, n_centers: int, pad: float, nugget: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kernel expansions a and b of ``certified_mismatch_constants``.

    Args:
        scale: Standardization scale s of the constraint.
        lengthscale: Kernel lengthscale on unit design coordinates.
        n_centers: Centers per axis.
        pad: Extension of the center grid beyond [0, 1] on each side.
        nugget: Ridge term of the coefficient fit.

    Returns:
        Centers Z, shape (n_centers^2, 2), and coefficients of a and b.
    """
    axis = np.linspace(-pad, 1.0 + pad, n_centers)
    Z = np.stack([g.ravel() for g in np.meshgrid(axis, axis, indexing="ij")], axis=1)
    A = rbf(Z, Z, lengthscale) + nugget * np.eye(len(Z))
    alpha_a = np.linalg.solve(A, -c_off_unit(Z) / scale)
    alpha_b = np.linalg.solve(A, np.full(len(Z), 1.0 / scale))
    return Z, alpha_a, alpha_b


def certified_mismatch_constants(
    scale: float,
    lengthscale: float,
    n_centers: int,
    pad: float,
    nugget: float,
    n_check: int,
    chunk: int = 4096,
) -> dict:
    """Norm bound and mismatch allowance that hold by construction.

    The standardized constraint is g(x, c) = (c - c_off_unit(x)) / scale, with
    stress c in [0, 1]. Under the kernel rbf(x, x') (1 + c c'), the function
    f(x, c) = a(x) + c b(x) has squared RKHS norm ||a||^2 + ||b||^2. Here a and
    b are finite kernel expansions on an n_centers x n_centers grid of
    [-pad, 1 + pad]^2, fitted to a* = -c_off_unit / scale and b* = 1 / scale by
    ridge regression with the given nugget. Any such expansion is a valid RKHS
    element; the fit only sets the trade-off between B and eta.

    Because g - f is affine in c, sup over X x C of |g - f| is the larger of
    sup |r_a| and sup |r_a + r_b|, with r_a = a - a* and r_b = b - b*. Each is
    bounded on [0, 1]^2 by a midpoint grid with n_check points per axis and the
    second-order Taylor bound |r(x)| <= |r(z)| + rho ||grad r(z)|| +
    rho^2 H / 2, where rho is the grid's covering radius and H bounds the
    Hessian: sqrt(3) ||.||_H / lengthscale^2 for the kernel part (the norm of the
    second directional derivative of the feature map) plus the closed-form bound
    PHI_HESSIAN_BOUND / (1.5 scale) for a*.

    Rounding: every float64 sum of n terms is off by at most gamma = 2 (n + 40) u
    times the sum of the absolute terms (u the unit roundoff; the 40 covers the
    kernel evaluation, whose relative error is below 20 u on these inputs). B and
    the Hessian norms add gamma |alpha|^T K |alpha| to each squared norm, and the
    grid residuals and gradients add gamma times the matching absolute sums, so
    B and eta bound the exact norm and mismatch of the stored expansion.

    Args:
        scale: Standardization scale s of the constraint.
        lengthscale: Kernel lengthscale on unit design coordinates.
        n_centers: Centers per axis.
        pad: Extension of the center grid beyond [0, 1] on each side.
        nugget: Ridge term of the coefficient fit (not the run's lambda).
        n_check: Midpoint-grid points per axis for the sup bound.
        chunk: Grid points per block.

    Returns:
        Dict with B, eta, the component norms, and the sup-bound terms.
    """
    Z, alpha_a, alpha_b = mismatch_fit(scale, lengthscale, n_centers, pad, nugget)
    K = rbf(Z, Z, lengthscale)
    gamma = 2.0 * (len(Z) + 40) * np.finfo(float).eps / 2.0
    alpha_ab = alpha_a + alpha_b

    def norm_bound(alpha: np.ndarray) -> float:
        abs_alpha = np.abs(alpha)
        sq = float(alpha @ K @ alpha) + gamma * float(abs_alpha @ K @ abs_alpha)
        return float(np.sqrt(max(sq, 0.0))) * (1.0 + 1e-12)

    norm_a, norm_b, norm_ab = (norm_bound(v) for v in (alpha_a, alpha_b, alpha_ab))
    abs_a, abs_ab = np.abs(alpha_a), np.abs(alpha_a) + np.abs(alpha_b)

    g1 = (np.arange(n_check) + 0.5) / n_check
    G = np.stack([g.ravel() for g in np.meshgrid(g1, g1, indexing="ij")], axis=1)
    rho = np.sqrt(2.0) / (2.0 * n_check)
    l2 = lengthscale**2
    first_a = first_ab = res_a = res_ab = 0.0
    for s in range(0, len(G), chunk):
        Gs = G[s : s + chunk]
        Kq = rbf(Gs, Z, lengthscale)
        diff = Z[None, :, :] - Gs[:, None, :]
        a, b = Kq @ alpha_a, Kq @ alpha_b
        grad_a = np.einsum("qn,qnd,n->qd", Kq, diff, alpha_a) / l2
        grad_b = np.einsum("qn,qnd,n->qd", Kq, diff, alpha_b) / l2
        a_star = -c_off_unit(Gs) / scale
        grad_a_star = -c_off_unit_grad(Gs) / scale
        r_a = a - a_star
        r_ab = r_a + b - 1.0 / scale
        g_a = np.linalg.norm(grad_a - grad_a_star, axis=1)
        g_ab = np.linalg.norm(grad_a + grad_b - grad_a_star, axis=1)
        # Rounding of the expansions, their gradients, and the subtractions.
        dist = np.linalg.norm(diff, axis=2) / l2
        err_a = gamma * (Kq @ abs_a + np.abs(a) + np.abs(a_star))
        err_ab = gamma * (Kq @ abs_ab + np.abs(a) + np.abs(b) + np.abs(a_star) + 1)
        gerr_a = gamma * ((Kq * dist) @ abs_a + np.linalg.norm(grad_a_star, axis=1))
        gerr_ab = gamma * ((Kq * dist) @ abs_ab + np.linalg.norm(grad_a_star, axis=1))
        first_a = max(
            first_a, float(np.max(np.abs(r_a) + err_a + rho * (g_a + gerr_a)))
        )
        first_ab = max(
            first_ab, float(np.max(np.abs(r_ab) + err_ab + rho * (g_ab + gerr_ab)))
        )
        res_a = max(res_a, float(np.max(np.abs(r_a))))
        res_ab = max(res_ab, float(np.max(np.abs(r_ab))))
    h_star = PHI_HESSIAN_BOUND / ((C_RANGE[1] - C_RANGE[0]) * scale)
    curv_a = 0.5 * rho**2 * (np.sqrt(3.0) * norm_a / l2 + h_star)
    curv_ab = 0.5 * rho**2 * (np.sqrt(3.0) * norm_ab / l2 + h_star)
    eta = max(first_a + curv_a, first_ab + curv_ab) * (1.0 + 1e-6) + 1e-12
    return {
        "B": float(np.sqrt(norm_a**2 + norm_b**2)) * (1.0 + 1e-12),
        "rounding_gamma": float(gamma),
        "eta": float(eta),
        "norm_a": norm_a,
        "norm_b": norm_b,
        "norm_a_plus_b": norm_ab,
        "grid_max_abs_r_a": res_a,
        "grid_max_abs_r_a_plus_b": res_ab,
        "curvature_term": float(max(curv_a, curv_ab)),
        "rho": float(rho),
        "n_centers": int(n_centers),
        "pad": float(pad),
        "nugget": float(nugget),
        "n_check": int(n_check),
    }
