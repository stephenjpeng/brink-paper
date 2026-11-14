"""Tests for the certified mismatch constants of the double integrator."""

import numpy as np

from brink.kernels import rbf
from brink.studies.double_integrator import (
    C_RANGE,
    PHI_HESSIAN_BOUND,
    c_off_unit,
    c_off_unit_grad,
    certified_mismatch_constants,
    mismatch_fit,
)

SCALE, LS = 0.35039964159972947, 1.0146321793283504


def test_c_off_grad_matches_finite_differences():
    X = np.random.default_rng(0).uniform(0.01, 0.99, size=(200, 2))
    h = 1e-6
    fd = np.stack(
        [(c_off_unit(X + h * e) - c_off_unit(X - h * e)) / (2 * h) for e in np.eye(2)],
        axis=1,
    )
    assert np.allclose(c_off_unit_grad(X), fd, atol=1e-7)


def test_phi_hessian_bound_covers_finite_differences():
    g = np.linspace(0.0, 1.0, 41)
    X = np.stack([a.ravel() for a in np.meshgrid(g, g, indexing="ij")], axis=1)
    h = 1e-4
    # c_off_unit = (1 - phi) / 1.5, so hess(phi) = -1.5 * hess(c_off_unit).
    H = np.empty((len(X), 2, 2))
    for i, e in enumerate(np.eye(2)):
        H[:, :, i] = -1.5 * (c_off_unit_grad(X + h * e) - c_off_unit_grad(X - h * e))
        H[:, :, i] /= 2 * h
    top = np.max(np.abs(np.linalg.eigvalsh(0.5 * (H + H.transpose(0, 2, 1)))))
    # Tight at the corner p = 0.5, v = 1 (eigenvalues 4 and 0); the central
    # difference steps just outside the box there.
    assert top <= PHI_HESSIAN_BOUND * (1 + 1e-6)
    assert top > 0.5 * PHI_HESSIAN_BOUND  # the bound is not vacuous


def test_certified_eta_bounds_the_mismatch_on_random_points():
    args = (SCALE, LS, 8, 0.0, 1e-8)
    r = certified_mismatch_constants(*args, n_check=256)
    Z, alpha_a, alpha_b = mismatch_fit(*args)
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(20000, 2))
    c = rng.uniform(size=20000)
    Kq = rbf(X, Z, LS)
    f = Kq @ alpha_a + c * (Kq @ alpha_b)
    g = (c - c_off_unit(X)) / SCALE
    assert np.max(np.abs(g - f)) <= r["eta"]
    K = rbf(Z, Z, LS)
    norm2 = alpha_a @ K @ alpha_a + alpha_b @ K @ alpha_b
    assert np.sqrt(norm2) <= r["B"]
    assert C_RANGE == (0.0, 1.5)
