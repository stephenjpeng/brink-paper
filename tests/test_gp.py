"""Tests for the fixed-kernel exact kernel regression and its confidence bounds."""

from types import SimpleNamespace

import numpy as np

from brink.gp import (
    VARIANT_DETERMINISTIC,
    VARIANT_NOISY,
    beta_sqrt,
    fit_gp,
    posterior,
)
from brink.kernels import rbf

SPEC = SimpleNamespace(m=2, lengthscale=0.3)
LENGTHSCALE = SPEC.lengthscale
LAM_DET = 1e-6
LAM_NOISY = 1.0
SIGMA = 0.01


def draw_environment(seed: int, spec=SPEC):
    """Two constraints in the joint RKHS with known norms (finite kernel sums)."""
    rng = np.random.default_rng(seed)
    Z, cz = rng.uniform(size=(12, 2)), rng.uniform(size=12)
    A = rng.standard_normal((spec.m, 12))
    Kz = rbf(Z, Z, spec.lengthscale) * (1.0 + np.outer(cz, cz))
    B = np.sqrt(np.einsum("ji,ik,jk->j", A, Kz, A))

    def g(X, c):
        c = np.broadcast_to(np.asarray(c, dtype=float), (len(X),))
        Kx = rbf(np.asarray(X), Z, spec.lengthscale) * (1.0 + np.outer(c, cz))
        return A @ Kx.T

    return SimpleNamespace(g=g, B=B)


def _separated_points(rng, n, min_dist):
    """Rejection-sample x-points in [0,1]^2 with a minimum separation."""
    pts = []
    while len(pts) < n:
        p = rng.uniform(size=2)
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    return np.array(pts)


def test_noiseless_model_interpolates_training_targets():
    env = draw_environment(1, SPEC)
    rng = np.random.default_rng(42)
    X = _separated_points(rng, 8, 0.35)
    c = rng.uniform(size=8)
    y = env.g(X, c)[0]
    model = fit_gp(
        X, y, lengthscale=LENGTHSCALE, lam=LAM_DET, B=float(env.B[0]), R=0.0, c=c
    )
    mu, _ = posterior(model, X, c)
    assert np.max(np.abs(mu - y)) < 1e-5


def _test_grid():
    xs = np.linspace(0.0, 1.0, 50)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    Xg = np.column_stack([gx.ravel(), gy.ravel()])
    cs = np.linspace(0.0, 1.0, 11)
    return Xg, cs


def test_certified_bound_deterministic_zero_violations():
    """|g_j - mu_n| <= B_j * sigma_n on a 50x50x11 grid, 50 environments."""
    Xg, cs = _test_grid()
    for env_seed in range(50):
        env = draw_environment(env_seed, SPEC)
        rng = np.random.default_rng(1000 + env_seed)
        X = rng.uniform(size=(30, 2))
        c = rng.uniform(size=30)
        G = env.g(X, c)
        for j in range(SPEC.m):
            model = fit_gp(
                X,
                G[j],
                lengthscale=LENGTHSCALE,
                lam=LAM_DET,
                B=float(env.B[j]),
                R=0.0,
                c=c,
            )
            b = beta_sqrt(model, 0.05, VARIANT_DETERMINISTIC)
            for c_slice in cs:
                mu, sd = posterior(model, Xg, float(c_slice))
                g_true = env.g(Xg, float(c_slice))[j]
                violations = np.abs(g_true - mu) > b * sd + 1e-10
                assert not violations.any(), (
                    f"certified-bound violation: env_seed={env_seed} j={j} c={c_slice}"
                )


def test_noisy_bound_violation_rate_within_nominal():
    """At most 10 of 50 replicates may violate anywhere (nominal rate 0.10)."""
    Xg, cs = _test_grid()
    env = draw_environment(12, SPEC)
    delta = 0.1
    delta_j = delta / SPEC.m
    n_violating = 0
    for rep in range(50):
        rng = np.random.default_rng(5000 + rep)
        X = rng.uniform(size=(30, 2))
        c = rng.uniform(size=30)
        G = env.g(X, c) + SIGMA * rng.standard_normal((SPEC.m, 30))
        violated = False
        for j in range(SPEC.m):
            model = fit_gp(
                X,
                G[j],
                lengthscale=LENGTHSCALE,
                lam=LAM_NOISY,
                B=float(env.B[j]),
                R=SIGMA,
                c=c,
            )
            b = beta_sqrt(model, delta_j, VARIANT_NOISY)
            for c_slice in cs:
                mu, sd = posterior(model, Xg, float(c_slice))
                g_true = env.g(Xg, float(c_slice))[j]
                if (np.abs(g_true - mu) > b * sd).any():
                    violated = True
                    break
            if violated:
                break
        n_violating += int(violated)
    assert n_violating <= 10, f"{n_violating} of 50 replicates violated"


def test_beta_sqrt_values():
    env = draw_environment(3, SPEC)
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(15, 2))
    c = rng.uniform(size=15)
    y = env.g(X, c)[0]
    model = fit_gp(
        X, y, lengthscale=LENGTHSCALE, lam=LAM_NOISY, B=float(env.B[0]), R=SIGMA, c=c
    )
    assert beta_sqrt(model, 0.05, VARIANT_DETERMINISTIC) == float(env.B[0])
    # Noisy formula: B + (R / sqrt(lam)) * sqrt(2 ln(2/delta_j) + logdet).
    K = np.exp(
        -((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) / (2 * LENGTHSCALE**2)
    ) * (1 + np.outer(c, c))
    sign, logdet = np.linalg.slogdet(np.eye(15) + K / LAM_NOISY)
    expected = float(env.B[0]) + (SIGMA / np.sqrt(LAM_NOISY)) * np.sqrt(
        2 * np.log(2 / 0.05) + logdet
    )
    assert sign > 0
    assert np.isclose(beta_sqrt(model, 0.05, VARIANT_NOISY), expected, atol=1e-8)


def test_empty_model_prior_posterior():
    model = fit_gp(
        np.empty((0, 2)),
        np.empty(0),
        lengthscale=LENGTHSCALE,
        lam=LAM_NOISY,
        B=2.0,
        R=SIGMA,
        c=None,
    )
    mu, sd = posterior(model, np.array([[0.5, 0.5]]))
    assert mu[0] == 0.0
    assert np.isclose(sd[0], 1.0)
    assert beta_sqrt(model, 0.05, VARIANT_NOISY) >= 2.0


def test_posterior_chunked_matches_unchunked(monkeypatch):
    """Row-chunked posterior equals the single-block posterior (P2)."""
    import brink.gp as gp

    env = draw_environment(6, SPEC)
    rng = np.random.default_rng(11)
    X = rng.uniform(size=(20, 2))
    c = rng.uniform(size=20)
    y = env.g(X, c)[1]
    kw = dict(lengthscale=LENGTHSCALE, lam=LAM_NOISY, B=float(env.B[1]), R=SIGMA)
    joint = fit_gp(X, y, c=c, **kw)
    x_only = fit_gp(np.column_stack([X, c]), y, **kw)
    Xq = rng.uniform(size=(50, 2))
    cq = rng.uniform(size=50)
    cases = [
        (joint, Xq, 0.4),
        (joint, Xq, cq),
        (x_only, np.column_stack([Xq, cq]), None),
    ]
    for model, Q, c_arg in cases:
        monkeypatch.setattr(gp, "POSTERIOR_CHUNK", 10**9)
        mu_full, sd_full = posterior(model, Q, c_arg)
        monkeypatch.setattr(gp, "POSTERIOR_CHUNK", 7)
        mu_chunk, sd_chunk = posterior(model, Q, c_arg)
        assert np.allclose(mu_chunk, mu_full, rtol=0, atol=1e-12)
        assert np.allclose(sd_chunk, sd_full, rtol=0, atol=1e-12)
