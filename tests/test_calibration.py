"""Phase 0 calibration-recipe tests."""

import numpy as np
import pytest

from brink.calibration import (
    LENGTHSCALE_MULTIPLIERS,
    N_SOBOL,
    calibrate,
    calibration_points,
    fit_scalers,
    load_calibration,
    median_pairwise_distance,
    save_calibration,
    standardize,
)


def smooth_oracle(x: np.ndarray, c: float):
    """Analytic two-channel raw oracle, smooth and c-monotone."""
    g1 = x[0] + 0.8 * c - 0.9
    g2 = 0.5 * x[1] + 0.4 * c - 0.6
    return np.array([g1, g2])


def crashing_oracle(x: np.ndarray, c: float):
    """Oracle that crashes in a corner region, else smooth."""
    if x[0] > 0.9 and c > 0.9:
        return None
    return smooth_oracle(x, c)


def test_calibration_points_registered_seeds_and_shapes():
    X, c = calibration_points(x_dim=4)
    assert X.shape == (512, 4)
    assert c.shape == (512,)
    assert np.all((X >= 0) & (X <= 1)) and np.all((c >= 0) & (c <= 1))
    X2, c2 = calibration_points(x_dim=4)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(c, c2)
    # The uniform block reproduces the seed-9002 stream.
    rng = np.random.default_rng(9002)
    u = rng.uniform(size=(256, 5))
    np.testing.assert_array_equal(X[N_SOBOL:], u[:, :4])
    np.testing.assert_array_equal(c[N_SOBOL:], u[:, 4])


def test_fit_scalers_and_standardize_crash_convention():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(100, 2))
    crashed = np.zeros(100, dtype=bool)
    crashed[:5] = True
    Y[crashed] = np.nan
    median, scale = fit_scalers(Y, crashed)
    good = Y[~crashed]
    np.testing.assert_allclose(median, np.median(good, axis=0))
    iqr = np.quantile(good, 0.75, axis=0) - np.quantile(good, 0.25, axis=0)
    np.testing.assert_allclose(scale, iqr / 1.349)
    Z = standardize(Y, crashed, scale, 3.0)
    np.testing.assert_array_equal(Z[crashed], 3.0)
    np.testing.assert_allclose(Z[~crashed], np.clip(good / scale, -3.0, 3.0))
    assert np.all((Z >= -3.0) & (Z <= 3.0))


def test_fit_scalers_rejects_degenerate_channel():
    Y = np.ones((50, 2))
    with pytest.raises(ValueError, match="degenerate"):
        fit_scalers(Y, np.zeros(50, dtype=bool))


def test_median_pairwise_distance_simple():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert median_pairwise_distance(X) == pytest.approx(1.0)


def test_calibrate_smooth_oracle_learnable():
    cal = calibrate("unit-test", smooth_oracle, x_dim=2, m=2)
    assert cal.n_crashed == 0
    assert cal.lengthscale_multiplier in LENGTHSCALE_MULTIPLIERS
    assert cal.lengthscale == pytest.approx(
        cal.median_pairwise_distance_x * cal.lengthscale_multiplier
    )
    assert all(b > 0 for b in cal.B)
    assert all(b > 0 for b in cal.B_slice)
    assert cal.R == 0.0 and cal.lam == 1.0e-6
    # A smooth linear oracle is trivially GP-learnable.
    assert cal.r2_ok
    assert all(r2 > 0.9 for r2 in cal.r2_holdout)


def test_p6_norm_fit_regularization_branches():
    deterministic = calibrate("deterministic", smooth_oracle, x_dim=2, m=2)
    stochastic = calibrate("stochastic", smooth_oracle, x_dim=2, m=2, R=0.2)
    assert deterministic.lam == pytest.approx(1.0e-6)
    assert stochastic.lam == pytest.approx(0.2**2)
    # The stochastic norm fit uses the same noise-matched lambda and is
    # less interpolation-driven than the deterministic 1e-6 branch.
    assert np.all(np.asarray(stochastic.B) < np.asarray(deterministic.B))
    assert np.all(np.asarray(stochastic.B_slice) < np.asarray(deterministic.B_slice))


def test_calibrate_crash_convention_and_roundtrip(tmp_path):
    cal = calibrate("unit-test-crash", crashing_oracle, x_dim=2, m=2)
    assert cal.n_crashed > 0
    assert cal.crash_margin == 3.0
    path = tmp_path / "frozen.json"
    save_calibration(cal, path)
    loaded = load_calibration(path)
    assert loaded == cal
