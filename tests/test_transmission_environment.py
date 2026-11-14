"""Tests for the grid-loadability v2 window-margin environment (P2)."""

import json

import numpy as np
import pytest

from brink.studies.transmission import (
    TransmissionEnvironment,
    load_environment,
)
from brink.window_margins import WindowMarginOracle


def make_env(window_oracle=None):
    return TransmissionEnvironment(
        scaler_scale=np.array([0.2, 0.4]),
        B=np.array([1.0, 1.0]),
        B_slice=np.array([1.0, 1.0]),
        window_oracle=window_oracle,
    )


def test_oracle_scale_only_unclipped_and_affine():
    # Window [0.2, 0.8] independent of x.
    oracle = WindowMarginOracle(lambda x, c: 0.2 <= c <= 0.8)
    env = make_env(window_oracle=oracle)
    g_mid = env.oracle(np.full(4, 0.5), 0.5)
    # g_lo = (0.2 - 0.5)/0.2, g_hi = (0.5 - 0.8)/0.4 within edge tolerance.
    np.testing.assert_allclose(g_mid, [-1.5, -0.75], atol=0.01)
    # Exact affinity in c after standardization: slopes -1/scale, +1/scale.
    g2 = env.oracle(np.full(4, 0.5), 0.75)
    np.testing.assert_allclose(g2 - g_mid, [-0.25 / 0.2, 0.25 / 0.4], atol=1e-12)
    # Values can exceed +-3 standardized (unclipped by design).
    g_far = env.oracle(np.full(4, 0.5), 1.0)
    assert g_far[0] < -3.0


def test_oracle_rejects_noise():
    env = make_env(WindowMarginOracle(lambda x, c: True))
    with pytest.raises(ValueError):
        env.oracle(np.zeros(4), 0.5, None, 0.1)


def test_window_counters_exposed():
    oracle = WindowMarginOracle(lambda x, c: c <= 0.6)
    env = make_env(window_oracle=oracle)
    env.oracle(np.zeros(4), 0.3)
    counters = env.window_counters()
    assert counters["n_calls"] == 1
    assert counters["n_raw_evals"] >= 9
    assert env.window_counters()["raw_evals_per_call"] > 0


def test_load_rejects_registration_mismatch(tmp_path):
    payload = {
        "environment": {
            "gen_rows": [0, 5, 6],
            "shunt_bus": 99,
            "c_range": [1.0, 1.35],
            "oracle_representation": "window-margin",
            "m": 2,
        },
        "calibration": {
            "scaler_scale": [0.2, 0.4],
            "B": [1.0, 1.0],
            "B_slice": [1.0, 1.0],
        },
    }
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="shunt_bus"):
        load_environment(path)


def test_load_roundtrip(tmp_path):
    payload = {
        "environment": {
            "gen_rows": [0, 5, 6],
            "shunt_bus": 24,
            "c_range": [1.0, 1.35],
            "oracle_representation": "window-margin",
            "m": 2,
        },
        "calibration": {
            "scaler_scale": [0.2, 0.4],
            "B": [2.0, 3.0],
            "B_slice": [1.5, 2.5],
        },
        "truth": {
            "v_min": 0.263,
            "c_star_unit": 0.4500737190246582,
            "beta_hat": 0.9472724478856274,
            "v_grid": [1.0, 0.5, 0.0],
            "v_grid_c": [0.0, 0.5, 1.0],
        },
    }
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload))
    env = load_environment(path)
    np.testing.assert_allclose(env.scaler_scale, [0.2, 0.4])
