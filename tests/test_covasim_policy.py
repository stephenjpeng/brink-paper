"""Tests for the S5 Covasim policy environment."""

import numpy as np
import pytest

from brink.studies import covasim_policy as cp


def test_every_cube_point_is_an_affordable_section8_policy():
    U = np.random.default_rng(0).uniform(size=(5000, 3))
    U = np.concatenate([U, np.array(np.meshgrid(*[[0, 1]] * 3)).reshape(3, -1).T])
    for u in U:
        t, r, d = cp.to_policy(u)
        assert 0.0 <= t <= cp.TEST_MAX and 0.0 <= r <= cp.TRACE_MAX + 1e-12
        assert 5 <= d <= 60 and isinstance(d, int)
        assert cp.policy_cost(t, r, d) <= cp.COST_CAP + 1e-9


def test_full_share_faces_spend_the_budget():
    # u3 = 1 spends what testing and tracing leave (to a whole day).
    t, r, d = cp.to_policy(np.array([0.5, 0.5, 1.0]))
    assert cp.COST_CAP - 1.0 < cp.policy_cost(t, r, d) <= cp.COST_CAP
    # Full testing leaves 10, so full tracing is r = 0.5 and nothing is left.
    assert cp.to_policy(np.array([1.0, 1.0, 1.0])) == pytest.approx((0.3, 0.5, 60))
    assert cp.to_policy(np.zeros(3)) == (0.0, 0.0, 60)
    assert cp.beta_multiplier(0.0) == 0.8 and cp.beta_multiplier(1.0) == 4.0
    assert cp.capacity() == pytest.approx(60.0)


def test_simulation_margin_and_seeded_predicate():
    out = cp.simulate(np.array([1.0, 1.0, 1.0]), 0.0, 123)
    assert out["margin"] == pytest.approx((out["peak_critical"] - 60.0) / 60.0)
    a = cp.margins(np.array([1.0, 1.0, 1.0]), 0.0, 2, np.random.default_rng(5))
    b = cp.margins(np.array([1.0, 1.0, 1.0]), 0.0, 2, np.random.default_rng(5))
    assert np.array_equal(a, b) and a.shape == (2,) and np.all(a >= -1.0)


def _frozen(tmp_path, env=None):
    import json

    payload = {
        "environment": env or cp.environment_registration(),
        "calibration": {"scaler_scale": [0.25], "B": [3.0], "lengthscale": 0.7},
        "noise_scale": {"by_knob": {"1": {"R": 0.4}}},
    }
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload))
    return path


def test_frozen_environment_window_channel(tmp_path, monkeypatch):
    env = cp.load_environment(_frozen(tmp_path), 1)
    assert (env.R, env.lengthscale, env.m, env.x_dim) == (0.4, 0.7, 1, 3)
    monkeypatch.setattr(cp, "mean_feasible", lambda x, c, k, rng: c <= 0.5)
    y = env.oracle(np.full(3, 0.5), 0.2, np.random.default_rng(0))
    assert y.shape == (1,) and y[0] == pytest.approx((0.2 - 0.5) / 0.25, abs=0.01)
    assert env.window_counters()["n_raw_evals"] == 17
    with pytest.raises(ValueError):
        env.oracle(np.full(3, 0.5), 0.2, None)
    with pytest.raises(ValueError):
        cp.load_environment(_frozen(tmp_path), 4)
    with pytest.raises(ValueError):
        cp.load_environment(_frozen(tmp_path, {"pop_size": 1}), 1)


def test_sim_metadata_starts_no_git_subprocess(monkeypatch):
    import subprocess

    def refuse(*args, **kwargs):
        raise AssertionError("a Covasim run started a subprocess")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    cp.simulate(np.array([1.0, 1.0, 1.0]), 0.0, 7)
