"""Checks for the the check layer verification layer."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from brink.verify import verify_and_adjust

ROOT = Path(__file__).resolve().parents[1]
LAYER = {"c_min": 0.0, "c_max": 1.0, "step0": 0.02, "resolution": 0.005}
X = np.zeros(4)


def _death_check(death: float):
    calls: list[float] = []

    def check(x, c):
        calls.append(c)
        return c <= death

    return check, calls


def test_pass_at_lower_steps_up_with_doubling_increments():
    check, _ = _death_check(0.3)
    res = verify_and_adjust(X, 0.1, check, n_max=12, **LAYER)
    cs = [c for c, _ in res.trace]
    assert cs[:5] == pytest.approx([0.1, 0.12, 0.16, 0.24, 0.40])
    assert [ok for _, ok in res.trace[:5]] == [True, True, True, True, False]
    assert 0.3 - 0.005 <= res.tested_lower <= 0.3


def test_fail_at_lower_steps_down_to_the_floor_like_the_naive_baseline():
    # Section 3 indicative path: fail at 0.25, 0.23, 0.19, 0.11, pass at 0.
    check, _ = _death_check(0.106)
    res = verify_and_adjust(X, 0.251, check, n_max=12, **LAYER)
    cs = [c for c, _ in res.trace]
    assert cs[:5] == pytest.approx([0.251, 0.231, 0.191, 0.111, 0.0])
    assert [ok for _, ok in res.trace[:5]] == [False, False, False, False, True]
    assert res.n_checks == 10  # five bisections from [0, 0.111] to <= 0.005
    assert 0.106 - 0.005 <= res.tested_lower <= 0.106


def test_reserve_is_never_exceeded_and_the_trace_is_complete():
    rng = np.random.default_rng(0)
    for _ in range(500):
        death, lower = rng.uniform(-0.1, 1.1), rng.uniform(0.0, 1.0)
        n_max = int(rng.integers(1, 15))
        check, calls = _death_check(death)
        res = verify_and_adjust(X, lower, check, n_max=n_max, **LAYER)
        assert res.n_checks == len(res.trace) == len(calls) <= n_max
        passes = [c for c, ok in res.trace if ok]
        assert res.tested_lower == (max(passes) if passes else None)
        assert all(LAYER["c_min"] <= c <= LAYER["c_max"] for c in calls)


def test_floor_failure_returns_no_tested_design():
    check, _ = _death_check(-0.01)
    res = verify_and_adjust(X, 0.3, check, n_max=12, **LAYER)
    assert res.tested_lower is None
    assert res.trace[-1] == (0.0, False)


def test_ceiling_pass_stops_at_c_max():
    check, _ = _death_check(2.0)
    res = verify_and_adjust(X, 0.9, check, n_max=12, **LAYER)
    assert res.tested_lower == 1.0
    assert res.trace == [(0.9, True), (0.92, True), (0.96, True), (1.0, True)]


def test_resolution_stops_before_the_reserve_and_lands_within_it():
    for death in (0.0137, 0.4, 0.777):
        check, _ = _death_check(death)
        res = verify_and_adjust(X, 0.2, check, n_max=100, **LAYER)
        assert res.n_checks < 100
        assert death - LAYER["resolution"] <= res.tested_lower <= death
        fails = [c for c, ok in res.trace if not ok]
        assert min(fails) - res.tested_lower <= LAYER["resolution"]


def test_three_se_rule_on_gaussian_noise_passes_below_and_fails_above():
    # S3 numbers (Section 0): se 0.004 at 4,096 snapshots, slope 0.85 per unit c.
    spec = importlib.util.spec_from_file_location(
        "he", ROOT / "scripts" / "feeder_setup.py"
    )
    he = importlib.util.module_from_spec(spec)
    cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        spec.loader.exec_module(he)
    finally:
        os.chdir(cwd)
    rng = np.random.default_rng(1)
    death, slope, sd = 0.11, 0.85, 0.004 * np.sqrt(4096)

    def check(x, c):
        g = rng.normal(slope * (c - death), sd, size=(4096, 2))
        return he.margin_check(g)["clear_3se"]

    def tested(lower):
        res = verify_and_adjust(X, lower, check, n_max=1, **LAYER)
        return res.tested_lower is not None

    below = sum(tested(death - 0.03) for _ in range(2000))
    above = sum(tested(death + 0.03) for _ in range(2000))
    assert below >= 0.99 * 2000
    assert above == 0
