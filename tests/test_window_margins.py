"""Tests for the window-margin oracle construction."""

import sys
from pathlib import Path

import numpy as np
import pytest

from brink.calibration import standardize
from brink.window_margins import (
    BISECT_STEPS,
    MAX_RAW_EVALS,
    N_SCAN,
    NO_WINDOW_C_OFF,
    NO_WINDOW_C_ON,
    WindowMarginOracle,
    extract_window,
    window_channels,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Bisection over one scan gap of width 1/8 with 8 steps resolves the edge
# to (1/8) / 2^8 ~ 4.9e-4; allow 1e-3.
EDGE_TOL = 1.0e-3


def interval_predicate(lo: float, hi: float):
    """Feasibility predicate: feasible iff c in [lo, hi]."""
    return lambda c: lo <= c <= hi


class CountingPredicate:
    """Wraps a predicate and counts raw evaluations."""

    def __init__(self, fn):
        self.fn = fn
        self.count = 0

    def __call__(self, c):
        self.count += 1
        return self.fn(c)


def test_interior_window():
    w = extract_window(interval_predicate(0.3, 0.7))
    assert abs(w.c_on - 0.3) < EDGE_TOL
    assert abs(w.c_off - 0.7) < EDGE_TOL
    assert not w.multi_window
    assert w.n_raw_evals == N_SCAN + 2 * BISECT_STEPS


def test_full_range_window():
    w = extract_window(lambda c: True)
    assert w.c_on == 0.0
    assert w.c_off == 1.0
    assert w.n_raw_evals == N_SCAN
    assert not w.multi_window


def test_no_window_constants():
    w = extract_window(lambda c: False)
    assert w.c_on == NO_WINDOW_C_ON == 1.25
    assert w.c_off == NO_WINDOW_C_OFF == -0.25
    assert w.n_raw_evals == N_SCAN
    assert not w.multi_window
    # No-window channels are infeasible at every c in [0, 1].
    for c in (0.0, 0.5, 1.0):
        g = window_channels(w.c_on, w.c_off, c)
        assert (g > 0.0).all()


def test_prefix_window():
    w = extract_window(interval_predicate(0.0, 0.6))
    assert w.c_on == 0.0
    assert abs(w.c_off - 0.6) < EDGE_TOL
    assert w.n_raw_evals == N_SCAN + BISECT_STEPS


def test_suffix_window():
    w = extract_window(interval_predicate(0.4, 1.0))
    assert abs(w.c_on - 0.4) < EDGE_TOL
    assert w.c_off == 1.0
    assert w.n_raw_evals == N_SCAN + BISECT_STEPS


def test_multi_window_keeps_widest():
    def feasible(c):
        return 0.11 <= c <= 0.14 or 0.35 <= c <= 0.9

    w = extract_window(feasible)
    assert w.multi_window
    assert abs(w.c_on - 0.35) < EDGE_TOL
    assert abs(w.c_off - 0.9) < EDGE_TOL


def test_window_narrower_than_scan_spacing_missed():
    # A window between scan points is invisible to the 9-point
    # scan: the construction returns the no-window constants (recorded
    # limitation of the construction).
    w = extract_window(interval_predicate(0.3, 0.35))
    assert w.c_on == NO_WINDOW_C_ON
    assert w.c_off == NO_WINDOW_C_OFF


def test_boundary_window_at_scan_point():
    # Window exactly one scan point wide.
    w = extract_window(interval_predicate(0.25, 0.25))
    assert abs(w.c_on - 0.25) < EDGE_TOL
    assert abs(w.c_off - 0.25) < EDGE_TOL


def test_raw_evaluation_cap():
    rng = np.random.default_rng(7)
    for _ in range(200):
        lo = float(rng.uniform(0.0, 1.0))
        hi = float(rng.uniform(lo, 1.0))
        pred = CountingPredicate(interval_predicate(lo, hi))
        w = extract_window(pred)
        assert pred.count == w.n_raw_evals
        assert w.n_raw_evals <= MAX_RAW_EVALS


def test_channels_exactly_affine_in_c():
    w = extract_window(interval_predicate(0.3, 0.7))
    cs = np.linspace(0.0, 1.0, 11)
    G = np.array([window_channels(w.c_on, w.c_off, float(c)) for c in cs])
    # g_lo has slope -1 and g_hi has slope +1, exactly.
    np.testing.assert_allclose(np.diff(G[:, 0]) / np.diff(cs), -1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.diff(G[:, 1]) / np.diff(cs), 1.0, rtol=0, atol=1e-12)
    # Feasibility matches the window membership at the report points.
    for c, g in zip(cs, G, strict=True):
        assert ((g <= 0.0).all()) == (w.c_on <= c <= w.c_off)


def test_p5_prefix_no_window_clamp_one_report_step():
    step = 1.0 / 200.0
    oracle = WindowMarginOracle(
        lambda x, c: False,
        prefix_only=True,
        prefix_no_window_report_grid_step=step,
    )
    g0 = oracle.raw_channels(np.zeros(1), 0.0)
    assert g0.shape == (1,)
    assert g0[0] == pytest.approx(step)
    # The m=2 default remains unchanged.
    w = extract_window(lambda c: False)
    assert w.c_off == NO_WINDOW_C_OFF


def test_p5_clamp_requires_prefix_mode_and_positive_step():
    with pytest.raises(ValueError, match="prefix_only"):
        WindowMarginOracle(
            lambda x, c: False,
            prefix_no_window_report_grid_step=0.005,
        )
    with pytest.raises(ValueError, match="positive"):
        WindowMarginOracle(
            lambda x, c: False,
            prefix_only=True,
            prefix_no_window_report_grid_step=0.0,
        )


def test_prefix_mode_single_channel():
    oracle = WindowMarginOracle(
        lambda x, c: c <= 0.5 + 0.2 * float(x[0]), prefix_only=True
    )
    g = oracle.raw_channels(np.array([0.0]), 0.25)
    assert g.shape == (1,)
    assert oracle.m == 1
    # g_hi = c - c_off; feasible below the frontier.
    assert g[0] < 0.0
    g2 = oracle.raw_channels(np.array([0.0]), 0.9)
    assert g2[0] > 0.0


def test_oracle_counters_and_multi_window_sidecar():
    def raw_feasible(x, c):
        return 0.11 <= c <= 0.14 or 0.35 <= c <= 0.9

    oracle = WindowMarginOracle(raw_feasible)
    assert oracle.m == 2
    for _ in range(3):
        oracle.raw_channels(np.zeros(2), 0.5)
    counters = oracle.counters()
    assert counters["n_calls"] == 3
    assert counters["n_multi_window"] == 3
    assert counters["n_raw_evals"] == 3 * (N_SCAN + 2 * BISECT_STEPS)
    assert counters["raw_evals_per_call"] == pytest.approx(N_SCAN + 2 * BISECT_STEPS)


def test_standardize_unclipped_window_channels():
    Y_raw = np.array([[1.25, -1.0], [-0.5, 0.75]])
    crashed = np.zeros(2, dtype=bool)
    scale = np.array([0.05, 0.1])
    Y = standardize(Y_raw, crashed, scale, 3.0, clip=False)
    # Values beyond +-3 survive (no clipping; affinity preserved).
    np.testing.assert_allclose(Y, [[25.0, -10.0], [-10.0, 7.5]])
    Y_clipped = standardize(Y_raw, crashed, scale, 3.0)
    assert Y_clipped.max() == 3.0
