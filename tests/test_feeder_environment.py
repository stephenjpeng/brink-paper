"""Tests for the hosting-capacity window-margin environment (Section 7)."""

import json

import numpy as np
import pytest

from brink.calibration import measure_noise_scale
from brink.studies.feeder import (
    PREFIX_NO_WINDOW_REPORT_GRID_STEP,
    SNAPSHOT_CRASH_MARGIN,
    HostingCapacityEnvironment,
    HostingCapacityOracle,
    _MeanPredicate,
    load_hosting_environment,
    select_pv_buses,
    verify_fast_path,
)
from brink.window_margins import WindowMarginOracle


@pytest.fixture(scope="module")
def raw_oracle():
    return HostingCapacityOracle()


def make_env(k=4, window_oracle=None):
    return HostingCapacityEnvironment(
        k_snapshots=k,
        pv_buses=[12, 1, 5, 8],
        prefix_only=False,
        prefix_no_window_report_grid_step=None,
        scaler_scale=np.array([0.2, 0.4]),
        B=np.array([1.0, 1.0]),
        B_slice=np.array([1.0, 1.0]),
        R_by_knob={16: 0.01, 4: 0.02, 1: 0.04},
        v_min=0.3,
        c_star=0.1,
        beta_hat=1.0,
        v_grid=np.array([0.75, 0.3, 0.0]),
        v_grid_c=np.array([0.0, 0.1, 1.0]),
        window_oracle=window_oracle,
    )


def test_registered_pv_bus_selection(raw_oracle):
    # The 4 load buses with highest total nominal load on CIGRE MV with DER.
    assert raw_oracle.pv_buses == [12, 1, 5, 8]
    assert select_pv_buses(raw_oracle.net) == [12, 1, 5, 8]


def test_oracle_rejects_wrong_bus_registration():
    with pytest.raises(ValueError, match="disagree"):
        HostingCapacityOracle(pv_buses=[1, 2, 3, 4])


def test_fast_path_matches_pandapower(raw_oracle):
    # Small-sample exact-equivalence check (the build stage runs 300).
    result = verify_fast_path(raw_oracle, n_samples=20, seed=555)
    assert result["n_convergence_flag_mismatches"] == 0
    assert result["max_margin_deviation"] < 1.0e-8


def test_snapshot_rng_reproducibility(raw_oracle):
    x = np.array([0.5, 0.25, 0.75, 0.1])
    g1 = raw_oracle.mean_margins(x, 0.2, 4, np.random.default_rng(99))
    g2 = raw_oracle.mean_margins(x, 0.2, 4, np.random.default_rng(99))
    g3 = raw_oracle.mean_margins(x, 0.2, 4, np.random.default_rng(100))
    np.testing.assert_array_equal(g1, g2)
    assert not np.array_equal(g1, g3)
    assert g1.shape == (2,)


def test_crash_margins_at_collapse(raw_oracle):
    # Far beyond the voltage-collapse boundary the snapshot must crash to
    # the (+3, +3) raw margins.
    g = raw_oracle.snapshot_margins(np.zeros(4), 8.0, np.random.default_rng(1))
    np.testing.assert_array_equal(g, np.full(2, SNAPSHOT_CRASH_MARGIN))


def test_env_oracle_standardizes_and_requires_rng():
    oracle = WindowMarginOracle(_StubPredicate())
    env = make_env(window_oracle=oracle)
    env._predicate = oracle.raw_feasible
    g = env.oracle(np.full(4, 0.5), 0.5, np.random.default_rng(0))
    np.testing.assert_allclose(g, [(0.2 - 0.5) / 0.2, (0.5 - 0.8) / 0.4], atol=0.01)
    with pytest.raises(ValueError, match="sigma"):
        env.oracle(np.zeros(4), 0.5, np.random.default_rng(0), 0.1)
    with pytest.raises(ValueError, match="RNG"):
        env.oracle(np.zeros(4), 0.5, None)


class _StubPredicate:
    """Deterministic stand-in predicate with the rebindable-RNG interface."""

    def __init__(self):
        self.rng = None

    def __call__(self, x, c):
        return 0.2 <= c <= 0.8


def test_mean_predicate_requires_rng(raw_oracle):
    pred = _MeanPredicate(raw_oracle, 1)
    with pytest.raises(ValueError, match="RNG"):
        pred(np.zeros(4), 0.0)


def test_raw_predicate_view_draws_snapshots_from_the_call_rng(raw_oracle):
    from brink.adapter import RawPredicateView

    view = RawPredicateView(
        lambda x, c, rng: raw_oracle.mean_feasible(x, c, 2, rng), 4, needs_rng=True
    )
    x = np.full(4, 0.5)
    for c in (0.0, 0.15, 0.3):
        rng = np.random.default_rng(5)
        got = view.oracle(x, c, rng)[0] <= 0.0
        assert got == raw_oracle.mean_feasible(x, c, 2, np.random.default_rng(5))
        assert rng.bit_generator.state != np.random.default_rng(5).bit_generator.state


def test_measure_noise_scale_known_sigma():
    # Synthetic replicated oracle: y = const + N(0, sigma^2) per channel.
    sigma = 0.05
    rng = np.random.default_rng(7)

    def rep(x, c):
        return np.array([1.0, 2.0]) + sigma * rng.standard_normal(2)

    record = measure_noise_scale(rep, x_dim=4, scale=np.ones(2))
    assert record["n_pairs"] == 100
    # R = 1.2 x sd(diff/sqrt(2)) estimates 1.2 x sigma.
    assert record["R"] == pytest.approx(1.2 * sigma, rel=0.25)
    assert record["lam"] == pytest.approx(record["R"] ** 2)
    assert len(record["per_channel_sd"]) == 2


def frozen_payload():
    return {
        "environment": {
            "network": (
                "pandapower.networks.create_cigre_network_mv(with_der='pv_wind')"
            ),
            "pv_buses": [12, 1, 5, 8],
            "c_range": [1.0, 2.0],
            "pv_size_range_mw": [0.0, 4.0],
            "sigma_ln": 0.35,
            "oracle_representation": "window-margin over the "
            "K-snapshot-mean feasibility predicate",
            "m": 2,
        },
        "calibration": {
            "scaler_scale": [0.2, 0.4],
            "B": [10.0, 20.0],
            "B_slice": [5.0, 8.0],
        },
        "noise_scale": {
            "by_knob": {
                "16": {"R": 0.01},
                "4": {"R": 0.02},
                "1": {"R": 0.04},
            }
        },
        "truth": {
            "v_min": 0.3,
            "c_star_unit": 0.1,
            "beta_hat": 1.0,
            "v_grid": [0.75, 0.3, 0.0],
            "v_grid_c": [0.0, 0.1, 1.0],
        },
    }


def test_load_p5_prefix_roundtrip(tmp_path):
    payload = frozen_payload()
    payload["environment"].update(
        {
            "prefix_only": True,
            "m": 1,
            "prefix_no_window_report_grid_step": (PREFIX_NO_WINDOW_REPORT_GRID_STEP),
        }
    )
    payload["calibration"].update(
        {"scaler_scale": [0.2], "B": [10.0], "B_slice": [5.0]}
    )
    path = tmp_path / "p5.json"
    path.write_text(json.dumps(payload))
    env = load_hosting_environment(path, k_snapshots=16)
    assert env.prefix_only
    assert env.m == 1
    assert env.prefix_no_window_report_grid_step == pytest.approx(0.005)


def test_load_roundtrip_and_knob_selection(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(frozen_payload()))
    env = load_hosting_environment(path, k_snapshots=4)
    assert env.k_snapshots == 4
    assert env.pv_buses == [12, 1, 5, 8]
    assert env.R_by_knob == {16: 0.01, 4: 0.02, 1: 0.04}
    assert env.v_min == 0.3
    assert env.c_star == 0.1
    assert env.true_volume(0.05) == pytest.approx(0.525)


def test_load_rejects_unknown_knob(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(frozen_payload()))
    with pytest.raises(ValueError, match="noise knob"):
        load_hosting_environment(path, k_snapshots=8)


def test_load_rejects_registration_mismatch(tmp_path):
    payload = frozen_payload()
    payload["environment"]["sigma_ln"] = 0.6
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sigma_ln"):
        load_hosting_environment(path, k_snapshots=4)
