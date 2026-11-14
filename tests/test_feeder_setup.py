"""Check for the S3 truth helper (witness mean-margin decision)."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "s3", ROOT / "scripts" / "feeder_setup.py"
    )
    m = importlib.util.module_from_spec(spec)
    cwd = Path.cwd()
    try:  # the T07 runner it imports reads its frozen config by relative path
        os.chdir(ROOT)
        spec.loader.exec_module(m)
    finally:
        os.chdir(cwd)
    return m


def test_margin_check_separates_feasible_borderline_and_infeasible():
    check = _module().margin_check
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((4096, 2)) * 0.1
    clear = check(noise + [-0.05, -0.05])
    assert clear["feasible"] and clear["clear_3se"]
    edge = check(noise + [-0.05, -0.002])  # mean below 0, within 3 se (se ~ 0.0016)
    assert edge["feasible"] and not edge["clear_3se"]
    assert not check(noise + [-0.05, 0.01])["feasible"]
    assert np.allclose(clear["se"], 0.1 / 64, rtol=0.05)


def test_s3_config_matches_the_registration():
    from brink.quantile_bound import required_sample_size

    cfg = _module().load_s3_config(ROOT / "configs" / "feeder.yaml")
    cols = cfg["columns"]
    assert cols["n"] == required_sample_size(0.01, cols["delta_order"], k=3) == 628
    assert cols["eps"] == [0.01, 0.05] and cols["clip"] == [-1.0, 2.0]
    assert cfg["budgets_raw"] == [5000, 17000]
    assert cfg["seeds"]["stop"] - cfg["seeds"]["start"] == 50
    assert [a["label"] for a in cfg["arms"]] == [
        "witness-focused", "greedy", "config",
        "baseline-k16", "baseline-k64", "baseline-k256",
    ]  # fmt: skip
    assert all(a.get("snapshots", 16) % cfg["k_snapshots"] == 0 for a in cfg["arms"])
    assert len(cfg["sha256"]) == 64


def test_design_draw_and_ladder_placement():
    he = _module()
    X = he.draw_designs([998, 0], 628)
    assert X.shape == (628, 4) and X.min() >= 0.0 and X.max() <= 1.0
    assert np.array_equal(X, he.draw_designs([998, 0], 628))
    assert not np.array_equal(X[:20], he.draw_designs([998, 9], 20))
    assert he.ladder_start(0.1287, 0.01) == 0.1287
    assert he.ladder_start(float("-inf"), 0.01) == -0.01  # first rung at c = 0
    assert he.ladder_start(float("inf"), 0.01) == 0.99  # first rung at c_max


def test_monotone_flags_detects_a_decrease_beyond_pooled_se():
    flags = _module().monotone_flags
    means = np.array([-0.13, -0.07, -0.03, 0.04, 0.10])
    ses = np.full(5, 0.009)
    flagged, min_z = flags(means, ses, 3.0)
    assert not flagged and min_z > 3.0
    dip = means.copy()
    dip[3] = -0.08  # a 0.05 drop against pooled se 0.0127
    flagged, min_z = flags(dip, ses, 3.0)
    assert flagged and min_z < -3.0
    assert flags(np.array([0.0, 0.0]), np.zeros(2), 3.0) == (False, 0.0)


def test_summarize_scores_applies_the_proposition_per_eps():
    he = _module()
    cols = {"eps": [0.01, 0.05], "delta_order": 0.05}
    s = np.arange(628.0)
    out = he.summarize_scores(s, cols)
    assert out["n"] == 628 and out["n_inf"] == 0
    assert out["by_eps"]["0.01"] == {"U_eps": 625.0, "k": 3, "finite": True}
    assert out["by_eps"]["0.05"]["k"] == 23
    s[:3] = np.inf
    assert not he.summarize_scores(s, cols)["by_eps"]["0.01"]["finite"]
    assert he.summarize_scores(s, cols)["by_eps"]["0.05"]["finite"]


def test_score_column_wiring_on_the_real_oracle_with_tiny_snapshots():
    """Streams, ladder placement and the row layout, at a budget that cannot fire."""
    he = _module()
    cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        he._set_truth_snapshots(8)
        cols = {
            "rough_index_base": 3_000_000, "ladder_step": 0.01, "ladder_rungs": 3,
            "test_snapshots": 16, "clip": [-1.0, 2.0],
        }  # fmt: skip
        row = he.score_column((5, [0.978, 0.951, 0.871, 0.875], cols, 1e-5, [998, 2]))
    finally:
        he._set_truth_snapshots(he.TRUTH_SNAPSHOTS)
        os.chdir(cwd)
    assert row["index"] == 5 and row["n_tests"] == 3 and row["score"] == np.inf
    start = he.ladder_start(row["rough_death"], 0.01)
    assert [r["c"] for r in row["rungs"]] == pytest.approx(
        [min(start + 0.01 * j, 1.0) for j in (1, 2, 3)]
    )
    assert all(not r["fired"] and r["t"] > 1.0 for r in row["rungs"])  # 16 snapshots
    assert all(-1.0 <= r["mean"] <= 2.0 for r in row["rungs"])
