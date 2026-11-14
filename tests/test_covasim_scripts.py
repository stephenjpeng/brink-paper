"""Tests for the S5 scripts: truth column, S5-S1 scores, config, sweep wiring."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import covasim_setup as ce  # noqa: E402
import covasim_sweep as cs  # noqa: E402

from brink.studies import covasim_policy as cp  # noqa: E402


def test_bisect_death_prefix_rules():
    d, n = ce.bisect_death(lambda c: c <= 0.37, 8)
    assert abs(d - 0.37) <= 0.5**9 + 1e-12 and n == 10
    assert ce.bisect_death(lambda c: False, 8) == (-np.inf, 1)
    assert ce.bisect_death(lambda c: True, 8) == (np.inf, 2)


def test_local_grid3_is_clipped_and_unique():
    g = ce.local_grid3(np.array([0.95, 0.5, 1.0]), 0.1)
    assert g.min() >= 0.0 and g.max() <= 1.0
    assert len(g) == 3 * 3 * 2  # the u3 axis collapses at the face


def test_clipped_test_rule_and_floor():
    g = np.full(256, 0.3)
    g[::2] = 0.25
    out = ce.clipped_test(g, [-1.0, 0.5], 1e-4)
    assert out["fired"] and out["t"] > 0.0
    assert not ce.clipped_test(-g, [-1.0, 0.5], 1e-4)["fired"]
    # clipping from above only lowers the mean
    big = np.full(256, 3.0)
    assert ce.clipped_test(big, [-1.0, 0.5], 1e-4)["mean"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        ce.clipped_test(np.full(8, -1.5), [-1.0, 0.5], 1e-4)


def test_mean_zero_bounded_margins_rarely_fire():
    rng = np.random.default_rng(0)
    fires = 0
    for _ in range(2000):  # right-skewed, mean exactly 0, in [-1, 2]
        g = rng.exponential(0.3, 64) - 0.3
        fires += ce.clipped_test(np.minimum(g, 2.0), [-1.0, 0.5], 0.01)["fired"]
    assert fires <= 20  # p = 0.01 per test


def test_score_column_on_a_synthetic_oracle(monkeypatch):
    death = 0.42  # expected margin 1.3 (c - death), noise sd 0.15

    def fake_margins(x, c, k, rng):
        return np.maximum(1.3 * (c - death) + 0.15 * rng.standard_normal(int(k)), -1.0)

    monkeypatch.setattr(cp, "margins", fake_margins)
    monkeypatch.setattr(
        cp, "mean_feasible", lambda x, c, k, rng: fake_margins(x, c, k, rng).mean() <= 0
    )
    cfg = ce.he.load_s3_config(ce.S5_CONFIG)
    row = ce.score_column(
        (0, [0.5, 0.5, 0.5], cfg["columns"], ce.column_p(cfg["columns"]))
    )  # noqa: E501
    assert death < row["score"] <= death + 0.3 and row["n_tests"] <= 8
    assert abs(row["rough_death"] - death) < 0.1


def test_config_matches_the_registration():
    cfg = ce.he.load_s3_config(ce.S5_CONFIG)
    assert [a["label"] for a in cfg["arms"]][:4] == [
        "witness-focused",
        "greedy",
        "config",
        "baseline-k1",
    ]
    assert [a.get("runs") for a in cfg["arms"][3:]] == [1, 4, 16]
    assert cfg["columns"]["n"] == 59 and cfg["columns"]["eps"] == [0.05, 0.10]
    assert (cfg["columns"]["test_runs"], cfg["columns"]["ladder_step"]) == (512, 0.04)
    assert cfg["budgets_raw"] == [1000, 3000] and cfg["truth_check"]["runs"] == 128
    assert ce.column_p(cfg["columns"]) == pytest.approx(0.05 / (59 * 8))
    tasks = cs.sweep_tasks(cfg)
    assert len(tasks) == (4 * 30 + 2 * 10) * 2 and tasks[0][1] == "witness-focused"
    assert {t[3] for t in tasks if t[1] == "greedy"} == set(range(600, 610))
    assert {t[3] for t in tasks if t[1] == "baseline-k1"} == set(range(600, 630))
    short = {
        **cfg,
        "arms": [
            {**a, "n_seeds": 10} if a["label"] == "greedy" else a for a in cfg["arms"]
        ],
    }
    greedy_seeds = {t[3] for t in cs.sweep_tasks(short) if t[1] == "greedy"}
    assert greedy_seeds == set(range(600, 610))
    assert [t[0] for t in tasks] == list(range(len(tasks)))


def test_hardened_view_charges_its_runs(monkeypatch):
    monkeypatch.setattr(cp, "mean_feasible", lambda x, c, k, rng: True)
    view = cs.make_view({"view": "predicate", "runs": 16}, None)
    view.oracle(np.zeros(3), 0.5, np.random.default_rng(0))
    assert view.n_raw_evals == 16 and view.log[0][2] is True


def test_analyze_replaces_g4_with_the_primary_eps():
    cfg = ce.he.load_s3_config(ce.S5_CONFIG)
    cols = {
        "by_eps": {
            "0.05": {"U_eps": 0.8, "k": 1, "finite": True},
            "0.1": {"U_eps": 0.75, "k": 2, "finite": True},
        }
    }
    gate = cs.analyze([], cfg, cols, {"passed": True})
    assert gate["G4"]["pass"] and gate["G4"]["U_primary"] == 0.8
    assert cs.analyze([], cfg, cols, {"passed": False})["G4"]["pass"] is False


def test_calibration_replay_returns_values_in_recipe_order():
    pts = [([0.0, 0.0, 0.0], 0.1), ([1.0, 1.0, 1.0], 0.2)]
    oracle = ce.replay([[0.5, 17], [0.7, 9]], pts)
    assert oracle(np.zeros(3), 0.1)[0] == 0.5 and oracle(np.ones(3), 0.2)[0] == 0.7
    with pytest.raises(RuntimeError):
        ce.replay([[0.5, 17]], pts[:1])(np.ones(3), 0.1)


def test_peak_rss_is_in_gib_on_this_platform():
    np.ones(10_000_000)  # 80 MB touched
    assert 0.05 < ce.peak_rss_gib() < 64.0


def test_sweep_parts_partition_every_cell():
    cfg = ce.he.load_s3_config(ce.S5_CONFIG)
    tasks = cs.sweep_tasks(cfg)
    a, b = cs.part_tasks(tasks, 0, 2), cs.part_tasks(tasks, 1, 2)
    assert sorted(a + b) == tasks and not set(a) & set(b)
    for lab in ("witness-focused", "greedy"):
        for bud in (1000, 3000):
            na = sum(t[1] == lab and t[2] == bud for t in a)
            nb = sum(t[1] == lab and t[2] == bud for t in b)
            assert abs(na - nb) <= 1
    with pytest.raises(ValueError):
        cs.part_tasks(tasks, 2, 2)


def test_merged_rows_need_every_task_exactly_once(tmp_path):
    import json

    cfg = ce.he.load_s3_config(ce.S5_CONFIG)
    tasks = cs.sweep_tasks(cfg)
    dirs = []
    for k in (0, 1):
        d = tmp_path / f"part{k}"
        d.mkdir()
        rows = [{"task_index": t[0]} for t in cs.part_tasks(tasks, k, 2)]
        (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        dirs.append(d)
    assert len(cs.merged_rows(cfg, dirs)) == len(tasks)
    with pytest.raises(SystemExit):
        cs.merged_rows(cfg, dirs[:1])
    with pytest.raises(SystemExit):
        cs.merged_rows(cfg, [dirs[0], dirs[0], dirs[1]])
