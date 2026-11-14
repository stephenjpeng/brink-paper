"""Checks for the S3 sweep script: config wiring, views, and the gate analysis."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from brink.bracket import SCAN_OFF

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        spec.loader.exec_module(m)
    finally:
        os.chdir(cwd)
    return m


@pytest.fixture(scope="module")
def sw():
    return _module("feeder_sweep")


@pytest.fixture(scope="module")
def cfg(sw):
    return sw.he.load_s3_config(ROOT / "configs" / "feeder.yaml")


def test_build_cfg_registers_scan_off_and_report_units(sw, cfg):
    win = sw.build_cfg(cfg, 17000, "window", 0.1173, 1.5457)
    pred = sw.build_cfg(cfg, 5000, "predicate", 0.1173, 1.5457)
    assert win.certificate_scan == pred.certificate_scan == SCAN_OFF
    assert (win.budget, win.report_every) == (17000, 10)
    assert (pred.budget, pred.report_every) == (5000, 170)
    assert win.lam == pytest.approx(0.1173**2) and win.variant == "noisy"
    assert win.cover_max_depth == 0  # the end scan sets its own depth


def test_task_lists_follow_the_registered_order(sw, cfg):
    tasks = sw.sweep_tasks(cfg)
    assert len(tasks) == 600 and [t[0] for t in tasks] == list(range(600))
    assert tasks[0][1:] == ("witness-focused", 5000, 500)
    assert tasks[50][1:] == ("witness-focused", 17000, 500)
    assert tasks[-1][1:] == ("baseline-k256", 17000, 549)


class _Raw:
    """Fake raw oracle recording the snapshot count of each predicate call."""

    def __init__(self):
        self.calls = []

    def mean_feasible(self, x, c, k, rng):
        self.calls.append(k)
        return c <= 0.5


def test_make_view_charges_snapshot_multiples(sw, cfg):
    raw = _Raw()
    spec = {"label": "baseline-k64", "view": "predicate", "snapshots": 64}
    view = sw.make_view(spec, env=None, raw_oracle=raw, k_snapshots=16)
    rng = np.random.default_rng(0)
    assert view.oracle(np.zeros(4), 0.2, rng)[0] <= 0
    assert view.oracle(np.zeros(4), 0.7, rng)[0] > 0
    assert view.n_raw_evals == 8 and raw.calls == [64, 64]
    assert view.log[0][2] and not view.log[1][2]
    with pytest.raises(ValueError, match="multiple"):
        sw.make_view({"view": "predicate", "snapshots": 24}, None, raw, 16)
    assert sw.best_x_from_log(view.log, 0.2) == [0.0] * 4
    assert sw.best_x_from_log(view.log, 0.7) is None


def _rows(label, view, budget, n, *, violate=0, no_witness=0, wrong=0, lower=0.08):
    rows = []
    for s in range(n):
        has_w = s >= no_witness
        rows.append({
            "label": label, "view": view, "budget": budget, "seed": 500 + s,
            "raw_evals": budget, "wall_s": 10.0, "final_lower": lower if has_w else 0.0,
            "final_upper": 1.0, "has_witness": has_w,
            "first_witness_raw": 1000 if has_w else None,
            "witness_x": [0.5] * 4 if has_w else None, "theta": 0.1287,
            "theta_minus_lower": 0.1287 - (lower if has_w else 0.0),
            "anytime_lower_violation": s < violate, "final_lower_violation": False,
            "witness_truly_feasible": (s >= wrong) if has_w else None,
            "end_scan": {"scan": {"has_certificate": False, "capped": False}}
            if view == "window" else None,
        })  # fmt: skip
    return rows


def _scenario(cfg, **over):
    rows = []
    for a in cfg["arms"]:
        for b in cfg["budgets_raw"]:
            kw = dict(over.get(a["label"], {}))
            rows += _rows(a["label"], a["view"], b, 50, **kw)
    return rows


def test_analyze_rows_passes_a_clean_scenario_and_reads_the_hardened_arms(sw, cfg):
    rows = _scenario(
        cfg,
        **{
            "greedy": {"no_witness": 50},
            "baseline-k16": {"wrong": 50, "lower": 0.25},
            "baseline-k64": {"wrong": 40, "lower": 0.2},
            "baseline-k256": {"wrong": 5, "lower": 0.03},
        },
    )
    cols = {"by_eps": {"0.01": {"U_eps": 0.14, "k": 3, "finite": True},
                       "0.05": {"U_eps": 0.12, "k": 23, "finite": True}}}  # fmt: skip
    gate = sw.analyze_rows(rows, cfg, cols, {"passed": True})
    assert gate["verdict"] == {
        "G1": True, "G2": True, "G3": True, "G4": True,
        "lower_bound_claim": True, "exposure_claim": True, "two_sided_statement": True,
    }  # fmt: skip
    assert gate["G1"]["greedy/17000"]["witness_feasible_rate"] is None
    assert gate["hardened"]["baseline-k64/5000"]["reading"].startswith("also fails")
    assert "trades the answer" in gate["hardened"]["baseline-k256/5000"]["reading"]
    assert gate["cells"]["witness-focused/17000"]["witness_rate"] == 1.0
    assert gate["paired"]["greedy-vs-witness-focused/17000"]["losses"] == 50
    assert gate["G4"]["U_0.01"] == 0.14


def test_analyze_rows_fails_the_predeclared_counts(sw, cfg):
    rows = _scenario(cfg, **{
        "config": {"violate": 10}, "witness-focused": {"no_witness": 4},
        "baseline-k16": {"wrong": 31},
    })  # fmt: skip
    gate = sw.analyze_rows(rows, cfg)
    assert not gate["G1"]["pass"] and gate["G1"]["config/5000"]["violations"]["reject"]
    assert not gate["G2"]["pass"] and gate["G2"]["5000"]["count"] == 46
    assert not gate["G3"]["pass"] and gate["G3"]["17000"]["count"] == 31
    assert gate["G4"]["pass"] is None  # no column or monotone summary given
    assert "straw man" in gate["hardened"]["baseline-k256/5000"]["reading"]
    ok = _scenario(cfg, **{
        "config": {"violate": 9}, "witness-focused": {"no_witness": 3},
        "baseline-k16": {"wrong": 32},
    })  # fmt: skip
    g2 = sw.analyze_rows(ok, cfg)
    assert g2["G1"]["pass"] and g2["G2"]["pass"] and g2["G3"]["pass"]


def test_remaining_tasks_skips_rows_already_written(sw, cfg, tmp_path):
    rows = tmp_path / "rows.jsonl"
    done = (0, 5, 599)
    rows.write_text("".join(json.dumps({"task_index": i}) + "\n" for i in done))
    remaining, done = sw.remaining_tasks(cfg, rows)
    assert done == [0, 5, 599] and len(remaining) == 597
    assert [t[0] for t in remaining[:6]] == [1, 2, 3, 4, 6, 7]
    assert remaining[-1][1:] == ("baseline-k256", 17000, 548)
