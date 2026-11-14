"""Checks for the transmission scripts: config, columns, views, and gate analysis."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from brink.bracket import SCAN_OFF
from brink.quantile_bound import required_sample_size

ROOT = Path(__file__).resolve().parents[1]
INF = float("inf")
THETA, ERR = 0.639168, 3.7e-4


@pytest.fixture(scope="module")
def sw():
    spec = importlib.util.spec_from_file_location(
        "transmission_sweep", ROOT / "scripts" / "transmission_sweep.py"
    )
    m = importlib.util.module_from_spec(spec)
    cwd = Path.cwd()
    try:  # imported runners read their frozen configs by relative path
        os.chdir(ROOT)
        spec.loader.exec_module(m)
    finally:
        os.chdir(cwd)
    return m


@pytest.fixture(scope="module")
def cfg(sw):
    return sw.load_cfg(ROOT / "configs" / "transmission.yaml")


def test_config_matches_the_registration(cfg):
    assert (cfg["seeds"]["start"], cfg["seeds"]["stop"]) == (600, 650)
    assert cfg["budgets_raw"] == [5000, 17000]
    assert [(a["name"], a["view"]) for a in cfg["arms"]] == [
        ("existence-witness-focused", "window"),
        ("existence-bracket-greedy", "window"),
        ("joint-config-existence", "window"),
        ("uncertified-sup-bisection", "predicate"),
        ("uncertified-direct-score", "score"),
    ]
    cols = cfg["columns"]
    assert cols["n"] == required_sample_size(0.01, 0.05, k=3) == 628
    assert cols["score_pad"] == 2.0**-13 == cfg["monotone_check"]["tol"]
    assert cols["design_stream"] == [997, 0]
    assert cfg["monotone_check"]["stream"] == [997, 8]
    assert cfg["arm_constants"]["end_scan"]["channels"] == [1]  # upper window only
    assert cfg["arm_constants"]["end_scan"]["second_order"] is True
    assert cfg["arm_constants"]["end_scan"]["cap_s"] == 2200  # amendment S2-A3
    t = cfg["truth"]
    assert t["theta_exists_unit"] + t["err_unit"] == pytest.approx(0.639538)
    assert t["theta_exists_unit"] - t["err_unit"] == pytest.approx(0.638798)


def test_build_cfg_registers_scan_off_and_report_units(sw, cfg):
    win = sw.build_cfg(cfg, 17000, "window", 1.5457)
    raw = sw.build_cfg(cfg, 5000, "score", 1.5457)
    assert win.certificate_scan == raw.certificate_scan == SCAN_OFF
    assert (win.budget, win.report_every) == (17000, 10)
    assert (raw.budget, raw.report_every) == (5000, 170)
    assert (win.lam, win.R, win.variant) == (1e-6, 0.0, "deterministic")


def test_task_lists_follow_the_registered_order(sw, cfg, tmp_path):
    tasks = sw.hs.sweep_tasks(cfg)
    assert len(tasks) == 500 and [t[0] for t in tasks] == list(range(500))
    assert tasks[0][1:] == ("witness-focused", 5000, 600)
    assert tasks[50][1:] == ("witness-focused", 17000, 600)
    assert tasks[-1][1:] == ("direct-score", 17000, 649)
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"task_index": i}) + "\n" for i in (0, 499)))
    remaining, done = sw.hs.remaining_tasks(cfg, rows)
    assert done == [0, 499] and len(remaining) == 498


def test_column_score_follows_construction_s2_s1(sw):
    pad = 2.0**-13
    assert sw.ts.column_score(0.0, INF, 0, pad) == 1.0  # feasible at c = 1
    assert sw.ts.column_score(0.0, 0.5, 0, pad) == 0.5 + pad
    assert sw.ts.column_score(0.2, 1.0 - pad / 2, 0, pad) == 1.0  # capped at c_max
    assert sw.ts.column_score(0.1, 0.5, 1, pad) == INF  # multi-window scan
    assert sw.ts.column_score(INF, -INF, 0, pad) == INF  # no window found


def test_window_pattern_flag_applies_the_s2_m1_rule(sw):
    c, i = np.linspace(0.0, 1.0, 201), np.arange(201)  # c[i] = i / 200
    tol = 2.0**-13
    ok = sw.ts.window_pattern_flag((i >= 20) & (i <= 120), c, 0.1, 0.6, tol)
    assert not ok["flagged"] and ok["contiguous"]
    gap = (i <= 120) & ~((i > 60) & (i < 64))
    assert sw.ts.window_pattern_flag(gap, c, 0.0, 0.6, tol)["flagged"]
    late = sw.ts.window_pattern_flag(i <= 130, c, 0.0, 0.6, tol)
    assert late["flagged"] and late["n_outside_feasible"] == 10
    early = sw.ts.window_pattern_flag(i <= 100, c, 0.0, 0.6, tol)
    assert early["flagged"] and early["n_inside_infeasible"] == 19
    assert not sw.ts.window_pattern_flag(i >= 80, c, 0.4, INF, tol)["flagged"]
    assert not sw.ts.window_pattern_flag(i < 0, c, INF, -INF, tol)["flagged"]
    assert sw.ts.window_pattern_flag(i == 100, c, INF, -INF, tol)["flagged"]


def test_designs_and_score_summary(sw, cfg):
    X = sw.he.draw_designs(cfg["columns"]["design_stream"], 628)
    assert X.shape == (628, 4) and X.min() >= 0.0 and X.max() < 1.0
    again = sw.he.draw_designs([997, 0], 628)
    assert sw.ts.designs_sha256(X) == sw.ts.designs_sha256(again)
    assert sw.ts.designs_sha256(X) != sw.ts.designs_sha256(
        sw.he.draw_designs([997, 9], 628)
    )
    scores = np.linspace(0.0, 0.6, 628)
    scores[:2] = INF
    s = sw.he.summarize_scores(scores, cfg["columns"])
    assert s["n_inf"] == 2 and s["by_eps"]["0.01"]["k"] == 3
    assert s["by_eps"]["0.01"]["finite"] and s["by_eps"]["0.05"]["k"] == 23
    scores[2] = INF
    assert not sw.he.summarize_scores(scores, cfg["columns"])["by_eps"]["0.01"][
        "finite"
    ]


class _Window:
    def __init__(self, c_off, n):
        self.c_off, self.n_raw_evals = c_off, n


class _Oracle:
    def __init__(self):
        self.n_raw_evals = 0
        self.answers = [(0.40, 17), (1.0, 9), (-0.25, 9)]

    def window(self, x):
        c_off, n = self.answers.pop(0)
        self.n_raw_evals += n
        return _Window(c_off, n)


class _Env:
    x_dim = 4

    def __init__(self):
        self.o = _Oracle()

    def _oracle(self):
        return self.o

    def window_counters(self):
        return {"n_raw_evals": self.o.n_raw_evals}


def test_window_score_view_reports_the_feasible_bracket_end(sw):
    view = sw.WindowScoreView(_Env())
    assert view.spec.x_dim == 4 and view.n_raw_evals == 0
    assert view.score(np.zeros(4)) == pytest.approx(0.40 - 0.125 * 0.5**8 / 2)
    assert view.n_raw_evals == 17
    assert view.score(np.zeros(4)) == 1.0  # feasible at c = 1
    assert view.score(np.zeros(4)) < 0.0 and view.n_raw_evals == 35  # no window


def test_witness_check_at_the_truth_corner(sw):
    x = [1.0, 1.0, 1.0, 0.0]
    inside = sw.witness_check(x, 0.639)
    assert inside["feasible"] and max(inside["margins"]) <= 0.0
    assert inside["death"] == pytest.approx(THETA, abs=2.0**-13)
    assert inside["start"] == pytest.approx(0.125816, abs=2.0**-13)  # delayed onset
    assert inside["multi_window"] == 0
    assert not sw.witness_check(x, 0.6396)["feasible"]


def _rows(
    label,
    view,
    budget,
    n,
    *,
    violate=0,
    no_witness=0,
    infeasible=0,
    lower=0.62,
    unsound=0,
    no_cert=0,
):
    rows = []
    for s in range(n):
        has_w = s >= no_witness
        low = lower if has_w else 0.0
        reports = [{"lower": 0.3, "upper": 1.0}, {"lower": low, "upper": low + 0.004}]
        if s < violate:
            reports[0]["lower"] = THETA + 2 * ERR
        row = {
            "task_index": s,
            "label": label,
            "view": view,
            "budget": budget,
            "seed": 600 + s,
            "raw_evals": budget,
            "wall_s": 10.0,
            "task_s": 12.0,
            "final_lower": low,
            "final_upper": 1.0,
            "has_witness": has_w,
            "first_witness_raw": 1000 if has_w else None,
            "witness_x": [0.9] * 4 if has_w else None,
            "witness_feasible": (s >= infeasible) if has_w else None,
            "truth_check": {"death": 0.63} if has_w else None,
            "reports": reports,
        }
        if view == "window":
            cert = s < n - no_cert
            up = THETA - 2 * ERR if s < unsound else 0.66
            row["end_scan"] = {
                "beta_sd_at_witness": [0.01, 0.02],
                "scan": {
                    "upper": up if cert else 1.0,
                    "has_certificate": cert,
                    "seconds": 30.0,
                    "capped": False,
                },
            }
        rows.append(row)
    return rows


def _scenario(cfg, **over):
    rows = []
    for a in cfg["arms"]:
        for b in cfg["budgets_raw"]:
            rows += _rows(a["label"], a["view"], b, 50, **over.get(a["label"], {}))
    return rows


def _truth(n_above: int = 0):
    death = np.full(6561, 0.5)
    death[:n_above] = 0.59  # above U_0.01 = 0.584
    return {"theta": THETA, "err": ERR, "grid_death": death}


COLS = {
    "by_eps": {
        "0.01": {"U_eps": 0.584, "k": 3, "finite": True},
        "0.05": {"U_eps": 0.541, "k": 23, "finite": True},
    }
}


def test_analyze_rows_passes_a_clean_scenario(sw, cfg):
    gate = sw.analyze_rows(_scenario(cfg), cfg, _truth(33), COLS, {"passed": True})
    assert gate["verdict"] == {
        "G1": True,
        "G2": True,
        "G3": True,
        "G4": {"5000": True, "17000": True},
        "lower_bound_claim": True,
        "volume_claim": True,
        "two_sided_bracket": {"5000": True, "17000": True},
    }
    c = gate["cells"]["greedy/17000"]
    assert c["end_scan_certificates"] == 50 and c["median_U"] == 0.66
    assert c["median_band_looseness"] == pytest.approx(0.01)
    assert c["median_search_gap"] == pytest.approx(THETA - 0.63)
    sup = gate["cells"]["sup-bisection/5000"]
    assert sup["bracket_contains_theta_final"] == 0
    assert gate["G3"]["grid_fraction_above_U_0.01"] == pytest.approx(33 / 6561)
    assert gate["paired"]["direct-score-vs-witness-focused/17000"]["ties"] == 50


def test_analyze_rows_fails_at_the_predeclared_counts(sw, cfg):
    bad = _scenario(
        cfg,
        **{
            "config": {"violate": 10},
            "witness-focused": {"no_witness": 4},
            "greedy": {"unsound": 10, "infeasible": 3},
        },
    )
    gate = sw.analyze_rows(bad, cfg, _truth(66), COLS, {"passed": True})
    assert gate["G1"]["config/5000"]["violations"]["reject"]
    assert gate["G1"]["greedy/5000"]["witness_feasible_rate"] == 0.94
    assert not gate["G1"]["pass"]
    assert not gate["G2"]["pass"] and gate["G2"]["5000"]["count"] == 46
    assert not gate["G3"]["pass"]  # 66 / 6561 > 0.01 of the truth grid above U_0.01
    assert gate["verdict"]["G4"] == {"5000": False, "17000": False}
    good = _scenario(
        cfg,
        **{
            "config": {"violate": 9},
            "witness-focused": {"no_witness": 3},
            "greedy": {"unsound": 9, "infeasible": 2, "no_cert": 3},
        },
    )
    g = sw.analyze_rows(good, cfg, _truth(65), COLS, {"passed": True})
    assert g["G1"]["pass"] and g["G2"]["pass"] and g["G3"]["pass"]
    assert g["verdict"]["G4"] == {"5000": True, "17000": True}
    few = _scenario(cfg, greedy={"no_cert": 4})
    assert sw.analyze_rows(few, cfg, _truth())["verdict"]["G4"]["5000"] is False


def test_analyze_rows_reads_the_median_bar_and_missing_columns(sw, cfg):
    low = _scenario(cfg, **{"witness-focused": {"lower": 0.5705}})
    gate = sw.analyze_rows(low, cfg, _truth())
    assert not gate["G2"]["pass"] and gate["G2"]["median_lower_top_budget"] == 0.5705
    assert gate["G3"]["pass"] is None  # no column or monotone summary given
    ok = _scenario(cfg, **{"witness-focused": {"lower": 0.5706}})
    assert sw.analyze_rows(ok, cfg, _truth())["G2"]["pass"]
    mono = sw.analyze_rows(ok, cfg, _truth(), COLS, {"passed": False})
    assert mono["G3"]["pass"] is False
