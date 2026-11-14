"""Checks for the S7 sweep script: config wiring, budgets, task order, gates."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from brink.arms import ARM_SUP_BISECTION, ExistenceConfig
from brink.bracket import MODE_AFFINE

ROOT = Path(__file__).resolve().parents[1]
BOUND = 0.1375  # S3 theta + err


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
def vr():
    return _module("check_layer_sweep")


@pytest.fixture(scope="module")
def cfg(vr):
    return vr.load_config(ROOT / "configs" / "check_layer.yaml")


def test_config_matches_the_registration(vr, cfg):
    assert range(cfg["seeds"]["start"], cfg["seeds"]["stop"]) == range(900, 950)
    assert cfg["arms"] == ["witness-focused", "sup-bisection"]
    assert cfg["gate_check_stream_offset"] == 3000
    s3, s4 = cfg["s3"], cfg["s4"]
    assert vr.proposer_budget(s3) == 17000 - 12 * 256 == 13928
    assert vr.proposer_budget(s4) == 6000 - 64 == 5936
    assert (s3["layer"]["step0"], s3["layer"]["resolution"]) == (0.02, 0.005)
    assert (s4["layer"]["step0"], s4["layer"]["resolution"]) == (0.02, 0.0005)
    assert s3["layer"]["stream_prefix"] == 995
    assert s3["layer"]["check_snapshots"] == 4096
    g = cfg["gates"]
    assert (g["alpha"], g["g1a_tested_rate_null"], g["g1b_violation_delta"]) == (
        0.05, 0.85, 0.10,
    )  # fmt: skip
    assert (g["g1b_feasible_min"], g["g2_sign_null"]) == (0.95, 0.5)


def test_arm_labels_point_at_the_frozen_base_arms(cfg):
    names = {
        "witness-focused": "existence-witness-focused",
        "sup-bisection": "uncertified-sup-bisection",
    }
    for env in ("s3", "s4"):
        base = yaml.safe_load((ROOT / cfg[env]["base_config"]).read_text())
        assert base["arm_constants"]["blas_threads"] == 1  # the script's thread pin
        base_seeds = set(range(base["seeds"]["start"], base["seeds"]["stop"]))
        assert not base_seeds & set(range(900, 950))
        arms = {a["label"]: a for a in base["arms"]}
        assert {arms[lab]["name"] for lab in cfg[env]["arm_labels"].values()} == set(
            names.values()
        )
        for s7, lab in cfg[env]["arm_labels"].items():
            assert arms[lab]["name"] == names[s7]
    s3 = yaml.safe_load((ROOT / cfg["s3"]["base_config"]).read_text())
    k16 = next(a for a in s3["arms"] if a["label"] == "baseline-k16")
    assert k16["snapshots"] == s3["k_snapshots"] == 16  # one raw per decision
    lay = cfg["s3"]["layer"]
    assert lay["raw_per_check"] * s3["k_snapshots"] == lay["check_snapshots"]


def test_task_order(vr, cfg):
    t = vr.sweep_tasks(cfg, "s3")
    assert len(t) == 100 and [x[1] for x in t] == list(range(100))
    assert t[0] == ("s3", 0, "witness-focused", 900)
    assert t[50] == ("s3", 50, "sup-bisection", 900)
    assert t[-1] == ("s3", 99, "sup-bisection", 949)


def test_budget_accounting_on_a_stub_view(vr):
    hs = _module("feeder_sweep")

    def feasible(x, c):
        return bool(c <= 0.2 + 0.5 * x[0])

    view = hs.LoggedPredicateView(feasible, 2, raw_evals_per_call=3)
    e = {
        "budget_total_raw": 400,
        "layer": {"raw_per_check": 10, "n_max": 6, "step0": 0.02,
                  "resolution": 0.005, "c_min": 0.0, "c_max": 1.0},
    }  # fmt: skip
    budget = vr.proposer_budget(e)
    run_cfg = ExistenceConfig(
        mode=MODE_AFFINE, budget=budget, report_every=30, bisect_depth=8
    )
    calls = []

    def check(x, c):
        calls.append(c)
        return feasible(x, c)

    row = vr.propose_and_verify(
        view, ARM_SUP_BISECTION, run_cfg, np.ones(1), 0, check, e, hs.best_x_from_log
    )
    assert budget == row["budget"] == 340 and row["raw_evals"] <= 340
    assert row["layer_raw"] == 10 * len(calls) <= row["reserve_raw"] == 60
    assert row["adjust"]["n_checks"] == len(calls) >= 1
    assert row["budget_total"] == 400 and vr.budget_ok(row)
    assert row["proposer_lower"] == row["final_lower"]
    assert feasible(row["witness_x"], row["final_lower"])
    assert feasible(row["witness_x"], row["tested_lower"])
    assert not vr.budget_ok({**row, "layer_raw": 70})
    assert not vr.budget_ok({**row, "raw_evals": 341})
    assert vr.budget_ok({**row, "view": "window", "raw_evals": 341})  # call cap


def _row(label, seed, tested, feasible=True):
    gate = None
    if tested is not None:
        gate = {"feasible": feasible, "clear_3se": feasible, "eb_feasible": feasible,
                "c_off_replicates": [0.11] * 4, "robust_fraction": 1.0}  # fmt: skip
    return {
        "task_index": seed - 900 + 50 * (label == "sup-bisection"), "label": label,
        "seed": seed, "view": "window" if label == "witness-focused" else "predicate",
        "budget": 13928, "budget_total": 17000, "reserve_raw": 3072,
        "raw_evals": 13928, "layer_raw": 1024, "final_lower": 0.085,
        "proposer_lower": 0.085, "tested_lower": tested,
        "adjust": {"n_checks": 4, "layer_raw": 1024},
        "gate_check": gate, "task_s": 600.0, "peak_rss_gib": 0.9,
    }  # fmt: skip


def _rows(untested=0, exceed=0, wins=40, infeasible=0):
    rows = []
    for i, s in enumerate(range(900, 950)):
        lt = 0.10
        if i < untested:
            lt = None
        elif i < untested + exceed:
            lt = 0.2  # above theta + err, gate check feasible
        rows.append(_row("witness-focused", s, lt, feasible=i < 50 - infeasible))
        rows.append(_row("sup-bisection", s, 0.09 if i < wins else 0.11))
    return rows


def test_clean_rows_pass_every_gate(vr, cfg):
    out = vr.analyze_rows(_rows(), cfg, "s3", bound=BOUND)
    assert out["verdict"] == {"G1": True, "G2": True}
    assert out["G2"]["wins"] == 40


@pytest.mark.parametrize(
    ("kwargs", "gate", "passes"),
    [
        ({"untested": 3}, "G1", True),
        ({"untested": 4}, "G1", False),  # 46 of 50 tested designs
        ({"exceed": 9}, "G1", True),
        ({"exceed": 10}, "G1", False),  # 10 of 50 violating runs
        ({"infeasible": 3}, "G1", False),  # 47 of 50 feasible < 95 percent
        ({"wins": 32}, "G2", True),
        ({"wins": 31}, "G2", False),  # 31 of 50 pairs
    ],
)
def test_gates_fail_at_exactly_the_predeclared_counts(vr, cfg, kwargs, gate, passes):
    out = vr.analyze_rows(_rows(**kwargs), cfg, "s3", bound=BOUND)
    assert out["verdict"][gate] is passes


def test_untested_arm_one_runs_count_against_it_in_g2(vr, cfg):
    out = vr.analyze_rows(_rows(untested=3, wins=34), cfg, "s3", bound=BOUND)
    assert out["G2"]["wins"] == 31 and out["verdict"]["G2"] is False


def test_s4_has_no_g2_gate_and_budget_violations_stop_the_analysis(vr, cfg):
    out = vr.analyze_rows(_rows(), cfg, "s4", t_ref=0.625)
    assert out["verdict"]["G2"] is None and out["verdict"]["G1"] is True
    bad = _rows()
    bad[0]["layer_raw"] = 4000
    out = vr.analyze_rows(bad, cfg, "s3", bound=BOUND)
    assert out["stopped"] and out["excluded"] == [bad[0]["task_index"]]
    assert "verdict" not in out
