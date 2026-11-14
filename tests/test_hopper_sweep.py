"""Tests for the S4 sweep script (scripts/hopper_sweep.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hopper_sweep as ms  # noqa: E402

from brink import bracket as br  # noqa: E402
from brink.arms import (  # noqa: E402
    ARM_DIRECT_SCORE,
    run_existence_arm,
)


@pytest.fixture(scope="module")
def cfg():
    return ms.me.load_s4_config()


def test_task_order_is_arm_major_then_budget_then_seed(cfg):
    tasks = ms.sweep_tasks(cfg)
    assert len(tasks) == 500
    assert [t[0] for t in tasks] == list(range(500))
    assert tasks[0] == (0, "witness-focused", 2000, 700)
    assert tasks[49] == (49, "witness-focused", 2000, 749)
    assert tasks[50] == (50, "witness-focused", 6000, 700)
    assert tasks[-1] == (499, "direct-score", 6000, 749)


def test_build_cfg_carries_the_registered_constants(cfg):
    window = ms.build_cfg(cfg, 6000, "window")
    assert window.mode == br.MODE_AFFINE and window.budget == 6000
    assert window.certificate_scan == br.SCAN_OFF
    assert window.variant == "deterministic" and window.R == 0.0
    assert window.report_every == 10
    assert window.lam == pytest.approx(1e-6)
    assert window.lengthscale == pytest.approx(1.3277678265153883)
    assert ms.build_cfg(cfg, 2000, "predicate").report_every == 170
    assert ms.build_cfg(cfg, 2000, "score").report_every == 170


class _FakeWindowEnv:
    """Window environment stub: c_off(x) = x[0] / 2, cost 9 + 8 rollouts."""

    x_dim, m = 3, 1

    def __init__(self):
        self.n = 0

    def _oracle(self):
        env = self

        class W:
            def window(self, x):
                env.n += 17
                return type("R", (), {"c_off": float(x[0]) / 2, "n_raw_evals": 17})()

        return W()


def test_window_score_view_charges_each_windows_rollouts():
    view = ms.WindowScoreView(_FakeWindowEnv())
    assert view.score(np.array([0.6, 0.1, 0.1])) == pytest.approx(0.3)
    assert view.n_raw_evals == 17
    cfg = ms.build_cfg(ms.me.load_s4_config(), 200, "score")
    view = ms.WindowScoreView(_FakeWindowEnv())
    res = run_existence_arm(view, ARM_DIRECT_SCORE, cfg, np.array([1.0]), 1)
    assert res.raw_evals == view.n_raw_evals >= 200
    assert res.raw_evals < 200 + 17
    best = max(s for _, s in view.log)
    assert res.final_lower == pytest.approx(best)
    assert res.final_witness_x == pytest.approx(
        next(x for x, s in view.log if s == best).tolist()
    )


def _row(label, budget, seed, *, witness=True, check=True, lower=0.3, mono=True):
    return {
        "label": label,
        "budget": budget,
        "seed": seed,
        "has_witness": witness,
        "final_lower": lower if witness else 0.0,
        "check_feasible": (check if witness else None),
        "reports_monotone": mono,
        "robust_fraction": 0.5 if witness else None,
    }


def _rows(cfg, overrides=None):
    rows = []
    for a in cfg["arms"]:
        for b in cfg["budgets_raw"]:
            for s in range(700, 750):
                lower = 0.45 if a["label"] == "sup-bisection" else 0.3
                rows.append(_row(a["label"], b, s, lower=lower))
    for i, kw in (overrides or {}).items():
        rows[i] = {**rows[i], **kw}
    return rows


def test_analyze_clean_scenario_passes_g1_g2_g4(cfg):
    columns = {"eps_0.01": {"U": 0.5, "k": 3}, "eps_0.05": {"U": 0.4, "k": 23}}
    out = ms.analyze_rows(_rows(cfg), cfg, columns=columns, desk={"n_flagged": 0})
    assert out["G1"]["pass"] and out["G2"]["pass"] and out["G4"]["pass"]
    assert out["t_ref"] == 0.45 and out["t_ref_source"].startswith("sup-bisection/")
    out2 = ms.analyze_rows(_rows(cfg), cfg, t_search=0.5, t_search_x=[0.1, 0.2, 0.3])
    assert (out2["t_ref"], out2["t_ref_x"]) == (0.5, [0.1, 0.2, 0.3])
    assert out2["t_ref_source"] == "truth search"
    cmp = out["search_comparison"]["6000"]["sup-bisection"]
    assert cmp["losses"] == 50 and cmp["p"] < 0.05  # arm 1 below sup-bisection


def test_analyze_g1_95_percent_rule_and_monotone_reports(cfg):
    two = {i: {"check_feasible": False} for i in (3, 7)}
    assert ms.analyze_rows(_rows(cfg, two), cfg)["G1"]["pass"]
    three = {i: {"check_feasible": False} for i in (3, 7, 9)}
    g1 = ms.analyze_rows(_rows(cfg, three), cfg)["G1"]
    assert not g1["pass"]
    assert g1["cells"]["witness-focused/2000"]["feasible_fraction"] == pytest.approx(
        47 / 50
    )
    ten = {i: {"check_feasible": False} for i in range(10)}
    cell = ms.analyze_rows(_rows(cfg, ten), cfg)["G1"]["cells"]["witness-focused/2000"]
    assert cell["failures"]["reject"]
    nine = {i: {"check_feasible": False} for i in range(9)}
    cell = ms.analyze_rows(_rows(cfg, nine), cfg)["G1"]["cells"]["witness-focused/2000"]
    assert not cell["failures"]["reject"]
    bad = ms.analyze_rows(_rows(cfg, {170: {"reports_monotone": False}}), cfg)["G1"]
    assert not bad["pass"] and bad["cells"]["greedy/6000"]["non_monotone"] == 1


def test_analyze_g2_threshold_is_47_of_50(cfg):
    three = {i: {"has_witness": False} for i in (1, 2, 3)}
    assert ms.analyze_rows(_rows(cfg, three), cfg)["G2"]["pass"]
    four = {i: {"has_witness": False} for i in (1, 2, 3, 4)}
    g2 = ms.analyze_rows(_rows(cfg, four), cfg)["G2"]
    assert not g2["pass"] and g2["cells"]["2000"]["count"] == 46


def test_analyze_g4_needs_informative_u_and_no_flags(cfg):
    good = {"eps_0.01": {"U": 0.5, "k": 3}}
    assert not ms.analyze_rows(_rows(cfg), cfg, good, {"n_flagged": 1})["G4"]["pass"]
    top = {"eps_0.01": {"U": 1.1, "k": 3}}
    assert not ms.analyze_rows(_rows(cfg), cfg, top, {"n_flagged": 0})["G4"]["pass"]
    assert ms.analyze_rows(_rows(cfg), cfg)["G4"]["pass"] is None
