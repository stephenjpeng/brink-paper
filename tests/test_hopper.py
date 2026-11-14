"""Tests for the S4 MuJoCo Hopper oracle."""

import io
import sys
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import pytest

from brink.studies import hopper as mh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hopper_setup import (  # noqa: E402
    grid_score,
    top_edge,
)

needs_policy = pytest.mark.skipif(
    not mh.POLICY_ZIP.exists(), reason="policy zip not downloaded (docs/data.md)"
)


def _compiled(*args):
    model = mujoco.MjModel.from_xml_string(mh.hopper_xml(*args))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_nominal_xml_reproduces_stock_geometry_and_masses():
    stock = mujoco.MjModel.from_xml_path(str(mh._stock_xml_path()))
    mod, _ = _compiled(1.0, 1.0, 1.0, 0.0)
    np.testing.assert_allclose(mod.body_mass, stock.body_mass)
    np.testing.assert_allclose(mod.body_pos, stock.body_pos, atol=1e-12)
    np.testing.assert_allclose(mod.jnt_pos, stock.jnt_pos, atol=1e-12)
    np.testing.assert_allclose(mod.qpos0, stock.qpos0)
    for name in ("torso_geom", "thigh_geom", "leg_geom", "foot_geom"):
        np.testing.assert_allclose(mod.geom(name).pos, stock.geom(name).pos, atol=1e-12)
        np.testing.assert_allclose(mod.geom(name).size, stock.geom(name).size)


@pytest.mark.parametrize("leg", [0.6, 1.4])
def test_leg_scale_keeps_the_chain_connected_and_foot_height(leg):
    model, data = _compiled(leg, 1.3, 1.0, 0.0)
    z = {n: data.xpos[model.body(n).id][2] for n in ("torso", "thigh", "leg")}
    knee = data.xanchor[model.joint("leg_joint").id][2]
    ankle = data.xanchor[model.joint("foot_joint").id][2]
    assert z["thigh"] - knee == pytest.approx(0.45 * leg)
    assert knee - ankle == pytest.approx(0.5 * leg)
    assert ankle == pytest.approx(0.1)
    assert data.geom_xpos[model.geom("foot_geom").id][2] == pytest.approx(0.1)
    assert data.qpos[1] == pytest.approx(z["torso"])  # rootz = absolute height


def test_foot_scale_scales_the_foot_capsule_about_the_ankle():
    model, data = _compiled(1.0, 1.4, 1.0, 0.0)
    ankle_x = data.xanchor[model.joint("foot_joint").id][0]
    centre_x = data.geom_xpos[model.geom("foot_geom").id][0]
    assert centre_x - ankle_x == pytest.approx(0.065 * 1.4)
    assert model.geom("foot_geom").size[1] == pytest.approx(0.195 * 1.4)


def test_torso_mass_scale_and_payload_add_up():
    nominal, _ = _compiled(1.0, 1.0, 1.0, 0.0)
    model, _ = _compiled(1.0, 1.0, 1.6, 7.5)
    torso = model.body("torso").id
    assert model.body_mass[torso] == pytest.approx(1.6 * nominal.body_mass[torso] + 7.5)


def test_to_physical_maps_the_unit_box():
    assert mh.to_physical(np.zeros(3), 0.0) == (0.6, 0.6, 0.6, 0.0)
    assert mh.to_physical(np.ones(3), 1.0) == pytest.approx((1.4, 1.4, 1.6, 20.0))


def test_crashed_rollout_is_infeasible():
    r = mh.Rollout(float("nan"), float("nan"), True)
    assert np.all(r.margins() > 0)
    ok = mh.Rollout(3.6, 0.71, False).margins()
    assert np.all(ok <= 0)


@needs_policy
def test_numpy_actor_matches_sb3_policy():
    import torch
    from gymnasium import spaces
    from stable_baselines3.sac.policies import SACPolicy

    pol = SACPolicy(
        spaces.Box(-np.inf, np.inf, (11,), np.float64),
        spaces.Box(-1, 1, (3,), np.float32),
        lambda _: 0.0,
    )
    blob = zipfile.ZipFile(mh.POLICY_ZIP).read("policy.pth")
    pol.load_state_dict(
        torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    )
    obs = np.random.default_rng(0).normal(size=(200, 11)) * 2
    a_sb3, _ = pol.predict(obs, deterministic=True)
    np.testing.assert_allclose(mh.actor_action(mh.load_actor(), obs), a_sb3, atol=1e-4)


@needs_policy
def test_rollout_is_bit_deterministic():
    oracle = mh.HopperOracle()
    x = np.array([0.5, 0.5, 0.4])
    runs = {oracle.rollout(x, 0.1) for _ in range(3)}
    assert len(runs) == 1


def test_top_edge_bisects_the_last_feasible_scan_cell():
    lo, hi, n, runs = top_edge(lambda c: c <= 0.3137)
    assert lo <= 0.3137 < hi
    assert hi - lo == pytest.approx((1 / 32) / 2**10)
    assert (n, runs) == (43, 1)


def test_top_edge_finds_an_island_on_the_scan_and_counts_runs():
    lo, hi, _, runs = top_edge(lambda c: c <= 0.2 or 0.5 <= c <= 0.51)
    assert lo <= 0.51 < hi and lo > 0.5
    assert runs == 2


def test_top_edge_never_and_always_feasible():
    assert top_edge(lambda c: False) == (-np.inf, -np.inf, 33, 0)
    assert top_edge(lambda c: True) == (1.0, np.inf, 33, 1)


def test_grid_score_adds_the_margin_to_the_last_feasible_point():
    grid = np.linspace(0, 1, 11)
    feas = np.zeros(11, bool)
    assert grid_score(feas, grid, 0.05) == -np.inf
    feas[[0, 1, 4]] = True
    assert grid_score(feas, grid, 0.05) == pytest.approx(0.45)


@needs_policy
def test_reset_seed_changes_the_initial_state_and_outcome():
    w = mh.load_actor()
    xml = mh.hopper_xml(1.0, 1.0, 1.0, 0.0)
    assert mh.rollout(xml, w) != mh.rollout(xml, w, reset_seed=7)


def test_desk_flag_rule():
    from hopper_setup import REPORT_GRID, desk_flag

    n = REPORT_GRID.size
    grid = np.arange(n) <= 40  # last feasible grid point 0.2; s = 0.3 at margin 0.1
    off = np.zeros(n - 1, bool)
    fine_grid = np.linspace(0.195, 0.35, 156)
    flag, excess = desk_flag(grid, off, fine_grid <= 0.29, fine_grid, 0.1)
    assert not flag and excess == pytest.approx(0.09)
    flag, excess = desk_flag(grid, off, fine_grid <= 0.31, fine_grid, 0.1)
    assert flag and excess == pytest.approx(0.11)
    off_hi = off.copy()
    off_hi[70] = True  # offset point 0.3525 > s
    assert desk_flag(grid, off_hi, fine_grid <= 0.2, fine_grid, 0.1)[0]
    never = np.zeros(n, bool)
    empty = np.zeros(0, bool), np.zeros(0)
    assert desk_flag(never, off, *empty, 0.1) == (False, -np.inf)
    off_any = off.copy()
    off_any[3] = True
    assert desk_flag(never, off_any, *empty, 0.1)[0]


def test_column_summary_reads_the_order_statistic():
    from hopper_setup import column_summary

    scores = np.full(628, -np.inf)
    scores[:5] = [0.5, 0.45, 0.42, 0.3, 0.2]
    out = column_summary(scores, [0.01, 0.05], 0.05)
    assert out["eps_0.01"]["k"] == 3
    assert out["eps_0.01"]["U"] == pytest.approx(0.42)
    assert out["eps_0.05"]["k"] == 23
    assert out["eps_0.05"]["U"] == -np.inf
    assert out["n_no_feasible_grid_point"] == 623


class _FakeRaw:
    """Synthetic column: feasible on [0, 0.2] and on an off-grid island."""

    def feasible(self, x, c):
        return c <= 0.2 or 0.3105 <= c <= 0.3115  # only the 0.001 scan sees it


def test_desk_task_wiring_finds_an_island_above_the_margin(monkeypatch):
    import hopper_setup as me

    monkeypatch.setattr(me, "_RAW", _FakeRaw())
    grid, off, fine_grid, fine = me._desk_task((0.5, 0.5, 0.5, 0.1, 0.001, 0.005, 0.05))
    assert grid.size == 201 and off.size == 200
    assert fine_grid[0] == pytest.approx(0.195) and fine_grid[-1] == pytest.approx(0.35)
    flag, excess = me.desk_flag(grid, off, fine, fine_grid, 0.1)
    assert flag and excess == pytest.approx(0.111)
    flag, _ = me.desk_flag(grid, off, fine, fine_grid, 0.12)
    assert not flag


def test_score_columns_and_volume_curve(monkeypatch):
    import hopper_setup as me

    class Pool:
        map = staticmethod(lambda f, it, chunksize=1: map(f, it))

    monkeypatch.setattr(me, "_RAW", _FakeRaw())
    feas, scores = me.score_columns(Pool(), np.full((3, 3), 0.5), 0.1)
    assert feas.shape == (3, 201)
    np.testing.assert_allclose(scores, 0.3)  # island missed by the grid; +margin
    vc = me.volume_curve(feas)
    assert vc["v_hat"][0] == 1.0 and vc["v_hat"][-1] == 0.0
    assert vc["ci_low"][0] < 1.0 <= vc["ci_high"][0]


def test_s4_config_matches_the_registration():
    import hopper_setup as me

    cfg = me.load_s4_config()
    assert cfg["budgets_raw"] == [2000, 6000]
    assert (cfg["seeds"]["start"], cfg["seeds"]["stop"]) == (700, 750)
    assert [a["name"] for a in cfg["arms"]] == [
        "existence-witness-focused",
        "existence-bracket-greedy",
        "joint-config-existence",
        "uncertified-sup-bisection",
        "uncertified-direct-score",
    ]
    assert cfg["policy_zip_sha256"] == mh.POLICY_ZIP_SHA256
    assert cfg["c_range_kg"] == list(mh.C_RANGE_KG)
    cc = cfg["columns"]
    assert (cc["n"], cc["margin"], cc["eps"]) == (628, 0.1, [0.01, 0.05])
    from brink.quantile_bound import required_sample_size

    assert required_sample_size(0.01, cc["delta_order"], k=3) == cc["n"]
    X = me.draw_designs(cfg)
    assert X.shape == (628, 3) and np.array_equal(X, me.draw_designs(cfg))
    widths = cfg["truth"]["refine_half_widths"]
    assert widths == pytest.approx([1 / 14, 1 / 56, 1 / 224])


def test_refine_points_keeps_rows_and_clips_to_the_box():
    from hopper_setup import refine_points

    top = np.array([[0.0, 0.5, 1.0], [0.5, 0.5, 0.5]])
    P = refine_points(top, 1 / 14)
    assert P.ndim == 2 and P.shape[1] == 3
    assert P.shape[0] == 3 * 5 * 3 + 125  # clipped faces collapse duplicate rows
    assert P.min() >= 0.0 and P.max() <= 1.0
    assert len(np.unique(P, axis=0)) == len(P)


class _InlinePool:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @staticmethod
    def map(f, it, chunksize=1):
        return map(f, it)


def test_truth_and_column_commands_end_to_end(monkeypatch, tmp_path):
    import json

    import hopper_setup as me

    cfg = me.load_s4_config()
    cfg["truth"]["grid_axis"] = 3
    cfg["columns"]["n"] = 12

    class Peak:  # death 0.6 - |x - 0.3|_1 / 2, highest at x = 0.3
        def feasible(self, x, c):
            return c <= 0.6 - 0.5 * float(np.abs(np.asarray(x) - 0.3).sum())

    monkeypatch.setattr(me, "load_s4_config", lambda: cfg)
    monkeypatch.setattr(me, "ProcessPoolExecutor", _InlinePool)
    monkeypatch.setattr(me, "_RAW", Peak())
    monkeypatch.setattr(me, "_manifest", lambda *a, **k: {"stub": True})
    monkeypatch.setattr(me, "_config_sha256", lambda: "x")
    dirs = iter(tmp_path / n for n in ("truth", "designs", "columns"))

    def new_dir(name):
        d = next(dirs)
        d.mkdir()
        return d

    monkeypatch.setattr(me, "_new_dir", new_dir)
    me.cmd_truth(1)
    truth = json.loads((tmp_path / "truth" / "truth.json").read_text())
    assert len(truth["stages"]) == 4
    grid_max = truth["stages"][0]["max"]
    assert grid_max == pytest.approx(0.3, abs=1e-3)  # best grid point (0.5, 0.5, 0.5)
    assert grid_max < truth["t_search_unit"] <= 0.6  # refinement moves toward 0.3
    me.cmd_draw_columns()
    me.cmd_score_columns(tmp_path / "designs", 1)
    cols = json.loads((tmp_path / "columns" / "columns.json").read_text())
    assert cols["n"] == 12 and "eps_0.01" in cols


def test_witness_diagnostic_classes_and_task(monkeypatch):
    import hopper_setup as me

    assert me.witness_class(0.30, 0.31) == "gap"
    assert me.witness_class(0.30, 0.30) == "gap"
    assert me.witness_class(0.30, 0.25) == "band-overreach"

    class Win:
        def window(self, x):
            return type("W", (), {"c_on": 0.0, "c_off": 0.3115})()

    monkeypatch.setattr(me, "_RAW", _FakeRaw())  # feasible on [0, 0.2] and an island
    monkeypatch.setattr(me, "_WIN", Win())
    out = me._diag_task((0.5, 0.5, 0.5, 0.25))
    assert out["check_feasible"] is False
    assert out["c_off_window"] == pytest.approx(0.3115)
    assert out["grid_top"] == pytest.approx(0.2)
    assert out["class"] == "gap"


def test_witness_diagnostic_command_end_to_end(monkeypatch, tmp_path):
    import json

    import hopper_setup as me

    class Win:
        def window(self, x):
            return type("W", (), {"c_on": 0.0, "c_off": 0.2})()

    rows = [
        {"task_index": i, "label": "greedy", "budget": 2000, "seed": 700 + i,
         "view": "window", "has_witness": True, "check_feasible": i % 3 != 0,
         "witness_x": [0.5, 0.5, 0.5], "final_lower": 0.1 if i % 3 else 0.25}
        for i in range(9)
    ]  # fmt: skip
    run = tmp_path / "run"
    run.mkdir()
    (run / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(me, "ProcessPoolExecutor", _InlinePool)
    monkeypatch.setattr(me, "_RAW", _FakeRaw())
    monkeypatch.setattr(me, "_WIN", Win())
    monkeypatch.setattr(me, "_manifest", lambda *a, **k: {"stub": True})
    monkeypatch.setattr(me, "_new_dir", lambda name: out)
    me.cmd_witness_diagnostic(run, 1)
    s = json.loads((out / "diagnostic.json").read_text())["summary"]
    assert s["fail"]["n"] == s["pass"]["n"] == 3
    assert s["fail"]["class_band_overreach"] == 3  # c_off 0.2 < L 0.25
    assert s["pass"]["class_gap"] == 3 and s["pass"]["recheck_feasible"] == 3
