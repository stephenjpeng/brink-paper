"""Tests for the search rules on the double integrator (exact limit, closed form)."""

from pathlib import Path

import numpy as np
import pytest

from brink.adapter import RawPredicateView, ScoreView, WindowExistenceView
from brink.arms import (
    ALL_EXISTENCE_ARMS,
    ARM_BRACKET_GREEDY,
    ARM_DIRECT_SCORE,
    ARM_SUP_BISECTION,
    ARM_WITNESS_FOCUSED,
    CERTIFIED_EXISTENCE_ARMS,
    ExistenceConfig,
    run_existence_arm,
)
from brink.bracket import MODE_AFFINE
from brink.studies.double_integrator import (
    THETA_EXISTS_UNIT,
    is_viable,
    load_double_integrator,
)

FROZEN = Path(__file__).resolve().parents[1] / "configs/frozen/double_integrator.json"
FAST = dict(
    budget=60,
    n_init=12,
    report_every=20,
    n_candidates=128,
    n_witness_candidates=256,
    report_grid_size=51,
    cover_n_per_axis=17,
)


@pytest.fixture(scope="module")
def env():
    return load_double_integrator(FROZEN)


def _cfg(env, **kw):
    return ExistenceConfig(
        mode=MODE_AFFINE, lengthscale=env.lengthscale, lam=1e-6, **{**FAST, **kw}
    )


def test_model_based_arms_contain_the_exact_limit(env):
    for arm in (*CERTIFIED_EXISTENCE_ARMS, ARM_WITNESS_FOCUSED):
        res = run_existence_arm(WindowExistenceView(env), arm, _cfg(env), env.B, 0)
        assert res.raw_evals == FAST["budget"]
        for r in res.reports:
            assert r.lower <= THETA_EXISTS_UNIT <= r.upper, arm
        if res.final_witness_x is not None:
            assert is_viable(np.asarray(res.final_witness_x), res.final_lower)[0]


def test_bracket_width_is_monotone_nonincreasing(env):
    res = run_existence_arm(
        WindowExistenceView(env), ARM_BRACKET_GREEDY, _cfg(env), env.B, seed=2
    )
    widths = [r.width for r in res.reports]
    assert all(w2 <= w1 + 1e-12 for w1, w2 in zip(widths, widths[1:], strict=False))


def test_determinism_same_seed(env):
    a = run_existence_arm(WindowExistenceView(env), ARM_BRACKET_GREEDY, _cfg(env),
                          env.B, seed=3)  # fmt: skip
    b = run_existence_arm(WindowExistenceView(env), ARM_BRACKET_GREEDY, _cfg(env),
                          env.B, seed=3)  # fmt: skip
    assert a.final_lower == b.final_lower and a.final_upper == b.final_upper


def test_baselines_respect_the_budget(env):
    cfg = _cfg(env, budget=200)
    res = run_existence_arm(RawPredicateView(env.raw_feasible, x_dim=env.x_dim),
                            ARM_SUP_BISECTION, cfg, env.B, seed=4)  # fmt: skip
    assert res.raw_evals <= cfg.budget and res.final_lower <= res.final_upper
    res = run_existence_arm(ScoreView(env.score_unit, x_dim=env.x_dim),
                            ARM_DIRECT_SCORE, cfg, env.B, seed=4)  # fmt: skip
    assert res.raw_evals <= cfg.budget and res.final_lower <= THETA_EXISTS_UNIT + 1e-9


def test_sup_bisection_never_exceeds_budget():
    """The c_max probe after a feasible incumbent probe is budget-checked too."""
    for budget in range(1, 25):
        view = RawPredicateView(lambda x, c: c <= 0.5, x_dim=2)  # feasible at c_min
        cfg = ExistenceConfig(mode=MODE_AFFINE, **{**FAST, "budget": budget})
        res = run_existence_arm(view, ARM_SUP_BISECTION, cfg, np.ones(1), seed=4)
        assert view.n_raw_evals == res.raw_evals <= budget, budget


class _CostlyView:
    """Each oracle call costs `cost` raw evaluations (window-oracle stand-in)."""

    def __init__(self, view, cost):
        self.view, self.spec, self.cost = view, view.spec, cost

    def oracle(self, x, c, rng=None, sigma=0.0):
        self.last_raw_evals = self.cost
        return self.view.oracle(x, c, rng, sigma)


def test_budget_counts_raw_evaluations(env):
    view = _CostlyView(WindowExistenceView(env), 2)
    res = run_existence_arm(view, ARM_BRACKET_GREEDY, _cfg(env), env.B, seed=0)
    assert res.raw_evals == FAST["budget"]  # 30 calls x 2 raw evaluations
    assert [r.calls for r in res.reports] == [20, 30]  # final off-schedule report
    assert [r.raw_evals for r in res.reports] == [40, 60]


def test_unknown_arm_name_raises(env):
    with pytest.raises(ValueError, match="unknown existence arm"):
        run_existence_arm(WindowExistenceView(env), "existence-bracket-greedy@x",
                          _cfg(env), env.B, 0)  # fmt: skip


class _SpyView:
    """Records every (x, c) the arm queries."""

    def __init__(self, view):
        self.view, self.spec, self.calls = view, view.spec, []

    def oracle(self, x, c, rng=None, sigma=0.0):
        self.calls.append((np.array(x), float(c)))
        return self.view.oracle(x, c, rng, sigma)


def test_witness_focused_arm_targets_the_stress_above_the_running_lower_bound(env):
    assert ARM_WITNESS_FOCUSED not in ALL_EXISTENCE_ARMS
    spy = _SpyView(WindowExistenceView(env))
    cfg = _cfg(env)
    res = run_existence_arm(spy, ARM_WITNESS_FOCUSED, cfg, env.B, seed=1)
    assert res.reports[-1].has_witness and res.final_lower > 0.0
    h_c = 1.0 / (cfg.report_grid_size - 1)
    lower_before = {r.calls + 1: r.lower for r in res.reports}  # L in force next call
    L = 0.0
    for i, (_, c) in enumerate(spy.calls[cfg.n_init :], start=cfg.n_init + 1):
        L = lower_before.get(i, L)
        assert abs(c - min(L + h_c, 1.0)) < 1e-12
    xs = [tuple(x) for x, _ in spy.calls]
    assert len(set(xs)) < len(xs)  # past designs are candidates: calls repeat


def test_path_out_records_queries_without_changing_the_run(env):
    a = run_existence_arm(WindowExistenceView(env), ARM_BRACKET_GREEDY, _cfg(env),
                          env.B, seed=5)  # fmt: skip
    path: dict = {}
    b = run_existence_arm(WindowExistenceView(env), ARM_BRACKET_GREEDY, _cfg(env),
                          env.B, seed=5, path_out=path)  # fmt: skip
    assert [(r.lower, r.upper) for r in a.reports] == [
        (r.lower, r.upper) for r in b.reports
    ]
    n = b.raw_evals
    assert path["X"].shape == (n, env.x_dim) and path["c"].shape == (n,)
    assert path["Y"].shape == (n, 1)
