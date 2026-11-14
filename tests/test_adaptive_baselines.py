"""Tests for the adaptive uncertified baselines."""

import numpy as np
import pytest

from brink.adapter import ExistenceSpec
from brink.adaptive_baselines import (
    ADAPTIVE_ARMS,
    ARM_DIFF_EVO,
    ARM_GP_UCB,
    MAX_RAW_PER_DESIGN,
    NO_WINDOW_SCORE,
    death_score,
    run_adaptive_baseline,
)
from brink.arms import ExistenceConfig
from brink.bracket import MODE_AFFINE

RESOLUTION = 0.125 * 0.5**8  # 9-point scan, then 8 bisection steps


class WindowPredicate:
    """Deterministic window predicate: feasible iff c_on <= c <= c_off(x)."""

    def __init__(self, x_dim, c_off, c_on=0.1):
        self.spec = ExistenceSpec(x_dim=x_dim, m=1)
        self.raw_evals_per_call = 1
        self.c_off, self.c_on = c_off, c_on
        self.n_calls = 0

    def feasible(self, x, c):
        return bool(self.c_on <= c <= self.c_off(np.asarray(x, dtype=float)))

    def oracle(self, x, c, rng=None, sigma=0.0):
        self.n_calls += 1
        return np.array([-1.0 if self.feasible(x, c) else 1.0])


def _peak(center, top=0.9, curv=4.0):
    center = np.asarray(center, dtype=float)
    return lambda x: top - curv * float(np.sum((x - center) ** 2))


def _cfg(budget, report_every=170):
    return ExistenceConfig(
        mode=MODE_AFFINE, budget=budget, n_init=20, report_every=report_every,
        n_candidates=512, delta=0.1,
    )  # fmt: skip


def _counting(pred):
    calls = []

    def feasible(c):
        calls.append(c)
        return pred(c)

    return feasible, calls


def test_death_score_finds_death_of_delayed_onset_window():
    # Infeasible at c = 0: a rule that bisects up from c_min would miss it.
    feasible, calls = _counting(lambda c: 0.1 <= c <= 0.63)
    s = death_score(feasible)
    assert 0.63 - RESOLUTION < s <= 0.63
    assert feasible(s)  # the reported point was observed feasible
    assert len(calls) - 1 <= MAX_RAW_PER_DESIGN == 17


def test_death_score_no_window_and_full_range():
    feasible, calls = _counting(lambda c: False)
    assert death_score(feasible) == NO_WINDOW_SCORE < 0.0
    assert len(calls) == 9
    assert death_score(lambda c: True) == 1.0


@pytest.mark.parametrize("arm", ADAPTIVE_ARMS)
def test_budget_reports_and_reported_design(arm):
    view = WindowPredicate(3, _peak([0.3, 0.7, 0.5]))
    res, info = run_adaptive_baseline(view, arm, _cfg(900, report_every=100), seed=3)
    assert res.raw_evals == view.n_calls
    assert 900 - MAX_RAW_PER_DESIGN < res.raw_evals <= 900
    assert [r.raw_evals for r in res.reports] == list(range(100, 901, 100))
    lowers = [r.lower for r in res.reports]
    assert lowers == sorted(lowers)
    assert not any(r.has_certificate for r in res.reports)
    assert res.final_lower == lowers[-1]
    x = np.asarray(res.final_witness_x)
    assert view.feasible(x, res.final_lower)  # sound on a deterministic predicate
    assert res.final_lower <= 0.9
    assert info["n_designs"] >= 900 // MAX_RAW_PER_DESIGN


@pytest.mark.parametrize("arm", ADAPTIVE_ARMS)
def test_deterministic_given_seed(arm):
    runs = [
        run_adaptive_baseline(
            WindowPredicate(2, _peak([0.2, 0.8])), arm, _cfg(400), seed=5
        )[0]
        for _ in range(2)
    ]
    assert runs[0].final_lower == runs[1].final_lower
    assert runs[0].final_witness_x == runs[1].final_witness_x


def _uniform_design_lower(view, budget, seed):
    rng = np.random.default_rng([seed, 99])
    best, used = NO_WINDOW_SCORE, 0
    while used + MAX_RAW_PER_DESIGN <= budget:
        x = rng.uniform(size=view.spec.x_dim)
        feasible, calls = _counting(lambda c, x=x: view.feasible(x, c))
        best = max(best, death_score(feasible))
        used += len(calls)
    return best


def test_gp_ucb_beats_uniform_designs_on_peaked_score():
    view = WindowPredicate(3, _peak([0.62, 0.35, 0.8]))
    for seed in (0, 1):
        res, _ = run_adaptive_baseline(view, ARM_GP_UCB, _cfg(700), seed=seed)
        assert res.final_lower > 0.88
        assert res.final_lower > _uniform_design_lower(view, 700, seed)


def test_diff_evo_improves_on_its_initial_population():
    view = WindowPredicate(2, _peak([0.62, 0.35], curv=8.0))
    res, info = run_adaptive_baseline(view, ARM_DIFF_EVO, _cfg(5100), seed=0)
    init_best = info["best_after_first_population"]
    assert res.final_lower > init_best
    assert res.final_lower > 0.89


def test_diff_evo_restarts_when_the_population_converges():
    # Every design scores 1.0, so the population spread is zero at once.
    view = WindowPredicate(2, lambda x: 2.0, c_on=0.0)
    res, info = run_adaptive_baseline(view, ARM_DIFF_EVO, _cfg(1200), seed=0)
    assert info["n_restarts"] >= 1
    assert res.raw_evals > 1200 - MAX_RAW_PER_DESIGN
    assert res.final_lower == 1.0


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        run_adaptive_baseline(
            WindowPredicate(2, _peak([0.5, 0.5])), "nope", _cfg(100), seed=0
        )
