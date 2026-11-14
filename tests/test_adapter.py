"""Tests for the portfolio-to-existence adapter (raw-evaluation accounting)."""

import numpy as np
import pytest

from brink.adapter import RawPredicateView, WindowExistenceView
from brink.arms import (
    ARM_BRACKET_GREEDY,
    ARM_SUP_BISECTION,
    ExistenceConfig,
    run_existence_arm,
)
from brink.bracket import MODE_AFFINE
from brink.window_margins import WindowMarginOracle

FAST = dict(
    budget=400,
    n_init=8,
    report_every=5,
    n_candidates=128,
    n_witness_candidates=128,
    report_grid_size=51,
    cover_n_per_axis=9,
)


def _feasible(x, c):
    return c <= 0.3 + 0.4 * float(x[0])  # theta_exists = 0.7 at x0 = 1


class _WindowEnv:
    """Minimal portfolio window environment over an analytic predicate."""

    x_dim, m = 2, 2

    def __init__(self):
        self.window_oracle = WindowMarginOracle(_feasible, prefix_only=False)

    def oracle(self, x, c, rng=None, sigma=0.0):
        return self.window_oracle.raw_channels(np.asarray(x, dtype=float), float(c))

    def window_counters(self):
        return self.window_oracle.counters()


def test_window_view_counts_raw_evaluations_per_call():
    env = _WindowEnv()
    view = WindowExistenceView(env)
    assert (view.spec.c_min, view.spec.c_max, view.spec.x_dim) == (0.0, 1.0, 2)
    cfg = ExistenceConfig(mode=MODE_AFFINE, lengthscale=0.5, lam=1e-6, **FAST)
    res = run_existence_arm(view, ARM_BRACKET_GREEDY, cfg, np.array([5.0, 5.0]), 0)
    counters = env.window_counters()
    assert res.raw_evals == counters["n_raw_evals"]  # exact, not 1 per call
    assert cfg.budget <= res.raw_evals < cfg.budget + 32  # cap 32 per call
    assert res.reports[-1].calls == counters["n_calls"] < res.raw_evals
    raws = [r.raw_evals for r in res.reports]
    assert raws == sorted(raws) and raws[-1] == res.raw_evals


def test_portfolio_norm_fit_uses_the_existence_affine_kernel():
    """S2 kernel match: frozen portfolio B is a norm under the arms' kernel."""
    from brink.bracket import fit_joint_bands
    from brink.calibration import _interpolant_norm

    rng = np.random.default_rng(3)
    X, c = rng.uniform(size=(40, 4)), rng.uniform(size=40)
    y = np.sin(3 * X[:, 0]) - c * X[:, 1]
    bands = fit_joint_bands(
        X, c, y[:, None], mode=MODE_AFFINE, lengthscale=1.5, lam=1e-6,
        B=np.array([1.0]), R=0.0, betas=[1.0],
    )  # fmt: skip
    # lipschitz = (||mu||_H + beta) * sqrt(1 + c_max^2) / l, so recover ||mu||_H.
    mu_norm = bands.lipschitz(0) * 1.5 / np.sqrt(2.0) - bands.betas[0]
    assert np.isclose(mu_norm, _interpolant_norm(X, y, 1.5, c, 1e-6), rtol=1e-8)


def test_raw_predicate_view_costs_one_per_predicate():
    view = RawPredicateView(_feasible, x_dim=2)
    cfg = ExistenceConfig(mode=MODE_AFFINE, **{**FAST, "budget": 60})
    res = run_existence_arm(view, ARM_SUP_BISECTION, cfg, np.array([1.0]), 0)
    assert res.raw_evals <= cfg.budget
    assert view.n_raw_evals == res.raw_evals
    assert view.oracle(np.array([1.0, 0.0]), 0.65)[0] <= 0.0
    assert view.oracle(np.array([1.0, 0.0]), 0.75)[0] > 0.0


def test_raw_predicate_view_passes_the_call_rng_when_stochastic():
    import pytest

    def noisy(x, c, rng):  # theta 0.7 plus call-RNG noise
        return c <= 0.3 + 0.4 * float(x[0]) + 0.05 * rng.standard_normal()

    cfg = ExistenceConfig(mode=MODE_AFFINE, **{**FAST, "budget": 60})
    runs = []
    for _ in range(2):
        view = RawPredicateView(noisy, x_dim=2, needs_rng=True)
        res = run_existence_arm(view, ARM_SUP_BISECTION, cfg, np.array([1.0]), 0)
        assert view.n_raw_evals == res.raw_evals <= cfg.budget
        runs.append(res.final_lower)
    assert runs[0] == runs[1]  # the arm's seeded noise stream drives the predicate
    other = run_existence_arm(
        RawPredicateView(noisy, x_dim=2, needs_rng=True), ARM_SUP_BISECTION, cfg,
        np.array([1.0]), 1,
    )  # fmt: skip
    assert other.final_lower != runs[0]
    view = RawPredicateView(noisy, x_dim=2, needs_rng=True)
    with pytest.raises(ValueError, match="RNG"):
        view.oracle(np.zeros(2), 0.5)
    assert view.n_raw_evals == 0


def test_score_view_counts_raw_evaluations():
    """ScoreView charges its raw cost per query."""
    from brink.adapter import ScoreView

    view = ScoreView(lambda x: float(x[0]), x_dim=2, raw_evals_per_call=3)
    assert view.n_raw_evals == 0
    assert view.score(np.array([0.4, 0.1])) == pytest.approx(0.4)
    assert view.n_raw_evals == 3


def test_direct_score_arm_is_sound_below_and_respects_budget():
    """The direct-score lower endpoint never exceeds the true supremum."""
    from brink.adapter import ScoreView
    from brink.arms import (
        ARM_DIRECT_SCORE,
        ExistenceConfig,
        run_existence_arm,
    )
    from brink.bracket import MODE_AFFINE

    # Score peaks at 0.75 on the unit square; the sup is attainable.
    def score(x):
        return 0.75 * float(np.exp(-((x[0] - 0.3) ** 2 + (x[1] - 0.6) ** 2) / 0.05))

    view = ScoreView(score, x_dim=2)
    cfg = ExistenceConfig(mode=MODE_AFFINE, budget=120, report_every=10)
    res = run_existence_arm(view, ARM_DIRECT_SCORE, cfg, np.array([1.0]), seed=5)
    assert res.raw_evals == 120
    assert res.final_lower <= 0.75 + 1e-12
    assert res.final_upper >= res.final_lower
    assert len(res.reports) == 12
    # The running maximum never decreases.
    lows = [r.lower for r in res.reports]
    assert all(a <= b + 1e-12 for a, b in zip(lows[:-1], lows[1:], strict=True))


@pytest.mark.parametrize("cost", [4, 16])
def test_raw_predicate_view_charges_the_registered_cost(cost):
    """A K'-snapshot predicate charged r raw evaluations stops on the raw budget."""
    view = RawPredicateView(_feasible, x_dim=2, raw_evals_per_call=cost)
    cfg = ExistenceConfig(
        mode=MODE_AFFINE, **{**FAST, "budget": 400, "report_every": 40}
    )
    res = run_existence_arm(view, ARM_SUP_BISECTION, cfg, np.array([1.0]), 0)
    assert view.n_raw_evals == res.raw_evals <= cfg.budget
    assert res.raw_evals % cost == 0 and res.raw_evals > cfg.budget - cost
    raws = [r.raw_evals for r in res.reports]
    assert raws == list(range(40, 401, 40))  # reports in raw units
    assert all(r.calls == r.raw_evals // cost for r in res.reports)
    assert 0.0 < res.final_lower <= 0.7


def test_scan_off_keeps_the_trivial_upper_endpoint():
    """certificate_scan='off' runs no certificate; the witness side still works."""
    from brink.bracket import SCAN_OFF

    view = WindowExistenceView(_WindowEnv())
    cfg = ExistenceConfig(
        mode=MODE_AFFINE, lengthscale=0.5, lam=1e-6, certificate_scan=SCAN_OFF,
        **{**FAST, "budget": 200},
    )  # fmt: skip
    res = run_existence_arm(view, ARM_BRACKET_GREEDY, cfg, np.array([5.0, 5.0]), 0)
    assert all(not r.has_certificate and r.upper == 1.0 for r in res.reports)
    assert res.final_upper == 1.0 and res.final_lower > 0.0
