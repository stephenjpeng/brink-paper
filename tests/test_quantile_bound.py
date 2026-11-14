"""Tests for the distribution-free order-statistic upper bound U_eps.

Proposition and proof: Proposition 1 and Appendix A.3 of the paper.
"""

import numpy as np
import pytest
from scipy.stats import binom, binomtest

from brink.quantile_bound import (
    empirical_bernstein_deviation,
    ladder_upper_score,
    order_statistic_upper,
    required_sample_size,
)


@pytest.mark.parametrize(
    ("eps", "expected"),
    [(0.05, (59, 124, 181)), (0.01, (299, 628, 913)), (0.001, (2995, 6294, 9151))],
)
def test_required_sample_size_matches_spec_table(eps, expected):
    assert tuple(required_sample_size(eps, 0.05, k=k) for k in (1, 3, 5)) == expected


def test_required_sample_size_is_the_smallest_qualifying_n():
    for eps, delta, k in [(0.05, 0.05, 2), (0.2, 0.1, 4), (0.01, 0.2, 1)]:
        n = required_sample_size(eps, delta, k=k)
        assert binom.cdf(k - 1, n, eps) <= delta
        assert n == k or binom.cdf(k - 1, n - 1, eps) > delta


def test_order_statistic_picks_largest_qualifying_k():
    u, k = order_statistic_upper(np.arange(628.0), eps=0.01, delta=0.05)
    assert k == 3
    assert u == 625.0
    u, k = order_statistic_upper(np.arange(300.0), eps=0.01, delta=0.05)
    assert (u, k) == (299.0, 1)


def test_chosen_k_is_maximal():
    rng = np.random.default_rng(1)
    for n in (37, 99, 200, 1000):  # 99 is the smallest N that qualifies at k = 1
        scores = rng.uniform(size=n)
        _, k = order_statistic_upper(scores, eps=0.03, delta=0.05)
        assert 0 <= k <= n
        if k == 0:
            assert binom.cdf(0, n, 0.03) > 0.05
        else:
            assert binom.cdf(k - 1, n, 0.03) <= 0.05
            assert k == n or binom.cdf(k, n, 0.03) > 0.05


def test_no_qualifying_k_returns_inf_and_zero():
    assert order_statistic_upper(np.arange(10.0), eps=0.01, delta=0.05) == (np.inf, 0)


def test_failed_runs_as_plus_inf_push_the_bound_up():
    u, k = order_statistic_upper(np.array([0.1, 0.2, np.inf]), eps=0.7, delta=0.05)
    assert k == 1
    assert u == np.inf


def test_all_infeasible_designs_give_minus_inf():
    u, k = order_statistic_upper(np.full(60, -np.inf), eps=0.05, delta=0.05)
    assert k == 1
    assert u == -np.inf


def test_bound_is_monotone_in_eps_and_delta():
    scores = np.random.default_rng(2).uniform(size=1000)
    u_small_eps, _ = order_statistic_upper(scores, eps=0.01, delta=0.05)
    u_large_eps, _ = order_statistic_upper(scores, eps=0.05, delta=0.05)
    assert u_small_eps >= u_large_eps
    u_small_delta, _ = order_statistic_upper(scores, eps=0.02, delta=0.01)
    u_large_delta, _ = order_statistic_upper(scores, eps=0.02, delta=0.20)
    assert u_small_delta >= u_large_delta


@pytest.mark.parametrize(
    ("bad_scores", "eps", "delta"),
    [
        (np.ones((3, 2)), 0.05, 0.05),
        (np.array([0.1, np.nan]), 0.05, 0.05),
        (np.ones(5), 0.0, 0.05),
        (np.ones(5), 1.0, 0.05),
        (np.ones(5), 0.05, 0.0),
        (np.ones(5), 0.05, 1.0),
        (np.ones(0), 0.05, 0.05),
    ],
)
def test_order_statistic_rejects_bad_inputs(bad_scores, eps, delta):
    with pytest.raises(ValueError):
        order_statistic_upper(bad_scores, eps=eps, delta=delta)


def _no_excess_violations(violations: int, reps: int, delta: float) -> bool:
    """One-sided exact binomial test at alpha 0.05 that the rate exceeds delta."""
    return binomtest(violations, reps, delta, alternative="greater").pvalue > 0.05


def _uniform_family(rng, n, eps):
    """s = x on [0, 1]: V(c) = 1 - c, theta_eps = 1 - eps. Tight: exact rate = cap."""
    return rng.uniform(size=n), 1.0 - eps


def _window_family(rng, n, eps):
    """Strict inclusion F(c) strictly inside {s >= c}.

    start = 1.2 x1, death = 0.2 + 0.8 x2 on [0, 1]^2, feasible when
    start <= c <= death, score s = death. V(c) = c (1 - c) / 0.96 on [0.2, 1], so
    theta_eps is the larger root of c (1 - c) = 0.96 eps. The exact violation rate
    is below the binomial cap. Any continuous score family of the form
    V(c) = mu(s >= c) is the uniform case under a monotone map, to which the bound
    is invariant, so only strict inclusion or atoms add evidence.
    """
    x = rng.uniform(size=(n, 2))
    return 0.2 + 0.8 * x[:, 1], (1.0 + np.sqrt(1.0 - 3.84 * eps)) / 2.0


@pytest.mark.parametrize("family", [_uniform_family, _window_family])
@pytest.mark.parametrize(("eps", "n"), [(0.05, 59), (0.01, 628)])
def test_coverage_against_exact_theta_eps(family, eps, n):
    rng = np.random.default_rng(20261026)
    reps, delta = 2000, 0.05
    violations = 0
    for _ in range(reps):
        scores, theta_eps = family(rng, n, eps)
        u, k = order_statistic_upper(scores, eps, delta)
        assert k >= 1
        violations += int(u < theta_eps)
    assert _no_excess_violations(violations, reps, delta)


def test_coverage_under_a_discrete_measure_with_ties():
    # mu puts mass (0.5, 0.3, 0.15, 0.04, 0.01) on the atoms below; ties are
    # the grid-v2 situation (scores sampled from a finite truth grid).
    atoms = np.array([-np.inf, 0.2, 0.5, 0.8, 1.0])
    weights = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    eps, delta, n, reps = 0.03, 0.05, 124, 2000
    tail = np.cumsum(weights[::-1])[::-1]  # P(s >= atom)
    theta_eps = atoms[tail > eps].max()  # = 0.8 (tail 0.05 > 0.03; 1.0 has 0.01)
    assert theta_eps == 0.8
    rng = np.random.default_rng(7)
    violations = 0
    for _ in range(reps):
        u, _ = order_statistic_upper(rng.choice(atoms, size=n, p=weights), eps, delta)
        violations += int(u < theta_eps)
    assert _no_excess_violations(violations, reps, delta)


def test_empirical_bernstein_deviation_formula_and_monotonicity():
    t = empirical_bernstein_deviation(0.09, 65_536, 3.0, 1e-5)
    L = np.log(2e5)
    assert t == pytest.approx(np.sqrt(2 * 0.09 * L / 65_536) + 7 * 3 * L / (3 * 65_535))
    assert 0.006 < t < 0.008  # the S3 registration's sizing figure
    assert empirical_bernstein_deviation(0.09, 2**14, 3.0, 1e-5) > t
    assert empirical_bernstein_deviation(0.09, 65_536, 3.0, 1e-3) < t
    assert empirical_bernstein_deviation(0.0, 65_536, 3.0, 1e-5) < t
    for bad in [(0.09, 1, 3.0, 0.01), (0.09, 10, 0.0, 0.01), (-1.0, 10, 3.0, 0.01),
                (0.09, 10, 3.0, 0.0), (0.09, 10, 3.0, 1.0)]:  # fmt: skip
        with pytest.raises(ValueError):
            empirical_bernstein_deviation(*bad)


def test_bernstein_test_false_fire_rate_within_budget():
    """A mean-zero, right-skewed bounded variable fires in at most p of the trials."""
    rng = np.random.default_rng(3)
    n, p, reps, b = 256, 0.01, 2000, 3.0
    fires = 0
    for _ in range(reps):
        x = b * (rng.random(n) < 0.02) - b * 0.02  # mean 0, range b, heavy right tail
        t = empirical_bernstein_deviation(float(x.var(ddof=1)), n, b, p)
        fires += x.mean() - t > 0
    assert _no_excess_violations(fires, reps, p)


def test_ladder_upper_score_rules():
    assert ladder_upper_score(0.30, 0.01, 8, lambda c: c >= 0.325) == (
        pytest.approx(0.33),
        3,
    )
    assert ladder_upper_score(0.30, 0.01, 8, lambda c: False) == (np.inf, 8)
    # Clipped to c_max: the rungs are 0.98, 0.99, 1.0 and the scan stops there.
    assert ladder_upper_score(0.97, 0.01, 8, lambda c: False) == (np.inf, 3)
    assert ladder_upper_score(0.97, 0.01, 8, lambda c: c >= 1.0) == (1.0, 3)
    with pytest.raises(ValueError):
        ladder_upper_score(0.3, 0.0, 8, lambda c: True)
    with pytest.raises(ValueError):
        ladder_upper_score(0.3, 0.01, 0, lambda c: True)


def test_ladder_scores_dominate_death_on_a_synthetic_noisy_oracle():
    """Construction S3-S1 on a bounded skewed oracle: domination within budget.

    Per-snapshot margin at c is (c - death) + 3 (Beta(0.5, 4.5) - 0.1): mean
    c - death, sd about 0.37, support width 3, right-skewed. The rough pass
    is death plus N(0, 0.01^2) noise; the ladder steps 0.01 for 8 rungs; each
    rung draws K = 16,384 snapshots, clips the margin from above at +2 (never
    from below: the floor -1.3 is the true support floor) and fires when the
    clipped mean exceeds its empirical-Bernstein deviation at p = 0.01 / 8, so
    each column fails with probability at most 0.01.
    """
    rng = np.random.default_rng(11)
    n_cols, K, step, rungs, p = 2000, 16_384, 0.01, 8, 0.01 / 8
    lo, hi = -1.3, 2.0
    violations, gaps, n_tests = 0, [], []
    for _ in range(n_cols):
        death = rng.uniform(0.05, 0.95)
        rough = death + 0.01 * rng.standard_normal()

        def fires(c, death=death):
            g = (c - death) + 3.0 * (rng.beta(0.5, 4.5, size=K) - 0.1)
            g = np.minimum(g, hi)
            t = empirical_bernstein_deviation(float(g.var(ddof=1)), K, hi - lo, p)
            return g.mean() - t > 0

        s, m = ladder_upper_score(rough, step, rungs, fires)
        violations += s < death
        gaps.append(s - death)
        n_tests.append(m)
    assert _no_excess_violations(violations, n_cols, 0.01)
    gaps = np.array(gaps)
    assert np.isfinite(gaps).mean() > 0.95  # the ladder rarely runs out
    assert 0.0 < np.median(gaps[np.isfinite(gaps)]) < 0.05  # and is not loose
    assert np.mean(n_tests) < 4
