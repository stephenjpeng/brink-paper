"""Tests for the metric rules of the quick evaluation dimensions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluation_dimensions as ed  # noqa: E402


def test_targets_inside_the_error_band_are_not_scored():
    theta, err = 0.50, 0.02
    assert ed.target_truth(0.47, theta, err) is True
    assert ed.target_truth(0.53, theta, err) is False
    assert ed.target_truth(0.49, theta, err) is None
    assert ed.target_truth(0.515, theta, err) is None


def test_a_best_verified_design_scores_only_targets_at_or_below_it():
    # Hopper: the reference is a lower bound on the limit, so a higher target is
    # neither achievable nor unachievable by the reference.
    assert ed.target_truth(0.60, 0.625, 0.0, lower_bound_only=True) is True
    assert ed.target_truth(0.625, 0.625, 0.0, lower_bound_only=True) is True
    assert ed.target_truth(0.65, 0.625, 0.0, lower_bound_only=True) is None


def test_only_our_upper_bound_can_answer_not_achievable():
    # A claimed interval [0.30, 0.31] says nothing proved about 0.40.
    assert ed.target_answer(0.30, True, 0.31, ed.CLAIMED, 0.40) == ed.UNDECIDED
    assert ed.target_answer(0.30, True, 0.31, ed.OURS, 0.40) == ed.NOT_ACHIEVABLE
    assert ed.target_answer(0.30, True, 0.31, ed.OURS, 0.25) == ed.ACHIEVABLE
    # The domain-minimum level of a run without a design asserts nothing.
    assert ed.target_answer(0.0, False, None, None, 0.0) == ed.UNDECIDED


def test_one_early_report_that_excludes_the_limit_breaks_every_report_containment():
    # Bisection pattern: narrow claimed intervals that start below the limit.
    reports = [(0.40, 0.41), (0.48, 0.49), (0.495, 0.505)]
    assert ed.contains(*reports[-1], theta=0.50, err=0.0)
    assert not ed.contains_every_report(reports, theta=0.50, err=0.0)
    # The stated error widens containment on both sides.
    assert ed.contains_every_report(reports[1:], theta=0.50, err=0.011)


def test_double_integrator_slack_uses_the_closed_form_limit():
    # x = (0.5, 0.5): p = 0.25, v = 0.5, c_off = 1 - 0.25 / 1.5 = 5/6 (physical),
    # 5/9 in unit stress on [0, 1.5].
    limit = ed.own_limit("s6b", None, [0.5, 0.5])
    assert limit == pytest.approx(5 / 9)
    assert ed.stress_slack(limit, 0.5) == pytest.approx(5 / 9 - 0.5)
    assert ed.own_limit("s6b", None, None) is None  # bisection stores no design


def test_transmission_slack_uses_the_exact_window_end():
    limit = ed.own_limit("s2", {"death": 0.631, "margins": [-0.1, -0.2, -0.3]}, None)
    assert ed.stress_slack(limit, 0.625) == pytest.approx(0.006)


def test_feeder_slack_uses_the_median_of_the_replicated_limits():
    check = {"c_off_replicates": [0.09, 0.12, 0.13, 0.30]}  # mean 0.16
    limit = ed.own_limit("s3", check, None)
    assert limit == pytest.approx(0.125)
    assert ed.stress_slack(limit, 0.20) == pytest.approx(-0.075)  # above its limit
    assert ed.own_limit("s5", {"mean": [-0.1], "se": [0.01]}, None) is None


def test_stopping_takes_the_first_report_within_tolerance_despite_float_noise():
    reports = [(10, 0.62, 1.0), (50, 0.664, 0.6700000000000001),
               (90, 0.665, 0.6700000000000044)]  # fmt: skip
    assert ed.first_stop(reports, 0.006) == reports[1]
    assert ed.first_stop(reports, 0.005) == reports[2]
    assert ed.first_stop(reports, 0.001) is None
