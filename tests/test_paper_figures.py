"""Tests for the manuscript exhibits' data transforms."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper_figures as pf  # noqa: E402


def test_reported_level_is_carried_forward_and_zero_before_first_report():
    # Run 1 reports at 170 and 340 (unsorted on input); run 2 first reports at 510.
    runs = [[(340, 0.3), (170, 0.1)], [(510, 0.5)], [(170, 0.2), (1000, 0.6)]]
    grid = np.array([100.0, 170.0, 339.0, 600.0, 1000.0])
    q = pf.step_quantiles(runs, grid, qs=(0.0, 0.5, 1.0))
    np.testing.assert_allclose(q[0], [0.0, 0.0, 0.0, 0.2, 0.3])  # min across runs
    np.testing.assert_allclose(q[1], [0.0, 0.1, 0.1, 0.3, 0.5])  # median
    np.testing.assert_allclose(q[2], [0.0, 0.2, 0.2, 0.5, 0.6])  # max
    # A width defaults to the full range before a run's first report.
    widths = pf.step_quantiles(runs, grid, qs=(0.5,), before=100.0)
    np.testing.assert_allclose(widths[0], [100.0, 0.2, 0.2, 0.3, 0.5])


def test_threshold_is_kth_largest_limit_and_survival_counts_ties():
    scores = np.array([0.1, 0.5, 0.5, 0.9])
    u = pf.kth_largest(scores, 2)
    assert u == 0.5
    # At the threshold, the tie keeps three designs; just above it only one remains.
    np.testing.assert_allclose(pf.survival(scores, np.array([u, u + 1e-9])),
                               [0.75, 0.25])  # fmt: skip


def test_table_numbers_use_physical_units_and_never_print_negative_zero():
    cfg = {"studies": {"s3": {"offset": 0.0, "scale": 100.0, "digits": 2},
                       "s2": {"offset": 1.0, "scale": 0.35, "digits": 4}}}  # fmt: skip
    assert pf._fmt(cfg, "s3", -0.1226, delta=True) == r"$-$12.26\%"
    assert pf._fmt(cfg, "s3", -0.00001, delta=True) == r"0.00\%"
    assert pf._fmt(cfg, "s2", 0.63) == "1.2205"
    assert pf._fmt(cfg, "s2", None) == "--"


def test_overview_never_hides_a_returned_design_and_counts_upper_bounds_off_axis():
    # A returned design beyond either edge would be clipped silently, so it stops.
    with pytest.raises(SystemExit):
        pf.check_on_axis(np.array([0.5, 1.2]), 0.0, 1.0, "design")
    with pytest.raises(SystemExit):
        pf.check_on_axis(np.array([-0.1]), 0.0, 1.0, "design")
    pf.check_on_axis(np.array([0.0, 1.0]), 0.0, 1.0, "design")  # edges are drawn
    # Upper bounds beyond the right edge are counted for the annotation, not dropped.
    shown, beyond = pf.split_upper(np.array([0.4, 1.0, 1.5, 2.0]), 0.0, 1.0)
    np.testing.assert_allclose(shown, [0.4, 1.0])
    assert beyond == 2
    with pytest.raises(SystemExit):
        pf.split_upper(np.array([-0.1, 0.5]), 0.0, 1.0)
