"""Tests for the cross-study comparison's summary rules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import method_comparison as mc  # noqa: E402


def _rec(L, verified, points, U=None):
    return {"L": L, "U": U, "U_kind": "claimed" if U is not None else None,
            "has_witness": True, "verified": verified, "points": points,
            "first_witness": 170, "raw": 1000, "wall_s": 10.0, "task_s": 11.0,
            "rss": 0.3, "robust": None}  # fmt: skip


def test_unsound_reports_count_neither_as_usable_nor_as_reaching_target():
    # An overshooting design that fails its check reaches the target early;
    # neither the verified L nor the sample-efficiency count may credit it.
    recs = [
        _rec(0.30, False, [(170, 0.30), (340, 0.30)], U=0.31),
        _rec(0.20, True, [(170, 0.10), (340, 0.20)], U=0.26),
        _rec(0.26, True, [(170, 0.15), (510, 0.26)], U=0.27),
    ]
    s = mc.summarize(recs, theta=0.25, err=0.0, target=0.25, r_sim=0.009)
    assert s["median_verified_L"] == 0.20  # (0.0, 0.20, 0.26)
    assert s["n_reach_target"] == 1 and s["median_raw_to_target"] == 510
    assert s["lower_ok_anytime"] == 1  # only the 0.20 run stays <= truth
    assert s["upper_covers"] == 1  # only [0.20, 0.26] contains 0.25
    assert s["median_overhead_s"] == 10.0 - 1000 * 0.009


def test_rows_without_report_traces_have_no_target_count():
    s = mc.summarize([_rec(0.2, True, [])], theta=0.25, err=0.0, target=0.1,
                     r_sim=None)  # fmt: skip
    assert s["n_reach_target"] is None
