"""Checks for the S2 truth helpers (nesting gaps, clipped local grids)."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("s2", ROOT / "scripts" / "common.py")
    m = importlib.util.module_from_spec(spec)
    cwd = Path.cwd()
    try:  # the T07 runner it imports reads its frozen config by relative path
        os.chdir(ROOT)
        spec.loader.exec_module(m)
    finally:
        os.chdir(cwd)
    return m


def test_union_gaps(mod):
    inf = np.inf
    # One interval from 0 (overlapping windows, a never-feasible column).
    assert (
        mod.window_union_gaps(
            np.array([0.0, 0.2, inf]), np.array([0.3, 0.6, -inf]), 1e-3
        )
        == []
    )
    # A delayed window above a dead region is a gap.
    assert mod.window_union_gaps(np.array([0.0, 0.5]), np.array([0.3, 0.6]), 1e-3) == [
        (0.3, 0.5)
    ]
    # No column feasible at c = 0.
    assert mod.window_union_gaps(np.array([0.1]), np.array([0.4]), 1e-3) == [(0.0, 0.1)]


def test_local_grid_clips_to_box_faces(mod):
    pts = mod.local_grid(np.array([0.05, 0.5, 0.95, 0.5]), 1 / 9)
    assert pts.min() == 0.0 and pts.max() == 1.0
    assert pts.shape[1] == 4 and len(pts) <= 5**4
