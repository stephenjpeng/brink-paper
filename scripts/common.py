"""Helpers shared by the pipeline scripts: run provenance and run-directory lookup.

Every stage writes ``results/<UTC timestamp>_<name>/`` with a ``manifest.json``.
Later stages find earlier results through ``configs/runs.yaml``. An entry there
may contain the token ``<latest>``, which resolves to the newest run directory
with that name, so a fresh replication needs no config edits. Replace an entry
with an explicit path to use a specific run (for example, the authors' runs).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import yaml

RUNS_PATH = Path("configs/runs.yaml")
LATEST = "<latest>"
_STAMP = re.compile(r"^\d{8}T\d{6}Z$")
REFINE_POINTS_PER_AXIS = 5  # local refinement grids of the reference searches


def git_state() -> tuple[str, bool]:
    """The HEAD commit and whether the working tree has uncommitted changes.

    Returns:
        (commit hash, dirty flag); ("", False) outside a git repository.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    return commit, dirty


def resolve(path: str | Path) -> Path:
    """Resolve one run path, replacing ``<latest>`` by the newest timestamp.

    Args:
        path: A path such as ``results/<latest>_feeder_truth/truth.json``.

    Returns:
        The concrete path.

    Raises:
        SystemExit: When no run directory matches the pattern.
    """
    parts = Path(path).parts
    hits = [i for i, p in enumerate(parts) if p.startswith(LATEST)]
    if not hits:
        return Path(path)
    i = hits[0]
    parent, suffix = Path(*parts[:i]) if i else Path("."), parts[i][len(LATEST) :]
    matches = sorted(
        d
        for d in parent.glob(f"*{suffix}")
        if d.is_dir() and _STAMP.match(d.name[: len(d.name) - len(suffix)])
    )
    if not matches:
        raise SystemExit(f"no run directory matches {path}; run that stage first")
    return Path(matches[-1], *parts[i + 1 :])


def run_path(key: str, path: Path = RUNS_PATH) -> Path | list[Path]:
    """The resolved run path (or list of paths) under ``key``.

    Args:
        key: Entry of ``configs/runs.yaml``.
        path: The runs file.

    Returns:
        One resolved path, or a list when the entry is a list.
    """
    entry = yaml.safe_load(Path(path).read_text())[key]
    if isinstance(entry, list):
        return [resolve(p) for p in entry]
    return resolve(entry)


def window_union_gaps(start: np.ndarray, death: np.ndarray, tol: float) -> list:
    """Gaps in the union of the column windows [start, death] (unit c).

    Args:
        start: Window onsets (inf when never feasible).
        death: Window deaths.
        tol: Bisection resolution; smaller gaps are not resolved.

    Returns:
        List of (gap_lo, gap_hi); empty when the union is one interval from 0.
    """
    ok = np.isfinite(start)
    order = np.argsort(start[ok])
    s, d = start[ok][order], death[ok][order]
    gaps = [] if s.size and s[0] <= tol else [(0.0, float(s[0]) if s.size else 1.0)]
    reach = -np.inf
    for lo, hi in zip(s, d, strict=True):
        if np.isfinite(reach) and lo > reach + tol:
            gaps.append((float(reach), float(lo)))
        reach = max(reach, hi)
    return gaps


def local_grid(center: np.ndarray, half_width: float) -> np.ndarray:
    """5^d grid of half-width ``half_width`` around ``center``, clipped to [0,1]."""
    ax = np.linspace(-half_width, half_width, REFINE_POINTS_PER_AXIS)
    mesh = np.meshgrid(*([ax] * center.size), indexing="ij")
    pts = center + np.stack([m.ravel() for m in mesh], axis=1)
    return np.unique(np.clip(pts, 0.0, 1.0), axis=0)


def filter_tasks(
    tasks: list[tuple],
    labels: list[str] | None = None,
    budgets: list[int] | None = None,
    seeds: list[int] | None = None,
    fields: tuple[int, int, int] = (1, 2, 3),
) -> list[tuple]:
    """Keep the tasks whose method label, budget, and seed are selected.

    Task indices, and so every random stream, are those of the full sweep, so
    a filtered run reproduces the same rows as the full sweep.

    Args:
        tasks: Task tuples of one sweep.
        labels: Method labels to keep; None keeps all.
        budgets: Budgets to keep; None keeps all.
        seeds: Seeds to keep; None keeps all.
        fields: Positions of (label, budget, seed) in a task tuple.

    Returns:
        The selected tasks in their original order.
    """
    li, bi, si = fields
    return [
        t
        for t in tasks
        if (labels is None or t[li] in labels)
        and (budgets is None or t[bi] in budgets)
        and (seeds is None or t[si] in seeds)
    ]


def add_filter_args(parser) -> None:
    """Add ``--labels``, ``--budgets``, and ``--seeds`` to a sweep parser."""
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
