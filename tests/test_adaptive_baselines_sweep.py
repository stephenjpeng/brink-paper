"""Tests for the S8 task list, job split, and row merge."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import adaptive_baselines_sweep as s8  # noqa: E402

CFG = s8.load_config()


def test_task_indices_do_not_depend_on_the_study_selection():
    # Resume and merge key on the index, so a job's selection must not shift it.
    alone = s8.sweep_tasks(CFG, ["s5"])
    together = [t for t in s8.sweep_tasks(CFG, ["s2", "s4", "s5"]) if t[0] == "s5"]
    assert alone == together
    every = s8.sweep_tasks(CFG, list(CFG["studies"]))
    assert len({t[1] for t in every}) == len(every) == 720


def test_parts_cover_every_task_once_and_balance_the_load():
    tasks = s8.sweep_tasks(CFG, ["s2", "s4", "s5"])
    parts = [s8.part_tasks(CFG, tasks, p, 2) for p in range(2)]
    assert sorted(parts[0] + parts[1]) == sorted(tasks)
    est = {k: float(v["est_s_per_raw"]) for k, v in CFG["studies"].items()}
    loads = [sum(t[3] * est[t[0]] for t in p) for p in parts]
    assert abs(loads[0] - loads[1]) / max(loads) < 0.05
    for p in parts:  # longest first, so the long S5 runs start at once
        costs = [t[3] * est[t[0]] for t in p]
        assert costs == sorted(costs, reverse=True)
    with pytest.raises(ValueError):
        s8.part_tasks(CFG, tasks, 2, 2)


def test_merge_rejects_a_missing_task(tmp_path):
    rows = [
        {"study": t[0], "task_index": t[1], "config_sha256": CFG["sha256"]}
        for t in s8.sweep_tasks(CFG, ["s5"])
    ]
    for i, chunk in enumerate((rows[:60], rows[60:])):
        d = tmp_path / f"part{i}"
        d.mkdir()
        (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in chunk))
    dirs = [tmp_path / "part0", tmp_path / "part1"]
    assert len(s8.merged_rows(CFG, dirs)["s5"]) == 120
    (tmp_path / "part1" / "rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows[60:-1])
    )
    with pytest.raises(SystemExit):
        s8.merged_rows(CFG, dirs)
