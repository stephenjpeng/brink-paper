"""GP-UCB and differential evolution on the four applied studies: sweep and analysis.

Constants in ``configs/adaptive_baselines.yaml``, which points at each study's
config for every environment, budget, seed, predicate, and design check. Both
methods (``brink.adaptive_baselines``) maximize each design's observed last
feasible stress on the same pass/fail checks as the study's single-check
bisection, at the study's budgets and seeds. The design each run returns is
checked at its reported level with the study's own check. ``analyze`` reads
the study sweeps (``configs/runs.yaml``) for the seed-paired comparisons.

A task is deterministic given (study, task index, method, budget, seed), so
the worker count and the split into parts are compute settings, and a stopped
sweep resumes from its ``rows.jsonl``. Check streams use task indices from
20,000 (feeder: ``feeder_setup._witness_check`` stream 1,000 + index; Covasim:
stream [994, 1,000 + index]); transmission and Hopper checks are
deterministic. The reported Hopper runs used x86-64.

Study names on the command line: transmission, feeder, hopper, covasim.

Usage:
    uv run python scripts/adaptive_baselines_sweep.py sweep --studies feeder \
        --workers N [--part K --n-parts N] [--resume DIR]
    uv run python scripts/adaptive_baselines_sweep.py analyze [DIR ...]
"""

from __future__ import annotations

import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # every base config: blas 1
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # Linux (docs/runbook.md)
os.environ.setdefault("OMP_NUM_THREADS", "1")  # before numpy loads

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from brink.adaptive_baselines import (  # noqa: E402
    run_adaptive_baseline,
)
from brink.memory import RssMonitor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import git_state, run_path  # noqa: E402

log = logging.getLogger("adaptive_baselines_sweep")
CFG_PATH = Path("configs/adaptive_baselines.yaml")
STUDY_KEYS = {"transmission": "s2", "feeder": "s3", "hopper": "s4", "covasim": "s5"}
STUDY_NAMES = {v: k for k, v in STUDY_KEYS.items()}
SPAWN = mp.get_context("spawn")  # same start method on macOS and Linux
NOISY = ("s3", "s5")
METHOD_LABEL = "witness-focused"


def load_config(path: Path = CFG_PATH) -> dict:
    """The baseline constants (YAML), with their SHA-256 under ``sha256``."""
    text = Path(path).read_text()
    return {**yaml.safe_load(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def _base(e: dict) -> dict:
    return yaml.safe_load(Path(e["base_config"]).read_text())


def study_cells(cfg: dict, study: str) -> list[tuple[str, int, int]]:
    """Cells of one study: arm-major, then budget, then seed."""
    base = _base(cfg["studies"][study])
    seeds = range(int(base["seeds"]["start"]), int(base["seeds"]["stop"]))
    return [
        (a["label"], int(b), int(s))
        for a in cfg["arms"]
        for b in base["budgets_raw"]
        for s in seeds
    ]


def _index_base(cfg: dict, study: str, base: int) -> int:
    pos = list(cfg["studies"]).index(study)  # position, not selection
    return int(base) + int(cfg["study_index_stride"]) * pos


def sweep_tasks(cfg: dict, studies: list[str]) -> list[tuple]:
    """Tasks (study, index, arm label, budget, seed) of the studies."""
    out = []
    for s in studies:
        i0 = _index_base(cfg, s, cfg["task_index_base"])
        out += [(s, i0 + i, *cell) for i, cell in enumerate(study_cells(cfg, s))]
    return out


def part_tasks(cfg: dict, tasks: list[tuple], part: int, n_parts: int) -> list:
    """One job's share: tasks by estimated cost, dealt round-robin, longest first."""
    if not 0 <= part < n_parts:
        raise ValueError(f"part {part} outside 0..{n_parts - 1}")
    est = {s: float(e["est_s_per_raw"]) for s, e in cfg["studies"].items()}
    order = sorted(tasks, key=lambda t: (-t[3] * est[t[0]], t[1]))
    return order[part::n_parts]


def _run(view, arm: str, run_cfg, seed: int):
    t0 = time.perf_counter()
    res, info = run_adaptive_baseline(view, arm, run_cfg, seed)
    return res, info, time.perf_counter() - t0


def _row(task: tuple, arm: str, res, info: dict, wall: float) -> dict:
    study, idx, label, budget, seed = task
    return {
        "study": study, "task_index": int(idx), "label": label, "arm": arm,
        "view": "predicate", "budget": int(budget), "seed": int(seed),
        "raw_evals": int(res.raw_evals), "wall_s": wall,
        "final_lower": float(res.final_lower), "final_upper": float(res.final_upper),
        "has_witness": res.final_witness_x is not None,
        "first_witness_raw": next(
            (r.raw_evals for r in res.reports if r.has_witness), None),
        "witness_x": res.final_witness_x, "info": info,
        "reports": [asdict(r) for r in res.reports],
    }  # fmt: skip


def _s2_task(e: dict, task: tuple, arm: str) -> dict:
    import transmission_sweep as gs  # pandapower only in transmission workers

    _, _, _, budget, seed = task
    base = gs.load_cfg(Path(e["base_config"]))
    env = gs.load_environment(base["frozen_environment"])
    cal = json.loads(Path(base["frozen_environment"]).read_text())["calibration"]
    run_cfg = gs.build_cfg(base, budget, "predicate", cal["lengthscale"])
    view = gs.make_view({"view": "predicate", **e["predicate"]}, env)
    res, info, wall = _run(view, arm, run_cfg, seed)
    row = _row(task, arm, res, info, wall)
    row["check"] = row["check_feasible"] = None
    if row["has_witness"]:
        t0 = time.perf_counter()
        row["check"] = gs.witness_check(row["witness_x"], row["final_lower"])
        row["check_feasible"] = bool(row["check"]["feasible"])
        row["check_s"] = time.perf_counter() - t0
    return row


def _s3_task(e: dict, task: tuple, arm: str) -> dict:
    import feeder_sweep as hs  # pandapower only in feeder workers

    _, idx, _, budget, seed = task
    he = hs.he
    base = he.load_s3_config(Path(e["base_config"]))
    K = int(base["k_snapshots"])
    env = hs.hc.load_hosting_environment(base["frozen_environment"], K)
    R = float(env.R_by_knob[K])
    cal = json.loads(Path(base["frozen_environment"]).read_text())["calibration"]
    run_cfg = hs.build_cfg(base, budget, "predicate", R, cal["lengthscale"])
    view = hs.make_view({"view": "predicate", **e["predicate"]}, env, he._oracle(), K)
    res, info, wall = _run(view, arm, run_cfg, seed)
    row = _row(task, arm, res, info, wall)
    theta, err = hs._truth(base)
    row.update(theta=theta, theta_err=err)
    row["check"] = row["check_feasible"] = None
    if row["has_witness"]:
        t0 = time.perf_counter()
        w = int(base["truth_check"]["stream_offset"]) + int(idx)
        row["check"] = he._witness_check((w, row["witness_x"], row["final_lower"]))
        row["check"]["stream_index"] = w
        row["check_feasible"] = bool(row["check"]["feasible"])
        row["check_s"] = time.perf_counter() - t0
    return row


def _s4_task(e: dict, task: tuple, arm: str) -> dict:
    import hopper_sweep as ms  # mujoco only in Hopper workers

    _, _, _, budget, seed = task
    base = _base(e)
    run_cfg = ms.build_cfg(base, budget, "predicate")
    view = ms.LoggedPredicateView(ms.mh.HopperOracle().feasible, ms.mh.X_DIM)
    res, info, wall = _run(view, arm, run_cfg, seed)
    row = _row(task, arm, res, info, wall)
    row["check"] = row["check_feasible"] = row["robust_fraction"] = None
    if row["has_witness"]:
        t0 = time.perf_counter()
        oracle, ch = ms.mh.HopperOracle(), base["checks"]
        x, L = np.asarray(row["witness_x"]), row["final_lower"]
        r0 = oracle.rollout(x, L, reset_seed=int(ch["reset_seed"]))
        row["check"] = asdict(r0)
        row["check_feasible"] = bool(np.all(r0.margins() <= 0.0))
        row["robust_fraction"] = float(np.mean([
            bool(np.all(oracle.rollout(x, L, int(s)).margins() <= 0.0))
            for s in ch["robustness_reset_seeds"]
        ]))  # fmt: skip
        row["check_s"] = time.perf_counter() - t0
    return row


def _s5_task(e: dict, task: tuple, arm: str) -> dict:
    import covasim_sweep as cs  # covasim only in Covasim workers

    _, idx, _, budget, seed = task
    base = cs.ce.he.load_s3_config(Path(e["base_config"]))
    env = cs.cp.load_environment(base["frozen_environment"], 1)
    run_cfg = cs.hs.build_cfg(base, budget, "predicate", env.R, env.lengthscale)
    view = cs.make_view({"view": "predicate", **e["predicate"]}, env)
    res, info, wall = _run(view, arm, run_cfg, seed)
    row = _row(task, arm, res, info, wall)
    theta, err = cs._truth(base)
    row.update(theta=theta, theta_err=err)
    row["check"] = row["check_feasible"] = None
    if row["has_witness"]:
        t0 = time.perf_counter()
        x = np.asarray(row["witness_x"])
        row["witness_policy"] = [float(v) for v in cs.cp.to_policy(x)]
        row["check"] = cs.truth_check(base, int(idx), row["witness_x"],
                                      row["final_lower"])  # fmt: skip
        row["check_feasible"] = bool(row["check"]["feasible"])
        row["check_s"] = time.perf_counter() - t0
    return row


TASKS = {"s2": _s2_task, "s3": _s3_task, "s4": _s4_task, "s5": _s5_task}


def run_task(task: tuple) -> dict:
    """One (study, task index, arm label, budget, seed) run in a fresh process."""
    t0 = time.perf_counter()
    cfg = load_config()
    study, _, label = task[:3]
    arm = next(a["name"] for a in cfg["arms"] if a["label"] == label)
    row = TASKS[study](cfg["studies"][study], tuple(task), arm)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes macOS, KiB Linux
    row["peak_rss_gib"] = rss / (1024**3 if sys.platform == "darwin" else 1024**2)
    row["config_sha256"] = cfg["sha256"]
    row["task_s"] = time.perf_counter() - t0
    return row


def _run_pool(tasks: list, workers: int, on_row, label: str) -> RssMonitor:
    """Run tasks in fresh spawned processes, handing each row to ``on_row``."""
    mon = RssMonitor()
    mon.start()
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=SPAWN, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result()
            on_row(row)
            log.info(
                "%s %d/%d %s %s b=%d s=%d: %.0fs L=%.4f check=%s raw=%d designs=%d",
                label, i, len(tasks), row["study"], row["label"], row["budget"],
                row["seed"], row["task_s"], row["final_lower"], row["check_feasible"],
                row["raw_evals"], row["info"]["n_designs"],
            )  # fmt: skip
            if i % 20 == 0:
                log.info(
                    "eta %.2f h", (len(tasks) - i) / (i / (time.time() - t0)) / 3600
                )
    mon.stop()
    if mon.missed_samples:
        log.warning("RSS monitor missed %d samples", mon.missed_samples)
    return mon


def _manifest(cfg: dict, studies: list[str], workers: int, kind: str, **extra) -> dict:
    commit, dirty = git_state()
    bases = {}
    for s in studies:
        p = Path(cfg["studies"][s]["base_config"])
        frozen = Path(_base(cfg["studies"][s])["frozen_environment"])
        bases[s] = {
            "base_config": str(p),
            "base_config_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "frozen_environment_sha256": hashlib.sha256(
                frozen.read_bytes()
            ).hexdigest(),
        }
    return {
        "experiment": f"adaptive_baselines_{kind}", "git_commit": commit,
        "git_dirty": dirty, "config_sha256": cfg["sha256"], "studies": studies,
        "bases": bases, "workers": workers,
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "started_utc": datetime.now(UTC).isoformat(), **extra,
    }  # fmt: skip


def _new_dir(name: str) -> Path:
    out = Path("results") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{name}"
    out.mkdir(parents=True)
    return out


def _append(path: Path, rows: list[dict]) -> None:
    with path.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _require_platform(studies: list[str]) -> None:
    if "s4" in studies and platform.machine() not in ("x86_64", "AMD64"):
        log.warning(
            "Hopper rollouts depend on the architecture; the reported runs "
            "used x86-64, so these rows can differ from them"
        )


def cmd_sweep(studies: list[str], workers: int, part: int, n_parts: int,
              resume: Path | None) -> None:  # fmt: skip
    """One part of the sweep of the studies; ``resume`` continues one."""
    cfg = load_config()
    _require_platform(studies)
    tasks = part_tasks(cfg, sweep_tasks(cfg, studies), part, n_parts)
    if resume is None:
        manifest = _manifest(cfg, studies, workers, "sweep", part=part,
                             n_parts=n_parts, n_tasks=len(tasks))  # fmt: skip
        names = "_".join(STUDY_NAMES[s] for s in studies)
        out = _new_dir(f"adaptive_baselines_{names}_part{part}of{n_parts}")
    else:
        out = resume
        manifest = json.loads((out / "manifest.json").read_text())
        key = (manifest["config_sha256"], manifest["studies"], manifest["part"])
        same = key == (cfg["sha256"], studies, part)
        same = same and manifest["n_parts"] == n_parts
        if not same:
            raise SystemExit("resume target used another config, studies, or part")
        lines = (out / "rows.jsonl").read_text().splitlines()
        done = {json.loads(ln)["task_index"] for ln in lines if ln.strip()}
        tasks = [t for t in tasks if t[1] not in done]
        commit, dirty = git_state()
        manifest.setdefault("resumes", []).append({
            "resumed_utc": datetime.now(UTC).isoformat(), "git_commit": commit,
            "git_dirty": dirty, "workers": workers, "rows_done": len(done),
            "rows_remaining": len(tasks),
        })  # fmt: skip
        log.info("resuming %s: %d done, %d remaining", out, len(done), len(tasks))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    t0 = time.time()
    mon = _run_pool(tasks, workers, lambda r: _append(out / "rows.jsonl", [r]), "sweep")
    (out / "rss_log.json").write_text(json.dumps(mon.samples))
    (out / "sweep_summary.json").write_text(json.dumps({
        "wall_h": (time.time() - t0) / 3600.0, "aggregate_peak_rss_gib": mon.peak_gib,
        "rss_missed_samples": mon.missed_samples, "n_rows": len(tasks),
        "finished_utc": datetime.now(UTC).isoformat(),
    }, indent=1))  # fmt: skip
    log.info("sweep done in %.2f h -> %s", (time.time() - t0) / 3600.0, out)


# ---------------------------------------------------------------- analysis


def _read_rows(paths) -> list[dict]:
    return [json.loads(ln) for p in paths for ln in Path(p).read_text().splitlines()
            if ln.strip()]  # fmt: skip


def merged_rows(cfg: dict, dirs: list[Path]) -> dict[str, list[dict]]:
    """S8 rows by study; every task of a present study exactly once."""
    rows = _read_rows([d / "rows.jsonl" for d in dirs])
    accepted = {cfg["sha256"], *cfg.get("equivalent_config_sha256", [])}
    by: dict[str, list[dict]] = {}
    for r in rows:
        if r["config_sha256"] not in accepted:
            raise SystemExit(f"row {r['task_index']} used another config")
        by.setdefault(r["study"], []).append(r)
    for s, rs in by.items():
        want = sorted(t[1] for t in sweep_tasks(cfg, [s]))
        got = sorted(r["task_index"] for r in rs)
        if got != want:
            raise SystemExit(f"{s}: {len(got)} rows, {len(want)} registered tasks")
    return by


def base_dirs(e: dict) -> list[Path]:
    """The study sweep directories (``configs/runs.yaml`` entry ``runs_key``)."""
    d = run_path(e["runs_key"])
    return d if isinstance(d, list) else [d]


def _truth(study: str, e: dict, rows: list[dict]) -> dict:
    """Reference value each study's records compare L against."""
    if study == "s2":
        g = json.loads((base_dirs(e)[0] / "gate.json").read_text())
        return {"value": float(g["theta"]), "err": float(g["err"]),
                "kind": "theta_exists"}  # fmt: skip
    if study == "s4":
        g = json.loads((base_dirs(e)[0] / "gate.json").read_text())
        return {"value": float(g["t_ref"]), "err": 0.0,
                "kind": "t_ref (S4 truth search)"}  # fmt: skip
    return {"value": float(rows[0]["theta"]), "err": float(rows[0]["theta_err"]),
            "kind": "theta_exists (replicated Monte Carlo)"}  # fmt: skip


def base_feasible(study: str, row: dict) -> bool | None:
    """The base sweep's check of its reported design at L (None: no design)."""
    if study == "s4":
        return row.get("check_feasible")
    return (row.get("truth_check") or {}).get("feasible")


def _summary(rs: list[dict], feas, truth: dict | None) -> dict:
    lowers = np.array([r["final_lower"] for r in rs])
    f = [feas(r) for r in rs]
    out = {
        "n": len(rs), "n_witness": int(sum(bool(r["has_witness"]) for r in rs)),
        "median_L": float(np.median(lowers)),
        "q25_L": float(np.quantile(lowers, 0.25)),
        "q75_L": float(np.quantile(lowers, 0.75)),
        "n_check_feasible": int(sum(v is True for v in f)),
        "n_check_infeasible": int(sum(v is False for v in f)),
    }  # fmt: skip
    if truth is not None:
        top = truth["value"] + truth["err"]
        out["n_L_above_truth"] = int((lowers > top).sum())
        out["median_L_minus_truth"] = float(np.median(lowers - truth["value"]))
    return out


def _paired(a: dict[int, float], b: dict[int, float]) -> dict:
    seeds = sorted(set(a) & set(b))
    hi = sum(a[s] > b[s] for s in seeds)
    lo = sum(a[s] < b[s] for s in seeds)
    p = binomtest(hi, hi + lo, 0.5).pvalue if hi + lo else 1.0
    return {"n": len(seeds), "higher": int(hi), "lower": int(lo),
            "ties": len(seeds) - hi - lo, "p_two_sided": float(p)}  # fmt: skip


def analyze_study(cfg: dict, study: str, rows: list[dict]) -> dict:
    """Paired comparisons of one study against its sweep (no gate)."""
    e, an = cfg["studies"][study], cfg["analysis"]
    paths = [d / "rows.jsonl" for d in base_dirs(e)]
    base_rows = [r for r in _read_rows(paths) if "label" in r]
    truth = _truth(study, e, rows)
    new_labels = [a["label"] for a in cfg["arms"]]
    base_labels = list(dict.fromkeys(r["label"] for r in base_rows))
    compare = [lab for lab in (METHOD_LABEL, e["naive_label"], "direct-score")
               if lab in base_labels]  # fmt: skip
    cells = []
    for b in _base(e)["budgets_raw"]:
        new = {lab: [r for r in rows if r["label"] == lab and r["budget"] == b]
               for lab in new_labels}  # fmt: skip
        old = {lab: [r for r in base_rows if r["label"] == lab and r["budget"] == b]
               for lab in base_labels}  # fmt: skip
        cell = {
            "budget": int(b),
            "arms": {lab: {**_summary(rs, lambda r: r["check_feasible"], truth),
                           "median_n_designs": float(np.median(
                               [r["info"]["n_designs"] for r in rs])),
                           "n_anytime_above_truth": int(sum(
                               any(p["lower"] > truth["value"] + truth["err"]
                                   for p in r["reports"]) for r in rs))}
                     for lab, rs in new.items()},
            "base_arms": {lab: _summary(rs, lambda r: base_feasible(study, r), truth)
                          for lab, rs in old.items() if rs},
            "paired_L": {
                lab: {c: _paired({r["seed"]: r["final_lower"] for r in new[lab]},
                                 {r["seed"]: r["final_lower"] for r in old[c]})
                      for c in compare if old[c]}
                for lab in new_labels
            },
        }  # fmt: skip
        if study == "s4":  # C14 diagnostic: the design hops from all 8 other states
            for key, groups in (("arms", new), ("base_arms", old)):
                for lab, rs in groups.items():
                    if rs:
                        cell[key][lab]["n_robust_all"] = int(
                            sum(r.get("robust_fraction") == 1.0 for r in rs)
                        )
        if study in NOISY:  # S3-G3 / S5-G3 measure: design infeasible at L, or none
            cell["wrong"] = {}
            for lab, rs in new.items():
                k = sum(r["check_feasible"] is not True for r in rs)
                p = binomtest(k, len(rs), float(an["wrong_rate_null"]),
                              alternative="greater").pvalue  # fmt: skip
                reject = bool(p < float(an["alpha"]))
                cell["wrong"][lab] = {
                    "count": int(k), "n": len(rs), "p": float(p), "reject": reject,
                }  # fmt: skip
        cells.append(cell)
    return {"study": study, "truth": truth, "compare": compare, "cells": cells}


def cmd_analyze(dirs: list[Path]) -> None:
    """Paired comparisons -> ``results/<UTC>_adaptive_baselines_analysis/``."""
    cfg = load_config()
    by = merged_rows(cfg, dirs)
    out = {"config_sha256": cfg["sha256"], "inputs": [str(d) for d in dirs],
           "studies": {s: analyze_study(cfg, s, by[s])
                       for s in cfg["studies"] if s in by}}  # fmt: skip
    commit, dirty = git_state()
    res = _new_dir("adaptive_baselines_analysis")
    (res / "manifest.json").write_text(json.dumps({
        "experiment": "adaptive_baselines_analysis", "git_commit": commit,
        "git_dirty": dirty,
        "config_sha256": cfg["sha256"], "inputs": out["inputs"],
        "started_utc": datetime.now(UTC).isoformat()}, indent=1))  # fmt: skip
    (res / "analysis.json").write_text(json.dumps(out, indent=1))
    for s, a in out["studies"].items():
        for c in a["cells"]:
            for lab, v in {**c["arms"], **c["base_arms"]}.items():
                log.info("%s %6d %-16s median L %.4f feasible %d/%d above-truth %s",
                         s, c["budget"], lab, v["median_L"], v["n_check_feasible"],
                         v["n"], v.get("n_L_above_truth"))  # fmt: skip
    log.info("analysis -> %s", res / "analysis.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("--studies", nargs="+", choices=tuple(STUDY_KEYS), required=True)
    sw.add_argument("--workers", type=int, required=True)
    sw.add_argument("--n-parts", type=int, default=1)
    sw.add_argument("--part", type=int, default=0)
    sw.add_argument("--resume", type=Path, default=None)
    an = sub.add_parser("analyze")
    an.add_argument("dirs", type=Path, nargs="*")
    args = ap.parse_args()
    if args.cmd == "sweep":
        keys = [STUDY_KEYS[s] for s in args.studies]
        cmd_sweep(keys, args.workers, args.part, args.n_parts, args.resume)
    else:
        cmd_analyze(args.dirs or run_path("adaptive_baselines"))
