"""Hopper (MuJoCo): the method and baseline sweep, and its gates.

Every constant comes from ``configs/hopper.yaml``. Five methods at two
rollout budgets, 50 seeds each: BRINK's proposed search (``witness-focused``),
BRINK's variance-seeking rule (``greedy``), the CONFIG adaptation
(``config``), bisection on random designs (``sup-bisection``), and direct
limit search (``direct-score``). No infeasibility test runs on Hopper. Every
returned design is checked by the rollout from the fixed initial state and by
rollouts from 8 other initial states. A task is deterministic given (task
index, method, budget, seed) on one processor architecture; the reported
runs used x86-64.

Usage:
    uv run python scripts/hopper_sweep.py sweep --workers N \
        [--labels L ...] [--budgets B ...] [--seeds S ...] [--resume DIR]
    uv run python scripts/hopper_sweep.py analyze [--run DIR] [--columns DIR] \
        [--desk DIR] [--truth DIR]
"""

from __future__ import annotations

import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # before numpy loads
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from brink import bracket as br  # noqa: E402
from brink.adapter import (  # noqa: E402
    ExistenceSpec,
    RawPredicateView,
    WindowExistenceView,
)
from brink.arms import ExistenceConfig, run_existence_arm  # noqa: E402
from brink.memory import RssMonitor  # noqa: E402
from brink.studies import hopper as mh  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hopper_setup as me  # noqa: E402
from common import add_filter_args, filter_tasks, run_path  # noqa: E402

log = logging.getLogger("hopper_sweep")
CERTIFIED = ("witness-focused", "greedy", "config")
METHOD = "witness-focused"


def _calibration(cfg: dict) -> dict:
    return json.loads(Path(cfg["frozen_environment"]).read_text())["calibration"]


def build_cfg(cfg: dict, budget: int, view_kind: str) -> ExistenceConfig:
    """Frozen arm constants at one rollout budget."""
    ac, cal = cfg["arm_constants"], _calibration(cfg)
    every = (
        ac["report_every_calls"] if view_kind == "window" else ac["report_every_raw"]
    )
    return ExistenceConfig(
        mode=br.MODE_AFFINE, budget=int(budget), n_init=int(ac["n_init"]),
        report_every=int(every), n_candidates=int(ac["n_candidates"]),
        n_witness_candidates=int(ac["n_witness_candidates"]),
        report_grid_size=int(ac["report_grid_size"]), delta=float(ac["delta"]),
        lengthscale=float(cal["lengthscale"]), lam=float(cal["lam"]), R=0.0,
        sigma=0.0, variant=str(ac["variant"]), bisect_depth=int(ac["bisect_depth"]),
        certificate_scan=br.SCAN_OFF,
    )  # fmt: skip


class LoggedWindowView(WindowExistenceView):
    """Window view that logs (x, c, y) for the end-of-run band refit."""

    def __init__(self, env) -> None:
        """Wrap the environment with an empty log."""
        super().__init__(env)
        self.log: list[tuple[np.ndarray, float, np.ndarray]] = []

    def oracle(self, x, c, rng=None, sigma=0.0):
        """One window call, logged."""
        y = super().oracle(x, c, rng, sigma)
        self.log.append((np.array(x, dtype=float), float(c), np.array(y)))
        return y


class LoggedPredicateView(RawPredicateView):
    """One-rollout predicate view that logs (x, c, feasible)."""

    def __init__(self, *args, **kwargs) -> None:
        """Bind the predicate with an empty log."""
        super().__init__(*args, **kwargs)
        self.log: list[tuple[np.ndarray, float, bool]] = []

    def oracle(self, x, c, rng=None, sigma=0.0):
        """One predicate call, logged."""
        y = super().oracle(x, c, rng, sigma)
        self.log.append((np.array(x, dtype=float), float(c), bool(y[0] <= 0.0)))
        return y


class WindowScoreView:
    """Direct-score view: the window death c_off(x), charged at the window's cost.

    Args:
        env: Environment whose ``_oracle()`` returns the window
            oracle (``window(x)`` with ``c_off`` and ``n_raw_evals``).
    """

    def __init__(self, env) -> None:
        """Bind the window oracle."""
        self.env = env
        self.spec = ExistenceSpec(x_dim=mh.X_DIM, m=1)
        self.n_raw_evals = 0
        self.log: list[tuple[np.ndarray, float]] = []

    def score(self, x: np.ndarray) -> float:
        """Window death of design x (unit c); clamp when no window."""
        w = self.env._oracle().window(np.asarray(x, dtype=float))
        self.n_raw_evals += int(w.n_raw_evals)
        self.log.append((np.array(x, dtype=float), float(w.c_off)))
        return float(w.c_off)


def best_x_from_log(log_rows: list, lower: float) -> list[float] | None:
    """Sup-bisection's reported design: an x observed feasible at its final lower."""
    return next((x.tolist() for x, c, ok in log_rows if ok and c == lower), None)


def beta_sd_at_witness(view_log: list, run_cfg: ExistenceConfig, B, x_w, lower):
    """Refit the final bands and return beta·sd at the witness (reported only)."""
    X = np.array([r[0] for r in view_log])
    C = np.array([r[1] for r in view_log])
    Y = np.array([r[2] for r in view_log])
    bands = br.fit_joint_bands(
        X, C, Y, mode=br.MODE_AFFINE, lengthscale=run_cfg.lengthscale,
        lam=run_cfg.lam, B=B, R=0.0, betas=[float(B[0])], c_max=1.0,
    )  # fmt: skip
    _, sd = bands._mu_sd(0, np.array([x_w]), float(lower))
    return float(B[0]) * float(sd[0])


def run_task(task: tuple[int, str, int, int]) -> dict:
    """One (task index, arm label, rollout budget, seed) run in a fresh process."""
    t_task = time.perf_counter()
    idx, label, budget, seed = task
    cfg = me.load_s4_config()
    spec = next(a for a in cfg["arms"] if a["label"] == label)
    cal = _calibration(cfg)
    B = np.asarray(cal["B"], dtype=float)
    env = mh.HopperEnvironment(scaler_scale=np.asarray(cal["scaler_scale"]))
    run_cfg = build_cfg(cfg, budget, spec["view"])
    if spec["view"] == "window":
        view = LoggedWindowView(env)
    elif spec["view"] == "predicate":
        view = LoggedPredicateView(mh.HopperOracle().feasible, mh.X_DIM)
    else:
        view = WindowScoreView(env)
    t0 = time.perf_counter()
    res = run_existence_arm(view, spec["name"], run_cfg, B, seed)
    wall = time.perf_counter() - t0
    has_w = bool(res.reports and res.reports[-1].has_witness)
    L = float(res.final_lower)
    if spec["view"] == "predicate":
        x_rep = best_x_from_log(view.log, L) if has_w else None
    else:
        x_rep = res.final_witness_x if has_w else None
    lowers = [r.lower for r in res.reports]
    row = {
        "task_index": idx, "label": label, "arm": spec["name"], "view": spec["view"],
        "budget": int(budget), "seed": int(seed), "raw_evals": int(res.raw_evals),
        "wall_s": wall, "final_lower": L, "has_witness": has_w,
        "first_witness_raw": next(
            (r.raw_evals for r in res.reports if r.has_witness), None),
        "witness_x": x_rep, "n_resets": int(res.n_resets),
        "reports_monotone": bool(all(np.diff(lowers) >= 0)),
    }  # fmt: skip
    if spec["view"] != "predicate":
        row["window_counters"] = env.window_counters()
    row["check_feasible"] = row["robust_fraction"] = None
    if x_rep is not None:
        t1 = time.perf_counter()
        oracle = mh.HopperOracle()
        ch = cfg["checks"]
        r0 = oracle.rollout(np.asarray(x_rep), L, reset_seed=int(ch["reset_seed"]))
        row["check"] = asdict(r0)
        row["check_feasible"] = bool(np.all(r0.margins() <= 0.0))
        robust = [
            bool(np.all(oracle.rollout(np.asarray(x_rep), L, s).margins() <= 0.0))
            for s in ch["robustness_reset_seeds"]
        ]
        row["robust_fraction"] = float(np.mean(robust))
        row["check_s"] = time.perf_counter() - t1
        if spec["view"] == "window":
            row["beta_sd_at_witness"] = beta_sd_at_witness(
                view.log, run_cfg, B, x_rep, L
            )
    row["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    if sys.platform != "darwin":  # ru_maxrss is KiB on Linux, bytes on macOS
        row["peak_rss_gib"] *= 1024
    row["task_s"] = time.perf_counter() - t_task
    row["reports"] = [asdict(r) for r in res.reports]
    return row


def sweep_tasks(cfg: dict) -> list[tuple[int, str, int, int]]:
    """Task list: arm-major, then budget, then seed."""
    seeds = range(int(cfg["seeds"]["start"]), int(cfg["seeds"]["stop"]))
    cells = [
        (a["label"], int(b), s)
        for a in cfg["arms"]
        for b in cfg["budgets_raw"]
        for s in seeds
    ]
    return [(i, *c) for i, c in enumerate(cells)]


def _run_pool(tasks, workers: int, out_rows, label: str) -> tuple[float, int]:
    """Run tasks in fresh processes; returns (aggregate peak RSS, missed samples)."""
    mon = RssMonitor()
    mon.start()
    t0 = time.time()
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result()
            out_rows(row)
            log.info(
                "%s %d/%d %s b=%d s=%d: %.0fs L=%.4f wit=%s check=%s robust=%s",
                label, i, len(tasks), row["label"], row["budget"], row["seed"],
                row["task_s"], row["final_lower"], row["has_witness"],
                row["check_feasible"], row["robust_fraction"],
            )  # fmt: skip
            if i % 25 == 0:
                log.info(
                    "eta %.2f h", (len(tasks) - i) / (i / (time.time() - t0)) / 3600
                )
    mon.stop()
    return mon.peak_gib, mon.missed_samples


def _manifest(name: str, cfg: dict, workers: int, **extra) -> dict:
    return me._manifest(
        name, workers, config_sha256=me._config_sha256(),
        hostname=socket.gethostname(),
        cpu_count=os.cpu_count(), blas_threads=os.environ["VECLIB_MAXIMUM_THREADS"],
        started_utc=datetime.now(UTC).isoformat(), **extra,
    )  # fmt: skip


def remaining_tasks(cfg: dict, rows_path: Path) -> tuple[list, list[int]]:
    """Tasks whose row is not yet in ``rows.jsonl``."""
    done = {
        json.loads(ln)["task_index"]
        for ln in rows_path.read_text().splitlines()
        if ln.strip()
    }
    return [t for t in sweep_tasks(cfg) if t[0] not in done], sorted(done)


def cmd_sweep(workers: int, filt: dict, resume: Path | None = None) -> None:
    """The sweep (or the selected cells); ``resume`` continues a stopped directory."""
    cfg = me.load_s4_config()
    if resume is None:
        tasks = filter_tasks(sweep_tasks(cfg), **filt)
        manifest = _manifest("hopper_sweep", cfg, workers, n_tasks=len(tasks),
                             filter=filt)  # fmt: skip
        out = me._new_dir("hopper_sweep")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    else:
        out = resume
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["config_sha256"] != me._config_sha256():
            raise SystemExit("resume target used another config")
        tasks, done = remaining_tasks(cfg, out / "rows.jsonl")
        tasks = filter_tasks(tasks, **manifest.get("filter", {}))
        commit, dirty = me.git_state()
        manifest.setdefault("resumes", []).append({
            "resumed_utc": datetime.now(UTC).isoformat(), "git_commit": commit,
            "git_dirty": dirty, "workers": workers, "rows_done": len(done),
            "rows_remaining": len(tasks),
        })  # fmt: skip
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        log.info("resuming %s: %d done, %d remaining", out, len(done), len(tasks))
    t0 = time.time()
    with (out / "rows.jsonl").open("a") as fh:

        def write(row: dict) -> None:
            fh.write(json.dumps(row) + "\n")
            fh.flush()

        peak, missed = _run_pool(tasks, workers, write, "sweep")
    (out / "sweep_summary.json").write_text(json.dumps({
        "wall_h": (time.time() - t0) / 3600, "aggregate_peak_rss_gib": peak,
        "rss_missed_samples": missed, "n_rows": len(tasks),
        "finished_utc": datetime.now(UTC).isoformat(),
    }, indent=1))  # fmt: skip
    log.info("sweep done in %.2f h -> %s", (time.time() - t0) / 3600, out)


def _reject(count: int, n: int, null: float, alpha: float) -> dict:
    p = binomtest(count, n, null, alternative="greater").pvalue if n else 1.0
    return {"count": count, "n": n, "p": float(p), "reject": bool(n and p < alpha)}


def analyze_rows(
    rows: list[dict],
    cfg: dict,
    columns: dict | None = None,
    desk: dict | None = None,
    t_search: float | None = None,
    t_search_x: list[float] | None = None,
) -> dict:
    """Predeclared gates S4-G1, G2, G4 and the reported metrics.

    Args:
        rows: Sweep rows (``run_task`` output).
        cfg: Config.
        columns: ``columns.json`` of the scored columns (U per eps).
        desk: ``desk_check.json`` of the S4-M1 desk check.
        t_search: The truth search's verified value (one term of T_ref).
        t_search_x: The design that attains it.

    Returns:
        Dict with each gate's cells and verdict, the search comparison, the
        robustness diagnostic, and T_ref with per-cell gaps.
    """
    g, alpha = cfg["gates"], float(cfg["gates"]["alpha"])
    cell = {}
    for r in rows:
        cell.setdefault((r["label"], int(r["budget"])), []).append(r)
    budgets = [int(b) for b in cfg["budgets_raw"]]
    g1 = {}
    for lab in CERTIFIED:
        for b in budgets:
            rs = cell[(lab, b)]
            wit = [r for r in rs if r["has_witness"]]
            fails = sum(1 for r in wit if not r["check_feasible"])
            frac = (len(wit) - fails) / len(wit) if wit else 0.0
            g1[f"{lab}/{b}"] = {
                "witnesses": len(wit),
                "feasible_fraction": frac,
                "failures": _reject(fails, len(rs), g["g1_violation_delta"], alpha),
                "non_monotone": sum(1 for r in rs if not r["reports_monotone"]),
            }
    g1_pass = all(
        c["witnesses"] > 0
        and c["feasible_fraction"] >= float(g["g1_witness_feasible_min"])
        and not c["failures"]["reject"]
        and c["non_monotone"] == 0
        for c in g1.values()
    )
    g2 = {}
    for b in budgets:
        rs = cell[(METHOD, b)]
        k = sum(r["has_witness"] for r in rs)
        p = binomtest(k, len(rs), g["g2_witness_rate_null"], alternative="greater")
        g2[str(b)] = {"count": k, "n": len(rs), "p": float(p.pvalue)}
    g2_pass = all(c["p"] < alpha for c in g2.values())
    if columns is None or desk is None:
        g4 = {"pass": None}
    else:
        U = float(columns["eps_0.01"]["U"])
        g4 = {
            "U_0.01": U,
            "U_0.05": columns.get("eps_0.05", {}).get("U"),
            "desk_flags": int(desk["n_flagged"]),
            "pass": bool(np.isfinite(U) and U < 1.0 and desk["n_flagged"] == 0),
        }
    comparison: dict = {}
    for b in budgets:
        m = {r["seed"]: r["final_lower"] for r in cell[(METHOD, b)]}
        comparison[str(b)] = {}
        for a in cfg["arms"]:
            if a["label"] == METHOD:
                continue
            o = {r["seed"]: r["final_lower"] for r in cell[(a["label"], b)]}
            d = np.array([m[s] - o[s] for s in m if s in o])
            w, lo = int((d > 0).sum()), int((d < 0).sum())
            comparison[str(b)][a["label"]] = {
                "wins": w, "losses": lo, "ties": int((d == 0).sum()),
                "p": float(binomtest(w, w + lo).pvalue) if w + lo else 1.0,
                "median_method": float(np.median(list(m.values()))),
                "median_other": float(np.median(list(o.values()))),
            }  # fmt: skip
    points = [
        (r["final_lower"], r.get("witness_x"), "{label}/{budget}/{seed}".format(**r))
        for r in rows
        if r.get("check_feasible")
    ]
    if t_search is not None:
        points.append((t_search, t_search_x, "truth search"))
    t_ref, t_ref_x, t_ref_source = max(points, key=lambda p: p[0])
    per_cell = {}
    for (lab, b), rs in cell.items():
        rob = [r["robust_fraction"] for r in rs if r["robust_fraction"] is not None]
        chk = [r["check_feasible"] for r in rs if r["check_feasible"] is not None]
        per_cell[f"{lab}/{b}"] = {
            "median_L": float(np.median([r["final_lower"] for r in rs])),
            "median_gap_to_t_ref": float(
                np.median([t_ref - r["final_lower"] for r in rs])
            ),
            "witnesses": len(chk),
            "check_pass": int(sum(chk)),
            "median_robust_fraction": float(np.median(rob)) if rob else None,
        }
    return {
        "G1": {"pass": g1_pass, "cells": g1},
        "G2": {"pass": g2_pass, "cells": g2},
        "G4": g4,
        "search_comparison": comparison,
        "t_ref": float(t_ref),
        "t_ref_x": t_ref_x,
        "t_ref_source": t_ref_source,
        "t_search": t_search,
        "per_cell": per_cell,
    }


def cmd_analyze(run: Path, columns: Path | None, desk: Path | None, truth: Path | None):
    """Gates and metrics of a finished sweep -> ``<run>/gate.json``."""
    cfg = me.load_s4_config()
    rows = [json.loads(ln) for ln in (run / "rows.jsonl").read_text().splitlines()]
    tr = json.loads((truth / "truth.json").read_text()) if truth else {}
    out = analyze_rows(
        rows,
        cfg,
        json.loads((columns / "columns.json").read_text()) if columns else None,
        json.loads((desk / "desk_check.json").read_text()) if desk else None,
        tr.get("t_search_unit"),
        tr.get("argmax_x_unit"),
    )
    out["config_sha256"] = me._config_sha256()
    (run / "gate.json").write_text(json.dumps(out, indent=1))
    log.info("G1 %s, G2 %s, G4 %s -> %s", out["G1"]["pass"], out["G2"]["pass"],
             out["G4"]["pass"], run / "gate.json")  # fmt: skip


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("--workers", type=int, default=30)
    sw.add_argument("--resume", type=Path)
    add_filter_args(sw)
    an = sub.add_parser("analyze")
    for opt in ("--run", "--columns", "--desk", "--truth"):
        an.add_argument(opt, type=Path)
    args = ap.parse_args()
    if args.cmd == "sweep":
        filt = {"labels": args.labels, "budgets": args.budgets, "seeds": args.seeds}
        cmd_sweep(args.workers, filt, args.resume)
    else:
        cmd_analyze(
            args.run or run_path("hopper_sweep"),
            args.columns or run_path("hopper_columns"),
            args.desk or run_path("hopper_desk_check"),
            args.truth or run_path("hopper_truth"),
        )
