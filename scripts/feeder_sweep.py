"""Distribution network (CIGRE MV feeder): the method and baseline sweep, and its gates.

Every constant comes from ``configs/feeder.yaml``. Six methods at two
budgets of pass/fail checks (one check = 16 sampled operating conditions),
50 seeds each: BRINK's proposed search (``witness-focused``), BRINK's
variance-seeking rule (``greedy``), the CONFIG adaptation (``config``), and
bisection on random designs with 1, 4, or 16 checks per decision
(``baseline-k16``, ``baseline-k64``, ``baseline-k256``; charged 1, 4, or 16
checks). The model-based methods run one time-capped infeasibility test at
the end of the search. Every returned design is checked with 4,096 new
conditions (``feeder_setup._witness_check``, stream index
``stream_offset + task index``). Rows record the reference limit of the
latest ``feeder_truth`` run (``configs/runs.yaml``), so run ``feeder_setup.py
truth`` first.

Usage:
    uv run python scripts/feeder_sweep.py sweep --workers N \
        [--labels L ...] [--budgets B ...] [--seeds S ...] [--resume DIR]
    uv run python scripts/feeder_sweep.py analyze [RESULTS_DIR] [--columns DIR] \
        [--monotone DIR]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

_CFG_PATH = Path("configs/feeder.yaml")
if _CFG_PATH.exists():  # BLAS thread count must be set before numpy loads
    os.environ.setdefault(
        "VECLIB_MAXIMUM_THREADS",
        str(yaml.safe_load(_CFG_PATH.read_text())["arm_constants"]["blas_threads"]),
    )

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from brink import bracket as br  # noqa: E402
from brink.adapter import (  # noqa: E402
    RawPredicateView,
    WindowExistenceView,
)
from brink.arms import ExistenceConfig, run_existence_arm  # noqa: E402
from brink.gp import beta_sqrt  # noqa: E402
from brink.memory import RssMonitor  # noqa: E402
from brink.studies import feeder as hc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feeder_setup as he  # noqa: E402
from common import add_filter_args, filter_tasks, git_state, run_path  # noqa: E402

log = logging.getLogger("feeder_sweep")


def build_cfg(cfg: dict, budget: int, view_kind: str, R: float, lengthscale: float):
    """Frozen arm constants at one raw budget."""
    ac = cfg["arm_constants"]
    return ExistenceConfig(
        mode=br.MODE_AFFINE, budget=int(budget), n_init=int(ac["n_init"]),
        report_every=int(ac["report_every_calls"] if view_kind == "window"
                         else ac["report_every_raw"]),
        n_candidates=int(ac["n_candidates"]),
        n_witness_candidates=int(ac["n_witness_candidates"]),
        report_grid_size=int(ac["report_grid_size"]),
        cover_n_per_axis=int(ac["cover_n_per_axis"]), delta=float(ac["delta"]),
        lengthscale=float(lengthscale), lam=float(R) ** 2, R=float(R), sigma=0.0,
        variant=str(ac["variant"]), bisect_depth=int(ac["bisect_depth"]),
        certificate_scan=br.SCAN_OFF,
    )  # fmt: skip


class LoggedWindowView(WindowExistenceView):
    """Window view that logs (x, c, y) for the end-of-run refit."""

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
    """Predicate view that logs (x, c, feasible) for the best-x recovery."""

    def __init__(self, *args, **kwargs) -> None:
        """Bind the predicate with an empty log."""
        super().__init__(*args, **kwargs)
        self.log: list[tuple[np.ndarray, float, bool]] = []

    def oracle(self, x, c, rng=None, sigma=0.0):
        """One predicate call, logged."""
        y = super().oracle(x, c, rng, sigma)
        self.log.append((np.array(x, dtype=float), float(c), bool(y[0] <= 0.0)))
        return y


def make_view(spec: dict, env, raw_oracle, k_snapshots: int):
    """The adapter view of one arm.

    Args:
        spec: Arm entry of the config (``view`` window or predicate, and the
            predicate's ``snapshots``).
        env: Frozen hosting environment (window oracle).
        raw_oracle: Raw hosting oracle with ``mean_feasible(x, c, k, rng)``.
        k_snapshots: Raw unit (snapshots per raw evaluation).

    Returns:
        A logged window or predicate view; the predicate view charges
        ``snapshots / k_snapshots`` raw evaluations per decision.
    """
    if spec["view"] == "window":
        return LoggedWindowView(env)
    snaps = int(spec["snapshots"])
    if snaps % int(k_snapshots):
        raise ValueError(f"snapshots {snaps} not a multiple of K = {k_snapshots}")
    return LoggedPredicateView(
        lambda x, c, rng: raw_oracle.mean_feasible(x, c, snaps, rng),
        he.X_DIM, needs_rng=True, raw_evals_per_call=snaps // int(k_snapshots),
    )  # fmt: skip


def best_x_from_log(log_rows: list, lower: float) -> list[float] | None:
    """The baseline's reported design: an x observed feasible at its final lower."""
    return next((x.tolist() for x, c, ok in log_rows if ok and c == lower), None)


def end_scan(
    view_log: list,
    run_cfg: ExistenceConfig,
    env,
    R: float,
    lower: float,
    witness_x: list[float] | None,
    ac: dict,
) -> dict:
    """Refit the final bands; beta·sd at the witness; one capped smallest scan."""
    X = np.array([r[0] for r in view_log])
    C = np.array([r[1] for r in view_log])
    Y = np.array([r[2] for r in view_log])
    bands = br.fit_joint_bands(
        X, C, Y, mode=br.MODE_AFFINE, lengthscale=run_cfg.lengthscale,
        lam=run_cfg.lam, B=env.B, R=R, betas=[float(env.B[0])], c_max=1.0,
    )  # fmt: skip
    beta = beta_sqrt(bands.models[0], run_cfg.delta, run_cfg.variant)
    bands.betas = [beta]
    out: dict = {"beta_end": float(beta)}
    if witness_x is not None:
        xw = np.array([witness_x])
        _, sd_w = bands._mu_sd(0, xw, lower)
        a, b = bands._mu_sd(0, xw, 0.0)[0][0], bands._mu_sd(0, xw, 1.0)[0][0]
        out["beta_sd_at_witness"] = float(beta * sd_w[0])
        out["c_off_hat_at_witness"] = float(-a / (b - a)) if b != a else None
    scan = ac["end_scan"]
    cover, rho = br.midpoint_cover(run_cfg.cover_n_per_axis, he.X_DIM)
    grid = np.linspace(0.0, 1.0, run_cfg.report_grid_size)
    adaptive = {
        "max_depth": int(scan["max_depth"]), "max_cells": int(scan["max_cells"]),
        "time_cap_s": float(scan["cap_s"]),
    }  # fmt: skip
    t0 = time.perf_counter()
    if witness_x is not None:  # smallest certified slice upward from L
        rule = "smallest-from-L"
        upper, ok = br.smallest_certified_slice(
            bands, grid, cover, rho, stop_below=lower, **adaptive
        )
    else:  # no design: descend from c_max while stresses are excluded
        rule = "top-down"
        upper, ok = br.certificate_upper_bound(bands, grid, cover, rho, **adaptive)
    seconds = time.perf_counter() - t0
    out["scan"] = {
        "rule": rule, "upper": float(upper), "has_certificate": bool(ok),
        "seconds": seconds, "capped": seconds >= float(scan["cap_s"]),
    }  # fmt: skip
    return out


def _truth(cfg: dict | None = None) -> tuple[float, float]:
    """Reference limit and its stated error from the latest ``feeder_truth`` run."""
    truth = json.loads((run_path("feeder_truth") / "truth.json").read_text())
    err = sum(truth["error_terms_unit_c"][k] for k in (
        "truth_bisection_half", "window_oracle_bisection_half",
        "x_refinement_last_improvement", "mc_standard_error_of_argmax_mean",
    ))  # fmt: skip
    return float(truth["theta_exists_unit"]), float(err)


def run_task(task: tuple[int, str, int, int]) -> dict:
    """One (task index, arm label, raw budget, seed) run in a fresh process."""
    idx, label, budget, seed = task
    cfg = he.load_s3_config()
    spec = next(a for a in cfg["arms"] if a["label"] == label)
    K = int(cfg["k_snapshots"])
    env = hc.load_hosting_environment(cfg["frozen_environment"], K)
    R = float(env.R_by_knob[K])
    cal = json.loads(Path(cfg["frozen_environment"]).read_text())["calibration"]
    run_cfg = build_cfg(cfg, budget, spec["view"], R, cal["lengthscale"])
    view = make_view(spec, env, he._oracle(), K)
    theta, err = _truth(cfg)
    t0 = time.perf_counter()
    res = run_existence_arm(view, spec["name"], run_cfg, env.B, seed)
    wall = time.perf_counter() - t0
    has_w = bool(res.reports and res.reports[-1].has_witness)
    if spec["view"] == "window":
        x_rep = res.final_witness_x
    else:
        x_rep = best_x_from_log(view.log, res.final_lower) if has_w else None
    row = {
        "task_index": idx, "label": label, "arm": spec["name"], "view": spec["view"],
        "budget": int(budget), "seed": int(seed), "raw_evals": int(res.raw_evals),
        "wall_s": wall, "final_lower": float(res.final_lower),
        "final_upper": float(res.final_upper), "has_witness": has_w,
        "first_witness_raw": next(
            (r.raw_evals for r in res.reports if r.has_witness), None),
        "witness_x": x_rep, "theta": theta, "theta_err": err,
        "theta_minus_lower": theta - float(res.final_lower),
        "anytime_lower_violation": bool(
            any(r.lower > theta + err for r in res.reports)),
        "final_lower_violation": bool(res.final_lower > theta + err),
        "n_resets": int(getattr(res, "n_resets", 0)),
    }  # fmt: skip
    if spec["view"] == "window":
        row["end_scan"] = end_scan(
            view.log, run_cfg, env, R, float(res.final_lower), x_rep,
            cfg["arm_constants"],
        )  # fmt: skip
    if x_rep is not None:
        t1 = time.perf_counter()
        w = int(cfg["truth_check"]["stream_offset"]) + idx
        row["truth_check"] = he._witness_check((w, x_rep, float(res.final_lower)))
        row["truth_check"]["stream_index"] = w
        row["truth_check_s"] = time.perf_counter() - t1
        row["witness_truly_feasible"] = bool(row["truth_check"]["feasible"])
    else:
        row["truth_check"] = None
        row["witness_truly_feasible"] = None
    row["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    row["reports"] = [asdict(r) for r in res.reports]
    return row


def sweep_tasks(cfg: dict) -> list[tuple[int, str, int, int]]:
    """Task list: arm-major, then budget, then seed."""
    seeds = range(int(cfg["seeds"]["start"]), int(cfg["seeds"]["stop"]))
    cells = [(a["label"], int(b), s) for a in cfg["arms"] for b in cfg["budgets_raw"]
             for s in seeds]  # fmt: skip
    return [(i, *c) for i, c in enumerate(cells)]


def _run_pool(tasks, workers: int, out_rows, label: str) -> float:
    """Run tasks in fresh processes, appending rows; returns aggregate peak RSS."""
    mon = RssMonitor()
    mon.start()
    t0 = time.time()
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result()
            out_rows(row)
            log.info(
                "%s %d/%d %s b=%d s=%d: %.0fs L=%.4f wit=%s feas=%s raw=%d",
                label, i, len(tasks), row["label"], row["budget"], row["seed"],
                row["wall_s"], row["final_lower"], row["has_witness"],
                row["witness_truly_feasible"], row["raw_evals"],
            )  # fmt: skip
            if i % 25 == 0:
                rate = i / (time.time() - t0)
                log.info("eta %.2f h", (len(tasks) - i) / rate / 3600)
    mon.stop()
    if mon.missed_samples:
        log.warning("RSS monitor missed %d samples", mon.missed_samples)
    return mon.peak_gib


def _manifest(out: Path, cfg: dict, workers: int, n_tasks: int, filt: dict) -> None:
    commit, dirty = git_state()
    theta, err = _truth(cfg)
    (out / "manifest.json").write_text(json.dumps({
        "experiment": "feeder_sweep", "git_commit": commit, "git_dirty": dirty,
        "config_sha256": cfg["sha256"],
        "frozen_calibration_sha256": hashlib.sha256(
            Path(cfg["frozen_environment"]).read_bytes()).hexdigest(),
        "truth_dir": str(run_path("feeder_truth")), "theta": theta, "theta_err": err,
        "filter": filt, "n_tasks": n_tasks, "workers": workers,
        "blas_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "python": sys.version, "numpy": np.__version__,
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "started_utc": datetime.now(UTC).isoformat(),
    }, indent=1))  # fmt: skip


def remaining_tasks(cfg: dict, rows_path: Path) -> tuple[list, list[int]]:
    """Tasks whose row is not yet in ``rows.jsonl`` (resume after a stop).

    Every task is deterministic given its (index, method, budget, seed), so a
    resumed sweep writes the same rows a clean run would, except that a
    time-capped end scan can stop at a different stress under different load.

    Args:
        cfg: Study config.
        rows_path: The sweep's ``rows.jsonl``.

    Returns:
        (remaining tasks in sweep order, sorted done task indices).
    """
    done = {
        json.loads(ln)["task_index"]
        for ln in rows_path.read_text().splitlines()
        if ln.strip()
    }
    return [t for t in sweep_tasks(cfg) if t[0] not in done], sorted(done)


def cmd_sweep(workers: int, filt: dict, resume: Path | None = None) -> None:
    """The sweep (or the selected cells) -> ``results/<UTC>_feeder_sweep/``.

    ``resume`` continues a stopped sweep directory: rows already written are
    kept, the remaining tasks run, and the manifest records the resume.
    """
    cfg = he.load_s3_config()
    if resume is None:
        out = Path("results") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_feeder_sweep"
        out.mkdir(parents=True)
        tasks = filter_tasks(sweep_tasks(cfg), **filt)
        _manifest(out, cfg, workers, len(tasks), filt)
    else:
        out = resume
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["config_sha256"] != cfg["sha256"]:
            raise SystemExit("resume target used another config")
        tasks, done = remaining_tasks(cfg, out / "rows.jsonl")
        tasks = filter_tasks(tasks, **manifest.get("filter", {}))
        commit, dirty = git_state()
        manifest.setdefault("resumes", []).append({
            "resumed_utc": datetime.now(UTC).isoformat(), "git_commit": commit,
            "git_dirty": dirty, "workers": workers, "rows_done": len(done),
            "rows_remaining": len(tasks),
        })  # fmt: skip
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        log.info("resuming %s: %d rows done, %d remaining", out, len(done), len(tasks))
    t0 = time.time()
    with (out / "rows.jsonl").open("a") as fh:

        def write(row: dict) -> None:
            fh.write(json.dumps(row) + "\n")
            fh.flush()

        peak = _run_pool(tasks, workers, write, "sweep")
    (out / "sweep_summary.json").write_text(json.dumps({
        "wall_h": (time.time() - t0) / 3600.0, "aggregate_peak_rss_gib": peak,
        "n_rows": len(tasks), "finished_utc": datetime.now(UTC).isoformat(),
    }, indent=1))  # fmt: skip
    log.info("sweep done in %.2f h, peak RSS %.2f GiB -> %s",
             (time.time() - t0) / 3600.0, peak, out)  # fmt: skip


def _reject(count: int, n: int, null: float, alpha: float) -> dict:
    p = binomtest(count, n, null, alternative="greater").pvalue if n else 1.0
    return {"count": count, "n": n, "p": float(p), "reject": bool(n and p < alpha)}


def analyze_rows(rows: list[dict], cfg: dict, columns: dict | None = None,
                 monotone: dict | None = None) -> dict:  # fmt: skip
    """Predeclared gates S3-G1 to G4 and the reported metrics.

    Args:
        rows: Sweep rows (``run_task`` output).
        cfg: Config.
        columns: ``score-columns`` summary (``by_eps``), if available.
        monotone: ``monotone-check`` summary (``passed``), if available.

    Returns:
        Gate dictionary with a ``verdict`` block.
    """
    g, alpha = cfg["gates"], float(cfg["gates"]["alpha"])
    labels = [a["label"] for a in cfg["arms"]]
    certified = [a["label"] for a in cfg["arms"] if a["view"] == "window"]
    method = labels[0]
    budgets = [int(b) for b in cfg["budgets_raw"]]
    cell = {(lab, b): [r for r in rows if r["label"] == lab and r["budget"] == b]
            for lab in labels for b in budgets}  # fmt: skip
    out: dict = {"n_rows": len(rows), "G1": {}, "G2": {}, "G3": {}, "G4": {},
                 "hardened": {}, "cells": {}, "paired": {}}  # fmt: skip
    g1 = True
    for lab in certified:
        for b in budgets:
            rs = cell[(lab, b)]
            if not rs:
                continue
            viol = _reject(sum(r["anytime_lower_violation"] for r in rs), len(rs),
                           float(g["g1_violation_delta"]), alpha)  # fmt: skip
            wit = [r for r in rs if r["witness_truly_feasible"] is not None]
            rate = (sum(r["witness_truly_feasible"] for r in wit) / len(wit)
                    if wit else None)  # fmt: skip
            ok_b = rate is None or rate >= float(g["g1_witness_feasible_min"])
            out["G1"][f"{lab}/{b}"] = {
                "violations": viol, "witness_feasible_rate": rate,
                "pass": bool(not viol["reject"] and ok_b),
            }  # fmt: skip
            g1 &= out["G1"][f"{lab}/{b}"]["pass"]
    out["G1"]["pass"] = bool(g1)
    g2 = True
    for b in budgets:
        rs = cell[(method, b)]
        t = _reject(sum(r["has_witness"] for r in rs), len(rs),
                    float(g["g2_witness_rate_null"]), alpha)  # fmt: skip
        out["G2"][str(b)] = {**t, "pass": t["reject"]}
        g2 &= t["reject"]
    out["G2"]["pass"] = bool(g2)

    def wrong_test(lab: str, b: int) -> dict:
        rs = [r for r in cell[(lab, b)] if r["witness_x"] is not None]
        return _reject(sum(not r["witness_truly_feasible"] for r in rs), len(rs),
                       float(g["g3_wrong_rate_null"]), alpha)  # fmt: skip

    naive = labels[3]
    g3 = True
    for b in budgets:
        t = wrong_test(naive, b)
        out["G3"][str(b)] = {**t, "pass": t["reject"]}
        g3 &= t["reject"]
    out["G3"]["pass"] = bool(g3)
    tol = float(g["hardened_gap_tolerance"])
    for lab in labels[4:]:
        for b in budgets:
            t = wrong_test(lab, b)
            med_h = float(
                np.median([r["final_lower"] for r in cell[(lab, b)]] or [np.nan])
            )
            med_m = float(
                np.median([r["final_lower"] for r in cell[(method, b)]] or [np.nan])
            )
            if t["reject"]:
                reading = "also fails: exposure robust to replication"
            elif med_h < med_m - tol:
                reading = "passes with a much lower L: replication trades the answer"
            else:
                reading = "passes with comparable L: naive baseline is a straw man"
            out["hardened"][f"{lab}/{b}"] = {**t, "median_lower": med_h,
                                             "method_median_lower": med_m,
                                             "reading": reading}  # fmt: skip
    by_eps = (columns or {}).get("by_eps", {})
    u = by_eps.get("0.01", {})
    out["G4"] = {
        "U_0.01": u.get("U_eps"), "k": u.get("k"),
        "U_0.05": by_eps.get("0.05", {}).get("U_eps"),
        "monotone_passed": None if monotone is None else bool(monotone["passed"]),
        "pass": None if not u or monotone is None
        else bool(u.get("finite") and monotone["passed"]),
    }  # fmt: skip
    for (lab, b), rs in cell.items():
        if not rs:
            continue
        fw = [r["first_witness_raw"] for r in rs if r["first_witness_raw"] is not None]
        scans = [r.get("end_scan", {}).get("scan") for r in rs if r.get("end_scan")]
        out["cells"][f"{lab}/{b}"] = {
            "n": len(rs),
            "median_lower": float(np.median([r["final_lower"] for r in rs])),
            "median_theta_minus_lower": float(np.median(
                [r["theta_minus_lower"] for r in rs])),
            "witness_rate": sum(r["has_witness"] for r in rs) / len(rs),
            "median_first_witness_raw": float(np.median(fw)) if fw else None,
            "truly_feasible": sum(bool(r["witness_truly_feasible"]) for r in rs),
            "anytime_violations": sum(r["anytime_lower_violation"] for r in rs),
            "final_violations": sum(r["final_lower_violation"] for r in rs),
            "end_scan_certificates": sum(
                bool(s and s["has_certificate"]) for s in scans),
            "end_scan_capped": sum(bool(s and s["capped"]) for s in scans),
            "median_wall_s": float(np.median([r["wall_s"] for r in rs])),
            "median_raw_evals": float(np.median([r["raw_evals"] for r in rs])),
        }  # fmt: skip
    for lab in certified[1:]:
        for b in budgets:
            by_seed = {r["seed"]: r["final_lower"] for r in cell[(method, b)]}
            d = [r["final_lower"] - by_seed[r["seed"]] for r in cell[(lab, b)]
                 if r["seed"] in by_seed]  # fmt: skip
            wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
            out["paired"][f"{lab}-vs-{method}/{b}"] = {
                "wins": wins, "losses": losses, "ties": len(d) - wins - losses,
                "sign_p": float(binomtest(wins, wins + losses, 0.5).pvalue)
                if wins + losses else None,
            }  # fmt: skip
    out["verdict"] = {"G1": out["G1"]["pass"], "G2": out["G2"]["pass"],
                      "G3": out["G3"]["pass"], "G4": out["G4"]["pass"],
                      "lower_bound_claim": bool(g1 and g2),
                      "exposure_claim": bool(g3),
                      "two_sided_statement": bool(out["G4"]["pass"])}  # fmt: skip
    return out


def cmd_analyze(results_dir: Path, columns_dir: Path | None,
                monotone_dir: Path | None) -> None:  # fmt: skip
    """Gate analysis of a sweep directory (plus the column and monotone summaries)."""
    cfg = he.load_s3_config()
    rows = [json.loads(x) for x in (results_dir / "rows.jsonl").read_text().splitlines()
            if x.strip()]  # fmt: skip
    cols = (
        json.loads((columns_dir / "columns.json").read_text()) if columns_dir else None
    )
    mono = (
        json.loads((monotone_dir / "check.json").read_text()) if monotone_dir else None
    )
    gate = analyze_rows(rows, cfg, cols, mono)
    gate["config_sha256"] = cfg["sha256"]
    (results_dir / "gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate["verdict"], indent=1))
    print(json.dumps({k: v["reading"] for k, v in gate["hardened"].items()}, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("--workers", type=int, required=True)
    sw.add_argument("--resume", type=Path, default=None)
    add_filter_args(sw)
    an = sub.add_parser("analyze")
    an.add_argument("results_dir", type=Path, nargs="?", default=None)
    an.add_argument("--columns", type=Path, default=None)
    an.add_argument("--monotone", type=Path, default=None)
    args = ap.parse_args()
    if args.cmd == "sweep":
        filt = {"labels": args.labels, "budgets": args.budgets, "seeds": args.seeds}
        cmd_sweep(args.workers, filt, args.resume)
    else:
        cmd_analyze(
            args.results_dir or run_path("feeder_sweep"),
            args.columns or run_path("feeder_columns"),
            args.monotone or run_path("feeder_monotone"),
        )
