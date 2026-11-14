"""Covasim epidemic policy: the method and baseline sweep, and its gates.

Every constant comes from ``configs/covasim.yaml``. Six methods at two run
budgets: BRINK's proposed search (``witness-focused``, 30 seeds), BRINK's
variance-seeking rule and the CONFIG adaptation (``greedy``, ``config``, 10
seeds each), and bisection on random policies with 1, 4, or 16 runs per
decision (``baseline-k1``, ``baseline-k4``, ``baseline-k16``, 30 seeds each,
charged 1, 4, or 16 runs). Task order, logged views, design recovery, and
the gates are the feeder sweep's (``feeder_sweep``); this file supplies the
Covasim environment, the 3-D end scan, and the 128-run design check. Rows
record the reference limit of the latest ``covasim_truth`` run, so run
``covasim_setup.py truth`` first.

The sweep may run as ``n`` parts on separate machines (``--part k/n``): each
task has fixed seeds and runs in a fresh process, so the union of the parts'
rows equals the one-job rows. List every part in ``configs/runs.yaml``.

Usage:
    uv run python scripts/covasim_sweep.py sweep --workers N [--part K/N] \
        [--labels L ...] [--budgets B ...] [--seeds S ...] [--resume DIR]
    uv run python scripts/covasim_sweep.py analyze [SWEEP_DIR ...] [--columns DIR] \
        [--monotone DIR]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # before numpy (S3 rule)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import covasim_setup as ce  # noqa: E402
import feeder_sweep as hs  # noqa: E402
import numpy as np  # noqa: E402
from common import add_filter_args, filter_tasks, git_state, run_path  # noqa: E402

from brink import bracket as br  # noqa: E402
from brink.arms import run_existence_arm  # noqa: E402
from brink.gp import beta_sqrt  # noqa: E402
from brink.memory import RssMonitor  # noqa: E402
from brink.studies import covasim_policy as cp  # noqa: E402

log = logging.getLogger("covasim_sweep")


def load_config() -> dict:
    """The Covasim constants with their SHA-256."""
    return ce.he.load_s3_config(ce.S5_CONFIG)


def make_view(spec: dict, env):
    """Logged window view (K = 1) or logged K'-run predicate view."""
    if spec["view"] == "window":
        return hs.LoggedWindowView(env)
    k = int(spec["runs"])
    return hs.LoggedPredicateView(
        lambda x, c, rng: cp.mean_feasible(x, c, k, rng), cp.X_DIM,
        needs_rng=True, raw_evals_per_call=k,
    )  # fmt: skip


def end_scan(view_log: list, run_cfg, env, lower: float, witness_x, ac: dict) -> dict:
    """The S3 end scan (beta·sd at the witness; one capped certificate scan) in 3-D."""
    X = np.array([r[0] for r in view_log])
    C = np.array([r[1] for r in view_log])
    Y = np.array([r[2] for r in view_log])
    bands = br.fit_joint_bands(
        X, C, Y, mode=br.MODE_AFFINE, lengthscale=run_cfg.lengthscale,
        lam=run_cfg.lam, B=env.B, R=env.R, betas=[float(env.B[0])], c_max=1.0,
    )  # fmt: skip
    beta = beta_sqrt(bands.models[0], run_cfg.delta, run_cfg.variant)
    bands.betas = [beta]
    out: dict = {"beta_end": float(beta)}
    if witness_x is not None:
        xw = np.array([witness_x])
        out["beta_sd_at_witness"] = float(beta * bands._mu_sd(0, xw, lower)[1][0])
    scan = ac["end_scan"]
    cover, rho = br.midpoint_cover(run_cfg.cover_n_per_axis, cp.X_DIM)
    grid = np.linspace(0.0, 1.0, run_cfg.report_grid_size)
    kw = {"max_depth": int(scan["max_depth"]), "max_cells": int(scan["max_cells"]),
          "time_cap_s": float(scan["cap_s"])}  # fmt: skip
    t0 = time.perf_counter()
    if witness_x is not None:
        rule = "smallest-from-L"
        upper, ok = br.smallest_certified_slice(
            bands, grid, cover, rho, stop_below=lower, **kw
        )
    else:
        rule = "top-down"
        upper, ok = br.certificate_upper_bound(bands, grid, cover, rho, **kw)
    sec = time.perf_counter() - t0
    out["scan"] = {"rule": rule, "upper": float(upper), "has_certificate": bool(ok),
                   "seconds": sec, "capped": sec >= float(scan["cap_s"])}  # fmt: skip
    return out


def truth_check(cfg: dict, w: int, x: list, lower: float) -> dict:
    """The large-replicate check at (x, L) on stream [994, offset + w]."""
    tc = cfg["truth_check"]
    rng = np.random.default_rng([*tc["stream"], int(tc["stream_offset"]) + w])
    g = cp.margins(np.asarray(x), float(lower), int(tc["runs"]), rng)
    return {
        **ce.he.margin_check(g[:, None]),
        "stream_index": int(tc["stream_offset"]) + w,
    }


def _truth(cfg: dict | None = None) -> tuple[float, float]:
    """Reference limit and its stated error from the latest ``covasim_truth`` run."""
    t = json.loads((run_path("covasim_truth") / "result.json").read_text())
    return float(t["theta_exists_unit"]), float(t["theta_err"])


def run_arm(cfg: dict, spec: dict, budget: int, seed: int):
    """One arm run on the frozen environment: (result, view, env, cfg, s)."""
    env = cp.load_environment(cfg["frozen_environment"], 1)
    run_cfg = hs.build_cfg(cfg, budget, spec["view"], env.R, env.lengthscale)
    view = make_view(spec, env)
    t0 = time.perf_counter()
    res = run_existence_arm(view, spec["name"], run_cfg, env.B, seed)
    return res, view, env, run_cfg, time.perf_counter() - t0


def run_task(task: tuple[int, str, int, int]) -> dict:
    """One (task index, arm label, run budget, seed) run in a fresh process."""
    idx, label, budget, seed = task
    cfg = load_config()
    spec = next(a for a in cfg["arms"] if a["label"] == label)
    theta, err = _truth(cfg)
    t0 = time.perf_counter()
    res, view, env, run_cfg, wall = run_arm(cfg, spec, budget, seed)
    has_w = bool(res.reports and res.reports[-1].has_witness)
    if spec["view"] == "window":
        x_rep = res.final_witness_x
    else:
        x_rep = hs.best_x_from_log(view.log, res.final_lower) if has_w else None
    row = {
        "task_index": idx, "label": label, "arm": spec["name"], "view": spec["view"],
        "budget": int(budget), "seed": int(seed), "raw_evals": int(res.raw_evals),
        "wall_s": wall, "final_lower": float(res.final_lower),
        "final_upper": float(res.final_upper), "has_witness": has_w,
        "first_witness_raw": next(
            (r.raw_evals for r in res.reports if r.has_witness), None),
        "witness_x": x_rep, "witness_policy": None if x_rep is None
        else list(cp.to_policy(np.asarray(x_rep))),
        "theta": theta, "theta_err": err,
        "theta_minus_lower": theta - float(res.final_lower),
        "anytime_lower_violation": bool(
            any(r.lower > theta + err for r in res.reports)),
        "final_lower_violation": bool(res.final_lower > theta + err),
        "n_resets": int(getattr(res, "n_resets", 0)),
    }  # fmt: skip
    if spec["view"] == "window":
        t1 = time.perf_counter()
        row["end_scan"] = end_scan(view.log, run_cfg, env, float(res.final_lower),
                                   x_rep, cfg["arm_constants"])  # fmt: skip
        row["end_scan_s"] = time.perf_counter() - t1
    if x_rep is not None:
        t1 = time.perf_counter()
        row["truth_check"] = truth_check(cfg, idx, x_rep, float(res.final_lower))
        row["truth_check_s"] = time.perf_counter() - t1
        row["witness_truly_feasible"] = bool(row["truth_check"]["feasible"])
    else:
        row["truth_check"], row["witness_truly_feasible"] = None, None
    row["task_s"] = time.perf_counter() - t0
    row["peak_rss_gib"] = ce.peak_rss_gib()
    row["reports"] = [asdict(r) for r in res.reports]
    return row


def run_pool(tasks: list, workers: int, out_rows, label: str) -> float:
    """Fresh process per task; rows appended as they finish; aggregate peak RSS."""
    mon = RssMonitor()
    mon.start()
    t0 = time.time()
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result()
            out_rows(row)
            log.info("%s %d/%d %s b=%d s=%d: %.0fs L=%.4f wit=%s feas=%s raw=%d",
                     label, i, len(tasks), row["label"], row["budget"], row["seed"],
                     row["task_s"], row["final_lower"], row["has_witness"],
                     row["witness_truly_feasible"], row["raw_evals"])  # fmt: skip
            if i % 10 == 0:
                log.info("eta %.2f h", (len(tasks) - i) * (time.time() - t0) / i / 3600)
    mon.stop()
    return mon.peak_gib


def arm_seeds(cfg: dict, arm: dict) -> range:
    """Seeds of one arm: the first ``n_seeds`` of the list, else all."""
    start, stop = int(cfg["seeds"]["start"]), int(cfg["seeds"]["stop"])
    return range(start, start + int(arm.get("n_seeds", stop - start)))


def sweep_tasks(cfg: dict) -> list[tuple[int, str, int, int]]:
    """Task list: arm-major, then budget, then seed (per-arm seed counts)."""
    cells = [(a["label"], int(b), s) for a in cfg["arms"] for b in cfg["budgets_raw"]
             for s in arm_seeds(cfg, a)]  # fmt: skip
    return [(i, *c) for i, c in enumerate(cells)]


def remaining_tasks(cfg: dict, rows_path: Path) -> tuple[list, list[int]]:
    """Tasks whose row is not yet in ``rows.jsonl`` (resume)."""
    done = {json.loads(ln)["task_index"] for ln in rows_path.read_text().splitlines()
            if ln.strip()}  # fmt: skip
    return [t for t in sweep_tasks(cfg) if t[0] not in done], sorted(done)


def part_tasks(tasks: list, part: int, n_parts: int) -> list:
    """Tasks of one sweep part: indices congruent to ``part`` mod n_parts.

    The sweep may run as ``n_parts`` jobs in separate job slots. Every
    task has fixed seeds and runs in a fresh process, so the union of the parts' rows
    equals the one-job sweep's rows (time-capped end scans aside).
    """
    if not 0 <= part < n_parts:
        raise ValueError(f"part {part} outside 0..{n_parts - 1}")
    return [t for t in tasks if t[0] % n_parts == part]


def merged_rows(cfg: dict, dirs: list[Path]) -> list[dict]:
    """Rows of all sweep parts; each task exactly once, else an error."""
    rows = [json.loads(x) for d in dirs for x in (d / "rows.jsonl").read_text()
            .splitlines() if x.strip()]  # fmt: skip
    got = sorted(r["task_index"] for r in rows)
    want = [t[0] for t in sweep_tasks(cfg)]
    if got != want:
        missing, extra = sorted(set(want) - set(got)), len(got) - len(set(got))
        raise SystemExit(
            f"sweep rows incomplete: {len(missing)} missing, {extra} repeated"
        )
    return rows


def cmd_sweep(workers: int, filt: dict, resume: Path | None,
              part: tuple[int, int] = (0, 1)) -> None:  # fmt: skip
    """The sweep, or part k of n, or the selected cells -> ``covasim_sweep``."""
    cfg = load_config()
    k, n = part
    commit, dirty = git_state()
    if resume is None:
        tag = "" if n == 1 else f"_part{k}of{n}"
        out = Path("results") / (
            f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_covasim_sweep{tag}"
        )
        out.mkdir(parents=True)
        tasks = filter_tasks(part_tasks(sweep_tasks(cfg), k, n), **filt)
        theta, err = _truth(cfg)
        import covasim  # noqa: PLC0415

        (out / "manifest.json").write_text(json.dumps({
            "experiment": "covasim_sweep",
            "git_commit": commit, "git_dirty": dirty, "config_sha256": cfg["sha256"],
            "truth_dir": str(run_path("covasim_truth")), "theta": theta,
            "theta_err": err, "filter": filt,
            "n_tasks": len(tasks), "part": k, "n_parts": n,
            "workers": workers, "covasim": covasim.__version__,
            "python": sys.version, "numpy": np.__version__,
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "started_utc": datetime.now(UTC).isoformat(),
        }, indent=1))  # fmt: skip
    else:
        out = resume
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["config_sha256"] != cfg["sha256"]:
            raise SystemExit("resume target used another config")
        if (manifest.get("part", 0), manifest.get("n_parts", 1)) != (k, n):
            raise SystemExit("resume target is another sweep part")
        tasks, done = remaining_tasks(cfg, out / "rows.jsonl")
        tasks = filter_tasks(part_tasks(tasks, k, n), **manifest.get("filter", {}))
        manifest.setdefault("resumes", []).append({
            "resumed_utc": datetime.now(UTC).isoformat(), "git_commit": commit,
            "git_dirty": dirty, "workers": workers, "rows_done": len(done),
            "rows_remaining": len(tasks),
        })  # fmt: skip
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    t0 = time.time()
    with (out / "rows.jsonl").open("a") as fh:

        def write(row: dict) -> None:
            fh.write(json.dumps(row) + "\n")
            fh.flush()

        peak = run_pool(tasks, workers, write, "sweep")
    (out / "sweep_summary.json").write_text(json.dumps({
        "wall_h": (time.time() - t0) / 3600.0, "aggregate_peak_rss_gib": peak,
        "n_rows": len(tasks), "finished_utc": datetime.now(UTC).isoformat(),
    }, indent=1))  # fmt: skip


def analyze(rows: list[dict], cfg: dict, columns: dict | None, monotone: dict | None):
    """The S3 gates (G1 to G3, hardened readings, paired tests) plus S5-G4."""
    gate = hs.analyze_rows(rows, cfg, None, None)
    eps = [str(e) for e in cfg["columns"]["eps"]]
    by_eps = (columns or {}).get("by_eps", {})
    u = by_eps.get(eps[0], {})
    gate["G4"] = {
        "primary_eps": eps[0], "U_primary": u.get("U_eps"), "k": u.get("k"),
        "secondary": {e: by_eps.get(e) for e in eps[1:]},
        "monotone_passed": None if monotone is None else bool(monotone["passed"]),
        "pass": None if not u or monotone is None
        else bool(u.get("finite") and monotone["passed"]),
    }  # fmt: skip
    gate["verdict"]["G4"] = gate["G4"]["pass"]
    gate["verdict"]["two_sided_statement"] = bool(gate["G4"]["pass"])
    if not cfg["gates"].get("register_g3", True):
        gate["verdict"]["G3"] = None
        gate["verdict"]["exposure_claim"] = None
    return gate


def cmd_analyze(dirs: list[Path], columns_dir: Path | None, mono_dir: Path | None):
    """Gate analysis of the sweep part directories -> ``covasim_gates``."""
    cfg = load_config()
    rows = merged_rows(cfg, dirs)

    def read(d):
        return json.loads((d / "result.json").read_text()) if d else None

    gate = analyze(rows, cfg, read(columns_dir), read(mono_dir))
    gate["config_sha256"] = cfg["sha256"]
    gate["sweep_dirs"] = [str(d) for d in dirs]
    out = ce.write_result("covasim_gates", gate, {
        "sweep_dirs": [str(d) for d in dirs], "columns_dir": str(columns_dir),
        "monotone_dir": str(mono_dir), "config_sha256": cfg["sha256"]}, 0)  # fmt: skip
    log.info("verdict %s -> %s", json.dumps(gate["verdict"]), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("--workers", type=int, required=True)
    sw.add_argument("--resume", type=Path, default=None)
    sw.add_argument("--part", default="0/1", help="k/n: run part k of n")
    add_filter_args(sw)
    an = sub.add_parser("analyze")
    an.add_argument("results_dirs", type=Path, nargs="*")
    an.add_argument("--columns", type=Path, default=None)
    an.add_argument("--monotone", type=Path, default=None)
    args = ap.parse_args()
    if args.cmd == "sweep":
        filt = {"labels": args.labels, "budgets": args.budgets, "seeds": args.seeds}
        cmd_sweep(args.workers, filt, args.resume,
                  tuple(int(v) for v in args.part.split("/")))  # fmt: skip
    else:
        cmd_analyze(
            args.results_dirs or run_path("covasim_sweep"),
            args.columns or run_path("covasim_columns"),
            args.monotone or run_path("covasim_monotone"),
        )
