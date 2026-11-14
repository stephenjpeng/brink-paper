"""Transmission network (IEEE 39-bus): the method and baseline sweep, and its gates.

Every constant comes from ``configs/transmission.yaml``. Five methods at two
budgets of power flows, 50 seeds each: BRINK's proposed search
(``witness-focused``), BRINK's variance-seeking rule (``greedy``), the CONFIG
adaptation (``config``), bisection on random designs (``sup-bisection``), and
direct limit search (``direct-score``). The three model-based methods run the
infeasibility test once at the end of the search, on the upper-window
constraint only (``end_scan.channels: [1]``). Every returned design is checked
by one power flow at its reported stress.

Usage:
    uv run python scripts/transmission_sweep.py sweep --workers N \
        [--labels L ...] [--budgets B ...] [--seeds S ...] [--resume DIR]
    uv run python scripts/transmission_sweep.py analyze [RESULTS_DIR] \
        [--truth DIR] [--columns DIR] [--monotone DIR]

``analyze`` reads the latest truth, column, and single-window-check runs from
``configs/runs.yaml`` unless given, and writes ``gate.json`` beside the rows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

_CFG_PATH = Path("configs/transmission.yaml")
if _CFG_PATH.exists():  # BLAS threads must be set before numpy loads (macOS, Linux)
    _n = str(yaml.safe_load(_CFG_PATH.read_text())["arm_constants"]["blas_threads"])
    for _var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(_var, _n)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from brink import bracket as br  # noqa: E402
from brink.adapter import ExistenceSpec  # noqa: E402
from brink.arms import ExistenceConfig, run_existence_arm  # noqa: E402
from brink.gp import beta_sqrt  # noqa: E402
from brink.memory import RssMonitor  # noqa: E402
from brink.studies.transmission import load_environment  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feeder_sweep as hs  # noqa: E402  (shared sweep helpers)
import transmission_setup as ts  # noqa: E402  (raw oracle and design windows)
from common import add_filter_args, filter_tasks, git_state, run_path  # noqa: E402

he = hs.he  # shared output helpers: load_s3_config, _write

log = logging.getLogger("transmission_sweep")
WINDOW_HALF = 0.125 * 0.5**8 / 2  # half-width of the window oracle's last bracket
SPAWN = mp.get_context("spawn")  # same start method on macOS and Linux


def load_cfg(path: Path = _CFG_PATH) -> dict:
    """The transmission constants (YAML), with their SHA-256 under ``sha256``."""
    return he.load_s3_config(path)


# --- methods ---


def build_cfg(cfg: dict, budget: int, view_kind: str, lengthscale: float):
    """Frozen arm constants at one raw budget."""
    ac = cfg["arm_constants"]
    return ExistenceConfig(
        mode=br.MODE_AFFINE,
        budget=int(budget),
        n_init=int(ac["n_init"]),
        report_every=int(
            ac["report_every_calls"]
            if view_kind == "window"
            else ac["report_every_raw"]
        ),
        n_candidates=int(ac["n_candidates"]),
        n_witness_candidates=int(ac["n_witness_candidates"]),
        report_grid_size=int(ac["report_grid_size"]),
        cover_n_per_axis=int(ac["cover_n_per_axis"]),
        delta=float(ac["delta"]),
        lengthscale=float(lengthscale),
        lam=float(ac["lam"]),
        R=float(ac["R"]),
        sigma=0.0,
        variant=str(ac["variant"]),
        bisect_depth=int(ac["bisect_depth"]),
        certificate_scan=br.SCAN_OFF,
    )


class WindowScoreView:
    """Arm 5's reduced score: one window call per design, charged its raw cost.

    The score is the feasible end of the window oracle's last bisection bracket
    (c_off - WINDOW_HALF), or 1.0 when the scan is feasible at c = 1; either
    point was evaluated feasible.
    """

    def __init__(self, env) -> None:
        """Wrap the frozen v2 environment."""
        self.env = env
        self.spec = ExistenceSpec(x_dim=int(env.x_dim), m=1)

    @property
    def n_raw_evals(self) -> int:
        """Cumulative raw evaluations of the environment's window oracle."""
        return int(self.env.window_counters()["n_raw_evals"])

    def score(self, x: np.ndarray) -> float:
        """The design's reduced score (unit c)."""
        w = self.env._oracle().window(np.asarray(x, dtype=float))
        return 1.0 if w.c_off >= 1.0 else float(w.c_off) - WINDOW_HALF


def make_view(spec: dict, env):
    """The adapter view of one arm."""
    if spec["view"] == "window":
        return hs.LoggedWindowView(env)
    if spec["view"] == "predicate":
        return hs.LoggedPredicateView(env._oracle().raw_feasible, int(env.x_dim))
    return WindowScoreView(env)


def end_scan(
    view_log: list,
    run_cfg: ExistenceConfig,
    env,
    lower: float,
    witness_x: list[float] | None,
    scan: dict,
) -> dict:
    """Refit the final bands, beta·sd at the witness, one capped end scan."""
    X = np.array([r[0] for r in view_log])
    C = np.array([r[1] for r in view_log])
    Y = np.array([r[2] for r in view_log])
    bands = br.fit_joint_bands(
        X,
        C,
        Y,
        mode=br.MODE_AFFINE,
        lengthscale=run_cfg.lengthscale,
        lam=run_cfg.lam,
        B=env.B,
        R=run_cfg.R,
        betas=[float(b) for b in env.B],
        c_max=1.0,
    )
    m = len(bands.models)
    bands.betas = [
        beta_sqrt(md, run_cfg.delta / m, run_cfg.variant) for md in bands.models
    ]
    out: dict = {"betas": [float(b) for b in bands.betas]}
    if witness_x is not None:
        xw = np.array([witness_x])
        out["beta_sd_at_witness"] = [
            float(bands.betas[j] * bands._mu_sd(j, xw, lower)[1][0]) for j in range(m)
        ]
    cover, rho = br.midpoint_cover(run_cfg.cover_n_per_axis, int(env.x_dim))
    grid = np.linspace(0.0, 1.0, run_cfg.report_grid_size)
    channels = scan.get("channels")  # [1]: the upper-window constraint only
    if channels is not None:
        bands = bands.subset(list(channels))
    kw = {
        "max_depth": int(scan["max_depth"]),
        "max_cells": int(scan["max_cells"]),
        "second_order": bool(scan["second_order"]),
        "time_cap_s": float(scan["cap_s"]),
    }
    t0 = time.perf_counter()
    if witness_x is not None:  # smallest excluded stress upward from L
        rule = "smallest-from-L"
        upper, ok = br.smallest_certified_slice(
            bands, grid, cover, rho, stop_below=lower, **kw
        )
    else:  # no design: descend from c_max while stresses are excluded
        rule = "top-down"
        upper, ok = br.certificate_upper_bound(bands, grid, cover, rho, **kw)
    seconds = time.perf_counter() - t0
    out["scan"] = {
        "rule": rule,
        "channels": channels,
        "upper": float(upper),
        "has_certificate": bool(ok),
        "seconds": seconds,
        "capped": seconds >= float(scan["cap_s"]),
    }
    return out


def witness_check(x: list[float], lower: float) -> dict:
    """Design check: raw predicate and margins at (x, L), and x's own window."""
    xa = np.asarray(x, dtype=float)
    g = ts._oracle().raw_margins(xa, float(lower))
    start, death, n_flows, multi = ts.truth_column(tuple(map(float, xa)))
    return {
        "feasible": bool(g is not None and (g <= 0.0).all()),
        "margins": None if g is None else g.tolist(),
        "start": float(start),
        "death": float(death),
        "multi_window": int(multi),
        "n_flows": int(n_flows) + 1,
    }


def run_task(task: tuple[int, str, int, int]) -> dict:
    """One (task index, method label, budget, seed) run in a fresh process."""
    t_task = time.perf_counter()
    idx, label, budget, seed = task
    cfg = load_cfg()
    spec = next(a for a in cfg["arms"] if a["label"] == label)
    env = load_environment(cfg["frozen_environment"])
    cal = json.loads(Path(cfg["frozen_environment"]).read_text())["calibration"]
    run_cfg = build_cfg(cfg, budget, spec["view"], cal["lengthscale"])
    view = make_view(spec, env)
    t0 = time.perf_counter()
    res = run_existence_arm(view, spec["name"], run_cfg, env.B, seed)
    wall = time.perf_counter() - t0
    has_w = bool(res.reports and res.reports[-1].has_witness)
    if spec["view"] == "predicate":
        x_rep = hs.best_x_from_log(view.log, res.final_lower) if has_w else None
    else:
        x_rep = res.final_witness_x if has_w else None
    row = {
        "task_index": idx,
        "label": label,
        "arm": spec["name"],
        "view": spec["view"],
        "budget": int(budget),
        "seed": int(seed),
        "raw_evals": int(res.raw_evals),
        "wall_s": wall,
        "final_lower": float(res.final_lower),
        "final_upper": float(res.final_upper),
        "has_witness": has_w,
        "first_witness_raw": next(
            (r.raw_evals for r in res.reports if r.has_witness), None
        ),
        "witness_x": x_rep,
        "window_counters": env.window_counters(),
    }
    if spec["view"] == "window":
        row["end_scan"] = end_scan(
            view.log,
            run_cfg,
            env,
            float(res.final_lower),
            x_rep,
            cfg["arm_constants"]["end_scan"],
        )
    row["truth_check"] = (
        None if x_rep is None else witness_check(x_rep, res.final_lower)
    )
    row["witness_feasible"] = None if x_rep is None else row["truth_check"]["feasible"]
    row["reports"] = [asdict(r) for r in res.reports]
    row["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**3 if sys.platform == "darwin" else 1024**2
    )  # ru_maxrss is bytes on macOS, KiB on Linux
    row["task_s"] = time.perf_counter() - t_task
    return row


# --- sweep ---


def _run_pool(tasks, workers: int, out_rows, label: str) -> float:
    """Run tasks in fresh processes, appending rows; returns aggregate peak RSS."""
    mon = RssMonitor()
    mon.start()
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=SPAWN, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result()
            out_rows(row)
            log.info(
                "%s %d/%d %s b=%d s=%d: %.0fs L=%.4f wit=%s feas=%s raw=%d",
                label,
                i,
                len(tasks),
                row["label"],
                row["budget"],
                row["seed"],
                row["task_s"],
                row["final_lower"],
                row["has_witness"],
                row["witness_feasible"],
                row["raw_evals"],
            )
            if i % 25 == 0:
                log.info(
                    "eta %.2f h", (len(tasks) - i) / (i / (time.time() - t0)) / 3600
                )
    mon.stop()
    if mon.missed_samples:
        log.warning("RSS monitor missed %d samples", mon.missed_samples)
    return mon.peak_gib


def _manifest(out: Path, cfg: dict, workers: int, n_tasks: int, filt: dict) -> None:
    commit, dirty = git_state()
    (out / "manifest.json").write_text(
        json.dumps(
            {
                **ts._host("transmission_sweep"),
                "git_commit": commit,
                "git_dirty": dirty,
                "config_sha256": cfg["sha256"],
                "frozen_calibration_sha256": hashlib.sha256(
                    Path(cfg["frozen_environment"]).read_bytes()
                ).hexdigest(),
                "seeds": cfg["seeds"],
                "filter": filt,
                "n_tasks": n_tasks,
                "workers": workers,
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "started_utc": datetime.now(UTC).isoformat(),
            },
            indent=1,
        )
    )


def cmd_sweep(workers: int, filt: dict, resume: Path | None = None) -> None:
    """The sweep (or the selected cells) -> ``results/<UTC>_transmission_sweep/``.

    ``resume`` continues a stopped sweep directory: rows already written are
    kept, the remaining tasks run, and the manifest records the resume.
    """
    cfg = load_cfg()
    if resume is None:
        out = Path("results") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_transmission_sweep"
        out.mkdir(parents=True)
        tasks = filter_tasks(hs.sweep_tasks(cfg), **filt)
        _manifest(out, cfg, workers, len(tasks), filt)
    else:
        out = resume
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["config_sha256"] != cfg["sha256"]:
            raise SystemExit("resume target used another config")
        tasks, done = hs.remaining_tasks(cfg, out / "rows.jsonl")
        tasks = filter_tasks(tasks, **manifest.get("filter", {}))
        commit, dirty = git_state()
        manifest.setdefault("resumes", []).append(
            {
                "resumed_utc": datetime.now(UTC).isoformat(),
                "git_commit": commit,
                "git_dirty": dirty,
                "workers": workers,
                "rows_done": len(done),
                "rows_remaining": len(tasks),
            }
        )
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        log.info("resuming %s: %d rows done, %d remaining", out, len(done), len(tasks))
    t0 = time.time()
    with (out / "rows.jsonl").open("a") as fh:

        def write(row: dict) -> None:
            fh.write(json.dumps(row) + "\n")
            fh.flush()

        peak = _run_pool(tasks, workers, write, "sweep")
    (out / "sweep_summary.json").write_text(
        json.dumps(
            {
                "wall_h": (time.time() - t0) / 3600.0,
                "aggregate_peak_rss_gib": peak,
                "n_rows": len(tasks),
                "finished_utc": datetime.now(UTC).isoformat(),
            },
            indent=1,
        )
    )
    log.info(
        "sweep done in %.2f h, peak RSS %.2f GiB -> %s",
        (time.time() - t0) / 3600.0,
        peak,
        out,
    )


# --- gates and reported metrics ---


def _med(values) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def analyze_rows(
    rows: list[dict],
    cfg: dict,
    truth: dict,
    columns: dict | None = None,
    monotone: dict | None = None,
) -> dict:
    """Predeclared gates S2-G1 to G4 and the reported metrics.

    Args:
        rows: Sweep rows (``run_task`` output).
        cfg: Config.
        truth: ``theta``, ``err``, and ``grid_death`` (the 9^4 truth grid).
        columns: ``score-columns`` summary (``by_eps``), if available.
        monotone: S2-M1 ``monotone-check`` summary (``passed``), if available.

    Returns:
        Gate dictionary with a ``verdict`` block.
    """
    g, alpha = cfg["gates"], float(cfg["gates"]["alpha"])
    theta, err = float(truth["theta"]), float(truth["err"])
    labels = [a["label"] for a in cfg["arms"]]
    certified = [a["label"] for a in cfg["arms"] if a["view"] == "window"]
    method, two_sided = labels[0], labels[1]
    budgets = [int(b) for b in cfg["budgets_raw"]]
    cell = {
        (lab, b): [r for r in rows if r["label"] == lab and r["budget"] == b]
        for lab in labels
        for b in budgets
    }

    def anytime_violation(r: dict) -> bool:
        return any(p["lower"] > theta + err for p in r["reports"])

    def scan(r: dict) -> dict:
        return r["end_scan"]["scan"]

    def unsound(r: dict) -> bool:
        return scan(r)["has_certificate"] and scan(r)["upper"] < theta - err

    out: dict = {
        "n_rows": len(rows),
        "theta": theta,
        "err": err,
        "G1": {},
        "G2": {},
        "G3": {},
        "G4": {},
        "cells": {},
        "paired": {},
    }
    g1 = True
    for lab in certified:
        for b in budgets:
            rs = cell[(lab, b)]
            viol = _rej(
                sum(map(anytime_violation, rs)), len(rs), g["g1_violation_delta"], alpha
            )
            wit = [r["witness_feasible"] for r in rs if r["witness_x"] is not None]
            rate = sum(wit) / len(wit) if wit else None
            ok = not viol["reject"] and (
                rate is None or rate >= float(g["g1_witness_feasible_min"])
            )
            out["G1"][f"{lab}/{b}"] = {
                "violations": viol,
                "witness_feasible_rate": rate,
                "pass": ok,
            }
            g1 &= ok
    out["G1"]["pass"] = bool(g1)
    g2 = True
    for b in budgets:
        rs = cell[(method, b)]
        t = _rej(
            sum(r["has_witness"] for r in rs), len(rs), g["g2_witness_rate_null"], alpha
        )
        out["G2"][str(b)] = {**t, "pass": t["reject"]}
        g2 &= t["reject"]
    med_top = _med(r["final_lower"] for r in cell[(method, max(budgets))])
    out["G2"]["median_lower_top_budget"] = med_top
    g2 &= med_top is not None and med_top >= float(g["g2_median_lower_min"])
    out["G2"]["pass"] = bool(g2)
    by_eps = (columns or {}).get("by_eps", {})
    grid_death = np.asarray(truth["grid_death"], dtype=float)
    out["G3"] = {
        "monotone_passed": None if monotone is None else bool(monotone["passed"])
    }
    for eps, u in by_eps.items():
        frac = float(np.mean(grid_death > u["U_eps"]))
        out["G3"] |= {
            f"U_{eps}": u["U_eps"],
            f"k_{eps}": u["k"],
            f"grid_fraction_above_U_{eps}": frac,
        }
    u01 = by_eps.get("0.01")
    out["G3"]["pass"] = (
        None
        if not u01 or monotone is None
        else bool(
            u01["finite"]
            and monotone["passed"]
            and out["G3"]["grid_fraction_above_U_0.01"]
            <= float(g["g3_grid_fraction_max"])
        )
    )
    for lab in certified[1:]:
        for b in budgets:
            rs = cell[(lab, b)]
            a = _rej(sum(map(unsound, rs)), len(rs), g["g4_violation_delta"], alpha)
            c = _rej(
                sum(scan(r)["has_certificate"] for r in rs),
                len(rs),
                g["g4_certificate_rate_null"],
                alpha,
            )
            out["G4"][f"{lab}/{b}"] = {
                "unsound": a,
                "certificates": c,
                "pass": bool(not a["reject"] and c["reject"]),
                "gated": lab == two_sided,
            }
    for (lab, b), rs in cell.items():
        if not rs:
            continue
        own = [
            (r["truth_check"]["death"], r["final_lower"])
            for r in rs
            if r["truth_check"] is not None
        ]
        c = {
            "n": len(rs),
            "median_lower": _med(r["final_lower"] for r in rs),
            "median_theta_minus_lower": _med(theta - r["final_lower"] for r in rs),
            "witness_rate": sum(r["has_witness"] for r in rs) / len(rs),
            "median_first_witness_raw": _med(r["first_witness_raw"] for r in rs),
            "witness_feasible": sum(bool(r["witness_feasible"]) for r in rs),
            "anytime_violations": sum(map(anytime_violation, rs)),
            "final_violations": sum(r["final_lower"] > theta + err for r in rs),
            "median_band_looseness": _med(d - low for d, low in own),
            "median_search_gap": _med(theta - d for d, _ in own),
            "median_wall_s": _med(r["wall_s"] for r in rs),
            "median_task_s": _med(r["task_s"] for r in rs),
            "median_raw_evals": _med(r["raw_evals"] for r in rs),
        }
        if lab in certified:
            cert = [r for r in rs if scan(r)["has_certificate"]]
            bsd = [
                r["end_scan"]["beta_sd_at_witness"]
                for r in rs
                if r["end_scan"].get("beta_sd_at_witness")
            ]
            c |= {
                "end_scan_certificates": len(cert),
                "end_scan_unsound": sum(map(unsound, rs)),
                "end_scan_capped": sum(scan(r)["capped"] for r in rs),
                "median_U": _med(scan(r)["upper"] for r in cert),
                "median_U_minus_L": _med(
                    scan(r)["upper"] - r["final_lower"] for r in cert
                ),
                "median_U_minus_theta": _med(scan(r)["upper"] - theta for r in cert),
                "median_scan_s": _med(scan(r)["seconds"] for r in rs),
                "median_beta_sd_at_witness": np.median(bsd, axis=0).tolist()
                if bsd
                else None,
            }
        else:
            c |= {
                "bracket_contains_theta_final": sum(  # at the final report
                    r["reports"][-1]["lower"] <= theta <= r["reports"][-1]["upper"]
                    for r in rs
                    if r["reports"]
                ),
                "bracket_contains_theta_every_report": sum(
                    all(p["lower"] <= theta <= p["upper"] for p in r["reports"])
                    for r in rs
                ),
            }
        out["cells"][f"{lab}/{b}"] = c
    for lab in labels[1:]:
        for b in budgets:
            by_seed = {r["seed"]: r["final_lower"] for r in cell[(method, b)]}
            d = [
                r["final_lower"] - by_seed[r["seed"]]
                for r in cell[(lab, b)]
                if r["seed"] in by_seed
            ]
            wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
            out["paired"][f"{lab}-vs-{method}/{b}"] = {
                "wins": wins,
                "losses": losses,
                "ties": len(d) - wins - losses,
                "sign_p": float(binomtest(wins, wins + losses, 0.5).pvalue)
                if wins + losses
                else None,
            }
    g4 = {str(b): out["G4"][f"{two_sided}/{b}"]["pass"] for b in budgets}
    out["verdict"] = {
        "G1": out["G1"]["pass"],
        "G2": out["G2"]["pass"],
        "G3": out["G3"]["pass"],
        "G4": g4,
        "lower_bound_claim": bool(g1 and g2),
        "volume_claim": bool(out["G3"]["pass"]),
        "two_sided_bracket": {b: bool(g1 and ok) for b, ok in g4.items()},
    }
    return out


def _rej(count: int, n: int, null, alpha: float) -> dict:
    return hs._reject(int(count), int(n), float(null), alpha)


def load_truth(truth_dir: Path, cfg: dict) -> dict:
    """The reference-limit run, checked against the reported value in the config."""
    t = json.loads((truth_dir / "truth.json").read_text())
    reg, err = cfg["truth"]["theta_exists_unit"], float(cfg["truth"]["err_unit"])
    theta = float(t["theta_exists_unit"])
    if abs(theta - reg) > err or t["nesting_union_gaps"]:
        raise SystemExit(
            f"reference {theta} differs from the reported {reg} or has gaps"
        )
    n = int(cfg["truth"]["grid_columns"])
    death = np.load(truth_dir / "truth_columns.npz")["death"][:n]
    manifest = json.loads((truth_dir / "manifest.json").read_text())
    return {
        "theta": theta,
        "err": err,
        "grid_death": death,
        "dir": str(truth_dir),
        "git_commit": manifest["git_commit"],
        "git_dirty": manifest["git_dirty"],
        "platform": manifest["platform"],
    }


def cmd_analyze(
    results_dir: Path,
    truth_dir: Path,
    columns_dir: Path | None,
    monotone_dir: Path | None,
) -> None:
    """Gate analysis of a sweep directory -> ``gate.json`` beside the rows."""
    cfg = load_cfg()
    rows = [
        json.loads(x)
        for x in (results_dir / "rows.jsonl").read_text().splitlines()
        if x.strip()
    ]
    truth = load_truth(truth_dir, cfg)
    cols = (
        json.loads((columns_dir / "columns.json").read_text()) if columns_dir else None
    )
    mono = (
        json.loads((monotone_dir / "check.json").read_text()) if monotone_dir else None
    )
    gate = analyze_rows(rows, cfg, truth, cols, mono)
    gate |= {
        "config_sha256": cfg["sha256"],
        "truth": {k: v for k, v in truth.items() if k != "grid_death"},
        "columns_dir": str(columns_dir),
        "monotone_dir": str(monotone_dir),
    }
    (results_dir / "gate.json").write_text(json.dumps(gate, indent=1))
    log.info("verdict %s", json.dumps(gate["verdict"]))


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
    an.add_argument("--truth", type=Path, default=None)
    an.add_argument("--columns", type=Path, default=None)
    an.add_argument("--monotone", type=Path, default=None)
    args = ap.parse_args()
    if args.cmd == "sweep":
        filt = {"labels": args.labels, "budgets": args.budgets, "seeds": args.seeds}
        cmd_sweep(args.workers, filt, args.resume)
    else:
        cmd_analyze(
            args.results_dir or run_path("transmission_sweep"),
            args.truth or run_path("transmission_truth"),
            args.columns or run_path("transmission_columns"),
            args.monotone or run_path("transmission_monotone"),
        )
