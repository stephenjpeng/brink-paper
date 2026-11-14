"""Double integrator (exact limit): calibration, the sweep, and its gates.

The design is an initial position and velocity; the stress is the bound on an
adversarial disturbance; the operating limit is exactly 1 (unit stress 2/3).
Five methods at budgets 100, 300, and 1,000 evaluations, 50 seeds each:
BRINK's variance-seeking rule (``existence-bracket-greedy``), the CONFIG
adaptation, random sampling with the same bounds, bisection, and direct limit
search. The infeasibility test runs at every report. Constants are module
constants below and ``configs/frozen/double_integrator.json``.

Usage:
    uv run python scripts/double_integrator_sweep.py calibrate
    uv run python scripts/double_integrator_sweep.py sweep --workers N \
        [--labels ARM ...] [--budgets B ...] [--seeds S ...]
    uv run python scripts/double_integrator_sweep.py analyze [RESULTS_DIR]

A run of 1,000 evaluations with the kernel bounds takes 0.5 to 1.9 core-hours;
the baselines take under a second.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import platform
import socket
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from brink.adapter import (  # noqa: E402
    RawPredicateView,
    ScoreView,
    WindowExistenceView,
)
from brink.arms import (  # noqa: E402
    ARM_BRACKET_GREEDY,
    ARM_DIRECT_SCORE,
    ARM_JOINT_CONFIG,
    ARM_RANDOM,
    ARM_SUP_BISECTION,
    ExistenceConfig,
    run_existence_arm,
)
from brink.bracket import (  # noqa: E402
    MODE_AFFINE,
    SCAN_SMALLEST,
)
from brink.gp import fit_gp  # noqa: E402
from brink.kernels import rbf  # noqa: E402
from brink.memory import RssMonitor  # noqa: E402
from brink.studies.double_integrator import (  # noqa: E402
    C_RANGE,
    THETA_EXISTS_UNIT,
    DoubleIntegratorEnvironment,
    c_off_unit,
    is_viable,
    load_double_integrator,
    theta_exists_truth,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import add_filter_args, filter_tasks, git_state, run_path  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("double_integrator_sweep")

FROZEN_PATH = Path("configs/frozen/double_integrator.json")

# Run constants.
BUDGETS = (100, 300, 1000)  # amendment double-integrator-A1
SEEDS = tuple(range(400, 450))
ARMS = (
    ARM_BRACKET_GREEDY,
    ARM_JOINT_CONFIG,
    ARM_RANDOM,
    ARM_SUP_BISECTION,
    ARM_DIRECT_SCORE,
)
CERTIFIED = (ARM_BRACKET_GREEDY, ARM_JOINT_CONFIG, ARM_RANDOM)
DELTA = 0.10
N_INIT = 20
REPORT_EVERY = 10
REPORT_GRID = 2001  # amendment double-integrator-A1
COVER_N_PER_AXIS = 61
COVER_MAX_DEPTH = 12
COVER_MAX_CELLS = 8_000_000
N_CANDIDATES = 2048
N_WITNESS = 2048
BISECT_DEPTH = 8
LAM = 1e-6
INFLATION = 2.0
CAL_N = 512
CAL_SEEDS = {"sobol": 9001, "uniform": 9002, "split": 9003}
NONTRIVIAL_FRACTION = 0.9


def build_cfg(budget: int) -> ExistenceConfig:
    """Frozen run constants at a budget."""
    cal = json.loads(FROZEN_PATH.read_text())["calibration"]
    return ExistenceConfig(
        mode=MODE_AFFINE,
        budget=budget,
        n_init=N_INIT,
        report_every=REPORT_EVERY,
        n_candidates=N_CANDIDATES,
        n_witness_candidates=N_WITNESS,
        report_grid_size=REPORT_GRID,
        cover_n_per_axis=COVER_N_PER_AXIS,
        cover_max_depth=COVER_MAX_DEPTH,
        cover_max_cells=COVER_MAX_CELLS,
        cover_second_order=True,
        certificate_scan=SCAN_SMALLEST,
        delta=DELTA,
        lengthscale=float(cal["lengthscale"]),
        lam=LAM,
        R=0.0,
        sigma=0.0,
        variant="deterministic",
        bisect_depth=BISECT_DEPTH,
    )


def _view_for(arm: str, env: DoubleIntegratorEnvironment):
    """The adapter view an arm consumes, with matched raw-evaluation cost."""
    if arm == ARM_SUP_BISECTION:
        return RawPredicateView(env.raw_feasible, x_dim=env.x_dim)
    if arm == ARM_DIRECT_SCORE:
        return ScoreView(env.score_unit, x_dim=env.x_dim)
    return WindowExistenceView(env)


def run_task(task: tuple[str, int, int]) -> dict:
    """One (arm, budget, seed) run -> JSON row."""
    arm, budget, seed = task
    env = load_double_integrator(FROZEN_PATH)
    cfg = build_cfg(budget)
    view = _view_for(arm, env)
    t0 = time.perf_counter()
    res = run_existence_arm(view, arm, cfg, env.B, seed=seed)
    theta = THETA_EXISTS_UNIT
    covered = all(r.lower <= theta <= r.upper for r in res.reports)
    witness_sound = None
    if res.final_witness_x is not None:
        witness_sound = bool(
            is_viable(np.asarray(res.final_witness_x), res.final_lower)[0]
        )
    return {
        "arm": arm,
        "budget": budget,
        "seed": seed,
        "wall_s": round(time.perf_counter() - t0, 3),
        "raw_evals": res.raw_evals,
        "final_lower": res.final_lower,
        "final_upper": res.final_upper,
        "final_width": res.final_upper - res.final_lower,
        "covered_anytime": bool(covered),
        "covered_final": bool(res.final_lower <= theta <= res.final_upper),
        "final_witness_x": res.final_witness_x,
        "witness_sound": witness_sound,
        "reports": [asdict(r) for r in res.reports],
    }


def cmd_calibrate() -> None:
    """Calibration and its five acceptance checks -> a new run directory."""
    truth = theta_exists_truth(n_per_axis=401)
    log.info("acceptance 1 crossing: %s", json.dumps(truth))
    if truth["columns_viable_at_c_max"] != 0:
        raise SystemExit("acceptance 1 FAILED: columns still viable at c_max")
    if abs(truth["theta_unit_including_face"] - THETA_EXISTS_UNIT) > 1e-9:
        raise SystemExit("acceptance 1 FAILED: analytic theta not reproduced")
    if truth["columns_viable_at_c_min"] != truth["n_columns"]:
        raise SystemExit("acceptance 2 FAILED: not a prefix family; register m = 2")
    log.info("acceptance 2 prefix: all %d columns viable at c_min", truth["n_columns"])

    rng_s = np.random.default_rng(CAL_SEEDS["sobol"])
    rng_u = np.random.default_rng(CAL_SEEDS["uniform"])
    X = np.concatenate([rng_s.uniform(size=(256, 2)), rng_u.uniform(size=(256, 2))])
    c = np.concatenate([rng_s.uniform(size=256), rng_u.uniform(size=256)])
    raw = c - c_off_unit(X)
    scale = float(np.subtract(*np.percentile(raw, [75, 25])) / 1.349)
    y = raw / scale
    dist = np.linalg.norm(X[:, None] - X[None], axis=2)
    lengthscale = 2.0 * float(np.median(dist[np.triu_indices(len(X), 1)]))

    split = np.random.default_rng(CAL_SEEDS["split"]).permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, te = split[:n_tr], split[n_tr:]
    m_tr = fit_gp(X[tr], y[tr], lengthscale=lengthscale, lam=LAM, B=1.0, R=0.0, c=c[tr])
    Kq = rbf(X[te], m_tr.X, lengthscale) * (1.0 + np.outer(c[te], m_tr.c))
    pred = Kq @ m_tr.w
    r2 = float(1.0 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
    log.info("acceptance 3 learnability: held-out R^2 = %.6f", r2)
    if r2 < 0.5:
        raise SystemExit("acceptance 3 FAILED: held-out R^2 below 0.5")

    m_full = fit_gp(X, y, lengthscale=lengthscale, lam=LAM, B=1.0, R=0.0, c=c)
    K = rbf(X, X, lengthscale) * (1.0 + np.outer(c, c))
    mu_norm = float(np.sqrt(max(m_full.w @ (K @ m_full.w), 0.0)))
    B = INFLATION * mu_norm
    m_slice = fit_gp(
        X, -c_off_unit(X) / scale, lengthscale=lengthscale, lam=LAM, B=1.0, R=0.0
    )
    slice_norm = float(
        np.sqrt(max(m_slice.w @ (rbf(X, X, lengthscale) @ m_slice.w), 0.0))
    )
    B_slice = INFLATION * slice_norm

    # acceptance 4: norm ladder at the frozen lengthscale (recorded, not gating)
    ladder = {}
    for n in (5, 9, 13, 17, 21):
        a = (np.arange(n) + 0.5) / n
        Xg = np.stack([g.ravel() for g in np.meshgrid(a, a, indexing="ij")], axis=1)
        Xj = np.concatenate([Xg, Xg])
        cj = np.concatenate([np.zeros(len(Xg)), np.ones(len(Xg))])
        yj = (cj - np.concatenate([c_off_unit(Xg)] * 2)) / scale
        mg = fit_gp(Xj, yj, lengthscale=lengthscale, lam=LAM, B=1.0, R=0.0, c=cj)
        Kg = rbf(Xj, Xj, lengthscale) * (1.0 + np.outer(cj, cj))
        ladder[f"{n}^2"] = float(np.sqrt(max(mg.w @ (Kg @ mg.w), 0.0)))
    log.info("acceptance 4 norm ladder at frozen lengthscale: %s", ladder)

    # acceptance 5: certificate conditions from the frozen constants
    kappa = float(np.sqrt(1.0 + 1.0**2))
    L = (mu_norm + B) * kappa / lengthscale
    Q = float(np.sqrt(3.0) * L / (2.0 * lengthscale))
    cert = []
    for c_slice in (0.70, 0.75, 0.80, 0.90, 1.0):
        margin = (c_slice - THETA_EXISTS_UNIT) / scale
        if margin <= 0:
            continue
        cert.append(
            {
                "c_unit": c_slice,
                "binding_margin": margin,
                "required_sd": margin / B,
                "second_order_cells": float(
                    (np.sqrt(2) / (2 * np.sqrt(margin / Q))) ** 2
                ),
            }
        )
    log.info("acceptance 5 certificate: %s", json.dumps(cert))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(f"results/{stamp}_double_integrator_calibration")
    out_dir.mkdir(parents=True)
    out_path = out_dir / "frozen.json"
    out_path.write_text(
        json.dumps(
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "environment": {
                    "kind": "double-integrator",
                    "u_max": 1.0,
                    "p_range": [0.0, 0.5],
                    "v_range": [0.0, 1.0],
                    "c_range": list(C_RANGE),
                    "x_dim": 2,
                    "m": 1,
                    "c_off_closed_form": "u_max - v^2 / (2 (1 - p))",
                    "oracle_note": "expensive oracle simulated by closed form",
                    "raw_evals_per_call": 1,
                },
                "truth": {
                    **truth,
                    "theta_unit": THETA_EXISTS_UNIT,
                    "stated_error": 0.0,
                },
                "acceptance": {
                    "crossing_ok": True,
                    "prefix_ok": True,
                    "learnability_r2": r2,
                    "norm_ladder_fixed_lengthscale": ladder,
                    "certificate_conditions": cert,
                },
                "calibration": {
                    "n_cal": CAL_N,
                    "seeds": CAL_SEEDS,
                    "scaler_scale": [scale],
                    "lengthscale": lengthscale,
                    "lam": LAM,
                    "inflation": INFLATION,
                    "mu_norm": mu_norm,
                    "B": [B],
                    "B_slice": [B_slice],
                    "r2_holdout": [r2],
                    "R": 0.0,
                },
            },
            indent=1,
        )
    )
    log.info(
        "calibration -> %s (compare with %s; B=%.3f, lengthscale=%.4f, scale=%.5f)",
        out_path,
        FROZEN_PATH,
        B,
        lengthscale,
        scale,
    )


def _tasks(seeds, budgets=BUDGETS):
    """All (arm, budget, seed) cells."""
    return [(a, b, s) for a in ARMS for b in budgets for s in seeds]


def _manifest(out_dir: Path, workers: int, n_tasks: int, filt: dict) -> None:
    """Write the run manifest."""
    commit, dirty = git_state()
    blob = FROZEN_PATH.read_text()
    out_dir.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "experiment": "double_integrator_sweep",
                "git_commit": commit,
                "git_dirty": dirty,
                "frozen_calibration_sha256": hashlib.sha256(blob.encode()).hexdigest(),
                "arms": list(ARMS),
                "budgets": list(BUDGETS),
                "seeds": list(SEEDS),
                "filter": filt,
                "n_tasks": n_tasks,
                "workers": workers,
                "theta_unit": THETA_EXISTS_UNIT,
                "python": sys.version,
                "numpy": np.__version__,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "started_utc": datetime.now(UTC).isoformat(),
            },
            indent=1,
        )
    )


def cmd_sweep(workers: int, filt: dict) -> None:
    """The 750-run sweep (or the selected cells) -> ``double_integrator_sweep``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(f"results/{stamp}_double_integrator_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = filter_tasks(_tasks(SEEDS), **filt, fields=(0, 1, 2))
    _manifest(out_dir, workers, len(tasks), filt)
    rows_path = out_dir / "rows.jsonl"
    mon = RssMonitor()
    mon.start()
    t0 = time.time()
    with rows_path.open("a") as fh, mp.Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_task, tasks), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 25 == 0:
                rate = i / (time.time() - t0)
                log.info(
                    "%d/%d eta %.2f h", i, len(tasks), (len(tasks) - i) / rate / 3600
                )
    mon.stop()
    log.info(
        "sweep done in %.2f h, peak RSS %.2f GiB",
        (time.time() - t0) / 3600,
        mon.peak_gib,
    )


def cmd_analyze(results_dir: str) -> None:
    """Containment (D-C1), contraction (D-C2), and design checks (D-C3) -> gate.json."""
    from scipy.stats import binomtest

    rows = [
        json.loads(x)
        for x in Path(results_dir, "rows.jsonl").read_text().splitlines()
        if x.strip()
    ]
    gate: dict = {"n_rows": len(rows), "D_C1": {}, "D_C2": {}, "D_C3": {}, "cells": {}}
    c1_ok = True
    for arm in CERTIFIED:
        for b in BUDGETS:
            cell = [r for r in rows if r["arm"] == arm and r["budget"] == b]
            if not cell:
                continue
            viol = sum(1 for r in cell if not r["covered_anytime"])
            p = binomtest(viol, len(cell), DELTA, alternative="greater").pvalue
            c1_ok &= not (p < 0.05)
            gate["D_C1"][f"{arm}/{b}"] = {
                "violations": viol,
                "n": len(cell),
                "p": float(p),
                "reject": bool(p < 0.05),
            }
    gate["D_C1"]["pass"] = bool(c1_ok)
    med = {}
    for arm in CERTIFIED:
        med[arm] = {
            b: float(
                np.median(
                    [
                        r["final_width"]
                        for r in rows
                        if r["arm"] == arm and r["budget"] == b
                    ]
                )
            )
            for b in BUDGETS
            if any(r["arm"] == arm and r["budget"] == b for r in rows)
        }
    widths = med.get(ARM_BRACKET_GREEDY, {})
    seq = [widths[b] for b in BUDGETS if b in widths]
    gate["D_C2"] = {
        "median_widths": med,
        "nontrivial_all": bool(
            all(w < NONTRIVIAL_FRACTION for d in med.values() for w in d.values())
        ),
        "contracting_method_arm": bool(
            len(seq) == len(BUDGETS)
            and all(a > b for a, b in zip(seq[:-1], seq[1:], strict=True))
        ),
    }
    gate["D_C2"]["pass"] = bool(
        gate["D_C2"]["nontrivial_all"] and gate["D_C2"]["contracting_method_arm"]
    )
    wit = [r for r in rows if r["arm"] in CERTIFIED and r["witness_sound"] is not None]
    sound = sum(1 for r in wit if r["witness_sound"])
    gate["D_C3"] = {
        "n_with_witness": len(wit),
        "sound": sound,
        "rate": sound / len(wit) if wit else None,
        "pass": bool(wit and sound / len(wit) >= 0.95),
    }
    for arm in ARMS:
        for b in BUDGETS:
            cell = [r for r in rows if r["arm"] == arm and r["budget"] == b]
            if not cell:
                continue
            gate["cells"][f"{arm}/{b}"] = {
                "median_width": float(np.median([r["final_width"] for r in cell])),
                "anytime_coverage": sum(r["covered_anytime"] for r in cell) / len(cell),
                "final_coverage": sum(r["covered_final"] for r in cell) / len(cell),
                "median_wall_s": float(np.median([r["wall_s"] for r in cell])),
            }
    gate["verdict"] = {
        "D_C1": gate["D_C1"]["pass"],
        "D_C2": gate["D_C2"]["pass"],
        "D_C3": gate["D_C3"]["pass"],
        "gate": "PASS" if gate["D_C1"]["pass"] and gate["D_C2"]["pass"] else "FAIL",
    }
    Path(results_dir, "gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate["verdict"], indent=1))
    print(json.dumps(gate["D_C2"]["median_widths"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    sw = sub.add_parser("sweep")
    sw.add_argument("--workers", type=int, required=True)
    add_filter_args(sw)
    an = sub.add_parser("analyze")
    an.add_argument("results_dir", nargs="?", default=None)
    args = ap.parse_args()
    if args.cmd == "calibrate":
        cmd_calibrate()
    elif args.cmd == "sweep":
        filt = {"labels": args.labels, "budgets": args.budgets, "seeds": args.seeds}
        cmd_sweep(args.workers, filt)
    else:
        cmd_analyze(args.results_dir or str(run_path("double_integrator_sweep")))
