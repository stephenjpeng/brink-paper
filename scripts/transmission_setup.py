"""Transmission network (IEEE 39-bus): calibration, reference limit, and thresholds.

Every constant comes from ``configs/transmission.yaml`` and
``configs/frozen/transmission.json``. Subcommands:

``calibrate``
    The 512-point calibration recipe (Appendix C.1) on the window channels.
    Writes ``frozen.json`` to a new run directory; the reported runs use the
    shipped ``configs/frozen/transmission.json``.
``truth``
    Reference limit: the 9^4 midpoint design grid, a 12-step bisection of
    each design's feasible window, then three clipped local 5^4 refinements
    around the best designs (Appendix C.5).
``monotone-check``
    Checks on 256 further designs that the feasible stresses form one
    interval (Appendix C.6).
``draw-columns`` and ``score-columns``
    The 628 independent designs behind the feasible-fraction thresholds
    U_0.01 and U_0.05 (Appendix C.6).

Usage:
    uv run python scripts/transmission_setup.py calibrate
    uv run python scripts/transmission_setup.py truth --workers N
    uv run python scripts/transmission_setup.py monotone-check --workers N
    uv run python scripts/transmission_setup.py draw-columns
    uv run python scripts/transmission_setup.py score-columns --workers N \
        [--designs DIR]
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
import socket  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from importlib.metadata import version  # noqa: E402

import numpy as np  # noqa: E402

from brink.calibration import calibrate  # noqa: E402
from brink.memory import RssMonitor  # noqa: E402
from brink.studies.transmission import (  # noqa: E402
    C_RANGE,
    GEN_BUSES_CASE39,
    GEN_ROWS,
    GEN_SCALE_RANGE,
    M_CHANNELS_WINDOW,
    SHUNT_BUS,
    SHUNT_RANGE_MVAR,
    SOLVER_SETTINGS,
    V_BAND,
    X_DIM,
    GridLoadabilityOracle,
    make_window_oracle,
)
from brink.window_margins import (  # noqa: E402
    BISECT_STEPS as WINDOW_BISECT_STEPS,
)
from brink.window_margins import (  # noqa: E402
    MAX_RAW_EVALS,
    N_SCAN,
    NO_WINDOW_C_OFF,
    NO_WINDOW_C_ON,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feeder_setup as he  # noqa: E402  (shared column and output helpers)
from common import git_state, local_grid, run_path, window_union_gaps  # noqa: E402

log = logging.getLogger("transmission_setup")
SPAWN = mp.get_context("spawn")  # same start method on macOS and Linux
FROZEN_PATH = Path("configs/frozen/transmission.json")

# Reference-limit construction (Appendix C.5).
TRUTH_GRID_AXIS = 9
BISECT_STEPS = 12
WINDOW_SCAN_POINTS = 16
REFINE_HALF_WIDTHS = (1 / 9, 1 / 36, 1 / 144)  # one stage each
REFINE_TOP_K = 4
TRUTH_BISECT_HALF = 0.5**BISECT_STEPS / 2  # midpoint of a 2^-12 bracket
WINDOW_BISECT_HALF = 0.125 * 0.5**8 / 2  # window oracle: 8 steps on a 1/8 cell


def load_cfg(path: Path = _CFG_PATH) -> dict:
    """The transmission constants (YAML), with their SHA-256 under ``sha256``."""
    return he.load_s3_config(path)


_ORACLE: GridLoadabilityOracle | None = None


def _oracle() -> GridLoadabilityOracle:
    """Per-process raw oracle, built lazily (spawn-safe)."""
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = GridLoadabilityOracle()
    return _ORACLE


def truth_column(
    args: tuple[float, float, float, float],
) -> tuple[float, float, int, int]:
    """Window-aware feasibility window of one x column, in unit c.

    Args:
        args: Unit design coordinates (x1, x2, x3, x4).

    Returns:
        Tuple (start, death, n_flows, multi_window_flag): the window
        [start, death] in unit c (start = inf when never feasible;
        death = inf when still feasible at c = 1), the number of power
        flows spent, and 1 when the coarse scan saw a non-contiguous
        feasible pattern (only the first window is kept).
    """
    x = np.asarray(args, dtype=float)
    oracle = _oracle()
    n_flows = 0

    def feas(c: float) -> bool:
        nonlocal n_flows
        n_flows += 1
        return oracle.feasible(x, c)

    def bisect(lo: float, hi: float, lo_feasible: bool) -> float:
        """Bisect the feasibility edge between lo and hi (12 steps)."""
        for _ in range(BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            if feas(mid) == lo_feasible:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    f0 = feas(0.0)
    multi = 0
    if f0:
        start = 0.0
        death = np.inf if feas(1.0) else bisect(0.0, 1.0, True)
        return start, death, n_flows, multi
    # Infeasible at c_min: coarse scan for a feasibility window.
    scan_c = (np.arange(WINDOW_SCAN_POINTS) + 0.5) / WINDOW_SCAN_POINTS
    scan_f = np.array([feas(float(c)) for c in scan_c])
    f1 = feas(1.0)
    if not scan_f.any():
        if not f1:
            return np.inf, -np.inf, n_flows, multi
        # Window confined between the last scan point and c = 1.
        start = bisect(float(scan_c[-1]), 1.0, False)
        return start, np.inf, n_flows, multi
    first = int(np.argmax(scan_f))
    run_end = first
    while run_end + 1 < WINDOW_SCAN_POINTS and scan_f[run_end + 1]:
        run_end += 1
    if scan_f[run_end + 1 :].any():
        multi = 1
    prev_c = float(scan_c[first - 1]) if first > 0 else 0.0
    start = bisect(prev_c, float(scan_c[first]), False)
    if run_end == WINDOW_SCAN_POINTS - 1:
        death = np.inf if f1 else bisect(float(scan_c[run_end]), 1.0, True)
    else:
        death = bisect(float(scan_c[run_end]), float(scan_c[run_end + 1]), True)
    return start, death, n_flows, multi


def volume_at(start: np.ndarray, death: np.ndarray, c: float) -> float:
    """Exact truth volume at a unit coefficient from the window arrays."""
    return float(((start <= c) & (c <= death)).mean())


# --- calibration ---


def environment_registration() -> dict:
    """The environment constants recorded in the frozen JSON."""
    return {
        "network": "pandapower.networks.case39 (IEEE 39-bus New England)",
        "gen_rows": list(GEN_ROWS),
        "gen_buses_case39": list(GEN_BUSES_CASE39),
        "gen_scale_range": list(GEN_SCALE_RANGE),
        "shunt_bus": SHUNT_BUS,
        "shunt_kind": "inductive reactor (absorbing MVAr)",
        "shunt_range_mvar": list(SHUNT_RANGE_MVAR),
        "c_range": list(C_RANGE),
        "voltage_band_pu": list(V_BAND),
        "voltage_margin_scope": "PQ (non-generator) buses only",
        "solver_settings": SOLVER_SETTINGS,
        "x_dim": X_DIM,
        "m": M_CHANNELS_WINDOW,
        "oracle_representation": "window-margin",
        "window_construction": {
            "n_scan": N_SCAN,
            "bisect_steps": WINDOW_BISECT_STEPS,
            "max_raw_evals_per_call": MAX_RAW_EVALS,
            "no_window_c_on": NO_WINDOW_C_ON,
            "no_window_c_off": NO_WINDOW_C_OFF,
            "multi_window_rule": "widest window kept; event counted in the run sidecar",
            "channels": "g_lo = c_on(x) - c, g_hi = c - c_off(x) in unit c",
            "standardization": "scale-only by calibration IQR/1.349, "
            "UNCLIPPED (clipping would break the registered exact affinity "
            "in c; raw channels are bounded in [-1.0, 1.25])",
            "crash_convention": "a non-converged power flow is infeasible "
            "inside the raw predicate; no crash offset exists in the "
            "window channels",
        },
    }


def cmd_calibrate() -> None:
    """Run the window-channel calibration -> ``results/<UTC>_transmission_calibration``.

    The shipped ``configs/frozen/transmission.json`` also carries a ``truth``
    block from an earlier positive-volume study on the same network; only its
    volume curve is used, as a consistency check in ``truth``.
    """
    oracle = make_window_oracle()

    def raw_window_channels(x_unit: np.ndarray, c_unit: float) -> np.ndarray:
        return oracle.raw_channels(x_unit, c_unit)

    t0 = time.perf_counter()
    cal = calibrate(
        "grid-loadability-v2",  # name stored in the frozen file
        raw_window_channels,
        x_dim=X_DIM,
        m=M_CHANNELS_WINDOW,
        clip=False,
    )
    wall = time.perf_counter() - t0
    counters = oracle.counters()
    log.info(
        "calibration oracle accounting: %d window calls, %d raw power flows, "
        "%.2f raw evals/call, %d multi-window events",
        counters["n_calls"],
        counters["n_raw_evals"],
        counters["raw_evals_per_call"],
        counters["n_multi_window"],
    )
    payload = {
        "environment": environment_registration(),
        "calibration": asdict(cal),
        "calibration_wall_time_s": wall,
        "calibration_window_oracle_counters": counters,
    }
    out = he._write("transmission_calibration", {"wall_s": wall}, {}, 1)
    (out / "frozen.json").write_text(json.dumps(payload, indent=2))
    log.info("calibration -> %s (compare with %s)", out / "frozen.json", FROZEN_PATH)
    log.info(
        "B=%s B_slice=%s R^2=%s lengthscale=%.6f (mult %.1f)",
        cal.B,
        cal.B_slice,
        cal.r2_holdout,
        cal.lengthscale,
        cal.lengthscale_multiplier,
    )
    if not cal.r2_ok:
        log.error("calibration R^2 %s below 0.5 on some channel", cal.r2_holdout)


# --- reference limit ---


def _columns(pool: ProcessPoolExecutor, X: np.ndarray) -> tuple[np.ndarray, ...]:
    out = list(pool.map(truth_column, [tuple(map(float, x)) for x in X], chunksize=8))
    return tuple(np.array([r[i] for r in out]) for i in range(4))


def cmd_truth(workers: int) -> None:
    """Reference limit, nesting check, and refinement -> ``transmission_truth``."""
    frozen = json.loads(FROZEN_PATH.read_text())["truth"]
    ax = (np.arange(TRUTH_GRID_AXIS) + 0.5) / TRUTH_GRID_AXIS
    X = np.stack([m.ravel() for m in np.meshgrid(ax, ax, ax, ax, indexing="ij")], 1)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        start, death, flows, multi = _columns(pool, X)
        log.info("grid: %d columns, %d flows", len(X), flows.sum())
        c_grid = np.asarray(frozen["v_grid_c"])
        v_grid = np.array([volume_at(start, death, float(c)) for c in c_grid])
        reproduces = bool(np.array_equal(v_grid, np.asarray(frozen["v_grid"])))
        gaps = window_union_gaps(start, death, 2 * TRUTH_BISECT_HALF)
        stages = [{"stage": "grid", "n_columns": len(X), "max_death": death.max()}]
        allX, alld, alls = X, death, start
        n_flows = int(flows.sum())
        for w in REFINE_HALF_WIDTHS:
            top = allX[np.argsort(alld)[::-1][:REFINE_TOP_K]]
            Xr = np.unique(np.concatenate([local_grid(t, w) for t in top]), axis=0)
            s_r, d_r, f_r, _ = _columns(pool, Xr)
            n_flows += int(f_r.sum())
            allX = np.concatenate([allX, Xr])
            alld, alls = np.concatenate([alld, d_r]), np.concatenate([alls, s_r])
            stages.append(
                {
                    "stage": f"local w={w:.5f}",
                    "n_columns": len(Xr),
                    "max_death": d_r.max(),
                }
            )
            log.info("refine w=%.5f: %d columns, max %.6f", w, len(Xr), d_r.max())
    wall = time.perf_counter() - t0
    best = int(np.argmax(alld))
    residual = float(stages[-1]["max_death"] - stages[-2]["max_death"])
    theta = float(alld[best])
    summary = {
        "theta_exists_unit": theta,
        "theta_exists_physical": 1.0 + 0.35 * theta,
        "argmax_x_unit": allX[best].tolist(),
        "argmax_start": float(alls[best]),
        "stages": [
            {k: float(v) if k == "max_death" else v for k, v in s.items()}
            for s in stages
        ],
        "last_stage_improvement": residual,
        "error_terms_unit_c": {
            "truth_bisection_half": TRUTH_BISECT_HALF,
            "window_oracle_bisection_half": WINDOW_BISECT_HALF,
            "x_refinement_last_improvement": max(residual, 0.0),
        },
        "frozen_v_grid_reproduced": reproduces,
        "nesting_union_gaps": gaps,
        "grid_multi_window_columns": int(multi.sum()),
        "grid_delayed_onset_columns": int(((start > 0) & np.isfinite(start)).sum()),
        "grid_never_feasible_columns": int((~np.isfinite(start)).sum()),
        "grid_still_feasible_at_c1": int(np.isposinf(death).sum()),
        "n_power_flows": n_flows,
        "wall_s": wall,
    }
    commit, dirty = git_state()
    out = Path("results") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_transmission_truth"
    out.mkdir(parents=True)
    np.savez_compressed(out / "truth_columns.npz", X=allX, start=alls, death=alld)
    (out / "truth.json").write_text(json.dumps(summary, indent=1))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "transmission_truth",
                "git_commit": commit,
                "git_dirty": dirty,
                "workers": workers,
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "seeds": "none (deterministic grids)",
            },
            indent=1,
        )
    )
    log.info(
        "theta_exists %.6f (gaps %s, v_grid reproduced %s) -> %s",
        theta,
        gaps,
        reproduces,
        out,
    )


# --- feasible-fraction thresholds (Appendix C.6) ---


def column_score(start: float, death: float, multi: int, pad: float) -> float:
    """Upper limit of one sampled design from its ``truth_column`` window.

    Args:
        start: Window onset (inf when no window was found).
        death: Window death (inf when feasible at c = 1).
        multi: 1 when the coarse scan saw more than one window.
        pad: At least the half-width of either final bisection bracket.

    Returns:
        1.0 when feasible at c = 1, min(death + pad, 1) for one window, and
        +inf for a multi-window or no-window column.
    """
    if multi or not np.isfinite(start):
        return float("inf")
    if np.isposinf(death):
        return 1.0
    return min(float(death) + pad, 1.0)


def window_pattern_flag(
    feasible, c_grid, start: float, death: float, tol: float
) -> dict:
    """Single-window check for one design (Appendix C.6).

    Args:
        feasible: Raw predicate at each grid coefficient, bool array.
        c_grid: The grid coefficients (unit c).
        start: The design's ``truth_column`` onset.
        death: The design's ``truth_column`` death.
        tol: Grid points within this distance of an edge are not judged.

    Returns:
        ``flagged`` when the feasible points are not one contiguous run, or a
        point more than tol inside the window is infeasible, or one more than
        tol outside it is feasible; plus the counts behind the decision.
    """
    f, c = np.asarray(feasible, dtype=bool), np.asarray(c_grid, dtype=float)
    idx = np.flatnonzero(f)
    contiguous = bool(idx.size == 0 or idx[-1] - idx[0] + 1 == idx.size)
    inside = (c > start + tol) & (c < death - tol)
    outside = (c < start - tol) | (c > death + tol)
    n_in, n_out = int((inside & ~f).sum()), int((outside & f).sum())
    return {
        "flagged": bool(not contiguous or n_in or n_out),
        "contiguous": contiguous,
        "n_inside_infeasible": n_in,
        "n_outside_feasible": n_out,
    }


def designs_sha256(X: np.ndarray) -> str:
    """SHA-256 of a design array's bytes (recorded with the columns)."""
    return hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()


def score_column(task: tuple[int, list, float]) -> dict:
    """One sampled design: ``truth_column``, then its upper limit."""
    i, x, pad = task
    t0 = time.perf_counter()
    start, death, n_flows, multi = truth_column(tuple(map(float, x)))
    return {
        "index": i,
        "x": list(x),
        "start": float(start),
        "death": float(death),
        "multi_window": int(multi),
        "n_flows": int(n_flows),
        "score": column_score(start, death, multi, pad),
        "wall_s": time.perf_counter() - t0,
    }


def monotone_design(task: tuple[int, list, int, float]) -> dict:
    """Raw predicate on the stress grid plus ``truth_column`` (single-window check)."""
    i, x, n_points, tol = task
    xa, oracle = np.asarray(x, dtype=float), _oracle()
    c = np.linspace(0.0, 1.0, int(n_points))
    feas = np.array([oracle.feasible(xa, float(v)) for v in c])
    start, death, n_flows, multi = truth_column(tuple(map(float, x)))
    return {
        "index": i,
        "x": list(x),
        "feasible": feas.astype(int).tolist(),
        "start": float(start),
        "death": float(death),
        "multi_window": int(multi),
        "n_flows": int(n_flows) + len(c),
        **window_pattern_flag(feas, c, start, death, tol),
    }


def _pool_map(fn, tasks: list, workers: int) -> tuple[list, float, float]:
    """Map in spawned workers; returns (rows, wall s, aggregate peak RSS GiB)."""
    mon = RssMonitor()
    mon.start()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=SPAWN) as pool:
        rows = list(pool.map(fn, tasks, chunksize=1))
    mon.stop()
    return rows, time.perf_counter() - t0, mon.peak_gib


def _host(name: str) -> dict:
    return {
        "experiment": name,
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "pandapower": version("pandapower"),
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def cmd_monotone_check(cfg: dict, workers: int) -> None:
    """Single-window check on 256 designs -> ``transmission_monotone_check``."""
    mc = cfg["monotone_check"]
    X = he.draw_designs(mc["stream"], mc["n_designs"])
    tasks = [
        (i, x.tolist(), int(mc["c_grid_points"]), float(mc["tol"]))
        for i, x in enumerate(X)
    ]
    rows, wall, peak = _pool_map(monotone_design, tasks, workers)
    flagged = [r["index"] for r in rows if r["flagged"]]
    summary = {
        "n_designs": len(rows),
        "n_flagged": len(flagged),
        "flagged": flagged,
        "passed": not flagged,
        "n_power_flows": sum(r["n_flows"] for r in rows),
        "n_multi_window": sum(r["multi_window"] for r in rows),
        "config_sha256": cfg["sha256"],
        "wall_s": wall,
        "aggregate_peak_rss_gib": peak,
        "rows": rows,
    }
    out = he._write(
        "transmission_monotone_check",
        summary,
        {
            **_host("transmission_monotone_check"),
            "monotone_check": mc,
            "streams": f"designs: default_rng({list(mc['stream'])})",
        },
        workers,
    )
    log.info("single-window check: %d/%d flagged -> %s", len(flagged), len(rows), out)


def cmd_draw_columns(cfg: dict) -> Path:
    """Draw and save the 628 independent designs -> ``transmission_designs``."""
    cols = cfg["columns"]
    X = he.draw_designs(cols["design_stream"], cols["n"])
    sha = designs_sha256(X)
    out = he._write(
        "transmission_designs",
        {
            "n": int(cols["n"]),
            "design_stream": list(cols["design_stream"]),
            "designs_sha256": sha,
            "config_sha256": cfg["sha256"],
        },
        {
            **_host("transmission_designs"),
            "purpose": "independent designs behind U_eps, drawn before any search",
        },
        0,
    )
    np.savez_compressed(out / "designs.npz", X=X)
    log.info("drew %d designs (sha256 %s) -> %s", len(X), sha[:12], out)
    return out


def cmd_score_columns(cfg: dict, designs_dir: Path, workers: int) -> None:
    """Upper limits of the sampled designs and U_eps -> ``transmission_columns``."""
    cols = cfg["columns"]
    meta = json.loads((designs_dir / "designs.json").read_text())
    X = np.load(designs_dir / "designs.npz")["X"]
    if designs_sha256(X) != meta["designs_sha256"]:
        raise ValueError("designs.npz does not match its recorded SHA-256")
    tasks = [(i, x.tolist(), float(cols["score_pad"])) for i, x in enumerate(X)]
    rows, wall, peak = _pool_map(score_column, tasks, workers)
    scores = np.array([r["score"] for r in rows])
    summary = {
        **he.summarize_scores(scores, cols),
        "designs_sha256": meta["designs_sha256"],
        "designs_dir": str(designs_dir),
        "config_sha256": cfg["sha256"],
        "n_power_flows": sum(r["n_flows"] for r in rows),
        "n_multi_window": sum(r["multi_window"] for r in rows),
        "n_no_window": int(sum(not np.isfinite(r["start"]) for r in rows)),
        "n_feasible_at_c1": int(sum(np.isposinf(r["death"]) for r in rows)),
        "wall_s": wall,
        "aggregate_peak_rss_gib": peak,
    }
    out = he._write(
        "transmission_columns",
        summary,
        {
            **_host("transmission_columns"),
            "columns": cols,
        },
        workers,
    )
    with (out / "rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    log.info("U_eps: %s -> %s", json.dumps(summary["by_eps"]), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    sub.add_parser("draw-columns")
    for name in ("truth", "monotone-check", "score-columns"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, required=True)
    sub.choices["score-columns"].add_argument("--designs", type=Path, default=None)
    args = ap.parse_args()
    if args.cmd == "calibrate":
        cmd_calibrate()
    elif args.cmd == "truth":
        cmd_truth(args.workers)
    elif args.cmd == "monotone-check":
        cmd_monotone_check(load_cfg(), args.workers)
    elif args.cmd == "draw-columns":
        cmd_draw_columns(load_cfg())
    else:
        designs = args.designs or run_path("transmission_designs")
        cmd_score_columns(load_cfg(), designs, args.workers)
