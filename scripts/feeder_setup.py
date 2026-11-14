"""Distribution network (CIGRE MV feeder): calibration, reference limit, and thresholds.

Every constant comes from ``configs/feeder.yaml`` and
``configs/frozen/feeder.json``. Subcommands:

``calibrate``
    The 512-point calibration recipe (Appendix C.1) on the upper-window
    channel at K = 16 sampled conditions per check, plus the noise scale R(K)
    for K in {16, 4, 1} from 100 replicate pairs. Writes ``frozen.json`` to a
    new run directory; the reported runs use the shipped frozen file.
``truth``
    Reference limit: the 7^4 midpoint design grid (256 conditions per check,
    10 bisection steps per window edge), three clipped local refinements, and
    16 replicated estimates at each of the 8 best designs (Appendix C.5).
``monotone-check``
    Persistence check: the expected loading margin increases with demand on
    256 further designs (Appendix C.6).
``draw-columns`` and ``score-columns``
    The 628 independent designs behind U_0.01 and U_0.05, each scored by a
    rough limit and a ladder of empirical-Bernstein loading tests.

Condition streams (all ``default_rng([991, index])``): grid column i: index
i; refinement stage s point j: 100_000 * s + j; replicate k of near-best
design r: 1_000_000 + 100 * r + k. The design check of sweep task i uses
``default_rng([993, offset + i])`` with 4,096 conditions.

Usage:
    uv run python scripts/feeder_setup.py calibrate
    uv run python scripts/feeder_setup.py truth --workers N
    uv run python scripts/feeder_setup.py monotone-check --workers N
    uv run python scripts/feeder_setup.py draw-columns
    uv run python scripts/feeder_setup.py score-columns --workers N [--designs DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from brink.calibration import (
    calibrate,
    calibration_points,
    evaluate_raw,
    fit_scalers,
    measure_noise_scale,
)
from brink.memory import RssMonitor
from brink.quantile_bound import (
    empirical_bernstein_deviation,
    ladder_upper_score,
    order_statistic_upper,
)
from brink.studies.feeder import (
    NOISE_KNOBS,
    PREFIX_NO_WINDOW_REPORT_GRID_STEP,
    X_DIM,
    HostingCapacityOracle,
    _MeanPredicate,
)
from brink.window_margins import WindowMarginOracle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import git_state, local_grid, run_path, window_union_gaps  # noqa: E402

log = logging.getLogger("feeder_setup")

# Reference-limit construction (Appendix C.5): 256-condition mean predicate,
# 10 bisection steps per window edge, 16-point scan, stream default_rng([991, i]).
TRUTH_GRID_AXIS = 7
TRUTH_SEED = 991
TRUTH_SNAPSHOTS = 256
TRUTH_BISECT_STEPS = 10
TRUTH_WINDOW_SCAN_POINTS = 16
# Calibration streams: the 512 calibration window calls consume one sequential
# stream seeded 9010; the replicate pairs of noise knob K consume [9011, K].
CAL_SNAPSHOT_SEED = 9010
REPLICATE_SNAPSHOT_SEED = 9011

_ORACLE: HostingCapacityOracle | None = None
_TRUTH_SNAPSHOTS = TRUTH_SNAPSHOTS


def _oracle() -> HostingCapacityOracle:
    """Per-process raw oracle, built lazily (spawn-safe)."""
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = HostingCapacityOracle()
    return _ORACLE


def truth_column(
    args: tuple[int, float, float, float, float],
) -> tuple[float, float, int, int]:
    """Window-aware feasibility window of one x column, in unit c.

    The raw predicate is the TRUTH_SNAPSHOTS-mean feasibility
    predicate with the dedicated per-column truth stream
    ``default_rng([991, column_index])``.

    Args:
        args: (column_index, x1, x2, x3, x4) unit design coordinates.

    Returns:
        Tuple (start, death, n_predicate_evals, multi_window_flag): the
        window [start, death] in unit c (start = inf when never feasible;
        death = inf when still feasible at c = 1).
    """
    col, *xs = args
    x = np.asarray(xs, dtype=float)
    oracle = _oracle()
    rng = np.random.default_rng([TRUTH_SEED, int(col)])
    n_evals = 0

    def feas(c: float) -> bool:
        nonlocal n_evals
        n_evals += 1
        return oracle.mean_feasible(x, float(c), _TRUTH_SNAPSHOTS, rng)

    def bisect(lo: float, hi: float, lo_feasible: bool) -> float:
        """Bisect the feasibility edge between lo and hi (10 steps)."""
        for _ in range(TRUTH_BISECT_STEPS):
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
        return start, death, n_evals, multi
    # Infeasible at c_min: coarse scan for a feasibility window.
    scan_c = (np.arange(TRUTH_WINDOW_SCAN_POINTS) + 0.5) / TRUTH_WINDOW_SCAN_POINTS
    scan_f = np.array([feas(float(c)) for c in scan_c])
    f1 = feas(1.0)
    if not scan_f.any():
        if not f1:
            return np.inf, -np.inf, n_evals, multi
        start = bisect(float(scan_c[-1]), 1.0, False)
        return start, np.inf, n_evals, multi
    first = int(np.argmax(scan_f))
    run_end = first
    while run_end + 1 < TRUTH_WINDOW_SCAN_POINTS and scan_f[run_end + 1]:
        run_end += 1
    if scan_f[run_end + 1 :].any():
        multi = 1
    prev_c = float(scan_c[first - 1]) if first > 0 else 0.0
    start = bisect(prev_c, float(scan_c[first]), False)
    if run_end == TRUTH_WINDOW_SCAN_POINTS - 1:
        death = np.inf if f1 else bisect(float(scan_c[run_end]), 1.0, True)
    else:
        death = bisect(float(scan_c[run_end]), float(scan_c[run_end + 1]), True)
    return start, death, n_evals, multi


def volume_at(start: np.ndarray, death: np.ndarray, c: float) -> float:
    """Truth volume at a unit coefficient from the window arrays."""
    return float(((start <= c) & (c <= death)).mean())


def _set_truth_snapshots(snapshots: int) -> None:
    """Worker initializer: set the truth snapshot count."""
    global _TRUTH_SNAPSHOTS
    _TRUTH_SNAPSHOTS = snapshots


S3_CONFIG = Path("configs/feeder.yaml")
FROZEN_PATH = Path("configs/frozen/feeder.json")
REFINE_HALF_WIDTHS = (1 / 7, 1 / 28, 1 / 112)  # one stage each (7^4 grid step 1/7)
REFINE_TOP_K = 4
N_NEAR_BEST = 8
N_REPLICATES = 16
N_WITNESS_REPLICATES = 4
WITNESS_SNAPSHOTS = 4096
TRUTH_BISECT_HALF = 0.5**TRUTH_BISECT_STEPS / 2  # midpoint of a 2^-10 bracket
WINDOW_BISECT_HALF = 0.125 * 0.5**8 / 2  # window oracle: 8 steps on a 1/8 cell


def _columns(pool, X: np.ndarray, first_index: int) -> tuple[np.ndarray, ...]:
    """Windows of the columns X with streams first_index, first_index + 1, ..."""
    args = [(first_index + i, *map(float, x)) for i, x in enumerate(X)]
    out = list(pool.map(truth_column, args, chunksize=4))
    return tuple(np.array([r[i] for r in out]) for i in range(4))


def margin_check(g: np.ndarray) -> dict:
    """Mean-margin feasibility from per-snapshot raw margins.

    Args:
        g: Raw margins, shape (n_snapshots, m).

    Returns:
        Per-channel means and standard errors, ``feasible`` (every mean <= 0),
        and ``clear_3se`` (every mean + 3 se <= 0).
    """
    mean = g.mean(axis=0)
    se = g.std(axis=0, ddof=1) / np.sqrt(len(g))
    return {
        "mean": mean.tolist(),
        "se": se.tolist(),
        "feasible": bool(np.all(mean <= 0.0)),
        "clear_3se": bool(np.all(mean + 3.0 * se <= 0.0)),
    }


def _write(name: str, payload: dict, extra: dict, workers: int) -> Path:
    commit, dirty = git_state()
    out = Path("results") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{name}"
    out.mkdir(parents=True)
    (out / f"{name.split('_')[-1]}.json").write_text(json.dumps(payload, indent=1))
    manifest = {
        "experiment": name, "git_commit": commit, "git_dirty": dirty,
        "workers": workers, "python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), **extra,
    }  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def cmd_truth(workers: int) -> None:
    """Grid truth, clipped refinement, replicates, nesting check -> results/."""
    frozen = json.loads(FROZEN_PATH.read_text())["truth"]
    ax = (np.arange(TRUTH_GRID_AXIS) + 0.5) / TRUTH_GRID_AXIS
    X = np.stack([m.ravel() for m in np.meshgrid(ax, ax, ax, ax, indexing="ij")], 1)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        start, death, evals, multi = _columns(pool, X, 0)
        c_grid = np.asarray(frozen["v_grid_c"])
        v_grid = np.array([volume_at(start, death, float(c)) for c in c_grid])
        reproduces = bool(np.array_equal(v_grid, np.asarray(frozen["v_grid"])))
        log.info("grid: %d columns, max death %.5f, frozen v_grid reproduced %s",
                 len(X), death.max(), reproduces)  # fmt: skip
        stages = [
            {"stage": "grid", "n_columns": len(X), "max_death": float(death.max())}
        ]
        allX, alls, alld, n_evals = X, start, death, int(evals.sum())
        for s, w in enumerate(REFINE_HALF_WIDTHS, start=1):
            top = allX[np.argsort(alld)[::-1][:REFINE_TOP_K]]
            Xr = np.unique(np.concatenate([local_grid(t, w) for t in top]), axis=0)
            s_r, d_r, e_r, _ = _columns(pool, Xr, 100_000 * s)
            n_evals += int(e_r.sum())
            allX = np.concatenate([allX, Xr])
            alls, alld = np.concatenate([alls, s_r]), np.concatenate([alld, d_r])
            stages.append({"stage": f"local w={w:.5f}", "n_columns": len(Xr),
                           "max_death": float(d_r.max())})  # fmt: skip
            log.info("refine w=%.5f: %d columns, max %.5f", w, len(Xr), d_r.max())
        near = allX[np.argsort(alld)[::-1][:N_NEAR_BEST]]
        reps = []
        for r, x in enumerate(near):
            Xk = np.repeat(x[None], N_REPLICATES, axis=0)
            _, d_k, e_k, _ = _columns(pool, Xk, 1_000_000 + 100 * r)
            n_evals += int(e_k.sum())
            reps.append(d_k)
    wall = time.perf_counter() - t0
    reps = np.array(reps)
    means = reps.mean(axis=1)
    best = int(np.argmax(means))
    sd = reps.std(axis=1, ddof=1)
    gaps = window_union_gaps(alls, alld, 2 * TRUTH_BISECT_HALF)
    residual = stages[-1]["max_death"] - stages[-2]["max_death"]
    theta = float(means[best])
    summary = {
        "theta_exists_unit": theta,
        "theta_exists_physical": 1.0 + theta,  # load factor c in [1, 2]
        "argmax_x_unit": near[best].tolist(),
        "single_stream_max_death": float(alld.max()),
        "single_stream_argmax_x": allX[int(np.argmax(alld))].tolist(),
        "near_best": [
            {
                "x": near[r].tolist(),
                "single_stream_death": float(np.sort(alld)[::-1][r]),
                "replicate_mean": float(means[r]),
                "replicate_sd": float(sd[r]),
                "replicate_min": float(reps[r].min()),
                "replicate_max": float(reps[r].max()),
            }
            for r in range(len(near))
        ],  # fmt: skip
        "stages": stages,
        "last_stage_improvement": float(residual),
        "error_terms_unit_c": {
            "truth_bisection_half": TRUTH_BISECT_HALF,
            "window_oracle_bisection_half": WINDOW_BISECT_HALF,
            "x_refinement_last_improvement": max(float(residual), 0.0),
            "mc_standard_error_of_argmax_mean": float(sd[best] / np.sqrt(N_REPLICATES)),
            "mc_single_evaluation_sd_at_argmax": float(sd[best]),
        },
        "frozen_v_grid_reproduced": reproduces,
        "nesting_union_gaps": gaps,
        "n_columns_total": len(allX),
        "grid_multi_window_columns": int(multi.sum()),
        "delayed_onset_columns": int(((alls > 0) & np.isfinite(alls)).sum()),
        "never_feasible_columns": int((~np.isfinite(alls)).sum()),
        "still_feasible_at_c1": int(np.isposinf(alld).sum()),
        "n_predicate_evals": n_evals,
        "n_power_flows": n_evals * TRUTH_SNAPSHOTS,
        "wall_s": wall,
    }
    out = _write("feeder_truth", summary, {
        "snapshots_per_predicate": TRUTH_SNAPSHOTS,
        "streams": "default_rng([991, index]); grid i; stage s point j: 100000*s+j; "
        "near-best r replicate k: 1000000+100*r+k",
    }, workers)  # fmt: skip
    np.savez_compressed(out / "truth_columns.npz", X=allX, start=alls, death=alld,
                        near=near, replicates=reps)  # fmt: skip
    log.info("theta_exists %.5f (sd %.5f, gaps %s) -> %s", theta, sd[best], gaps, out)


def _witness_check(task: tuple[int, list, float]) -> dict:
    """High-snapshot check at (x, L) plus replicate c_off of the witness column."""
    w, x, lower = task
    x = np.asarray(x, dtype=float)
    oracle, rng = _oracle(), np.random.default_rng([993, w])
    g = np.array(
        [oracle.snapshot_margins(x, lower, rng) for _ in range(WITNESS_SNAPSHOTS)]
    )
    deaths = [truth_column((2_000_000 + 100 * w + k, *x))[1]
              for k in range(N_WITNESS_REPLICATES)]  # fmt: skip
    return {**margin_check(g), "c_off_replicates": [float(d) for d in deaths]}


def load_s3_config(path: Path = S3_CONFIG) -> dict:
    """A study config (YAML), with its SHA-256 under ``sha256``.

    Args:
        path: Config path.

    Returns:
        The parsed mapping plus ``sha256`` of the file text.
    """
    text = Path(path).read_text()
    cfg = yaml.safe_load(text)
    cfg["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return cfg


def draw_designs(stream: list[int], n: int, x_dim: int = X_DIM) -> np.ndarray:
    """N i.i.d. uniform unit-box designs from a dedicated stream (use rule 1).

    Args:
        stream: Seed sequence of the design stream.
        n: Number of designs.
        x_dim: Design dimension.

    Returns:
        Array of shape (n, x_dim).
    """
    return np.random.default_rng(list(stream)).uniform(size=(int(n), int(x_dim)))


def ladder_start(rough_death: float, step: float, c_max: float = 1.0) -> float:
    """Ladder placement from a rough death: the first rung is start + step.

    A never-feasible rough pass (death = -inf) puts the first rung at 0; a
    still-feasible-at-c_max pass (death = +inf) puts it at c_max.
    """
    return float(np.clip(rough_death, -step, c_max - step))


def monotone_flags(
    means: np.ndarray, ses: np.ndarray, se_multiplier: float
) -> tuple[bool, float]:
    """S3-M1 desk-check decision for one design's loading-mean curve.

    Args:
        means: Loading-margin means along the coefficient grid.
        ses: Their standard errors.
        se_multiplier: Flag a decrease larger than this many pooled se.

    Returns:
        (flagged, smallest standardized increment between consecutive points).
    """
    inc = np.diff(np.asarray(means, dtype=float))
    pooled = np.sqrt(ses[1:] ** 2 + ses[:-1] ** 2)
    z = inc / np.where(pooled > 0, pooled, np.inf)
    return bool(np.any(z < -se_multiplier)), float(z.min())


def _loading_test(x: np.ndarray, c: float, rng, cols: dict, p: float) -> dict:
    """One conservative loading test at (x, c): S3-S1 step 3."""
    oracle = _oracle()
    g = np.array(
        [oracle.snapshot_margins(x, c, rng) for _ in range(int(cols["test_snapshots"]))]
    )
    lo, hi = (float(v) for v in cols["clip"])
    g2 = np.minimum(g[:, 1], hi)  # clip from above only; the floor is physical
    if g2.min() < lo:
        raise ValueError(f"loading margin {g2.min():.4f} below the registered floor")
    var = float(g2.var(ddof=1))
    t = empirical_bernstein_deviation(var, len(g2), hi - lo, p)
    return {
        "c": float(c), "mean": float(g2.mean()), "var": var, "t": t,
        "fired": bool(g2.mean() - t > 0.0),
        "voltage_mean": float(g[:, 0].mean()),
        "crash_fraction": float(np.mean(g[:, 0] == 3.0)),
    }  # fmt: skip


def score_column(task: tuple[int, list, dict, float, list]) -> dict:
    """Upper score of one column.

    Args:
        task: (column index i, design x, columns config, per-test p,
            test stream prefix). The rough pass uses ``truth_column`` with the
            worker's snapshot count (set by ``_set_truth_snapshots``) on
            stream index ``rough_index_base + i``; rung j draws its snapshots
            from ``default_rng([*test_stream, i, j])``.

    Returns:
        Row with the score, the rough window, every rung's statistics, and
        the wall time.
    """
    i, x, cols, p, stream = task
    x = np.asarray(x, dtype=float)
    t0 = time.perf_counter()
    start, death, n_evals, multi = truth_column((int(cols["rough_index_base"]) + i, *x))
    rungs: list[dict] = []

    def fires(c: float) -> bool:
        rng = np.random.default_rng([*stream, i, len(rungs) + 1])
        rungs.append(_loading_test(x, c, rng, cols, p))
        return rungs[-1]["fired"]

    score, n_tests = ladder_upper_score(
        ladder_start(death, float(cols["ladder_step"])),
        float(cols["ladder_step"]), int(cols["ladder_rungs"]), fires,
    )  # fmt: skip
    return {
        "index": i, "x": x.tolist(), "score": score, "n_tests": n_tests,
        "rough_start": float(start), "rough_death": float(death),
        "rough_predicate_evals": int(n_evals), "rough_multi_window": int(multi),
        "rungs": rungs, "wall_s": time.perf_counter() - t0,
    }  # fmt: skip


def cmd_draw_columns(cfg: dict) -> Path:
    """Draw and save the columns before any arm run (use rule 1)."""
    cols = cfg["columns"]
    X = draw_designs(cols["design_stream"], cols["n"])
    sha = hashlib.sha256(X.tobytes()).hexdigest()
    out = _write("feeder_designs", {
        "n": int(cols["n"]), "design_stream": list(cols["design_stream"]),
        "designs_sha256": sha, "config_sha256": cfg["sha256"],
    }, {"purpose": "independent designs behind U_eps, drawn before any search"}, 0)  # fmt: skip  # noqa: E501
    np.savez_compressed(out / "designs.npz", X=X)
    log.info("drew %d designs (sha256 %s) -> %s", len(X), sha[:12], out)
    return out


def summarize_scores(scores: np.ndarray, cols: dict) -> dict:
    """Proposition U-eps on the column scores at every eps."""
    s = np.asarray(scores, dtype=float)
    out = {"n": int(s.size), "n_inf": int(np.isinf(s).sum()), "by_eps": {}}
    for eps in cols["eps"]:
        u, k = order_statistic_upper(s, float(eps), float(cols["delta_order"]))
        out["by_eps"][str(eps)] = {"U_eps": u, "k": k, "finite": bool(np.isfinite(u))}
    return out


def cmd_score_columns(cfg: dict, designs_dir: Path, workers: int) -> None:
    """Score the columns and apply Proposition U-eps -> results/."""
    cols = cfg["columns"]
    meta = json.loads((designs_dir / "designs.json").read_text())
    X = np.load(designs_dir / "designs.npz")["X"]
    if hashlib.sha256(X.tobytes()).hexdigest() != meta["designs_sha256"]:
        raise ValueError("designs.npz does not match its recorded SHA-256")
    p = float(cols["delta_score"]) / (int(cols["n"]) * int(cols["ladder_rungs"]))
    tasks = [
        (i, x.tolist(), cols, p, list(cols["test_stream"])) for i, x in enumerate(X)
    ]
    mon = RssMonitor()
    mon.start()
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_set_truth_snapshots,
        initargs=(int(cols["rough_snapshots"]),),
    ) as pool:  # fmt: skip
        for n_done, row in enumerate(pool.map(score_column, tasks, chunksize=1), 1):
            rows.append(row)
            if n_done % 25 == 0 or n_done == len(tasks):
                log.info("%d/%d columns, %.0f s", n_done, len(tasks),
                         time.perf_counter() - t0)  # fmt: skip
    mon.stop()
    scores = np.array([r["score"] for r in rows])
    summary = {
        **summarize_scores(scores, cols),
        "per_test_p": p, "designs_sha256": meta["designs_sha256"],
        "designs_dir": str(designs_dir), "config_sha256": cfg["sha256"],
        "ladder_depth_histogram": {
            str(k): int(v) for k, v in zip(*np.unique(
                [r["n_tests"] for r in rows], return_counts=True), strict=True)},
        "rough_no_window": int(sum(not np.isfinite(r["rough_death"]) for r in rows)),
        "median_score_minus_rough_death": float(np.nanmedian([
            r["score"] - r["rough_death"] for r in rows
            if np.isfinite(r["score"]) and np.isfinite(r["rough_death"])])),
        "wall_s": time.perf_counter() - t0, "aggregate_peak_rss_gib": mon.peak_gib,
        "rss_missed_samples": mon.missed_samples,
    }  # fmt: skip
    out = _write("feeder_columns", summary, {
        "columns": cols,
        "streams": "rough: default_rng([991, rough_index_base + i]); rung j of "
        "column i: default_rng([998, 2, i, j])",
    }, workers)  # fmt: skip
    with (out / "rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    log.info("U_eps: %s -> %s", json.dumps(summary["by_eps"]), out)


def monotone_design(task: tuple[int, list, dict]) -> dict:
    """Loading-mean curve of one design along the coefficient grid."""
    i, x, mc = task
    x = np.asarray(x, dtype=float)
    oracle = _oracle()
    rng = np.random.default_rng([*mc["stream"], i])
    grid = np.linspace(0.0, 1.0, int(mc["c_grid_points"]))
    means, ses = [], []
    for c in grid:
        g2 = np.array([oracle.snapshot_margins(x, float(c), rng)[1]
                       for _ in range(int(mc["snapshots"]))])  # fmt: skip
        means.append(float(g2.mean()))
        ses.append(float(g2.std(ddof=1) / np.sqrt(len(g2))))
    flagged, min_z = monotone_flags(
        np.array(means), np.array(ses), float(mc["se_multiplier"])
    )
    slope = float(np.polyfit(grid, means, 1)[0])
    return {"index": i, "x": x.tolist(), "means": means, "ses": ses,
            "flagged": flagged, "min_standardized_increment": min_z,
            "fitted_slope_per_unit_c": slope}  # fmt: skip


def cmd_monotone_check(cfg: dict, workers: int) -> None:
    """Persistence check on 256 designs -> ``feeder_monotone_check``."""
    mc = cfg["monotone_check"]
    X = draw_designs(mc["stream"], mc["n_designs"])
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(monotone_design,
                             [(i, x.tolist(), mc) for i, x in enumerate(X)]))  # fmt: skip  # noqa: E501
    flagged = [r["index"] for r in rows if r["flagged"]]
    summary = {
        "n_designs": len(rows), "n_flagged": len(flagged), "flagged": flagged,
        "min_standardized_increment": min(
            r["min_standardized_increment"] for r in rows),
        "min_fitted_slope_per_unit_c": min(r["fitted_slope_per_unit_c"] for r in rows),
        "median_fitted_slope_per_unit_c": float(np.median(
            [r["fitted_slope_per_unit_c"] for r in rows])),
        "passed": not flagged, "config_sha256": cfg["sha256"],
        "wall_s": time.perf_counter() - t0, "rows": rows,
    }  # fmt: skip
    out = _write("feeder_monotone_check", summary, {
        "monotone_check": mc,
        "streams": "design i snapshots: default_rng([998, 8, i])",
    }, workers)  # fmt: skip
    log.info("persistence check: %d/%d flagged, min slope %.3f -> %s", len(flagged),
             len(rows), summary["min_fitted_slope_per_unit_c"], out)  # fmt: skip


# --- calibration ---


def cmd_calibrate() -> None:
    """Calibration at K = 16 with lam = R(16)^2 -> ``feeder_calibration/frozen.json``.

    The environment block (network, plant buses, stress range) is read from the
    shipped frozen file; everything the calibration fits is recomputed.
    """
    shipped = json.loads(FROZEN_PATH.read_text())
    payload = {"environment": dict(shipped["environment"])}
    pv_buses = [int(b) for b in payload["environment"]["pv_buses"]]

    # First obtain the fresh calibration scaler, then measure fresh R(16),
    # which is required before the norm fit can run.
    prelim_raw = HostingCapacityOracle(pv_buses=pv_buses)
    prelim_pred = _MeanPredicate(prelim_raw, 16)
    prelim_oracle = WindowMarginOracle(
        prelim_pred,
        prefix_only=True,
        prefix_no_window_report_grid_step=PREFIX_NO_WINDOW_REPORT_GRID_STEP,
    )
    prelim_rng = np.random.default_rng(CAL_SNAPSHOT_SEED)

    def prelim_window_channel(x_unit: np.ndarray, c_unit: float) -> np.ndarray:
        prelim_pred.rng = prelim_rng
        return prelim_oracle.raw_channels(x_unit, c_unit)

    X, c = calibration_points(X_DIM)
    Y_raw, crashed = evaluate_raw(prelim_window_channel, X, c, 1)
    _, prelim_scale = fit_scalers(Y_raw, crashed)
    r_pred = _MeanPredicate(prelim_raw, 16)
    r_oracle = WindowMarginOracle(
        r_pred,
        prefix_only=True,
        prefix_no_window_report_grid_step=PREFIX_NO_WINDOW_REPORT_GRID_STEP,
    )
    r_rng = np.random.default_rng([REPLICATE_SNAPSHOT_SEED, 16])

    def prelim_rep(x_unit: np.ndarray, c_unit: float) -> np.ndarray:
        r_pred.rng = r_rng
        return r_oracle.raw_channels(x_unit, c_unit)

    fresh_r16 = measure_noise_scale(prelim_rep, X_DIM, prelim_scale)
    log.info(
        "preliminary R(16)=%.6f, norm lam=%.7f",
        fresh_r16["R"],
        fresh_r16["lam"],
    )

    # Rerun the 512-point calibration fresh from its seed; only
    # the norm-bound fit consumes the fresh R(16).
    raw = HostingCapacityOracle(pv_buses=pv_buses)
    predicate = _MeanPredicate(raw, 16)
    oracle = WindowMarginOracle(
        predicate,
        prefix_only=True,
        prefix_no_window_report_grid_step=PREFIX_NO_WINDOW_REPORT_GRID_STEP,
    )
    cal_rng = np.random.default_rng(CAL_SNAPSHOT_SEED)

    def raw_window_channel(x_unit: np.ndarray, c_unit: float) -> np.ndarray:
        predicate.rng = cal_rng
        return oracle.raw_channels(x_unit, c_unit)

    t0 = time.perf_counter()
    cal = calibrate(
        "hosting-capacity-exposure-p6",  # name stored in the frozen file
        raw_window_channel,
        x_dim=X_DIM,
        m=1,
        R=fresh_r16["R"],
        clip=False,
    )
    cal_wall = time.perf_counter() - t0
    counters = oracle.counters()

    scale = np.asarray(cal.scaler_scale, dtype=float)
    by_knob: dict[str, dict] = {}
    for k in NOISE_KNOBS:
        pred_k = _MeanPredicate(raw, k)
        oracle_k = WindowMarginOracle(
            pred_k,
            prefix_only=True,
            prefix_no_window_report_grid_step=(PREFIX_NO_WINDOW_REPORT_GRID_STEP),
        )
        rep_rng = np.random.default_rng([REPLICATE_SNAPSHOT_SEED, k])

        def rep_oracle(
            x_unit: np.ndarray,
            c_unit: float,
            pred=pred_k,
            wmo=oracle_k,
            rng=rep_rng,
        ) -> np.ndarray:
            pred.rng = rng
            return wmo.raw_channels(x_unit, c_unit)

        t0 = time.perf_counter()
        record = measure_noise_scale(rep_oracle, X_DIM, scale)
        record["wall_time_s"] = time.perf_counter() - t0
        record["window_oracle_counters"] = oracle_k.counters()
        record["replicate_snapshot_seed"] = [REPLICATE_SNAPSHOT_SEED, k]
        by_knob[str(k)] = record
        log.info("R(K=%d)=%.6f (lam %.7f)", k, record["R"], record["lam"])
    if not np.isclose(by_knob["16"]["R"], fresh_r16["R"], rtol=0, atol=1e-12):
        raise RuntimeError("fresh preliminary and final R(16) disagree")

    payload["calibration"] = asdict(cal)
    payload["calibration_wall_time_s"] = cal_wall
    payload["calibration_window_oracle_counters"] = counters
    payload["calibration_snapshot_seed"] = CAL_SNAPSHOT_SEED
    payload["norm_fit_lam"] = fresh_r16["lam"]
    payload["noise_scale"] = {
        "recipe": "Section 3.2 step 5 on the P5 standardized g_hi channel",
        "measured_on": "P5 standardized prefix g_hi window channel",
        "by_knob": by_knob,
    }
    out = _write("feeder_calibration", {"wall_s": cal_wall}, {}, 1)
    (out / "frozen.json").write_text(json.dumps(payload, indent=2))
    log.info(
        "calibration: B=%s B_slice=%s R^2=%s R16=%.6f -> %s (compare with %s)",
        cal.B,
        cal.B_slice,
        cal.r2_holdout,
        fresh_r16["R"],
        out / "frozen.json",
        FROZEN_PATH,
    )
    if not cal.r2_ok:
        log.error("calibration R^2 %s below 0.5", cal.r2_holdout)


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
        cmd_monotone_check(load_s3_config(), args.workers)
    elif args.cmd == "draw-columns":
        cmd_draw_columns(load_s3_config())
    else:
        designs = args.designs or run_path("feeder_designs")
        cmd_score_columns(load_s3_config(), designs, args.workers)
