"""Covasim epidemic policy: calibration, reference limit, and thresholds.

Every constant comes from ``configs/covasim.yaml`` and
``configs/frozen/covasim.json``. Subcommands:

``calibrate``
    The 512-point calibration recipe on the K = 1 upper-window channel and
    the noise scale from 100 replicate pairs. Window calls run in parallel on
    per-call streams (``[9010, i]`` and ``[9011, 1, j]``) and are replayed
    into the recipe in its order. Writes ``result.json`` (the frozen file) to
    a new run directory.
``truth``
    Reference limit: a 7^3 policy grid, two clipped local refinements, and
    replicated bisections at the 8 best policies (Appendix C.5).
``monotone-check``
    Persistence check: mean peak demand increases with transmissibility on
    64 policies (Appendix C.6).
``draw-columns`` and ``score-columns``
    The 59 sampled policies behind U_0.05 and U_0.10.

Usage:
    uv run python scripts/covasim_setup.py calibrate --workers N
    uv run python scripts/covasim_setup.py truth --workers N
    uv run python scripts/covasim_setup.py monotone-check --workers N
    uv run python scripts/covasim_setup.py draw-columns
    uv run python scripts/covasim_setup.py score-columns --workers N [--designs DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from brink.memory import RssMonitor
from brink.quantile_bound import (
    empirical_bernstein_deviation,
    ladder_upper_score,
)
from brink.studies import covasim_policy as cp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feeder_setup as he  # noqa: E402  (shared column and config helpers)
from common import git_state, run_path  # noqa: E402

log = logging.getLogger("covasim_setup")
FROZEN = Path("configs/frozen/covasim.json")
S5_CONFIG = Path("configs/covasim.yaml")


def write_result(name: str, payload: dict, extra: dict, workers: int) -> Path:
    """Write ``results/<UTC>_<name>/`` with the payload and a manifest."""
    commit, dirty = git_state()
    out = Path("results") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{name}"
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(payload, indent=1))
    import covasim  # noqa: PLC0415  (version only)

    manifest = {
        "experiment": name, "git_commit": commit, "git_dirty": dirty,
        "workers": workers, "python": sys.version, "numpy": np.__version__,
        "covasim": covasim.__version__, "platform": platform.platform(), **extra,
    }  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def peak_rss_gib() -> float:
    """This process's peak RSS in GiB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1024**3


def _pool_map(fn, tasks: list, workers: int) -> tuple[list, float, float]:
    mon = RssMonitor()
    mon.start()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(fn, tasks, chunksize=1))
    mon.stop()
    return rows, time.perf_counter() - t0, mon.peak_gib


def window_call(task: tuple[list, list, float, int]) -> list[float]:
    """One window call at (x, c) with K runs per predicate on its own stream."""
    stream, x, c, k = task
    oracle = cp.window_oracle(k, [np.random.default_rng(list(stream))])
    return [*oracle.raw_channels(np.asarray(x), float(c)), oracle.n_raw_evals]


def replay(values: list[list[float]], points: list[tuple[list, float]]):
    """Oracle that returns precomputed window channels in the recipe's order."""
    it = iter(zip(values, points, strict=True))

    def oracle(x: np.ndarray, c: float) -> np.ndarray:
        v, (px, pc) = next(it)
        if not (np.allclose(x, px) and np.isclose(c, pc)):
            raise RuntimeError("calibration replay out of order")
        return np.asarray(v[:1])

    return oracle


def cmd_calibrate(workers: int) -> None:
    """Calibration recipe (upper window, lam = R^2) on the K = 1 channel.

    Window calls run in parallel on per-call streams (512 calibration calls on
    ``[9010, i]``; 200 replicate calls on ``[9011, 1, j]``) and are replayed
    into the frozen recipe functions in their own order.
    """
    from brink import calibration as cal_mod  # noqa: PLC0415

    X, c = cal_mod.calibration_points(cp.X_DIM)
    pts = [(x.tolist(), float(cc)) for x, cc in zip(X, c, strict=True)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cal_vals = list(pool.map(window_call, [([9010, i], x, cc, 1)
                                               for i, (x, cc) in enumerate(pts)]))  # fmt: skip  # noqa: E501
        rng = np.random.default_rng(cal_mod.REPLICATE_POINT_SEED)
        rx = rng.uniform(size=(cal_mod.REPLICATE_N_POINTS, cp.X_DIM))
        rc = rng.uniform(size=cal_mod.REPLICATE_N_POINTS)
        per = 2 * (cal_mod.REPLICATE_N_PAIRS // cal_mod.REPLICATE_N_POINTS)
        rep_pts = [(rx[i].tolist(), float(rc[i])) for i in range(len(rc))
                   for _ in range(per)]  # fmt: skip
        rep_vals = list(pool.map(window_call, [([9011, 1, j], x, cc, 1)
                                               for j, (x, cc) in enumerate(rep_pts)]))  # fmt: skip  # noqa: E501
    Y = np.array([v[:1] for v in cal_vals])
    _, scale = cal_mod.fit_scalers(Y, np.zeros(len(Y), dtype=bool))
    noise = cal_mod.measure_noise_scale(replay(rep_vals, rep_pts), cp.X_DIM, scale)
    cal = cal_mod.calibrate("covasim-policy-s5", replay(cal_vals, pts), x_dim=cp.X_DIM,
                            m=1, R=noise["R"], clip=False)  # fmt: skip
    raw = [v[1] for v in cal_vals + rep_vals]
    payload = {
        "environment": cp.environment_registration(),
        "calibration": asdict(cal),
        "noise_scale": {"recipe": "Section 3.2 step 5 on the K = 1 prefix channel",
                        "by_knob": {"1": noise}},
        "calibration_window_calls": len(raw), "calibration_raw_evals": int(sum(raw)),
        "calibration_no_window_fraction": float(np.mean(
            [cc - v[0] < 0.0 for v, (_, cc) in zip(cal_vals, pts, strict=True)])),
        "streams": {"calibration": "[9010, i]", "replicates": "[9011, 1, j]"},
        "wall_s": time.perf_counter() - t0,
    }  # fmt: skip
    out = write_result("covasim_calibration", payload, {}, workers)
    log.info("calibration: B=%s R=%.4f ls=%.4f R^2=%s ok=%s -> %s (compare with %s)",
             cal.B, noise["R"], cal.lengthscale, cal.r2_holdout, cal.r2_ok,
             out / "result.json", FROZEN)  # fmt: skip
    if not cal.r2_ok:
        log.error("calibration R^2 %s below 0.5", cal.r2_holdout)


def bisect_death(feas, steps: int) -> tuple[float, int]:
    """Last feasible unit c of a prefix predicate, with its eval count.

    Infeasible at c = 0 gives -inf (never feasible); feasible at c = 1 gives
    +inf; otherwise ``steps`` bisection steps and the bracket midpoint.

    Args:
        feas: Predicate on unit c.
        steps: Bisection steps.

    Returns:
        (death, number of predicate evaluations).
    """
    if not feas(0.0):
        return -np.inf, 1
    if feas(1.0):
        return np.inf, 2
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if feas(mid) else (lo, mid)
    return 0.5 * (lo + hi), 2 + steps


def truth_column(task: tuple[list, list, int, int]) -> tuple[float, int]:
    """Death of one column: (stream, x, runs per predicate, bisection steps)."""
    stream, x, k, steps = task
    rng = np.random.default_rng(list(stream))
    x = np.asarray(x, dtype=float)
    return bisect_death(lambda c: cp.mean_feasible(x, c, k, rng), steps)


def local_grid3(center: np.ndarray, half_width: float) -> np.ndarray:
    """3^d grid of half-width ``half_width`` around ``center``, clipped to [0, 1]."""
    ax = np.array([-half_width, 0.0, half_width])
    mesh = np.meshgrid(*([ax] * center.size), indexing="ij")
    pts = center + np.stack([m.ravel() for m in mesh], axis=1)
    return np.unique(np.clip(pts, 0.0, 1.0), axis=0)


def clipped_test(g: np.ndarray, clip: list[float], p: float) -> dict:
    """S5-S1 step 3: empirical-Bernstein infeasibility test on clipped margins.

    Args:
        g: Per-run margins.
        clip: Registered [lo, hi]; lo is the physical floor (peak >= 0).
        p: Per-test error probability.

    Returns:
        Mean, variance, deviation t, and ``fired`` (mean - t > 0).

    Raises:
        ValueError: When a margin lies below the floor.
    """
    lo, hi = (float(v) for v in clip)
    if np.min(g) < lo:
        raise ValueError(f"margin {np.min(g):.4f} below the registered floor {lo}")
    gc = np.minimum(g, hi)
    var = float(gc.var(ddof=1))
    t = empirical_bernstein_deviation(var, len(gc), hi - lo, p)
    return {"mean": float(gc.mean()), "var": var, "t": t,
            "fired": bool(gc.mean() - t > 0.0)}  # fmt: skip


def score_column(task: tuple[int, list, dict, float]) -> dict:
    """Upper score of one column.

    Args:
        task: (column index i, design x, columns config, per-test p). The
            rough pass bisects the ``rough_runs``-run predicate on
            ``[*rough_stream, i]``; rung j draws ``test_runs`` runs from
            ``[*test_stream, i, j]``.

    Returns:
        Row with the score, rough death, rung statistics, and wall time.
    """
    i, x, cols, p = task
    x = np.asarray(x, dtype=float)
    t0 = time.perf_counter()
    death, n_evals = truth_column(([*cols["rough_stream"], i], x.tolist(),
                                   int(cols["rough_runs"]), int(cols["bisect_steps"])))  # fmt: skip  # noqa: E501
    rungs: list[dict] = []

    def fires(c: float) -> bool:
        rng = np.random.default_rng([*cols["test_stream"], i, len(rungs) + 1])
        g = cp.margins(x, c, int(cols["test_runs"]), rng)
        rungs.append({"c": float(c), **clipped_test(g, cols["clip"], p)})
        return rungs[-1]["fired"]

    step = float(cols["ladder_step"])
    score, n_tests = ladder_upper_score(
        he.ladder_start(death, step), step, int(cols["ladder_rungs"]), fires
    )
    return {"index": i, "x": x.tolist(), "score": score, "n_tests": n_tests,
            "rough_death": float(death), "rough_predicate_evals": n_evals,
            "rungs": rungs, "wall_s": time.perf_counter() - t0}  # fmt: skip


def monotone_design(task: tuple[int, list, dict]) -> dict:
    """Mean-margin curve of one design along the coefficient grid."""
    i, x, mc = task
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng([*mc["stream"], i])
    grid = np.linspace(0.0, 1.0, int(mc["c_grid_points"]))
    gs = [cp.margins(x, float(c), int(mc["runs"]), rng) for c in grid]
    means = np.array([g.mean() for g in gs])
    ses = np.array([g.std(ddof=1) / np.sqrt(len(g)) for g in gs])
    flagged, min_z = he.monotone_flags(means, ses, float(mc["se_multiplier"]))
    return {"index": i, "x": x.tolist(), "means": means.tolist(),
            "ses": ses.tolist(), "flagged": flagged,
            "min_standardized_increment": min_z,
            "fitted_slope_per_unit_c": float(np.polyfit(grid, means, 1)[0])}  # fmt: skip  # noqa: E501


def _deaths(pool, X: np.ndarray, stream: list, base: int, k: int, steps: int):
    tasks = [([*stream, base + j], x.tolist(), k, steps) for j, x in enumerate(X)]
    out = list(pool.map(truth_column, tasks, chunksize=1))
    return np.array([d for d, _ in out]), int(sum(n for _, n in out)) * k


def cmd_truth(cfg: dict, workers: int) -> None:
    """Reference limit: grid, two clipped refinements, near-best replicates."""
    tr = cfg["truth"]
    ax = (np.arange(int(tr["grid_axis"])) + 0.5) / int(tr["grid_axis"])
    X = np.stack([m.ravel() for m in np.meshgrid(ax, ax, ax, indexing="ij")], 1)
    steps, stream = int(tr["bisect_steps"]), list(tr["stream"])
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        death, runs = _deaths(pool, X, stream, 0, int(tr["grid_runs"]), steps)
        stages = [{"stage": "grid", "n_columns": len(X), "runs": runs,
                   "max_death": float(death[np.isfinite(death)].max())}]  # fmt: skip
        log.info("truth grid: %s", stages[-1])
        allX, alld, total = X, death, runs
        for s, st in enumerate(tr["refine"], start=1):
            top = allX[np.argsort(np.where(np.isfinite(alld), alld, -1))[::-1]
                       [: int(st["top"])]]  # fmt: skip
            Xr = np.unique(np.concatenate(
                [local_grid3(t, float(st["half_width"])) for t in top]), axis=0)  # fmt: skip  # noqa: E501
            d_r, runs = _deaths(pool, Xr, stream, 100_000 * s, int(st["runs"]), steps)
            total += runs
            allX, alld = np.concatenate([allX, Xr]), np.concatenate([alld, d_r])
            stages.append({"stage": f"local w={st['half_width']:.5f}",
                           "n_columns": len(Xr), "runs": runs,
                           "max_death": float(d_r[np.isfinite(d_r)].max())})  # fmt: skip  # noqa: E501
            log.info("truth refine: %s", stages[-1])
        n_rep, near = int(tr["replicates"]), int(tr["near_best"])
        refined = np.arange(len(X), len(allX))
        order = refined[np.argsort(np.where(np.isfinite(alld[refined]),
                                            alld[refined], -1))[::-1]][:near]  # fmt: skip  # noqa: E501
        reps = []
        for r, ix in enumerate(order):
            Xk = np.repeat(allX[ix][None], n_rep, axis=0)
            d_k, runs = _deaths(pool, Xk, stream, 1_000_000 + 100 * r,
                                int(tr["replicate_runs"]), steps)  # fmt: skip
            total += runs
            reps.append(d_k)
    reps = np.array(reps)
    means, sd = reps.mean(axis=1), reps.std(axis=1, ddof=1)
    best = int(np.argmax(means))
    err = {"bisection_half": 0.5**steps / 2,
           "x_refinement_last_improvement": max(
               stages[-1]["max_death"] - stages[-2]["max_death"], 0.0),
           "mc_standard_error_of_argmax_mean": float(sd[best] / np.sqrt(n_rep))}  # fmt: skip  # noqa: E501
    grid_c = np.linspace(0.0, 1.0, 21)
    summary = {
        "theta_exists_unit": float(means[best]),
        "theta_exists_beta_multiplier": cp.beta_multiplier(float(means[best])),
        "theta_err": float(sum(err.values())), "error_terms_unit_c": err,
        "argmax_x_unit": allX[order[best]].tolist(),
        "argmax_policy": list(cp.to_policy(allX[order[best]])),
        "near_best": [{"x": allX[ix].tolist(), "policy": list(cp.to_policy(allX[ix])),
                       "single_stream_death": float(alld[ix]),
                       "replicate_mean": float(means[r]), "replicate_sd": float(sd[r]),
                       "replicates": reps[r].tolist()}
                      for r, ix in enumerate(order)],
        "stages": stages, "n_columns_total": len(allX), "n_runs": total,
        "grid_never_feasible": int(np.isneginf(death).sum()),
        "grid_feasible_at_c1": int(np.isposinf(death).sum()),
        "grid_volume_curve": {"c": grid_c.tolist(), "v": [
            float((death >= c).mean()) for c in grid_c]},
        "wall_s": time.perf_counter() - t0, "config_sha256": cfg["sha256"],
    }  # fmt: skip
    out = write_result("covasim_truth", summary, {"truth": tr}, workers)  # fmt: skip  # noqa: E501
    np.savez_compressed(out / "truth_columns.npz", X=allX, death=alld, replicates=reps)
    log.info("theta_exists %.4f +/- %.4f at %s -> %s", summary["theta_exists_unit"],
             summary["theta_err"], summary["argmax_policy"], out)  # fmt: skip


def cmd_monotone_check(cfg: dict, workers: int) -> None:
    """Assumption S5-M1 desk check."""
    mc = cfg["monotone_check"]
    X = he.draw_designs(mc["stream"], mc["n_designs"], cp.X_DIM)
    rows, wall, rss = _pool_map(
        monotone_design, [(i, x.tolist(), mc) for i, x in enumerate(X)], workers
    )
    flagged = [r["index"] for r in rows if r["flagged"]]
    summary = {
        "n_designs": len(rows), "n_flagged": len(flagged), "flagged": flagged,
        "min_standardized_increment": min(r["min_standardized_increment"] for r in rows),  # noqa: E501
        "min_fitted_slope_per_unit_c": min(r["fitted_slope_per_unit_c"] for r in rows),
        "median_fitted_slope_per_unit_c": float(np.median(
            [r["fitted_slope_per_unit_c"] for r in rows])),
        "passed": not flagged, "config_sha256": cfg["sha256"], "wall_s": wall,
        "aggregate_peak_rss_gib": rss, "rows": rows,
    }  # fmt: skip
    out = write_result("covasim_monotone_check", summary, {"monotone_check": mc},
                       workers)  # fmt: skip  # noqa: E501
    log.info("S5-M1: %d/%d flagged, median slope %.3f -> %s", len(flagged), len(rows),
             summary["median_fitted_slope_per_unit_c"], out)  # fmt: skip


def cmd_draw_columns(cfg: dict) -> Path:
    """Draw and hash the U_eps columns before any arm run."""
    cols = cfg["columns"]
    X = he.draw_designs(cols["design_stream"], cols["n"], cp.X_DIM)
    sha = hashlib.sha256(X.tobytes()).hexdigest()
    out = write_result("covasim_designs", {
        "n": int(cols["n"]), "design_stream": list(cols["design_stream"]),
        "designs_sha256": sha, "config_sha256": cfg["sha256"],
    }, {"purpose": "registered U_eps columns, drawn before any arm run"}, 0)  # fmt: skip  # noqa: E501
    np.savez_compressed(out / "designs.npz", X=X)
    log.info("drew %d designs (sha256 %s) -> %s", len(X), sha[:12], out)
    return out


def column_p(cols: dict) -> float:
    """Per-test error probability: delta_2 over all tests."""
    return float(cols["delta_score"]) / (int(cols["n"]) * int(cols["ladder_rungs"]))


def cmd_score_columns(cfg: dict, designs_dir: Path, workers: int) -> None:
    """Score the columns and apply Proposition U-eps."""
    cols = cfg["columns"]
    meta = json.loads((designs_dir / "result.json").read_text())
    X = np.load(designs_dir / "designs.npz")["X"]
    if hashlib.sha256(X.tobytes()).hexdigest() != meta["designs_sha256"]:
        raise ValueError("designs.npz does not match its recorded SHA-256")
    p = column_p(cols)
    rows, wall, rss = _pool_map(
        score_column, [(i, x.tolist(), cols, p) for i, x in enumerate(X)], workers
    )
    scores = np.array([r["score"] for r in rows])
    summary = {
        **he.summarize_scores(scores, cols), "per_test_p": p,
        "designs_sha256": meta["designs_sha256"], "config_sha256": cfg["sha256"],
        "ladder_depths": [r["n_tests"] for r in rows],
        "rough_no_window": int(sum(np.isneginf(r["rough_death"]) for r in rows)),
        "n_runs": int(sum(r["rough_predicate_evals"] * int(cols["rough_runs"])
                          + r["n_tests"] * int(cols["test_runs"]) for r in rows)),
        "wall_s": wall, "aggregate_peak_rss_gib": rss, "rows": rows,
    }  # fmt: skip
    out = write_result("covasim_columns", summary, {"columns": cols}, workers)  # fmt: skip  # noqa: E501
    log.info("U_eps: %s -> %s", json.dumps(summary["by_eps"]), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "truth", "monotone-check", "score-columns"):
        sp = sub.add_parser(name)
        sp.add_argument("--workers", type=int, required=True)
    sub.choices["score-columns"].add_argument("--designs", type=Path, default=None)
    sub.add_parser("draw-columns")
    args = ap.parse_args()
    cfg = he.load_s3_config(S5_CONFIG)
    if args.cmd == "calibrate":
        cmd_calibrate(args.workers)
    elif args.cmd == "truth":
        cmd_truth(cfg, args.workers)
    elif args.cmd == "monotone-check":
        cmd_monotone_check(cfg, args.workers)
    elif args.cmd == "draw-columns":
        cmd_draw_columns(cfg)
    else:
        designs = args.designs or run_path("covasim_designs")
        cmd_score_columns(cfg, designs, args.workers)
