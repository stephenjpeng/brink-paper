"""Hopper (MuJoCo, pretrained SAC policy): policy, calibration, reference, and checks.

Every constant comes from ``configs/hopper.yaml`` and
``configs/frozen/hopper.json``. Rollouts are chaotic, so results depend on the
processor architecture: the reported runs used x86-64 (Appendix C.5).
Subcommands:

``download-policy``
    Fetches ``sac-Hopper-v3.zip`` from the Hugging Face model repository
    ``sb3/sac-Hopper-v3`` at the pinned revision into ``data/raw/`` and
    checks its SHA-256.
``calibrate``
    The 512-point calibration recipe on the upper-window channel; writes
    ``frozen.json`` to a new run directory.
``truth``
    The best-design search behind the 12.5 kg reference: a 15^3 grid, a
    verified top edge per design, and three clipped local refinements.
``desk-check``
    The finer scan of 256 designs that tests the margin of the planned
    limits (Appendix C.6; 4 of 256 designs violate it).
``draw-columns`` and ``score-columns``
    The 628 sampled designs and their payload-grid scans (Appendix C.6).
``witness-diagnostic``
    Why returned designs fail the rollout (Appendix D.5); reads a sweep.

Usage:
    uv run python scripts/hopper_setup.py download-policy
    uv run python scripts/hopper_setup.py calibrate --workers N
    uv run python scripts/hopper_setup.py truth --workers N
    uv run python scripts/hopper_setup.py desk-check --workers N
    uv run python scripts/hopper_setup.py draw-columns
    uv run python scripts/hopper_setup.py score-columns --workers N [--designs DIR]
    uv run python scripts/hopper_setup.py witness-diagnostic --workers N [--run DIR]
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
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from brink.calibration import calibrate, calibration_points, save_calibration
from brink.quantile_bound import order_statistic_upper
from brink.studies import hopper as mh
from brink.window_margins import WindowMarginOracle, window_channels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import git_state, local_grid, run_path  # noqa: E402

log = logging.getLogger("hopper_setup")

FROZEN = Path("configs/frozen/hopper.json")
M_CHANNELS = 1  # prefix death channel

REPORT_GRID = np.linspace(0.0, 1.0, 201)  # unit c; step 0.005 = 0.1 kg
TRUTH_COARSE = 33  # truth column scan, step 1/32 = 0.625 kg
TRUTH_BISECT = 10

_WIN: WindowMarginOracle | None = None
_RAW: mh.HopperOracle | None = None


def top_edge(
    feasible, n_coarse: int = TRUTH_COARSE, n_bisect: int = TRUTH_BISECT
) -> tuple[float, float, int, int]:
    """Verified top feasible edge of one design column.

    Scans ``n_coarse`` equally spaced unit coefficients, then bisects between
    the last feasible scan point and the next one. The lower end is always a
    coefficient observed feasible, so it is a verified lower bound on the
    column's death; islands narrower than the scan step above it are missed.

    Args:
        feasible: Predicate on the unit coefficient.
        n_coarse: Scan points on [0, 1], endpoints included.
        n_bisect: Bisection steps on the top edge.

    Returns:
        (last feasible c, first infeasible c above it, rollouts, feasible runs
        in the scan); (-inf, -inf, n_coarse, 0) when no scan point is feasible
        and (1, inf, n_coarse, runs) when c = 1 is feasible.
    """
    coarse = np.linspace(0.0, 1.0, n_coarse)
    f = np.array([bool(feasible(float(c))) for c in coarse])
    runs = int(f[0]) + int(np.sum(f[1:] & ~f[:-1]))
    if not f.any():
        return -np.inf, -np.inf, n_coarse, 0
    k = int(np.flatnonzero(f)[-1])
    if k == n_coarse - 1:
        return 1.0, np.inf, n_coarse, runs
    lo, hi = float(coarse[k]), float(coarse[k + 1])
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi, n_coarse + n_bisect, runs


def grid_score(feas: np.ndarray, c_grid: np.ndarray, margin: float) -> float:
    """U_eps upper score of one column.

    Args:
        feas: Feasibility at every grid coefficient.
        c_grid: The scanned unit coefficients, increasing.
        margin: Margin added above the last feasible grid point.

    Returns:
        Last feasible grid coefficient plus ``margin``; -inf when no grid
        point is feasible.
    """
    idx = np.flatnonzero(feas)
    return float(c_grid[idx[-1]] + margin) if idx.size else -np.inf


OFFSET_GRID = 0.5 * (REPORT_GRID[:-1] + REPORT_GRID[1:])  # halfway points


def desk_flag(
    grid_feas: np.ndarray,
    offset_feas: np.ndarray,
    fine_feas: np.ndarray,
    fine_grid: np.ndarray,
    margin: float,
) -> tuple[bool, float]:
    """S4-M1 desk-check verdict for one design.

    Args:
        grid_feas: Feasibility on the report grid.
        offset_feas: Feasibility at the halfway points ``OFFSET_GRID``.
        fine_feas: Feasibility on the fine grid near the grid top (may be empty).
        fine_grid: The fine unit coefficients.
        margin: Score margin Delta.

    Returns:
        (flagged, excess): flagged when an offset or fine point is feasible
        above the S4-S1 score; excess is the largest feasible offset or fine
        coefficient minus the last feasible grid point (-inf when none, +inf
        when only off-grid points are feasible).
    """
    s = grid_score(grid_feas, REPORT_GRID, margin)
    extra = np.concatenate([OFFSET_GRID[offset_feas], fine_grid[fine_feas]])
    if not extra.size:
        return False, -np.inf
    top = grid_score(grid_feas, REPORT_GRID, 0.0)
    excess = float(extra.max() - top) if np.isfinite(top) else np.inf
    return bool(extra.max() > s), excess


def column_summary(scores: np.ndarray, eps_list, delta: float) -> dict:
    """Proposition U-eps at each eps from S4-S1 scores.

    Args:
        scores: Column scores (-inf for columns with no feasible grid point).
        eps_list: Eps values.
        delta: Order-statistic failure probability per statement.

    Returns:
        Dict with U and k per eps and the score counts.
    """
    out = {
        "n": int(scores.size),
        "n_no_feasible_grid_point": int(np.isneginf(scores).sum()),
        "max_score": float(scores.max()),
    }
    for eps in eps_list:
        U, k = order_statistic_upper(scores, float(eps), float(delta))
        out[f"eps_{eps}"] = {"U": float(U), "k": int(k)}
    return out


def _raw() -> mh.HopperOracle:
    global _RAW
    if _RAW is None:
        _RAW = mh.HopperOracle()
    return _RAW


def _scan_task(args: tuple) -> np.ndarray:
    """Worker: feasibility of design x at every unit c of linspace(lo, hi, n)."""
    x, (lo, hi, n) = np.asarray(args[:3], dtype=float), args[3:]
    return np.array([_raw().feasible(x, c) for c in np.linspace(lo, hi, int(n))])


def _truth_task(args: tuple) -> tuple:
    """Worker: verified top edge of the column at unit design x."""
    x = np.asarray(args, dtype=float)
    return top_edge(lambda c: _raw().feasible(x, c))


def _window(x: tuple) -> tuple:
    """Worker: window (c_on, c_off, raw rollouts, multi) of one x."""
    global _WIN
    if _WIN is None:
        _WIN = mh.make_window_oracle()
    w = _WIN.window(np.asarray(x, dtype=float))
    return w.c_on, w.c_off, w.n_raw_evals, w.multi_window


def _manifest(name: str, workers: int, **extra) -> dict:
    commit, dirty = git_state()
    return {
        "experiment": name,
        "git_commit": commit,
        "git_dirty": dirty,
        "workers": workers,
        "policy_zip_sha256": mh.POLICY_ZIP_SHA256,
        "python": sys.version,
        "numpy": np.__version__,
        "mujoco": mh.mujoco.__version__,
        "platform": platform.platform(),
        **extra,
    }


def cmd_calibrate(workers: int) -> None:
    """Window-channel calibration -> ``results/<UTC>_hopper_calibration/``."""
    manifest = _manifest("hopper_calibration", workers)  # before any output is written
    X, c = calibration_points(mh.X_DIM)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        win = np.array(list(pool.map(_window, map(tuple, X), chunksize=4)))
    lookup = {tuple(x): w for x, w in zip(X, win, strict=True)}

    def raw_oracle(x: np.ndarray, cc: float) -> np.ndarray:
        c_on, c_off = lookup[tuple(x)][:2]
        return window_channels(c_on, c_off, cc, prefix_only=True)

    cal = calibrate("mujoco_hopper_s4", raw_oracle, mh.X_DIM, M_CHANNELS, clip=False)
    wall = time.perf_counter() - t0
    out = Path("results") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_hopper_calibration"
    out.mkdir(parents=True)
    save_calibration(cal, out / "frozen.json")
    no_window = win[:, 0] > 1.0  # NO_WINDOW_C_ON
    summary = {
        "n_points": len(X),
        "n_rollouts": int(win[:, 2].sum()),
        "rollouts_per_window": float(win[:, 2].mean()),
        "n_no_window": int(no_window.sum()),
        "n_delayed_onset": int(((win[:, 0] > 0) & ~no_window).sum()),
        "n_multi_window": int(win[:, 3].sum()),
        "max_c_off": float(win[:, 1].max()),
        "r2_ok": cal.r2_ok,
        "wall_s": wall,
    }
    np.savez_compressed(out / "windows.npz", X=X, c=c, windows=win)
    (out / "calibration.json").write_text(json.dumps(summary, indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("calibration %s -> %s (compare with %s)", summary, out, FROZEN)


S4_CONFIG = Path("configs/hopper.yaml")


def load_s4_config() -> dict:
    """Hopper constants (``configs/hopper.yaml``)."""
    return yaml.safe_load(S4_CONFIG.read_text())


def _config_sha256() -> str:
    return hashlib.sha256(S4_CONFIG.read_bytes()).hexdigest()


def _new_dir(name: str) -> Path:
    out = Path("results") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{name}"
    out.mkdir(parents=True)
    return out


def _desk_task(args: tuple) -> tuple:
    """Worker: grid, offset, and fine scans of one desk-check design."""
    x, margin, step, below, above = np.asarray(args[:3]), *args[3:]
    grid = _scan_task((*x, 0.0, 1.0, REPORT_GRID.size))
    off = np.array([_raw().feasible(x, c) for c in OFFSET_GRID])
    top = grid_score(grid, REPORT_GRID, 0.0)
    if np.isfinite(top):
        lo, hi = max(0.0, top - below), min(1.0, top + margin + above)
        fine_grid = np.arange(round(lo / step), round(hi / step) + 1) * step
        fine = np.array([_raw().feasible(x, c) for c in fine_grid])
    else:
        fine_grid, fine = np.zeros(0), np.zeros(0, bool)
    return grid, off, fine_grid, fine


def cmd_desk_check(workers: int) -> None:
    """Assumption S4-M1 desk check -> results/."""
    cfg = load_s4_config()
    dc, margin = cfg["desk_check"], float(cfg["columns"]["margin"])
    manifest = _manifest("hopper_desk_check", workers, config_sha256=_config_sha256())
    X = np.random.default_rng(dc["stream"]).uniform(size=(int(dc["n_designs"]), 3))
    args = [
        (*x, margin, dc["fine_step"], dc["fine_below"], dc["fine_above_margin"])
        for x in X
    ]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(_desk_task, args, chunksize=1))
    wall = time.perf_counter() - t0
    verdicts = [desk_flag(g, o, f, fg, margin) for g, o, fg, f in res]
    flags = [i for i, (fl, _) in enumerate(verdicts) if fl]
    excess = np.array([e for _, e in verdicts])
    summary = {
        "n_designs": len(X),
        "n_flagged": len(flags),
        "flagged_indices": flags,
        "flag_rate_upper_95": float(
            stats.binomtest(len(flags), len(X)).proportion_ci(0.95).high
        ),
        "n_grid_feasible": int(sum(g.any() for g, *_ in res)),
        "max_finite_excess": float(excess[np.isfinite(excess)].max(initial=-np.inf)),
        "n_rollouts": int(sum(g.size + o.size + f.size for g, o, _, f in res)),
        "wall_s": wall,
    }
    out = _new_dir("hopper_desk_check")
    np.savez_compressed(
        out / "desk.npz",
        X=X,
        grid=np.array([g for g, *_ in res]),
        offset=np.array([o for _, o, *_ in res]),
        excess=excess,
    )
    (out / "desk_check.json").write_text(json.dumps(summary, indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("S4-M1 desk check %s -> %s", summary, out)


def refine_points(top: np.ndarray, half_width: float) -> np.ndarray:
    """Unique rows of the clipped 5^3 local grids around each top design.

    Args:
        top: Designs to refine around, shape (k, 3).
        half_width: Local grid half-width (unit box).

    Returns:
        Distinct designs, shape (n, 3), inside the unit box.
    """
    return np.unique(np.concatenate([local_grid(t, half_width) for t in top]), axis=0)


def cmd_truth(workers: int) -> None:
    """Verified-lower-bound truth search -> results/."""
    tc = load_s4_config()["truth"]
    manifest = _manifest("hopper_truth", workers, config_sha256=_config_sha256())
    ax = np.linspace(0.0, 1.0, int(tc["grid_axis"]))
    X = np.stack([m.ravel() for m in np.meshgrid(ax, ax, ax, indexing="ij")], 1)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:

        def columns(P: np.ndarray) -> np.ndarray:
            return np.array(list(pool.map(_truth_task, map(tuple, P), chunksize=4)))

        out_cols = columns(X)
        allX, allc = X, out_cols
        stages = [{"stage": "grid", "n": len(X), "max": float(out_cols[:, 0].max())}]
        for w in tc["refine_half_widths"]:
            top = allX[np.argsort(allc[:, 0])[::-1][: int(tc["refine_top_k"])]]
            P = refine_points(top, float(w))
            cols = columns(P)
            allX, allc = np.concatenate([allX, P]), np.concatenate([allc, cols])
            stages.append({"stage": f"w={w:.5f}", "n": len(P), "max": cols[:, 0].max()})
            log.info("refine w=%.5f: %d columns, max %.6f", w, len(P), cols[:, 0].max())
    best = int(np.argmax(allc[:, 0]))
    summary = {
        "t_search_unit": float(allc[best, 0]),
        "t_search_kg": 20.0 * float(allc[best, 0]),
        "argmax_x_unit": allX[best].tolist(),
        "argmax_first_infeasible_above": float(allc[best, 1]),
        "stages": [
            {k: (float(v) if k == "max" else v) for k, v in s.items()} for s in stages
        ],
        "grid_never_feasible": int(np.isneginf(out_cols[:, 0]).sum()),
        "grid_multi_run_columns": int((out_cols[:, 3] > 1).sum()),
        "n_rollouts": int(allc[:, 2].sum()),
        "wall_s": time.perf_counter() - t0,
    }
    out = _new_dir("hopper_truth")
    np.savez_compressed(out / "truth_columns.npz", X=allX, columns=allc)
    (out / "truth.json").write_text(json.dumps(summary, indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("T_search %.4f at %s -> %s", summary["t_search_unit"], allX[best], out)


def draw_designs(cfg: dict) -> np.ndarray:
    """The N U_eps columns, i.i.d. uniform on the unit box."""
    cc = cfg["columns"]
    return np.random.default_rng(cc["design_stream"]).uniform(size=(int(cc["n"]), 3))


def cmd_draw_columns() -> None:
    """Draw, save, and hash the columns (before any arm run)."""
    cfg = load_s4_config()
    manifest = _manifest("hopper_designs", 0, config_sha256=_config_sha256())
    X = draw_designs(cfg)
    sha = hashlib.sha256(X.tobytes()).hexdigest()
    out = _new_dir("hopper_designs")
    np.savez_compressed(out / "designs.npz", X=X)
    (out / "designs.json").write_text(json.dumps({"n": len(X), "sha256": sha}))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("%d designs, sha256 %s -> %s", len(X), sha, out)


def score_columns(pool, X: np.ndarray, margin: float) -> tuple:
    """S4-S1 grid scans and scores of the columns X."""
    args = [(*x, 0.0, 1.0, REPORT_GRID.size) for x in X]
    feas = np.array(list(pool.map(_scan_task, args, chunksize=1)))
    return feas, np.array([grid_score(f, REPORT_GRID, margin) for f in feas])


def volume_curve(feas: np.ndarray) -> dict:
    """Grid volume estimate with Clopper-Pearson 95 percent intervals."""
    k, n = feas.sum(0), feas.shape[0]
    ci = [stats.binomtest(int(kk), n).proportion_ci(0.95) for kk in k]
    return {
        "c_unit": REPORT_GRID.tolist(),
        "v_hat": (k / n).tolist(),
        "ci_low": [c.low for c in ci],
        "ci_high": [c.high for c in ci],
    }


def cmd_score_columns(designs: Path, workers: int) -> None:
    """Construction S4-S1 on the columns, then Proposition U-eps."""
    cfg = load_s4_config()
    cc = cfg["columns"]
    manifest = _manifest("hopper_columns", workers, config_sha256=_config_sha256())
    X = np.load(designs / "designs.npz")["X"]
    sha = hashlib.sha256(X.tobytes()).hexdigest()
    if sha != json.loads((designs / "designs.json").read_text())["sha256"]:
        raise ValueError("column designs do not match their recorded SHA-256")
    if not np.array_equal(X, draw_designs(cfg)):
        raise ValueError("column designs are not the registered draw")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        feas, scores = score_columns(pool, X, float(cc["margin"]))
    summary = column_summary(scores, cc["eps"], float(cc["delta_order"]))
    summary |= {"designs_sha256": sha, "wall_s": time.perf_counter() - t0}
    out = _new_dir("hopper_columns")
    np.savez_compressed(out / "columns.npz", X=X, feasible=feas, scores=scores)
    (out / "columns.json").write_text(json.dumps(summary, indent=1))
    (out / "volume_curve.json").write_text(json.dumps(volume_curve(feas)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("columns %s -> %s", summary, out)


def witness_class(lower: float, c_off_window: float) -> str:
    """Post-hoc class of a band witness (x, L) by the window death of x.

    Args:
        lower: The witness's reported unit coefficient L.
        c_off_window: The window oracle's death c_off(x).

    Returns:
        "gap" when the window's prefix channel itself claims feasibility through
        L (so a failure at L is a gap in a ragged column), else "band-overreach"
        (the band certified more than the oracle's own window reported).
    """
    return "gap" if c_off_window >= lower else "band-overreach"


def _diag_task(args: tuple) -> dict:
    """Worker: rollout check, report-grid scan, and window death of one witness."""
    global _WIN
    x, lower = np.asarray(args[:3], dtype=float), float(args[3])
    if _WIN is None:
        _WIN = mh.make_window_oracle()
    grid = np.array([_raw().feasible(x, c) for c in REPORT_GRID])
    c_off = float(_WIN.window(x).c_off)
    return {
        "check_feasible": bool(_raw().feasible(x, lower)),
        "c_off_window": c_off,
        "grid_top": grid_score(grid, REPORT_GRID, 0.0),
        "grid_feasible_above_L": bool(grid[lower < REPORT_GRID].any()),
        "grid_runs": int(grid[0]) + int(np.sum(grid[1:] & ~grid[:-1])),
        "class": witness_class(lower, c_off),
    }


def cmd_witness_diagnostic(run: Path, workers: int) -> None:
    """Post-hoc, no gate weight: why returned designs fail (Appendix D.5).

    Every GP-arm witness that failed its rollout check, plus as many passing
    ones drawn from ``default_rng([996, 10])``, gets a rollout at (x, L), a
    report-grid scan, and the window death c_off(x).
    """
    manifest = _manifest("hopper_witness_diagnostic", workers, sweep=str(run))
    rows = [json.loads(ln) for ln in (run / "rows.jsonl").read_text().splitlines()]
    gp = [r for r in rows if r["view"] == "window" and r["has_witness"]]
    fail = [r for r in gp if not r["check_feasible"]]
    ok = [r for r in gp if r["check_feasible"]]
    pick = np.random.default_rng([996, 10]).choice(len(ok), len(fail), replace=False)
    chosen = [("fail", r) for r in fail] + [("pass", ok[i]) for i in sorted(pick)]
    args = [(*r["witness_x"], r["final_lower"]) for _, r in chosen]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(_diag_task, args, chunksize=1))
    detail = [
        {"group": g, "task_index": r["task_index"], "label": r["label"],
         "budget": r["budget"], "seed": r["seed"], "L": r["final_lower"], **o}
        for (g, r), o in zip(chosen, out, strict=True)
    ]  # fmt: skip
    summary = {"wall_s": time.perf_counter() - t0}
    for g in ("fail", "pass"):
        d = [x for x in detail if x["group"] == g]
        summary[g] = {
            "n": len(d),
            "recheck_feasible": sum(x["check_feasible"] for x in d),
            "class_gap": sum(x["class"] == "gap" for x in d),
            "class_band_overreach": sum(x["class"] == "band-overreach" for x in d),
            "grid_feasible_above_L": sum(x["grid_feasible_above_L"] for x in d),
            "median_grid_runs": float(np.median([x["grid_runs"] for x in d])),
        }
    out_dir = _new_dir("hopper_witness_diagnostic")
    (out_dir / "diagnostic.json").write_text(
        json.dumps({"summary": summary, "detail": detail}, indent=1)
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("witness diagnostic %s -> %s", summary, out_dir)


POLICY_URL = (
    "https://huggingface.co/sb3/sac-Hopper-v3/resolve/"
    "8346bf5c56f201f0e38ad9acb06e093aad582a44/sac-Hopper-v3.zip"
)


def cmd_download_policy() -> None:
    """Fetch the pretrained policy at the pinned revision and check its SHA-256."""
    import urllib.request  # noqa: PLC0415

    dest = Path(mh.POLICY_ZIP)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", POLICY_URL)
        urllib.request.urlretrieve(POLICY_URL, dest)  # noqa: S310
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    if sha != mh.POLICY_ZIP_SHA256:
        raise SystemExit(f"{dest}: SHA-256 {sha} does not match the pinned file")
    log.info("policy ok: %s (sha256 %s)", dest, sha)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "desk-check", "truth", "score-columns"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=4)
    sub.add_parser("download-policy")
    sub.add_parser("draw-columns")
    wd = sub.add_parser("witness-diagnostic")
    wd.add_argument("--run", type=Path, default=None)
    wd.add_argument("--workers", type=int, default=4)
    sub.choices["score-columns"].add_argument("--designs", type=Path, default=None)
    args = ap.parse_args()
    if args.cmd == "download-policy":
        cmd_download_policy()
    elif args.cmd == "draw-columns":
        cmd_draw_columns()
    elif args.cmd == "score-columns":
        cmd_score_columns(args.designs or run_path("hopper_designs"), args.workers)
    elif args.cmd == "witness-diagnostic":
        cmd_witness_diagnostic(args.run or run_path("hopper_sweep"), args.workers)
    else:
        cmds = {"calibrate": cmd_calibrate, "desk-check": cmd_desk_check,
                "truth": cmd_truth}  # fmt: skip
        cmds[args.cmd](args.workers)
