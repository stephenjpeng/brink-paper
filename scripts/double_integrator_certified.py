"""Double integrator: intervals recomputed with certified constants (Appendix B).

Regenerates the 50 sampling paths of BRINK's variance-seeking rule at budget
1,000, computes a norm bound B and mismatch allowance eta that hold by
construction for the closed-form constraint, and recomputes the intervals on
those paths with the widened bounds, so that Theorem 1 covers them.

Usage (one results directory for every step):
    uv run python scripts/double_integrator_certified.py paths --workers N
    uv run python scripts/double_integrator_certified.py constants [DIR] --workers N
    uv run python scripts/double_integrator_certified.py select [DIR] --workers N
    uv run python scripts/double_integrator_certified.py replay [DIR] --workers N
    uv run python scripts/double_integrator_certified.py analyze [DIR]

``DIR`` defaults to the latest ``double_integrator_certified`` run. ``analyze``
compares the regenerated paths with the stored sweep rows of the latest
``double_integrator_sweep`` run (``configs/runs.yaml``).
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import run_path  # noqa: E402
from double_integrator_sweep import FROZEN_PATH, build_cfg  # noqa: E402

from brink.adapter import WindowExistenceView  # noqa: E402
from brink.arms import run_existence_arm  # noqa: E402
from brink.bracket import (  # noqa: E402
    MODE_AFFINE,
    compute_bracket,
    fit_joint_bands,
    midpoint_cover,
)
from brink.studies.double_integrator import (  # noqa: E402
    THETA_EXISTS_UNIT,
    certified_mismatch_constants,
    is_viable,
    load_double_integrator,
)

log = logging.getLogger("double_integrator_certified")
CONFIG_PATH = Path("configs/double_integrator_certified.yaml")


def _cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _report_seeds(cfg: dict) -> list[int]:
    return list(range(cfg["report_seeds"]["start"], cfg["report_seeds"]["stop"]))


def _pool(workers: int):
    return mp.get_context("spawn").Pool(workers)


def _manifest(out_dir: Path, step: str, workers: int) -> None:
    """Write ``manifest_<step>.json`` with code, config, and host provenance."""
    git = ["git", "-C", str(Path(__file__).resolve().parent.parent)]
    commit = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            [*git, "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    sha = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (CONFIG_PATH, FROZEN_PATH)
    }
    out_dir.joinpath(f"manifest_{step}.json").write_text(
        json.dumps(
            {
                "experiment": "double_integrator_certified",
                "step": step,
                "git_commit": commit,
                "git_dirty": dirty,
                "input_sha256": sha,
                "workers": workers,
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


# --- paths -----------------------------------------------------------------


def _path_task(seed: int, out_dir: str) -> dict:
    """Regenerate one budget-1000 greedy path and save it."""
    cfg = _cfg()
    env = load_double_integrator(FROZEN_PATH)
    path: dict = {}
    t0 = time.perf_counter()
    res = run_existence_arm(
        WindowExistenceView(env),
        cfg["arm"],
        build_cfg(cfg["budget"]),
        env.B,
        seed=seed,
        path_out=path,
    )
    np.savez(Path(out_dir) / "paths" / f"greedy_{seed}.npz", **path)
    return {
        "seed": seed,
        "wall_s": round(time.perf_counter() - t0, 3),
        "final_lower": res.final_lower,
        "final_upper": res.final_upper,
        "reports": [asdict(r) for r in res.reports],
    }


def cmd_paths(workers: int, resume: str | None) -> None:
    """Regenerate the select and report paths."""
    cfg = _cfg()
    if resume:
        out_dir = Path(resume)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path(f"results/{stamp}_double_integrator_certified")
    (out_dir / "paths").mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("config.yaml").write_text(CONFIG_PATH.read_text())
    _manifest(out_dir, "paths", workers)
    seeds = [
        s
        for s in [*cfg["select_seeds"], *_report_seeds(cfg)]
        if not (out_dir / "paths" / f"greedy_{s}.npz").exists()
    ]
    log.info("%s: %d paths to run", out_dir, len(seeds))
    with (out_dir / "paths_rows.jsonl").open("a") as fh, _pool(workers) as pool:
        tasks = [(s, str(out_dir)) for s in seeds]
        for row in pool.starmap(_path_task, tasks, chunksize=1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    log.info("paths done")


# --- constants and certified brackets ----------------------------------------


def _constants_task(n_centers: int, pad: float, nugget: float, n_check: int) -> dict:
    cal = json.loads(FROZEN_PATH.read_text())["calibration"]
    return certified_mismatch_constants(
        float(cal["scaler_scale"][0]),
        float(cal["lengthscale"]),
        n_centers,
        pad,
        nugget,
        n_check,
    )


def _pareto(rows: list[dict]) -> list[dict]:
    """Candidates not dominated in (B, eta)."""
    return [
        r
        for r in rows
        if not any(
            o["B"] <= r["B"] and o["eta"] <= r["eta"] and o is not r and o != r
            for o in rows
        )
    ]


def cmd_constants(out: str, workers: int) -> None:
    """Certified (B, eta) candidates."""
    cfg = _cfg()
    cand = cfg["candidates"]
    grid = list(itertools.product(cand["n_centers"], cand["pad"], cand["nugget"]))
    _manifest(Path(out), "constants", workers)
    with _pool(workers) as pool:
        rows = pool.starmap(
            _constants_task, [(m, p, n, cand["n_check"]) for m, p, n in grid]
        )
    Path(out, "constants_candidates.json").write_text(
        json.dumps({"all": rows, "pareto": _pareto(rows)}, indent=1)
    )
    log.info("%d candidates, %d on the Pareto front", len(rows), len(_pareto(rows)))


def certified_bracket(
    path: dict, n: int, B: float, eta: float, lam: float, seed: int
) -> dict:
    """Certified bracket on the first n path points.

    Args:
        path: Arrays ``X``, ``c``, ``Y`` of one regenerated path.
        n: Number of path points used.
        B: Certified norm bound.
        eta: Certified mismatch allowance.
        lam: Regularization of the certified model.
        seed: Path seed (witness stream).

    Returns:
        Row with the bracket, its flags, and the witness soundness check.
    """
    cfg = _cfg()
    run = build_cfg(cfg["budget"])
    X, c, Y = path["X"][:n], path["c"][:n], path["Y"][:n]
    beta = B + eta * np.sqrt(n / lam)
    bands = fit_joint_bands(
        X,
        c,
        Y,
        mode=MODE_AFFINE,
        lengthscale=run.lengthscale,
        lam=lam,
        B=np.array([B]),
        R=0.0,
        betas=[beta],
        c_max=1.0,
    )
    bands.offsets = [eta]
    rng = np.random.default_rng([seed, cfg["witness_stream_tag"], n])
    wit = np.concatenate([X, rng.uniform(size=(run.n_witness_candidates, X.shape[1]))])
    cover, rho = midpoint_cover(run.cover_n_per_axis, X.shape[1])
    t0 = time.perf_counter()
    br = compute_bracket(
        bands,
        wit,
        np.linspace(0.0, 1.0, run.report_grid_size),
        cover,
        rho,
        refine_factor=run.cover_refine_factor,
        scan=run.certificate_scan,
        max_depth=run.cover_max_depth,
        max_cells=run.cover_max_cells,
        second_order=run.cover_second_order,
    )
    sound = None
    if br.has_witness:
        sound = bool(is_viable(np.asarray(br.witness_x), br.lower)[0])
    return {
        "seed": seed,
        "n": n,
        "B": B,
        "eta": eta,
        "lam": lam,
        "beta": float(beta),
        "lower": br.lower,
        "upper": br.upper,
        "has_witness": br.has_witness,
        "has_certificate": br.has_certificate,
        "witness_x": None if br.witness_x is None else br.witness_x.tolist(),
        "witness_sound": sound,
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def _bracket_task(out: str, seed: int, n: int, B: float, eta: float, lam: float):
    path = dict(np.load(Path(out) / "paths" / f"greedy_{seed}.npz"))
    return certified_bracket(path, n, B, eta, lam, seed)


def cmd_select(out: str, workers: int) -> None:
    """Choose (B, eta, lam) on the select seed and freeze it."""
    cfg = _cfg()
    sel = Path(out, "selection.json")
    if sel.exists():
        raise SystemExit("selection.json exists; the selection is frozen")
    pareto = json.loads(Path(out, "constants_candidates.json").read_text())["pareto"]
    tasks = [
        (out, s, cfg["budget"], r["B"], r["eta"], lam)
        for s in cfg["select_seeds"]
        for r in pareto
        for lam in cfg["lam_cert"]
    ]
    _manifest(Path(out), "select", workers)
    with _pool(workers) as pool:
        rows = pool.starmap(_bracket_task, tasks, chunksize=1)
    best = min(rows, key=lambda r: (r["upper"] - r["lower"], r["B"]))
    fitted_B = float(json.loads(FROZEN_PATH.read_text())["calibration"]["B"][0])
    stop = best["beta"] > cfg["stop_multiplier"] * fitted_B
    sel.write_text(
        json.dumps(
            {
                "chosen": {k: best[k] for k in ("B", "eta", "lam")},
                "chosen_width": best["upper"] - best["lower"],
                "chosen_beta_at_budget": best["beta"],
                "fitted_B": fitted_B,
                "stop_rule_triggered": bool(stop),
                "candidates": rows,
                "frozen_utc": datetime.now(UTC).isoformat(),
            },
            indent=1,
        )
    )
    log.info("chosen %s width %.5f stop %s", best, best["upper"] - best["lower"], stop)


def cmd_replay(out: str, workers: int) -> None:
    """Certified brackets at every checkpoint of every report seed."""
    cfg = _cfg()
    chosen = json.loads(Path(out, "selection.json").read_text())["chosen"]
    tasks = [
        (out, s, n, chosen["B"], chosen["eta"], chosen["lam"])
        for s in _report_seeds(cfg)
        for n in cfg["checkpoints"]
    ]
    _manifest(Path(out), "replay", workers)
    with (Path(out) / "replay_rows.jsonl").open("w") as fh, _pool(workers) as pool:
        for row in pool.starmap(_bracket_task, tasks, chunksize=1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    log.info("replay done: %d rows", len(tasks))


# --- analysis ----------------------------------------------------------------


def running_endpoints(rows: list[dict]) -> list[dict]:
    """Running max of certified L and min of certified U over checkpoints.

    Args:
        rows: Checkpoint rows of one seed.

    Returns:
        Rows in checkpoint order with ``run_lower``, ``run_upper`` added.
    """
    lo, hi, out = 0.0, 1.0, []
    for r in sorted(rows, key=lambda r: r["n"]):
        if r["has_witness"]:
            lo = max(lo, r["lower"])
        if r["has_certificate"]:
            hi = min(hi, r["upper"])
        out.append({**r, "run_lower": lo, "run_upper": hi})
    return out


def cmd_analyze(out: str) -> None:
    """Gates R-G0 to R-G3."""
    cfg = _cfg()
    theta = THETA_EXISTS_UNIT
    seeds = _report_seeds(cfg)
    stored = {}
    stored = run_path("double_integrator_sweep") / "rows.jsonl"
    for line in stored.read_text().splitlines():
        r = json.loads(line)
        if r["arm"] == cfg["arm"] and r["budget"] == cfg["budget"]:
            stored[r["seed"]] = [(x["lower"], x["upper"]) for x in r["reports"]]
    regen = {}
    for line in Path(out, "paths_rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        regen[r["seed"]] = [(x["lower"], x["upper"]) for x in r["reports"]]
    match = {s: regen.get(s) == stored.get(s) for s in seeds}
    by_seed: dict[int, list] = {}
    for line in Path(out, "replay_rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        by_seed.setdefault(r["seed"], []).append(r)
    runs = {s: running_endpoints(by_seed[s]) for s in seeds}
    misses = [
        (s, r["n"])
        for s in seeds
        for r in runs[s]
        if not r["run_lower"] <= theta <= r["run_upper"]
    ]
    unsound = [
        (s, r["n"]) for s in seeds for r in runs[s] if r["witness_sound"] is False
    ]
    widths = {}
    for n in cfg["checkpoints"]:
        w = [next(r for r in runs[s] if r["n"] == n) for s in seeds]
        q25, q50, q75 = np.quantile(
            [r["run_upper"] - r["run_lower"] for r in w], [0.25, 0.5, 0.75]
        )
        widths[n] = {
            "median_width": float(q50),
            "q25": float(q25),
            "q75": float(q75),
            "with_certificate": int(sum(r["has_certificate"] for r in w)),
            "with_witness": int(sum(r["has_witness"] for r in w)),
        }
    fitted = {}
    for n in (100, 300, 1000):
        fw = [stored[s][n // 10 - 1] for s in seeds]
        fitted[n] = float(np.median([u - lo for lo, u in fw]))
    summary = {
        "R-G0_paths_match_stored": int(sum(match.values())),
        "R-G0_mismatched_seeds": [s for s, ok in match.items() if not ok],
        "R-G1_pass": not misses,
        "R-G1_misses": misses,
        "R-G2_pass": not unsound,
        "R-G2_unsound": unsound,
        "R-G3_certified": widths,
        "R-G3_fitted_median_width": fitted,
        "selection": json.loads(Path(out, "selection.json").read_text())["chosen"],
        "n_runs": len(seeds),
        "theta_unit": theta,
    }
    Path(out, "summary.json").write_text(json.dumps(summary, indent=1))
    _manifest(Path(out), "analyze", 1)
    log.info(json.dumps({k: v for k, v in summary.items() if k != "R-G3_certified"}))


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paths")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", default=None)
    for name in ("constants", "select", "replay"):
        q = sub.add_parser(name)
        q.add_argument("out", nargs="?", default=None)
        q.add_argument("--workers", type=int, default=8)
    sub.add_parser("analyze").add_argument("out", nargs="?", default=None)
    a = ap.parse_args()
    if a.cmd == "paths":
        cmd_paths(a.workers, a.resume)
        return
    out = a.out or str(run_path("double_integrator_certified"))
    if a.cmd == "constants":
        cmd_constants(out, a.workers)
    elif a.cmd == "select":
        cmd_select(out, a.workers)
    elif a.cmd == "replay":
        cmd_replay(out, a.workers)
    else:
        cmd_analyze(out)


if __name__ == "__main__":
    main()
