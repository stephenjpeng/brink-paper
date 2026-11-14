"""Check layer (Appendix C.7) on the feeder and Hopper studies: sweep and gates.

Constants in ``configs/check_layer.yaml``, which points at ``configs/feeder.yaml``
and ``configs/hopper.yaml`` for every method constant and environment. Two
methods per study (BRINK's proposed search and bisection) run unchanged at
budget N - R; the check layer (``brink.verify``) then spends at most R
evaluations testing the returned design, and every tested design is checked
again independently. A task is deterministic given (study, task index,
method, seed), so the worker count is a compute setting and a stopped sweep
resumes from its ``rows.jsonl``.

Streams: feeder check k (1-based) of task i draws 4,096 conditions from
``default_rng([995, i, k])`` and passes on ``margin_check(g)["clear_3se"]``;
the final feeder check is ``feeder_setup._witness_check`` on stream index
3,000 + i. Hopper checks are the fixed-state rollout (reset seed 0); the final
Hopper check adds the 8 robustness rollouts. The reported Hopper runs used
x86-64.

Usage:
    uv run python scripts/check_layer_sweep.py sweep --study feeder --workers N \
        [--seeds S ...] [--resume DIR]
    uv run python scripts/check_layer_sweep.py analyze --study feeder [RESULTS_DIR]
"""

from __future__ import annotations

import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # both base configs: blas 1
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # Linux (docs/runbook.md)
os.environ.setdefault("OMP_NUM_THREADS", "1")  # before numpy loads

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from brink.arms import (  # noqa: E402
    ARM_SUP_BISECTION,
    ExistenceConfig,
    run_existence_arm,
)
from brink.memory import RssMonitor  # noqa: E402
from brink.quantile_bound import (  # noqa: E402
    empirical_bernstein_deviation,
)
from brink.verify import verify_and_adjust  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import git_state, run_path  # noqa: E402

log = logging.getLogger("check_layer_sweep")
CFG_PATH = Path("configs/check_layer.yaml")
STUDY_KEYS = {"feeder": "s3", "hopper": "s4"}  # internal study keys in rows
WINDOW_CALL_CAP = 32  # a window call may overshoot by less than one call (S3, S4)
S3_GATE_STREAM = 993  # the snapshot stream hosting_existence._witness_check draws


def load_config(path: Path = CFG_PATH) -> dict:
    """The check-layer constants (YAML), with their SHA-256 under ``sha256``."""
    text = Path(path).read_text()
    return {**yaml.safe_load(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def proposer_budget(e: dict) -> int:
    """Proposer budget N - R of one environment block (R = n_max x raw per check)."""
    lay = e["layer"]
    return int(e["budget_total_raw"]) - int(lay["n_max"]) * int(lay["raw_per_check"])


def propose_and_verify(
    view,
    arm: str,
    run_cfg: ExistenceConfig,
    B: np.ndarray,
    seed: int,
    check: Callable[[np.ndarray, float], bool],
    e: dict,
    best_x_from_log: Callable[[list, float], list[float] | None],
) -> dict:
    """One proposer run at budget N - R, then the check layer on its design.

    Args:
        view: Adapter view of the arm (logged window or logged predicate).
        arm: Arm name.
        run_cfg: Frozen arm constants at budget N - R.
        B: Norm bounds passed to ``run_existence_arm``.
        seed: Run seed.
        check: The layer's system check ``check(x, c) -> passed``.
        e: Environment block of the S7 config (``budget_total_raw``, ``layer``).
        best_x_from_log: The base sweep's design recovery for sup-bisection.

    Returns:
        The row fields both environments share.
    """
    lay = e["layer"]
    t0 = time.perf_counter()
    res = run_existence_arm(view, arm, run_cfg, B, seed)
    wall = time.perf_counter() - t0
    has_w = bool(res.reports and res.reports[-1].has_witness)
    L = float(res.final_lower)
    x = None
    if has_w:
        sup = arm == ARM_SUP_BISECTION
        x = best_x_from_log(view.log, L) if sup else res.final_witness_x
    t1 = time.perf_counter()
    adj = None
    if x is not None:
        adj = verify_and_adjust(
            np.asarray(x, dtype=float), L, check, int(lay["n_max"]),
            float(lay["c_min"]), float(lay["c_max"]), float(lay["step0"]),
            float(lay["resolution"]),
        )  # fmt: skip
    layer_s = time.perf_counter() - t1
    n, r = (adj.n_checks if adj else 0), int(lay["raw_per_check"])
    return {
        "arm": arm, "view": "predicate" if arm == ARM_SUP_BISECTION else "window",
        "seed": int(seed), "budget": int(run_cfg.budget),
        "budget_total": int(e["budget_total_raw"]),
        "reserve_raw": int(lay["n_max"]) * r, "raw_evals": int(res.raw_evals),
        "layer_raw": n * r, "wall_s": wall, "layer_s": layer_s,
        "s_per_check": layer_s / n if n else None,
        "final_lower": L, "final_upper": float(res.final_upper), "has_witness": has_w,
        "first_witness_raw": next(
            (rp.raw_evals for rp in res.reports if rp.has_witness), None),
        "witness_x": x, "n_resets": int(getattr(res, "n_resets", 0)),
        "proposer_lower": L, "tested_lower": adj.tested_lower if adj else None,
        "adjust": None if adj is None else {
            "trace": adj.trace, "n_checks": n, "layer_raw": n * r},
        "reports": [asdict(rp) for rp in res.reports],
    }  # fmt: skip


def eb_verdict(g: np.ndarray, eb: dict) -> dict:
    """Empirical-Bernstein feasibility verdict at the gate check (co-reported).

    Per channel the margins are clipped to ``eb["clip"]``; the verdict is
    feasible when every clipped mean plus its deviation at ``eb["p"]`` is
    <= 0. Clipping from above lowers the mean, so a nonzero
    ``eb_n_clipped_above`` means the verdict may understate the true mean.

    Args:
        g: Raw margins, shape (n_snapshots, m).
        eb: ``{"p": ..., "clip": [lo, hi]}``.

    Returns:
        Upper confidence values per channel, the verdict, and clip counts.
    """
    lo, hi = (float(v) for v in eb["clip"])
    gc = np.clip(g, lo, hi)
    t = np.array([
        empirical_bernstein_deviation(float(col.var(ddof=1)), len(col), hi - lo,
                                      float(eb["p"])) for col in gc.T
    ])  # fmt: skip
    upper = gc.mean(axis=0) + t
    return {
        "eb_upper": upper.tolist(), "eb_feasible": bool(np.all(upper <= 0.0)),
        "eb_n_clipped_above": int((g > hi).sum()),
        "eb_n_clipped_below": int((g < lo).sum()),
    }  # fmt: skip


def _s3_task(cfg: dict, e: dict, idx: int, label: str, seed: int) -> dict:
    import feeder_sweep as hs  # pandapower only in feeder workers

    he = hs.he
    base = he.load_s3_config(Path(e["base_config"]))
    spec = next(a for a in base["arms"] if a["label"] == e["arm_labels"][label])
    K = int(base["k_snapshots"])
    env = hs.hc.load_hosting_environment(base["frozen_environment"], K)
    R = float(env.R_by_knob[K])
    cal = json.loads(Path(base["frozen_environment"]).read_text())["calibration"]
    run_cfg = hs.build_cfg(
        base, proposer_budget(e), spec["view"], R, cal["lengthscale"]
    )
    view = hs.make_view(spec, env, he._oracle(), K)
    oracle, lay = he._oracle(), e["layer"]

    def margins(x, c: float, rng) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = int(lay["check_snapshots"])
        return np.array([oracle.snapshot_margins(x, c, rng) for _ in range(n)])

    k = 0

    def check(x: np.ndarray, c: float) -> bool:
        nonlocal k
        k += 1
        rng = np.random.default_rng([int(lay["stream_prefix"]), idx, k])
        return he.margin_check(margins(x, c, rng))["clear_3se"]

    row = propose_and_verify(
        view, spec["name"], run_cfg, env.B, seed, check, e, hs.best_x_from_log
    )
    theta, err = hs._truth(base)
    row.update(theta=theta, theta_err=err, base_label=spec["label"])
    row["gate_check"] = None
    if row["tested_lower"] is not None:
        t0 = time.perf_counter()
        w = int(cfg["gate_check_stream_offset"]) + idx
        x, c = row["witness_x"], float(row["tested_lower"])
        gate = he._witness_check((w, x, c))
        g = margins(x, c, np.random.default_rng([S3_GATE_STREAM, w]))
        gate["eb_redraw_matches"] = bool(
            np.allclose(g.mean(axis=0), gate["mean"], rtol=0.0, atol=1e-12)
        )
        row["gate_check"] = {**gate, **eb_verdict(g, e["eb"]), "stream_index": w}
        row["gate_check_s"] = time.perf_counter() - t0
    return row


def _s4_task(cfg: dict, e: dict, idx: int, label: str, seed: int) -> dict:
    import hopper_sweep as ms  # mujoco only in Hopper workers

    base = yaml.safe_load(Path(e["base_config"]).read_text())
    spec = next(a for a in base["arms"] if a["label"] == e["arm_labels"][label])
    cal = ms._calibration(base)
    B = np.asarray(cal["B"], dtype=float)
    if spec["view"] == "window":
        scale = np.asarray(cal["scaler_scale"])
        view = ms.LoggedWindowView(ms.mh.HopperEnvironment(scaler_scale=scale))
    else:
        view = ms.LoggedPredicateView(ms.mh.HopperOracle().feasible, ms.mh.X_DIM)
    run_cfg = ms.build_cfg(base, proposer_budget(e), spec["view"])
    oracle, ch = ms.mh.HopperOracle(), base["checks"]

    def passes(x, c: float, reset_seed: int) -> bool:
        r = oracle.rollout(np.asarray(x, dtype=float), c, reset_seed=int(reset_seed))
        return bool(np.all(r.margins() <= 0.0))

    def check(x: np.ndarray, c: float) -> bool:
        return passes(x, c, ch["reset_seed"])

    row = propose_and_verify(
        view, spec["name"], run_cfg, B, seed, check, e, ms.best_x_from_log
    )
    row["base_label"], row["gate_check"] = spec["label"], None
    if row["tested_lower"] is not None:
        t0 = time.perf_counter()
        x, c = row["witness_x"], float(row["tested_lower"])
        r0 = oracle.rollout(np.asarray(x), c, reset_seed=int(ch["reset_seed"]))
        robust = [passes(x, c, s) for s in ch["robustness_reset_seeds"]]
        row["gate_check"] = {
            "check": asdict(r0), "feasible": bool(np.all(r0.margins() <= 0.0)),
            "robust_fraction": float(np.mean(robust)),
        }  # fmt: skip
        row["gate_check_s"] = time.perf_counter() - t0
    return row


def run_task(task: tuple[str, int, str, int]) -> dict:
    """One (environment, task index, S7 arm label, seed) run in a fresh process."""
    t0 = time.perf_counter()
    env, idx, label, seed = task
    cfg = load_config()
    row = (_s3_task if env == "s3" else _s4_task)(cfg, cfg[env], idx, label, seed)
    row.update(task_index=idx, env=env, label=label, config_sha256=cfg["sha256"])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes macOS, KiB Linux
    row["peak_rss_gib"] = rss / (1024**3 if sys.platform == "darwin" else 1024**2)
    row["task_s"] = time.perf_counter() - t0
    return row


def _tasks(cfg: dict, env: str, seeds, base: int) -> list[tuple[str, int, str, int]]:
    cells = [(lab, int(s)) for lab in cfg["arms"] for s in seeds]
    return [(env, base + i, lab, s) for i, (lab, s) in enumerate(cells)]


def sweep_tasks(cfg: dict, env: str) -> list[tuple[str, int, str, int]]:
    """Task list: arm-major, then seed (heavy method arm first)."""
    seeds = range(int(cfg["seeds"]["start"]), int(cfg["seeds"]["stop"]))
    return _tasks(cfg, env, seeds, 0)


def budget_ok(row: dict) -> bool:
    """Budget rule: proposer <= N - R, layer <= R, total <= N (+ one call)."""
    cap = WINDOW_CALL_CAP if row["view"] == "window" else 0
    return bool(
        row["raw_evals"] <= row["budget"] + cap
        and row["layer_raw"] <= row["reserve_raw"]
        and row["raw_evals"] + row["layer_raw"] <= row["budget_total"] + cap
    )


def _require_platform(env: str) -> None:
    if env == "s4" and platform.machine() not in ("x86_64", "AMD64"):
        log.warning(
            "Hopper rollouts depend on the architecture; the reported runs "
            "used x86-64, so these rows can differ from them"
        )


def _run_pool(tasks, workers: int, on_row, label: str) -> RssMonitor:
    """Run tasks in fresh processes, handing each row to ``on_row``."""
    mon = RssMonitor()
    mon.start()
    t0 = time.time()
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result()
            on_row(row)
            gate = (row["gate_check"] or {}).get("feasible")
            log.info(
                "%s %d/%d %s s=%d: %.0fs L=%.4f L_T=%s checks=%d gate=%s raw=%d+%d",
                label, i, len(tasks), row["label"], row["seed"], row["task_s"],
                row["proposer_lower"], row["tested_lower"],
                (row["adjust"] or {}).get("n_checks", 0), gate, row["raw_evals"],
                row["layer_raw"],
            )  # fmt: skip
            if i % 25 == 0:
                rate = i / (time.time() - t0)
                log.info("eta %.2f h", (len(tasks) - i) / rate / 3600)
    mon.stop()
    if mon.missed_samples:
        log.warning("RSS monitor missed %d samples", mon.missed_samples)
    return mon


def _manifest(cfg: dict, env: str, workers: int, kind: str, **extra) -> dict:
    commit, dirty = git_state()
    base_path = Path(cfg[env]["base_config"])
    frozen = Path(yaml.safe_load(base_path.read_text())["frozen_environment"])
    return {
        "experiment": f"check_layer_{env}_{kind}", "git_commit": commit,
        "git_dirty": dirty, "config_sha256": cfg["sha256"], "env": env,
        "base_config": str(base_path),
        "base_config_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "frozen_environment_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "seeds": {"start": cfg["seeds"]["start"], "stop": cfg["seeds"]["stop"]},
        "workers": workers, "blas_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "python": sys.version, "numpy": np.__version__,
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "started_utc": datetime.now(UTC).isoformat(), **extra,
    }  # fmt: skip


def _new_dir(name: str) -> Path:
    out = Path("results") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{name}"
    out.mkdir(parents=True)
    return out


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def cmd_sweep(env: str, workers: int, seeds: list[int] | None = None,
              resume: Path | None = None) -> None:  # fmt: skip
    """The 100-run sweep of one study (or the selected seeds); ``resume`` continues."""
    cfg = load_config()
    _require_platform(env)
    tasks = [t for t in sweep_tasks(cfg, env) if seeds is None or t[3] in seeds]
    if resume is None:
        manifest = _manifest(cfg, env, workers, "sweep", n_tasks=len(tasks),
                             seeds_filter=seeds)  # fmt: skip
        name = {v: k for k, v in STUDY_KEYS.items()}[env]
        out = _new_dir(f"check_layer_{name}_sweep")
    else:
        out = resume
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["config_sha256"] != cfg["sha256"] or manifest["env"] != env:
            raise SystemExit("resume target used another config or environment")
        lines = (out / "rows.jsonl").read_text().splitlines()
        done = {json.loads(ln)["task_index"] for ln in lines if ln.strip()}
        keep = manifest.get("seeds_filter")
        tasks = [t for t in sweep_tasks(cfg, env)
                 if t[1] not in done and (keep is None or t[3] in keep)]  # fmt: skip
        commit, dirty = git_state()
        manifest.setdefault("resumes", []).append({
            "resumed_utc": datetime.now(UTC).isoformat(), "git_commit": commit,
            "git_dirty": dirty, "workers": workers, "rows_done": len(done),
            "rows_remaining": len(tasks),
        })  # fmt: skip
        log.info("resuming %s: %d done, %d remaining", out, len(done), len(tasks))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    t0 = time.time()
    mon = _run_pool(tasks, workers, lambda r: _write_rows(out / "rows.jsonl", [r]),
                    "sweep")  # fmt: skip
    (out / "rss_log.json").write_text(json.dumps(mon.samples))
    (out / "sweep_summary.json").write_text(json.dumps({
        "wall_h": (time.time() - t0) / 3600.0, "aggregate_peak_rss_gib": mon.peak_gib,
        "rss_missed_samples": mon.missed_samples, "n_rows": len(tasks),
        "finished_utc": datetime.now(UTC).isoformat(),
    }, indent=1))  # fmt: skip
    log.info("sweep done in %.2f h -> %s", (time.time() - t0) / 3600.0, out)


def _binom(count: int, n: int, null: float, alpha: float) -> dict:
    p = binomtest(int(count), int(n), null, alternative="greater").pvalue if n else 1.0
    return {"count": int(count), "n": int(n), "p": float(p),
            "reject": bool(n and p < alpha)}  # fmt: skip


def _med(values) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def _reported(rs: list[dict], tested: list[dict], env: str,
              t_ref: float | None) -> dict:  # fmt: skip
    """Reported quantities of one method (no gate weight)."""
    checks = [r["adjust"]["n_checks"] for r in rs if r["adjust"]]
    gates = [r["gate_check"] for r in tested]
    out = {
        "n": len(rs), "tested": len(tested),
        "median_checks": _med(checks), "max_checks": max(checks, default=None),
        "median_layer_raw": _med([r["layer_raw"] for r in rs]),
        "median_proposer_lower": _med([r["proposer_lower"] for r in rs]),
        "median_tested_lower": _med([r["tested_lower"] for r in tested]),
        "median_adjustment": _med(
            [r["tested_lower"] - r["proposer_lower"] for r in tested]),
        "adjusted_up": sum(r["tested_lower"] > r["proposer_lower"] for r in tested),
        "adjusted_down": sum(r["tested_lower"] < r["proposer_lower"] for r in tested),
        "median_task_s": _med([r["task_s"] for r in rs]),
        "max_peak_rss_gib": max((r["peak_rss_gib"] for r in rs), default=None),
    }  # fmt: skip
    if env == "s3":
        deaths = [float(np.median(gc["c_off_replicates"])) for gc in gates]
        out.update({
            "median_theta_minus_tested": _med(
                [r["theta"] - r["tested_lower"] for r in tested if "theta" in r]),
            "gate_clear_3se": sum(gc["clear_3se"] for gc in gates),
            "gate_eb_feasible": sum(bool(gc.get("eb_feasible")) for gc in gates),
            "replicated_death_ge_tested": sum(
                d >= r["tested_lower"] for d, r in zip(deaths, tested, strict=True)),
            "median_replicated_death_minus_tested": _med(
                [d - r["tested_lower"] for d, r in zip(deaths, tested, strict=True)]),
        })  # fmt: skip
    else:
        out["median_robust_fraction"] = _med([gc["robust_fraction"] for gc in gates])
        out["median_t_ref_minus_tested"] = (
            None if t_ref is None else _med([t_ref - r["tested_lower"] for r in tested])
        )
    return out


def analyze_rows(
    rows: list[dict],
    cfg: dict,
    env: str,
    bound: float | None = None,
    t_ref: float | None = None,
) -> dict:
    """Gates S7-G1 and S7-G2 and the reported quantities.

    A violating run is a tested design whose gate check is infeasible or whose
    tested endpoint exceeds ``bound``. In G2 a seed pair counts for arm 1
    only when both arms have a tested design and arm 1's is higher; ties and
    pairs with a missing tested design count against arm 1.

    Args:
        rows: Sweep rows (``run_task`` output) of one environment.
        cfg: S7 config.
        env: ``"s3"`` or ``"s4"``; G2 gates on S3 only.
        bound: S3 theta + err; None on S4 (no truth).
        t_ref: S4 verified reference, for the reported gap only.

    Returns:
        Gate dictionary with a ``verdict`` block, or ``stopped`` with the
        excluded task indices when any row breaks the budget.
    """
    bad = [r["task_index"] for r in rows if not budget_ok(r)]
    if bad:
        return {"n_rows": len(rows), "excluded": bad,
                "stopped": "budget accounting failed"}  # fmt: skip
    g, alpha = cfg["gates"], float(cfg["gates"]["alpha"])
    by = {
        lab: sorted((r for r in rows if r["label"] == lab), key=lambda r: r["seed"])
        for lab in cfg["arms"]
    }
    out: dict = {"n_rows": len(rows), "G1": {}, "arms": {}}
    for lab, rs in by.items():
        tested = [r for r in rs if r["tested_lower"] is not None]
        bad_gate = [not r["gate_check"]["feasible"] for r in tested]
        above = [bound is not None and r["tested_lower"] > bound for r in tested]
        a = _binom(len(tested), len(rs), float(g["g1a_tested_rate_null"]), alpha)
        v = _binom(sum(x or y for x, y in zip(bad_gate, above, strict=True)), len(rs),
                   float(g["g1b_violation_delta"]), alpha)  # fmt: skip
        frac = 1.0 - sum(bad_gate) / len(tested) if tested else None
        ok_frac = frac is not None and frac >= float(g["g1b_feasible_min"])
        b_pass = bool(ok_frac and not v["reject"])
        out["G1"][lab] = {
            "a_tested": {**a, "pass": a["reject"]},
            "b_sound": {"feasible_fraction": frac, "infeasible": sum(bad_gate),
                        "above_bound": sum(above), "violations": v, "pass": b_pass},
            "pass": bool(a["reject"] and b_pass),
        }  # fmt: skip
        out["arms"][lab] = _reported(rs, tested, env, t_ref)
    out["G1"]["pass"] = all(out["G1"][lab]["pass"] for lab in by)
    a1, a2 = cfg["arms"]
    l2 = {r["seed"]: r["tested_lower"] for r in by[a2]}
    pairs = [(r["tested_lower"], l2[r["seed"]]) for r in by[a1] if r["seed"] in l2]
    both = [(x, y) for x, y in pairs if x is not None and y is not None]
    wins, losses = sum(x > y for x, y in both), sum(x < y for x, y in both)
    sign = _binom(wins, len(pairs), float(g["g2_sign_null"]), alpha)
    out["G2"] = {
        "wins": wins, "losses": losses, "n_pairs": len(pairs),
        "ties_or_untested": len(pairs) - wins - losses, "sign_test": sign,
        "sign_p_two_sided_excluding_ties": float(
            binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else None,
        "median_tested_lower": {lab: out["arms"][lab]["median_tested_lower"]
                                for lab in by},
        "gating": env == "s3", "pass": sign["reject"] if env == "s3" else None,
    }  # fmt: skip
    out["verdict"] = {"G1": out["G1"]["pass"], "G2": out["G2"]["pass"]}
    return out


def cmd_analyze(results_dir: Path, env: str) -> None:
    """Gates and reported quantities of a finished sweep -> ``gate.json``."""
    cfg = load_config()
    e = cfg[env]
    lines = (results_dir / "rows.jsonl").read_text().splitlines()
    rows = [json.loads(ln) for ln in lines if ln.strip()]
    bound = t_ref = None
    if env == "s3":
        import feeder_sweep as hs

        theta, err = hs._truth(hs.he.load_s3_config(Path(e["base_config"])))
        bound = theta + err
    else:
        gate = json.loads((run_path("hopper_sweep") / "gate.json").read_text())
        t_ref = float(gate["t_ref"])
    gate = analyze_rows(rows, cfg, env, bound=bound, t_ref=t_ref)
    gate.update(config_sha256=cfg["sha256"], bound=bound, t_ref=t_ref)
    (results_dir / "gate.json").write_text(json.dumps(gate, indent=1))
    if "stopped" in gate:
        raise SystemExit(f"{gate['stopped']}: tasks {gate['excluded']}")
    log.info("verdict %s -> %s", gate["verdict"], results_dir / "gate.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("--study", choices=tuple(STUDY_KEYS), required=True)
    sw.add_argument("--workers", type=int, required=True)
    sw.add_argument("--seeds", type=int, nargs="+", default=None)
    sw.add_argument("--resume", type=Path, default=None)
    an = sub.add_parser("analyze")
    an.add_argument("results_dir", type=Path, nargs="?", default=None)
    an.add_argument("--study", choices=tuple(STUDY_KEYS), required=True)
    args = ap.parse_args()
    env = STUDY_KEYS[args.study]
    if args.cmd == "sweep":
        cmd_sweep(env, args.workers, args.seeds, args.resume)
    else:
        cmd_analyze(args.results_dir or run_path(f"check_layer_{args.study}"), env)
