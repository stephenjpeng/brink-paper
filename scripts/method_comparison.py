"""Cross-study comparison of BRINK against every baseline on the applied studies.

Inputs: the study sweeps, the check-layer sweeps, and the GP-UCB and
differential-evolution runs named in ``configs/runs.yaml``; nothing reruns.
For each study, budget, and method it reports:

* validity: returned design passes the study's own check at L; L at or below
  the reference (plus stated error) at every report; [L, U] containing the
  reference where an upper bound exists;
* accuracy: reference minus L, and the passing level (L when the check
  passed, else 0);
* upper side: how often an upper bound exists, its containment, and U - L;
* sample efficiency: evaluations to the first design, and the evaluations at
  which a run with a passing design first reports L at or above the target
  (the proposed search's median final L at the top budget);
* compute: wall seconds per run, core-hours per cell, and peak RSS per run.

Usage:
    uv run python scripts/method_comparison.py [ADAPTIVE_BASELINE_DIR ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_baselines_sweep as s8  # noqa: E402
from common import git_state, run_path  # noqa: E402

log = logging.getLogger("method_comparison")
CFG_PATH = Path("configs/method_comparison.yaml")

ROLE = {
    "existence-witness-focused": "proposed: witness search (lower side)",
    "existence-bracket-greedy": "proposed: two-sided bracket (greedy)",
    "existence-bracket-greedy@smallest": "proposed: bracket, smallest-slice scan",
    "existence-bracket-greedy@top-down": "proposed: bracket, top-down scan",
    "joint-config-existence": "certified comparator: joint CONFIG",
    "joint-config-existence@smallest": "certified comparator: joint CONFIG",
    "random-joint-existence": "ablation: passive sampling, same bracket",
    "uncertified-sup-bisection": "baseline: random designs, bisection",
    "uncertified-direct-score": "baseline: random designs, reduced score",
    "uncertified-gp-ucb": "baseline: adaptive, GP-UCB (S8)",
    "uncertified-diff-evo": "baseline: adaptive, differential evolution (S8)",
}
GUARANTEE = {
    "existence-witness-focused": "witness verified on the system (no RKHS assumption)",
    "existence-bracket-greedy": "bracket: Theorem 1 on env A; certified form "
    "(A1eps, fitted B) elsewhere",
    "joint-config-existence": "same bracket layer as greedy",
    "random-joint-existence": "same bracket layer as greedy",
    "s7": "tested on the system (S7 layer)",
    "uncertified": "none (L is a measured outcome)",
}
NAIVE = {"s2": "sup-bisection", "s3": "baseline-k16", "s4": "sup-bisection",
         "s5": "baseline-k1"}  # fmt: skip


def applied_inputs() -> dict:
    """Rows and reference files of each applied study, from ``configs/runs.yaml``."""

    def rows(key: str) -> list[str]:
        d = run_path(key)
        return [str(x / "rows.jsonl") for x in (d if isinstance(d, list) else [d])]

    return {
        "s2": {"rows": rows("transmission_sweep"),
               "truth_gate": str(run_path("transmission_sweep") / "gate.json")},
        "s3": {"rows": rows("feeder_sweep")},
        "s4": {"rows": rows("hopper_sweep"),
               "truth_gate": str(run_path("hopper_sweep") / "gate.json")},
        "s5": {"rows": rows("covasim_sweep")},
        "s7": {"s3": rows("check_layer_feeder")[0],
               "s4": rows("check_layer_hopper")[0]},
    }  # fmt: skip


def load_config(path: Path = CFG_PATH) -> dict:
    """Comparison settings (YAML, SHA-256 under ``sha256``) plus the run inputs."""
    text = Path(path).read_text()
    cfg = {**yaml.safe_load(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    cfg["applied"] = applied_inputs()
    return cfg


def _rows(paths) -> list[dict]:
    return [json.loads(ln) for p in paths for ln in Path(p).read_text().splitlines()
            if ln.strip()]  # fmt: skip


def guarantee(arm: str) -> str:
    """Claim-stack label of an arm's output."""
    base = arm.split("@")[0]
    if base.startswith("uncertified"):
        return GUARANTEE["uncertified"]
    return GUARANTEE.get(base, GUARANTEE["s7"] if arm.endswith("S7 layer") else "")


def _report_points(row: dict) -> list[tuple[int, float]]:
    return [(int(r.get("raw_evals") or r["calls"]), float(r["lower"]))
            for r in row["reports"]]  # fmt: skip


def _applied_verified(study: str, row: dict) -> bool | None:
    if "check_feasible" in row and study != "s4" and "truth_check" not in row:
        return row["check_feasible"]  # S8 rows
    if study == "s4":
        return row.get("check_feasible")
    return (row.get("truth_check") or {}).get("feasible")


def _scan_upper(row: dict) -> float | None:
    scan = (row.get("end_scan") or {}).get("scan") or {}
    return float(scan["upper"]) if scan.get("has_certificate") else None


def _check_max_mean(row: dict) -> float | None:
    """Largest channel mean of the independent design check (<= 0 passes; noisy)."""
    chk = row.get("truth_check") or row.get("gate_check") or row.get("check") or {}
    m = chk.get("mean")
    return max(float(v) for v in m) if m else None


def applied_records(cfg: dict, s8_by: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Normalized run records of S2 to S5 (base and S8) and S7, by study."""
    out: dict[str, list[dict]] = {}
    for study, e in cfg["applied"].items():
        if study == "s7":
            continue
        replay = {}
        for r in _rows(e.get("upper_rows", [])):
            replay[(r["label"], r["budget"], r["seed"])] = _scan_upper(r)
        recs = []
        for r in [x for x in _rows(e["rows"]) if "label" in x] + s8_by.get(study, []):
            unc = r["arm"].startswith("uncertified")
            if unc:
                fu = r.get("final_upper")  # S4 rows do not store it
                U, kind = (None if fu is None else float(fu)), "claimed"
            elif (r["label"], r["budget"], r["seed"]) in replay:
                U, kind = replay[(r["label"], r["budget"], r["seed"])], "certified form"
            else:
                U, kind = _scan_upper(r), "certified form"
            recs.append({
                "cell": int(r["budget"]), "label": r["label"], "arm": r["arm"],
                "seed": int(r["seed"]), "L": float(r["final_lower"]), "U": U,
                "U_kind": kind if U is not None else None,
                "has_witness": bool(r["has_witness"]),
                "verified": _applied_verified(study, r),
                "check_max_mean": _check_max_mean(r),
                "points": _report_points(r),
                "first_witness": r.get("first_witness_raw"),
                "raw": int(r["raw_evals"]), "wall_s": float(r["wall_s"]),
                "task_s": float(r.get("task_s") or r["wall_s"]),
                "rss": r.get("peak_rss_gib"), "robust": r.get("robust_fraction"),
            })  # fmt: skip
        out[study] = recs
    for study, path in cfg["applied"]["s7"].items():
        for r in _rows([path]):
            L = r["tested_lower"]
            verified = (r.get("gate_check") or {}).get("feasible")
            out[study].append({
                "cell": int(r["budget_total"]), "label": f"{r['label']} + S7 layer",
                "arm": f"{r['arm']} + S7 layer", "seed": int(r["seed"]),
                "L": float(L) if L is not None else 0.0, "U": None, "U_kind": None,
                "has_witness": L is not None, "verified": verified if L is not None
                else None, "check_max_mean": _check_max_mean(r) if L is not None
                else None, "points": [], "first_witness": r.get("first_witness_raw"),
                "raw": int(r["raw_evals"]) + int(r["layer_raw"]),
                "wall_s": float(r["wall_s"]) + float(r["layer_s"]),
                "task_s": float(r["task_s"]), "rss": r.get("peak_rss_gib"),
                "robust": (r.get("gate_check") or {}).get("robust_fraction"),
            })  # fmt: skip
    return out


def applied_truth(cfg: dict, study: str) -> tuple[float, float, str]:
    """(value, stated error, kind) each applied study compares L against."""
    e = cfg["applied"][study]
    if study == "s2":
        g = json.loads(Path(e["truth_gate"]).read_text())
        return float(g["theta"]), float(g["err"]), "theta_exists"
    if study == "s4":
        g = json.loads(Path(e["truth_gate"]).read_text())
        return float(g["t_ref"]), 0.0, "T_ref (truth search; not an upper bound)"
    r = next(x for x in _rows(e["rows"]) if "theta" in x)
    return float(r["theta"]), float(r["theta_err"]), "theta_exists (Monte Carlo)"


def _med(v) -> float | None:
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def summarize(recs: list[dict], theta: float, err: float, target: float | None,
              r_sim: float | None) -> dict:  # fmt: skip
    """Every comparison dimension for one (study, cell, arm) group."""
    n = len(recs)
    ver = [r["verified"] for r in recs]
    with_u = [r for r in recs if r["U"] is not None]
    sound = [r for r in recs if r["verified"] is True]
    reach: list[int] = []
    traced = all(r["points"] for r in recs)  # S7 rows keep no report trace
    if target is not None and traced:
        for r in sound:
            hit = next((raw for raw, lo in r["points"] if lo >= target - 1e-12), None)
            if hit is not None:
                reach.append(hit)
    return {
        "n": n, "witness": sum(r["has_witness"] for r in recs),
        "verified": sum(v is True for v in ver),
        "failed_check": sum(v is False for v in ver),
        "lower_ok_anytime": sum(all(lo <= theta + err for _, lo in r["points"])
                                and r["L"] <= theta + err for r in recs),
        "median_truth_minus_L": _med([theta - r["L"] for r in recs]),
        "median_verified_L": _med([r["L"] if r["verified"] is True else 0.0
                                   for r in recs]),
        "n_upper": len(with_u), "upper_kind": with_u[0]["U_kind"] if with_u else None,
        "upper_covers": sum(r["U"] >= theta - err and r["L"] <= theta + err
                            for r in with_u),
        "median_width": _med([r["U"] - r["L"] for r in with_u]),
        "median_first_witness_raw": _med([r["first_witness"] for r in recs]),
        "target": target, "n_reach_target": len(reach) if traced else None,
        "median_raw_to_target": _med(reach),
        "median_wall_s": _med([r["wall_s"] for r in recs]),
        "median_overhead_s": None if r_sim is None else _med(
            [r["wall_s"] - r["raw"] * r_sim for r in recs]),
        "core_h": float(sum(r["task_s"] for r in recs) / 3600.0),
        "median_peak_rss_gib": _med([r["rss"] for r in recs]),
        "robust_all": sum(r["robust"] == 1.0 for r in recs)
        if any(r["robust"] is not None for r in recs) else None,
    }  # fmt: skip


def compare_applied(cfg: dict, recs_by: dict[str, list[dict]]) -> dict:
    """Applied studies: one summary per (cell, arm label)."""
    out = {}
    for study, recs in recs_by.items():
        theta, err, kind = applied_truth(cfg, study)
        cells = sorted({r["cell"] for r in recs})
        top = max(r["cell"] for r in recs if r["label"] == cfg["target_label"])
        tgt = _med(
            [
                r["L"]
                for r in recs
                if r["label"] == cfg["target_label"] and r["cell"] == top
            ]
        )
        rows = []
        for c in cells:
            naive = [r for r in recs if r["cell"] == c and r["label"] == NAIVE[study]]
            r_sim = _med([r["wall_s"] / r["raw"] for r in naive]) if naive else None
            for lab in dict.fromkeys(r["label"] for r in recs if r["cell"] == c):
                g = [r for r in recs if r["cell"] == c and r["label"] == lab]
                arm = g[0]["arm"]
                rows.append({"cell": c, "label": lab, "arm": arm,
                             "role": ROLE.get(arm.split(" + ")[0], "")
                             + (" + S7 layer" if " + S7" in arm else ""),
                             "guarantee": guarantee(arm),
                             **summarize(g, theta, err, tgt, r_sim)})  # fmt: skip
        out[study] = {"truth": {"value": theta, "err": err, "kind": kind},
                      "target": tgt, "rows": rows}  # fmt: skip
    return out


def _fmt(v, nd=3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(header: list[str], rows: list[list]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
    ]


def markdown(result: dict) -> str:
    """Tables for the results report (one validity and one efficiency table each)."""
    lines = []
    for study, s in result["applied"].items():
        t = s["truth"]
        lines += [
            f"\n### {study.upper()}: {t['kind']} = {t['value']:.4f} "
            f"(error {t['err']:.4f}); target L = {_fmt(s['target'])}\n"
        ]
        valid = []
        for r in s["rows"]:
            up = "-"
            if r["n_upper"]:
                up = (f"{r['n_upper']} ({r['upper_kind']}), {r['upper_covers']}, "
                      f"{_fmt(r['median_width'], 4)}")  # fmt: skip
            valid.append([
                f"{r['cell']:,}", r["label"], r["role"], f"{r['verified']}/{r['n']}",
                f"{r['lower_ok_anytime']}/{r['n']}", _fmt(r["median_truth_minus_L"]),
                _fmt(r["median_verified_L"]), up,
            ])  # fmt: skip
        lines += _table(
            ["Budget", "Arm", "Role", "Design verified at L",
             "L <= truth, every report", "Median truth - L", "Median verified L",
             "Upper: runs, covers, median width"], valid,
        )  # fmt: skip
        s4 = study == "s4"
        eff = []
        for r in s["rows"]:
            eff.append([
                f"{r['cell']:,}", r["label"], _fmt(r["median_first_witness_raw"], 0),
                f"{_fmt(r['n_reach_target'])}, {_fmt(r['median_raw_to_target'], 0)}",
                _fmt(r["median_wall_s"], 0), _fmt(r["median_overhead_s"], 0),
                _fmt(r["core_h"], 2), _fmt(r["median_peak_rss_gib"], 2),
                *([_fmt(r["robust_all"])] if s4 else []),
            ])  # fmt: skip
        lines += [""] + _table(
            ["Budget", "Arm", "First witness (raw)",
             "Sound runs reaching target: n, median raw", "Median wall s",
             "Median overhead s", "Core-h", "Peak RSS GiB",
             *(["Robust all 8"] if s4 else [])], eff,
        )  # fmt: skip
    return "\n".join(lines) + "\n"


def main(s8_dirs: list[Path]) -> None:
    """Write ``results/<UTC>_method_comparison/`` (JSON, Markdown, manifest)."""
    cfg = load_config()
    s8_by = s8.merged_rows(s8.load_config(), s8_dirs)
    result = {
        "config_sha256": cfg["sha256"], "s8_inputs": [str(d) for d in s8_dirs],
        "applied": compare_applied(cfg, applied_records(cfg, s8_by)),
    }  # fmt: skip
    out = Path("results") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_method_comparison"
    out.mkdir(parents=True)
    commit, dirty = git_state()
    (out / "manifest.json").write_text(json.dumps({
        "experiment": "method_comparison (no new runs)", "git_commit": commit,
        "git_dirty": dirty, "config_sha256": cfg["sha256"],
        "s8_inputs": result["s8_inputs"], "started_utc": datetime.now(UTC).isoformat(),
    }, indent=1))  # fmt: skip
    (out / "comparison.json").write_text(json.dumps(result, indent=1))
    (out / "comparison.md").write_text(markdown(result))
    log.info("comparison -> %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("s8_dirs", type=Path, nargs="*")
    main(ap.parse_args().s8_dirs or run_path("adaptive_baselines"))
