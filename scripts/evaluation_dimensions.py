"""Quick evaluation dimensions of the L4DC results briefing, from stored runs only.

Inputs: ``configs/evaluation_dimensions.yaml`` (targets, stopping tolerances, slack
buffer) and, through it, ``configs/paper_figures.yaml`` (every stored run row, the
reference limits, stress units, and method names). Nothing reruns. For the double
integrator and the applied studies S2 to S5 (base, S7, and S8 rows) it computes:

* T1 validity: returned designs passing the study's independent check; L at or below
  the limit (plus its stated error) at every report; the final [L, U] containing the
  limit, with U labeled ``ours`` (from our infeasibility test) or ``claimed`` (an
  uncertified method's own interval). Applied counts come from
  ``method_comparison.summarize``;
* T2 containment of the uncertified methods' claimed intervals at every report;
* T3 stress slack (a returned design's own last feasible stress minus its reported
  level, where stored) and the binding signed constraint value of its check;
* T4 variability across seeds: quantiles, extremes, and the worst run per cell;
* T5 decision value: achievable / not achievable / undecided answers to target
  stresses scored against the reference limit, and stopping the double integrator
  when U - L reaches a tolerance.

Every result is observed in simulation (or checked by simulation), never proved.
Writes ``results/<UTC>_evaluation_dimensions/`` with ``dimensions.json``,
``dimensions.md``, and ``manifest.json``.

Usage:
    uv run python scripts/evaluation_dimensions.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_baselines_sweep as s8  # noqa: E402
import method_comparison as mc  # noqa: E402
import paper_figures as pf  # noqa: E402
from common import git_state  # noqa: E402

from brink.studies.double_integrator import c_off_unit  # noqa: E402

log = logging.getLogger("evaluation_dimensions")
CFG_PATH = Path("configs/evaluation_dimensions.yaml")
OURS, CLAIMED = "ours", "claimed"  # who produced an upper value
ACHIEVABLE, NOT_ACHIEVABLE, UNDECIDED = "achievable", "not achievable", "undecided"
EPS = 1e-9  # float tolerance on comparisons of stored levels (grid steps >= 5e-4)
# Signed constraint channels of each study's design check (<= 0 meets the constraint).
CHANNELS = {
    # GridLoadabilityOracle.raw_margins: 0.94 - min PQ-bus voltage (pu), max PQ-bus
    # voltage - 1.06 (pu), (max line or transformer loading - 100 %) / 100 %.
    "s2": ("low voltage", "high voltage", "loading"),
    # hosting_capacity: max |V - 1| - 0.06 (pu), (max loading - 100 %) / 100 %, as
    # means over 4,096 fresh operating conditions.
    "s3": ("voltage", "loading"),
    # covasim_policy: (peak critical cases - capacity) / capacity, mean of 128 runs.
    "s5": ("critical care",),
}


def load_config(path: Path = CFG_PATH) -> dict:
    """Evaluation settings (YAML) with their SHA-256 under ``sha256``."""
    text = Path(path).read_text()
    return {**yaml.safe_load(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}


# ------------------------------------------------------------------ metric rules


def contains(lower: float, upper: float, theta: float, err: float) -> bool:
    """Whether [lower, upper] contains the limit within its stated error."""
    return lower <= theta + err and upper >= theta - err


def contains_every_report(
    intervals: list[tuple[float, float]], theta: float, err: float
) -> bool:
    """Whether every reported interval contains the limit within its stated error.

    Args:
        intervals: (lower, upper) per report, in report order.
        theta: Reference limit (unit stress).
        err: Stated error of the reference limit (unit stress).

    Returns:
        True if every interval contains the limit (vacuously true for none).
    """
    return all(contains(lo, up, theta, err) for lo, up in intervals)


def own_limit(study: str, check: dict | None, x: list[float] | None) -> float | None:
    """Last feasible stress of one returned design (unit stress), from stored fields.

    Args:
        study: ``s6b`` (closed form ``c_off_unit``), ``s2`` (the check's ``death``,
            the design's exact window end), or ``s3`` (median of the check's
            replicated last feasible levels); other studies store none.
        check: The design's independent check record (S2, S3), else None.
        x: The returned design in unit coordinates (double-integrator), else None.

    Returns:
        The design's own limit, or None when it is not stored.
    """
    if study == "s6b":
        return None if x is None else float(c_off_unit(np.asarray(x, dtype=float))[0])
    if not check:
        return None
    if study == "s2" and check.get("death") is not None:
        return float(check["death"])
    if study == "s3" and check.get("c_off_replicates"):
        return float(np.median(check["c_off_replicates"]))
    return None


def stress_slack(limit: float | None, level: float) -> float | None:
    """Own limit minus reported level; negative means reported above its own limit."""
    return None if limit is None else limit - level


def target_truth(
    target: float, theta: float, err: float, lower_bound_only: bool = False
) -> bool | None:
    """Whether the reference says the target stress is achievable.

    Args:
        target: Target stress (unit).
        theta: Reference limit (unit).
        err: Stated error of the reference (unit).
        lower_bound_only: The reference is a verified design, a lower bound on the
            limit (Hopper): only targets at or below it can be scored.

    Returns:
        True (achievable), False (not achievable), or None (inside the error band,
        or not scoreable).
    """
    if lower_bound_only:
        return True if target <= theta + EPS else None
    if target < theta - err:
        return True
    if target > theta + err:
        return False
    return None


def target_answer(
    level: float,
    has_design: bool,
    upper: float | None,
    upper_kind: str | None,
    target: float,
) -> str:
    """A run's final answer to "can the stress reach ``target``?".

    "Achievable" needs a returned design at a level at or above the target. "Not
    achievable" needs our upper bound, from the infeasibility test, below the
    target; an uncertified method's claimed upper value never decides it.

    Args:
        level: Reported level L (unit).
        has_design: Whether the run returned a design at L.
        upper: Upper value at the final report (unit), or None.
        upper_kind: ``ours`` or ``claimed`` (or None).
        target: Target stress (unit).

    Returns:
        ``achievable``, ``not achievable``, or ``undecided``.
    """
    if has_design and level >= target - EPS:
        return ACHIEVABLE
    if upper_kind == OURS and upper is not None and upper < target - EPS:
        return NOT_ACHIEVABLE
    return UNDECIDED


def first_stop(
    reports: list[tuple[int, float, float]], tol: float
) -> tuple[int, float, float] | None:
    """First report (raw evaluations, lower, upper) with U - L at most ``tol``."""
    return next((r for r in reports if r[2] - r[1] <= tol + EPS), None)


def _q(values) -> dict | None:
    """Median, quartiles, and extremes (np.median, as the comparison tables use)."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return None
    q25, q75 = np.quantile(v, [0.25, 0.75])
    return {"median": float(np.median(v)), "q25": float(q25), "q75": float(q75),
            "min": float(v.min()), "max": float(v.max())}  # fmt: skip


# ------------------------------------------------------------------------- runs


def _binding(study: str, check: dict | None) -> dict:
    """Largest signed constraint value of a design check, its channel, and its se."""
    if not check or study not in CHANNELS:
        return {"binding": None, "binding_se": None, "binding_channel": None}
    vals = check.get("margins") if study == "s2" else check.get("mean")
    if not vals:
        return {"binding": None, "binding_se": None, "binding_channel": None}
    i = int(np.argmax(vals))
    se = check.get("se")
    return {"binding": float(vals[i]), "binding_se": float(se[i]) if se else None,
            "binding_channel": CHANNELS[study][i]}  # fmt: skip


def _design_check(row: dict) -> dict | None:
    """The check record behind a run's design (base, S8, or S7 row)."""
    return row.get("gate_check") or row.get("truth_check") or row.get("check")


def _intervals(row: dict) -> list[tuple[int, float, float]]:
    """(raw evaluations, lower, upper) per report, then the final values if stored.

    The final values can follow the last report (a run ends between two reports);
    Hopper rows store no final upper value, so their last report stands for it.
    """
    out = [(int(r.get("raw_evals") or r["calls"]), float(r["lower"]), float(r["upper"]))
           for r in row["reports"]]  # fmt: skip
    if row.get("final_upper") is not None:
        out.append((int(row["raw_evals"]), float(row["final_lower"]),
                    float(row["final_upper"])))  # fmt: skip
    return out


def _group(label: str, arm: str, ours: bool) -> str:
    if " + S7 layer" in label:
        return "tested"
    if ours:
        return "ours"
    return "uncertified" if arm.startswith("uncertified") else "shared bounds"


GROUPS = ("ours", "shared bounds", "uncertified", "tested")


def _rank(cfg: dict, study: str, run: dict) -> tuple:
    """Budget, then method group, then the method order of the figure config."""
    base = run["label"].split(" + ")[0]
    key = f"{base}@{study}" if f"{base}@{study}" in cfg["arms"] else base
    return (run["cell"], GROUPS.index(run["group"]), list(cfg["arms"]).index(key),
            run["seed"])  # fmt: skip


def _name(cfg: dict, study: str, label: str) -> tuple[str, bool]:
    st = pf.arm_style(cfg, study, label.split(" + ")[0])
    return st["name"] + (", tested" if " + S7 layer" in label else ""), bool(
        st.get("ours")
    )


def applied_runs(cfg: dict, data: dict) -> dict[str, list[dict]]:
    """Normalized records of S2 to S5 joined to their raw rows by (label, budget, seed).

    Args:
        cfg: ``configs/paper_figures.yaml``.
        data: ``paper_figures.load_data(cfg)``.

    Returns:
        Per study, one dict per run (unit stress).
    """
    mcfg = data["mcfg"]
    s8_by = s8.merged_rows(s8.load_config(), [Path(d) for d in cfg["s8_dirs"]])
    out = {}
    for study, recs in data["recs"].items():
        raw = {}
        base = [r for r in mc._rows(mcfg["applied"][study]["rows"]) if "label" in r]
        for r in base + s8_by.get(study, []):
            raw[(r["label"], int(r["budget"]), int(r["seed"]))] = r
        for r in data["s7"].get(study, []):
            key = (f"{r['label']} + S7 layer", int(r["budget_total"]), int(r["seed"]))
            raw[key] = r
        if len(raw) != len(recs):
            raise SystemExit(f"{study}: {len(raw)} raw rows for {len(recs)} records")
        runs = []
        for rec in recs:
            row = raw[(rec["label"], rec["cell"], rec["seed"])]
            check = _design_check(row) if rec["has_witness"] else None
            name, ours = _name(cfg, study, rec["label"])
            tested = " + S7 layer" in rec["label"]
            limit = own_limit(study, check, None)
            runs.append({
                "label": rec["label"], "name": name, "arm": rec["arm"],
                "group": _group(rec["label"], rec["arm"], ours), "cell": rec["cell"],
                "seed": rec["seed"], "L": rec["L"], "U": rec["U"],
                "U_kind": {"certified form": OURS, "claimed": CLAIMED}.get(
                    rec["U_kind"]),
                "design": rec["has_witness"], "passed": rec["verified"],
                "intervals": [] if tested else _intervals(row),
                "limit": limit, "slack": stress_slack(limit, rec["L"])
                if rec["has_witness"] else None,
                **_binding(study, check), "robust": rec["robust"],
            })  # fmt: skip
        out[study] = runs
    return out


def s6b_runs(cfg: dict, data: dict) -> list[dict]:
    """Double-integrator runs (unit stress) with the closed-form own limit."""
    labels = cfg["double_integrator"]["labels"]
    out = []
    for r in data["s6b"]:
        label, x = labels[r["arm"]], r["final_witness_x"]
        st = cfg["arms"][label]
        unc = r["arm"].startswith("uncertified")
        limit = own_limit("s6b", None, x)
        out.append({
            "label": label, "name": st["name"], "arm": r["arm"],
            "group": _group(label, r["arm"], bool(st.get("ours"))),
            "cell": int(r["budget"]), "seed": int(r["seed"]),
            "L": float(r["final_lower"]), "U": float(r["final_upper"]),
            "U_kind": CLAIMED if unc else OURS,
            # the bisection rows report a design but do not store it
            "design": bool(r["reports"][-1]["has_witness"]), "x_stored": x is not None,
            "passed": r["witness_sound"],
            "intervals": [(int(p["raw_evals"]), float(p["lower"]), float(p["upper"]))
                          for p in r["reports"]],
            "limit": limit, "slack": stress_slack(limit, float(r["final_lower"])),
            "binding": None, "binding_se": None, "binding_channel": None,
            "robust": None, "covered_anytime": bool(r["covered_anytime"]),
            "covered_final": bool(r["covered_final"]),
        })  # fmt: skip
    return out


def _cells(runs: list[dict]) -> list[tuple[int, str, list[dict]]]:
    """(budget, label, runs in seed order), budgets ascending, labels as stored."""
    out = []
    for c in sorted({r["cell"] for r in runs}):
        for lab in dict.fromkeys(r["label"] for r in runs if r["cell"] == c):
            g = sorted((r for r in runs if r["cell"] == c and r["label"] == lab),
                       key=lambda r: r["seed"])  # fmt: skip
            out.append((c, lab, g))
    return out


def _head(study: str, cell: int, g: list[dict]) -> dict:
    return {"study": study, "cell": cell, "label": g[0]["label"],
            "name": g[0]["name"], "group": g[0]["group"], "n": len(g)}  # fmt: skip


# ------------------------------------------------------------------- dimensions


def t1_validity(data: dict, runs: dict, theta: dict) -> list[dict]:
    """Designs passing, L <= limit at every report, final [L, U] containing it."""
    out = []
    for study, rs in runs.items():
        if study == "s6b":
            th, _ = theta[study]
            for c, _, g in _cells(rs):
                stored = [r for r in g if r["x_stored"]]
                out.append({**_head(study, c, g),
                            "design": sum(r["design"] for r in g),
                            "design_stored": len(stored),
                            "passed": sum(r["passed"] is True for r in g),
                            "failed_check": sum(r["passed"] is False for r in g),
                            "lower_ok_anytime": sum(
                                all(lo <= th for _, lo, _ in r["intervals"])
                                for r in g),
                            "n_upper": len(g), "upper_kind": g[0]["U_kind"],
                            "upper_covers": sum(r["covered_final"] for r in g),
                            "upper_covers_anytime": sum(r["covered_anytime"]
                                                        for r in g)})  # fmt: skip
            continue
        by = {(r["cell"], r["label"]): r for r in data["comparison"][study]["rows"]}
        for c, lab, g in _cells(rs):
            s = by[(c, lab)]
            out.append({**_head(study, c, g), "design": s["witness"],
                        "passed": s["verified"], "failed_check": s["failed_check"],
                        "lower_ok_anytime": s["lower_ok_anytime"],
                        "n_upper": s["n_upper"],
                        "upper_kind": {"certified form": OURS, "claimed": CLAIMED}.get(
                            s["upper_kind"]),
                        "upper_covers": s["upper_covers"]})  # fmt: skip
    return out


def t2_anytime(runs: dict, theta: dict) -> dict:
    """Claimed intervals of the uncertified methods at every report."""
    rows, recheck = [], {}
    for study, rs in runs.items():
        th, err = theta[study]
        best_found = study == pf.BEST_FOUND
        if study == "s6b":
            mism = [r for r in rs if contains_every_report(
                [(lo, up) for _, lo, up in r["intervals"]], th, 0.0)
                != r["covered_anytime"]]  # fmt: skip
            recheck = {"runs": len(rs), "mismatches": len(mism),
                       "reproduced": not mism}  # fmt: skip
        for c, _, g in _cells(rs):
            if g[0]["group"] != "uncertified":
                continue
            iv = [[(lo, up) for _, lo, up in r["intervals"]] for r in g]
            row = {**_head(study, c, g), "n_reports_median": float(
                np.median([len(v) for v in iv]))}  # fmt: skip
            if best_found:  # an interval wholly below a verified design misses
                row |= {"below_best_found_final": sum(v[-1][1] < th for v in iv),
                        "below_best_found_any": sum(any(up < th for _, up in v)
                                                    for v in iv)}  # fmt: skip
            else:
                row |= {"contains_final": sum(contains(*v[-1], th, err) for v in iv),
                        "contains_every_report": sum(contains_every_report(v, th, err)
                                                     for v in iv)}  # fmt: skip
            rows.append(row)
    return {"rows": rows, "s6b_recheck": recheck}


def t3_slack(runs: dict, buffer_frac: float) -> dict:
    """Stress slack and binding constraint value of every returned design."""
    designs, rows, cross = [], [], {}
    for study, rs in runs.items():
        top = max(r["cell"] for r in rs)
        with_d = [
            r
            for r in rs
            if r["design"]
            and (
                r["limit"] is not None
                or r["binding"] is not None
                or r["robust"] is not None
            )
        ]
        for r in with_d:
            designs.append(
                {
                    k: r[k]
                    for k in (
                        "label",
                        "cell",
                        "seed",
                        "L",
                        "limit",
                        "slack",
                        "passed",
                        "binding",
                        "binding_se",
                        "binding_channel",
                        "robust",
                    )
                }
                | {"study": study}
            )
        tab = {"neg_failed": 0, "neg_passed": 0, "nonneg_failed": 0,
               "nonneg_passed": 0}  # fmt: skip
        for r in with_d:
            if r["slack"] is None or r["passed"] is None:
                continue
            tab[
                ("neg_" if r["slack"] < 0 else "nonneg_")
                + ("passed" if r["passed"] else "failed")
            ] += 1
        if any(tab.values()):
            cross[study] = tab
        for c, _, g in _cells(rs):
            if c != top:
                continue
            d = [r for r in g if r["design"]]
            sl = [r["slack"] for r in d if r["slack"] is not None]
            b = [r for r in d if r["binding"] is not None]
            worst = max(b, key=lambda r: r["binding"]) if b else None
            rob = [r["robust"] for r in d if r["robust"] is not None]
            stored = sum(r["limit"] is not None or r["binding"] is not None
                         or r["robust"] is not None for r in d)  # fmt: skip
            rows.append({**_head(study, c, g), "designs": len(d), "stored": stored,
                         "with_limit": len(sl), "slack": _q(sl),
                         "slack_below_0": sum(s < -EPS for s in sl),
                         "slack_below_buffer": sum(s < buffer_frac for s in sl),
                         "binding": _q([r["binding"] for r in b]),
                         "binding_max_se": worst and worst["binding_se"],
                         "binding_max_channel": worst and worst["binding_channel"],
                         "robust": _q(rob), "robust_all": sum(x == 1.0 for x in rob)
                         if rob else None})  # fmt: skip
    return {"rows": rows, "crosstab_all_budgets": cross, "designs": designs}


def t4_variability(runs: dict, theta: dict) -> list[dict]:
    """Spread of the reported level and of the bracket across seeds."""
    out = []
    for study, rs in runs.items():
        th, _ = theta[study]
        for c, _, g in _cells(rs):
            gap = [th - r["L"] for r in g]
            worst = max(g, key=lambda r: abs(th - r["L"]))
            row = {
                **_head(study, c, g),
                "designs": sum(r["design"] for r in g),
                "level": _q([r["L"] for r in g]),
                "gap": _q(gap),
                "worst": {
                    "seed": worst["seed"],
                    "L": worst["L"],
                    "gap": th - worst["L"],
                    "design": worst["design"],
                },
            }
            u = [r for r in g if r["U"] is not None]
            if u:
                wide = max(u, key=lambda r: r["U"] - r["L"])
                row["bracket"] = {
                    "n": len(u),
                    "kind": u[0]["U_kind"],
                    "width": _q([r["U"] - r["L"] for r in u]),
                    "widest_seed": wide["seed"],
                    "upper_minus_theta_max": max(r["U"] - th for r in u),
                }
            out.append(row)  # fmt: skip
    return out


def t5_targets(runs: dict, theta: dict, targets: dict) -> dict:
    """Target decisions at the final report, scored against the reference.

    Args:
        runs: Per study, the run records (unit stress).
        theta: Per study, (reference limit, stated error) in unit stress.
        targets: Per study, (physical, unit) target stresses.

    Returns:
        One row per (study, budget, method), and every wrong "not achievable".
    """
    rows, ours_wrong = [], []
    for study, rs in runs.items():
        th, err = theta[study]
        physical = {t: p for p, t in targets[study]}
        truth = {t: target_truth(t, th, err, study == pf.BEST_FOUND)
                 for t in physical}  # fmt: skip
        scored = [t for t, v in truth.items() if v is not None]
        for c, _, g in _cells(rs):
            counts = {"achievable_correct": 0, "achievable_wrong": 0,
                      "achievable_passing": 0, "achievable_failing": 0,
                      "achievable_unchecked": 0, "achievable_correct_passing": 0,
                      "not_correct": 0, "not_wrong": 0, "undecided": 0}  # fmt: skip
            per_t = []
            for t in scored:
                pt = {"target": t, "target_physical": physical[t], "truth": truth[t],
                      ACHIEVABLE: 0, NOT_ACHIEVABLE: 0, UNDECIDED: 0}  # fmt: skip
                for r in g:
                    a = target_answer(r["L"], r["design"], r["U"], r["U_kind"], t)
                    pt[a] += 1
                    if a == UNDECIDED:
                        counts["undecided"] += 1
                        continue
                    ok = truth[t] == (a == ACHIEVABLE)
                    key = "achievable" if a == ACHIEVABLE else "not"
                    counts[f"{key}_{'correct' if ok else 'wrong'}"] += 1
                    if a == ACHIEVABLE:
                        counts[
                            {
                                True: "achievable_passing",
                                False: "achievable_failing",
                                None: "achievable_unchecked",
                            }[r["passed"]]
                        ] += 1
                        if ok and r["passed"] is True:
                            counts["achievable_correct_passing"] += 1
                    elif not ok:
                        ours_wrong.append({"study": study, "label": r["label"],
                                           "cell": c, "seed": r["seed"], "target": t,
                                           "U": r["U"]})  # fmt: skip
                per_t.append(pt)
            n = len(g) * len(scored)
            rows.append({**_head(study, c, g), "targets": scored, "answers": n,
                         "correct": counts["achievable_correct"]
                         + counts["not_correct"],
                         "wrong": counts["achievable_wrong"] + counts["not_wrong"],
                         **counts, "per_target": per_t})  # fmt: skip
    return {"rows": rows, "ours_not_achievable_wrong": ours_wrong}


def t5_stopping(runs: list[dict], theta: float, tols: list[float]) -> list[dict]:
    """Double integrator: stop at the first report with U - L <= tol (exact limit)."""
    out = []
    th = theta
    for c, _, g in _cells(runs):
        for tol in tols:
            stops = [(r, first_stop(r["intervals"], tol)) for r in g]
            hit = [(r, s) for r, s in stops if s is not None]
            out.append({**_head("s6b", c, g), "tol": tol, "upper_kind": g[0]["U_kind"],
                        "stopped": len(hit),
                        "raw_at_stop": _q([s[0] for _, s in hit]),
                        "contains_at_stop": sum(s[1] <= th <= s[2] for _, s in hit),
                        "misses_at_stop": sum(not s[1] <= th <= s[2]
                                              for _, s in hit)})  # fmt: skip
    return out


def acceptance(result: dict, runs: dict) -> dict:
    """Handoff acceptance checks that the stored data can falsify."""
    s6b_bad = [r["seed"] for r in runs["s6b"]
               if r["passed"] is True and r["slack"] is not None
               and r["slack"] < -EPS]  # fmt: skip
    s2_bad = [(r["label"], r["cell"], r["seed"], r["slack"]) for r in runs["s2"]
              if r["passed"] is True and r["slack"] is not None
              and r["slack"] < -EPS]  # fmt: skip
    return {
        "t2_s6b_covered_anytime_reproduced": result["t2"]["s6b_recheck"]["reproduced"],
        "t3_s6b_sound_designs_with_negative_slack": s6b_bad,
        "t3_s2_passing_designs_with_negative_slack": s2_bad,
        "t5_ours_not_achievable_all_correct": not result["t5_targets"][
            "ours_not_achievable_wrong"
        ],
    }


# --------------------------------------------------------------------- markdown


def _f(cfg: dict, study: str, v: float | None, delta: bool = False) -> str:
    """Physical value (or difference) of unit stress ``v`` as text."""
    if v is None:
        return "-"
    m = pf.study_meta(cfg, study)
    x = m["scale"] * v if delta else m["offset"] + m["scale"] * v
    return f"{x:.{m['digits']}f}" + ("%" if study == "s3" else "")


def _iqr(cfg: dict, study: str, q: dict | None, delta: bool = False) -> str:
    if not q:
        return "-"
    return (
        f"{_f(cfg, study, q['median'], delta)} "
        f"({_f(cfg, study, q['q25'], delta)} to {_f(cfg, study, q['q75'], delta)})"
    )


def _table(header: list[str], rows: list[list]) -> list[str]:
    return mc._table(header, rows) + [""]


def markdown(cfg: dict, ecfg: dict, result: dict) -> str:
    """Tables of every dimension, in physical units, each count with its runs."""
    order = result["studies"]
    short = {s: pf.study_meta(cfg, s)["short"] for s in order}
    lines = ["# Quick evaluation dimensions (stored runs only; observed, not proved)",
             "", "Reference limits (physical): " + "; ".join(
                 f"{short[s]} {_f(cfg, s, t['value'])} ± {_f(cfg, s, t['err'], True)}"
                 + (" (best verified design, a lower bound)" if s == pf.BEST_FOUND
                    else "") for s, t in result["reference"].items()), ""]  # fmt: skip
    lines += ["## T1 Validity", ""]
    rows = []
    for r in result["t1"]:
        up, every = "-", "-"
        if r["n_upper"]:
            up = f"{r['upper_covers']}/{r['n_upper']} ({r['upper_kind']})"
            every = "end of run only" if r["upper_kind"] == OURS else "see T2"
        if "upper_covers_anytime" in r:
            every = f"{r['upper_covers_anytime']}/{r['n']}"
        passed = (f"{r['passed']}/{r['n']}" if r.get("design_stored", r["n"])
                  else "not stored")  # fmt: skip
        rows.append([short[r["study"]], f"{r['cell']:,}", r["name"], r["group"],
                     passed, f"{r['lower_ok_anytime']}/{r['n']}", up,
                     every])  # fmt: skip
    lines += _table(["Study", "Budget", "Method", "Group", "Design passes",
                     "L ≤ θ, every report", "Final [L, U] contains θ (U by)",
                     "Contains θ, every report"], rows)  # fmt: skip
    lines += ["## T2 Claimed intervals of the uncertified methods, every report", ""]
    rows = []
    for r in result["t2"]["rows"]:
        n = r["n"]
        if "contains_final" in r:
            fin = f"{r['contains_final']}/{n}"
            every = f"{r['contains_every_report']}/{n}"
        else:  # Hopper: only a miss below the best verified design is scoreable
            k = r["below_best_found_final"]
            fin = f"wholly below the best verified design: {k}/{n}"
            every = f"below it at some report: {r['below_best_found_any']}/{n}"
        rows.append([short[r["study"]], f"{r['cell']:,}", r["name"], fin, every,
                     f"{r['n_reports_median']:.0f}"])  # fmt: skip
    lines += _table(
        [
            "Study",
            "Budget",
            "Method",
            "Contains θ, final",
            "Contains θ, every report",
            "Reports (median)",
        ],
        rows,
    )
    rc = result["t2"]["s6b_recheck"]
    lines += [
        f"Double integrator `covered_anytime` recomputed from `reports[]`: "
        f"{rc['runs'] - rc['mismatches']}/{rc['runs']} runs match.",
        "",
    ]
    buf = ecfg["slack_buffer_frac"]
    lines += [f"## T3 Stress slack and binding constraint value (top budget; buffer "
              f"{100 * buf:g} % of the stress range)", ""]  # fmt: skip
    rows = []
    for r in result["t3"]["rows"]:
        s = r["study"]
        b = r["binding"]
        bind = "-" if not b else (
            f"{b['median']:.4f} / {b['max']:.4f}"
            + (f" ± {r['binding_max_se']:.4f}" if r["binding_max_se"] else "")
            + f" ({r['binding_max_channel']})")  # fmt: skip
        rob = (
            "-"
            if r["robust"] is None
            else (
                f"{r['robust']['median']:.3f}; all 8: {r['robust_all']}/{r['designs']}"
            )
        )
        sl = r["slack"]
        nd = f"{r['designs']}/{r['n']}"
        if r["stored"] < r["designs"]:
            nd += f" ({r['stored']} stored)"
        rows.append([short[s], f"{r['cell']:,}", r["name"], nd,
                     _iqr(cfg, s, sl, True),
                     f"{r['slack_below_0']}/{r['with_limit']}" if sl else "-",
                     f"{r['slack_below_buffer']}/{r['with_limit']}" if sl else "-",
                     bind, rob])  # fmt: skip
    lines += _table(
        [
            "Study",
            "Budget",
            "Method",
            "Designs/runs",
            "Slack median (IQR)",
            "Slack < 0",
            "Slack < buffer",
            "Binding value median / max (channel)",
            "Robust (Hopper)",
        ],
        rows,
    )
    for s, t in result["t3"]["crosstab_all_budgets"].items():
        lines.append(f"* {short[s]}, all budgets and methods: slack < 0 with a failed "
                     f"check {t['neg_failed']}, slack < 0 with a passed check "
                     f"{t['neg_passed']}, slack ≥ 0 with a failed check "
                     f"{t['nonneg_failed']}, slack ≥ 0 with a passed check "
                     f"{t['nonneg_passed']}.")  # fmt: skip
    lines += ["", "## T4 Variability across seeds", ""]
    rows = []
    for r in result["t4"]:
        s, w, br = r["study"], r["worst"], r.get("bracket")
        rows.append([short[s], f"{r['cell']:,}", r["name"], r["n"],
                     _iqr(cfg, s, r["level"]),
                     f"{_f(cfg, s, r['level']['min'])} to "
                     f"{_f(cfg, s, r['level']['max'])}",
                     _f(cfg, s, r["gap"]["median"], True),
                     f"{_f(cfg, s, w['gap'], True)} (seed {w['seed']}"
                     + ("" if w["design"] else ", no design") + ")",
                     "-" if not br else _f(cfg, s, br["width"]["median"], True),
                     "-" if not br else f"{_f(cfg, s, br['width']['max'], True)} "
                     f"(seed {br['widest_seed']})",
                     "-" if not br else _f(cfg, s, br["upper_minus_theta_max"], True),
                     ])  # fmt: skip
    lines += _table(
        [
            "Study",
            "Budget",
            "Method",
            "Runs",
            "L median (IQR)",
            "L min to max",
            "Median θ − L",
            "Worst θ − L (seed)",
            "Median U − L",
            "Max U − L (seed)",
            "Max U − θ",
        ],
        rows,
    )
    lines += ["## T5 (a) Target decisions at the final report", ""]
    for s in order:
        tg = [r for r in result["t5_targets"]["rows"] if r["study"] == s]
        if tg:
            lines.append(f"* {short[s]} scored targets: " + ", ".join(
                _f(cfg, s, t) for t in tg[0]["targets"]))  # fmt: skip
    lines.append("")
    rows = []
    for r in result["t5_targets"]["rows"]:
        a = r["achievable_correct"] + r["achievable_wrong"]
        rows.append([short[r["study"]], f"{r['cell']:,}", r["name"], r["answers"],
                     r["correct"], r["wrong"], r["undecided"],
                     f"{a} ({r['achievable_wrong']} wrong; passing design "
                     f"{r['achievable_passing']}, failing {r['achievable_failing']})",
                     f"{r['not_correct'] + r['not_wrong']} ({r['not_wrong']} wrong)",
                     ])  # fmt: skip
    lines += _table(["Study", "Budget", "Method", "Answers (runs × targets)",
                     "Correct", "Wrong", "Undecided", "Achievable answers",
                     "Not-achievable answers"], rows)  # fmt: skip
    lines += [
        "Answers per target at the largest budget (A achievable, N not "
        "achievable; ✓ the reference says achievable, ✗ not):",
        "",
    ]
    for s in order:
        tg = [r for r in result["t5_targets"]["rows"] if r["study"] == s]
        top = [r for r in tg if r["cell"] == max(x["cell"] for x in tg)]
        head = [f"{_f(cfg, s, p['target'])} {'✓' if p['truth'] else '✗'}"
                for p in top[0]["per_target"]]  # fmt: skip
        cells = [[" ".join(f"{p[k]}{tag}" for k, tag in ((ACHIEVABLE, "A"),
                                                        (NOT_ACHIEVABLE, "N"))
                           if p[k]) or "-" for p in r["per_target"]]
                 for r in top]  # fmt: skip
        lines += _table(
            [f"{short[s]}, {top[0]['cell']:,}", *head],
            [[r["name"], *c] for r, c in zip(top, cells, strict=True)],
        )
    lines += ["## T5 (b) Double integrator: stop when U − L ≤ tol", ""]
    rows = []
    for r in result["t5_stopping"]:
        q = r["raw_at_stop"]
        rows.append([f"{r['cell']:,}", r["name"], r["upper_kind"],
                     f"{100 * r['tol']:g} %", f"{r['stopped']}/{r['n']}",
                     "-" if not q else f"{q['median']:.0f}",
                     f"{r['contains_at_stop']}/{r['stopped']}",
                     f"{r['misses_at_stop']}/{r['stopped']}"])  # fmt: skip
    lines += _table(["Budget", "Method", "U by", "tol (of range)", "Stopped",
                     "Median evaluations at stop", "Contains θ at stop",
                     "Misses θ at stop"], rows)  # fmt: skip
    lines += ["## Acceptance checks", ""]
    lines += [f"* `{k}`: {v}" for k, v in result["acceptance"].items()]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- main


def compute(cfg: dict, ecfg: dict, data: dict) -> dict:
    """Every dimension from the loaded stored runs."""
    runs = {"s6b": s6b_runs(cfg, data), **applied_runs(cfg, data)}
    studies = cfg["cross_study"]["deterministic"] + cfg["cross_study"]["noisy"]
    runs = {s: sorted(runs[s], key=lambda r, s=s: _rank(cfg, s, r)) for s in studies}
    theta = {s: pf.study_truth(data, s) for s in studies}
    targets = {}
    for s in studies:
        m = pf.study_meta(cfg, s)
        targets[s] = [(t, (t - m["offset"]) / m["scale"]) for t in ecfg["targets"][s]]
    result = {
        "studies": studies,
        "reference": {s: {"value": theta[s][0], "err": theta[s][1]} for s in studies},
        "t1": t1_validity(data, runs, theta),
        "t2": t2_anytime(runs, theta),
        "t3": t3_slack(runs, ecfg["slack_buffer_frac"]),
        "t4": t4_variability(runs, theta),
        "t5_targets": t5_targets(runs, theta, targets),
        "t5_stopping": t5_stopping(
            runs["s6b"], theta["s6b"][0], ecfg["stop_tolerance_frac"]
        ),
    }
    result["acceptance"] = acceptance(result, runs)
    return result


def main() -> None:
    """Write ``results/<UTC>_evaluation_dimensions/`` (JSON, Markdown, manifest)."""
    commit, dirty = git_state()
    ecfg = load_config()
    cfg = pf.load_config(Path(ecfg["paper_figures_config"]))
    data = pf.load_data(cfg)
    result = compute(cfg, ecfg, data)
    out = Path("results") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_evaluation_dimensions"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({
        "experiment": "evaluation_dimensions (no new runs)", "git_commit": commit,
        "git_dirty": dirty, "config_sha256": ecfg["sha256"],
        "paper_figures_config_sha256": cfg["sha256"],
        "comparison_config_sha256": data["mcfg"]["sha256"],
        "python": platform.python_version(), "numpy": np.__version__,
        "inputs": pf.provenance(pf.input_files(cfg, data["mcfg"])),
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }, indent=1) + "\n")  # fmt: skip
    (out / "dimensions.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "dimensions.md").write_text(markdown(cfg, ecfg, result))
    for k, v in result["acceptance"].items():
        log.info("acceptance %s: %s", k, v)
    log.info("dimensions -> %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    main()
