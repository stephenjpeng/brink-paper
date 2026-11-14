"""Figures and tables of the L4DC paper from stored run rows; nothing reruns.

Inputs: ``configs/runs.yaml`` names the runs (study sweeps, check-layer
sweeps, GP-UCB and differential-evolution runs, sampled-design thresholds, and
the double-integrator sweep); ``configs/evaluation_dimensions.yaml`` holds the
target stresses of ``fig_decisions``. Display names, colors, and stress units
come from ``configs/paper_figures.yaml``, so every exhibit uses one label set.

Outputs, sized for the PMLR text width (6.0 in) with 7 to 8 pt Times text:

* ``exhibits/figures/<name>.pdf`` (vector, TrueType fonts embedded);
* ``exhibits/tables/<name>.tex`` (booktabs ``tabular`` bodies; captions are in
  the paper);
* ``exhibits/figures/manifest.json``: git state, config SHA-256, SHA-256 and
  run-manifest commit of every input file, package versions, and per exhibit
  the numbers its caption may quote (``facts``).

Stress levels are shown in physical units (``offset + scale * c``). Before its first
report a run's level is the domain minimum c = 0, which asserts nothing.

Usage:
    uv run python scripts/paper_figures.py [--only NAME ...] [--preview DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator, NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_baselines_sweep as s8  # noqa: E402
import method_comparison as mc  # noqa: E402
from common import git_state, run_path  # noqa: E402

from brink.studies.double_integrator import THETA_EXISTS_UNIT  # noqa: E402

log = logging.getLogger("paper_figures")
CFG_PATH = Path("configs/paper_figures.yaml")
TEXT_WIDTH_IN = 6.0  # l4dc2027.cls (jmlr, pmlr): \textwidth = 433.62 pt
NAIVE = mc.NAIVE  # the single-check bisection arm of each study
REPLICATED_16 = {"s3": "baseline-k256", "s5": "baseline-k16"}
BEST_FOUND = "s4"  # Hopper's reference is the best verified design, not a limit
RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "lines.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 2.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}
REF_COLOR, REF_BAND, FAIL_FILL = "#000000", "#C8C8C8", "#FBE3D6"
BEST_LS, THRESH_LS = (0, (4, 2)), (0, (1, 1.2))


def load_config(path: Path = CFG_PATH) -> dict:
    """Display settings (YAML, SHA-256 under ``sha256``) plus the run inputs."""
    text = Path(path).read_text()
    cfg = {**yaml.safe_load(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    s8_dirs = run_path("adaptive_baselines")
    cfg["s8_dirs"] = [
        str(d) for d in (s8_dirs if isinstance(s8_dirs, list) else [s8_dirs])
    ]
    cfg["columns"] = {
        s: str(run_path(f"{name}_columns"))
        for s, name in (("s2", "transmission"), ("s3", "feeder"), ("s4", "hopper"),
                        ("s5", "covasim"))
    }  # fmt: skip
    cfg["double_integrator"]["dir"] = str(run_path("double_integrator_sweep"))
    return cfg


# ----------------------------------------------------------------------------- data


def load_data(cfg: dict) -> dict:
    """Normalized run records, truths, comparison summaries, columns, and S7 rows."""
    mcfg = mc.load_config(Path(cfg["comparison_config"]))
    s8_by = s8.merged_rows(s8.load_config(), [Path(d) for d in cfg["s8_dirs"]])
    recs = mc.applied_records(mcfg, s8_by)
    columns = {}
    for s, d in cfg["columns"].items():
        d = Path(d)
        if s == "s5":
            res = json.loads((d / "result.json").read_text())
            scores, by_eps = [float(r["score"]) for r in res["rows"]], res["by_eps"]
        elif s == "s4":
            scores = np.load(d / "columns.npz")["scores"].tolist()
            meta = json.loads((d / "columns.json").read_text())
            by_eps = {k[4:]: {"U_eps": v["U"], "k": v["k"]}
                      for k, v in meta.items() if k.startswith("eps_")}  # fmt: skip
        else:
            scores = [float(r["score"]) for r in mc._rows([d / "rows.jsonl"])]
            by_eps = json.loads((d / "columns.json").read_text())["by_eps"]
        columns[s] = {"scores": np.asarray(scores, dtype=float), "by_eps": by_eps}
    return {
        "mcfg": mcfg,
        "recs": recs,
        "truth": {s: mc.applied_truth(mcfg, s) for s in recs},
        "comparison": mc.compare_applied(mcfg, recs),
        "columns": columns,
        "s7": {s: mc._rows([p]) for s, p in mcfg["applied"]["s7"].items()},
        "s6b": mc._rows([Path(cfg["double_integrator"]["dir"]) / "rows.jsonl"]),
    }


def input_files(cfg: dict, mcfg: dict) -> list[Path]:
    """Every stored file an exhibit reads."""
    a = mcfg["applied"]
    files = [
        Path(cfg["comparison_config"]),
        Path(cfg["evaluation_config"]),
        s8.CFG_PATH,
    ]
    for s in ("s2", "s3", "s4", "s5"):
        files += [Path(p) for p in a[s]["rows"] + a[s].get("upper_rows", [])]
        if "truth_gate" in a[s]:
            files.append(Path(a[s]["truth_gate"]))
    files += [Path(p) for p in a["s7"].values()]
    files += [Path(d) / "rows.jsonl" for d in cfg["s8_dirs"]]
    names = {"s5": ["result.json"], "s4": ["columns.npz", "columns.json"]}
    for s, d in cfg["columns"].items():
        files += [Path(d) / n for n in names.get(s, ["rows.jsonl", "columns.json"])]
    di = Path(cfg["double_integrator"]["dir"])
    return files + [di / "rows.jsonl", di / "gate.json"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(files: list[Path]) -> list[dict]:
    """SHA-256 of each input and the git state recorded by its run manifest."""
    out = []
    for f in files:
        man = f.parent / "manifest.json"
        m = json.loads(man.read_text()) if f.parent.name and man.exists() else {}
        out.append({"path": str(f), "sha256": _sha256(f),
                    "run_git_commit": m.get("git_commit"),
                    "run_git_dirty": m.get("git_dirty")})  # fmt: skip
    return out


def stress(cfg: dict, study: str, c):
    """Physical stress level of unit stress ``c``."""
    s = cfg["studies"][study]
    return s["offset"] + s["scale"] * np.asarray(c, dtype=float)


def arm_style(cfg: dict, study: str, label: str) -> dict:
    """Display name, color, marker, and line style of one stored run label."""
    st = dict(cfg["arms"].get(f"{label}@{study}") or cfg["arms"][label])
    if isinstance(st["ls"], list):
        st["ls"] = (0, tuple(st["ls"]))
    return st


def tex_name(name: str) -> str:
    """LaTeX form of a display name (ASCII macros only)."""
    for k in ("16x", "4x"):
        name = name.replace(k, k[:-1] + r"\ensuremath{\times}")
    return name


def plot_name(name: str) -> str:
    """Figure form of a display name."""
    return name.replace("16x", "16\u00d7").replace("4x", "4\u00d7")


def display_name(st: dict) -> str:
    """Figure form of an arm's display name (the paper's own arms are named BRINK)."""
    return plot_name(st["name"])


def group(recs: list[dict], label: str, cell: int) -> list[dict]:
    """Runs of one stored label at one budget, seed order."""
    return sorted((r for r in recs if r["label"] == label and r["cell"] == cell),
                  key=lambda r: r["seed"])  # fmt: skip


def top_budget(recs: list[dict]) -> int:
    """Largest budget of the proposed search in a study."""
    return max(r["cell"] for r in recs if r["label"] == "witness-focused")


def step_quantiles(
    runs: list[list[tuple[int, float]]],
    grid: np.ndarray,
    qs=(0.25, 0.5, 0.75),
    before: float = 0.0,
) -> np.ndarray:
    """Quantiles across runs of the last value each run reported by each count.

    ``runs`` holds one list of (raw evaluations, reported value) per run. A run that
    has not reported by a grid count contributes ``before`` (for a level, the domain
    minimum 0; for a width, the full range).
    """
    vals = np.zeros((len(runs), len(grid)))
    for i, pts in enumerate(runs):
        raw = np.array([p[0] for p in pts], dtype=float)
        lev = np.array([p[1] for p in pts], dtype=float)
        order = np.argsort(raw, kind="stable")
        raw, lev = raw[order], lev[order]
        idx = np.searchsorted(raw, grid, side="right") - 1
        vals[i] = np.where(idx >= 0, lev[np.clip(idx, 0, None)], before)
    return np.quantile(vals, qs, axis=0)


def survival(scores: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Fraction of sampled designs whose computed limit is at least each ``c``."""
    s = np.sort(scores)
    return 1.0 - np.searchsorted(s, c, side="left") / len(s)


def kth_largest(scores: np.ndarray, k: int) -> float:
    """The k-th largest computed design limit (the feasible-fraction threshold)."""
    return float(np.sort(scores)[::-1][k - 1])


def _threshold(cols: dict, eps: float) -> dict:
    """Stored threshold at ``eps``, checked against the stored design limits."""
    e = cols["by_eps"][f"{eps}"]
    if abs(kth_largest(cols["scores"], int(e["k"])) - float(e["U_eps"])) > 1e-12:
        raise SystemExit(f"stored U_{eps} is not the k-th largest design limit")
    return {"U_eps": float(e["U_eps"]), "k": int(e["k"])}


# ------------------------------------------------------------------------- drawing


def _reference(ax, cfg, data, study, axis="y") -> float:
    theta, err, _ = data["truth"][study]
    v, lo, hi = stress(cfg, study, [theta, theta - err, theta + err])
    span, line = (ax.axhspan, ax.axhline) if axis == "y" else (ax.axvspan, ax.axvline)
    if err > 0:
        span(lo, hi, color=REF_BAND, lw=0, zorder=0)
    line(v, color=REF_COLOR, lw=0.8, zorder=1, ls=BEST_LS if study == BEST_FOUND
         else "-")  # fmt: skip
    return float(v)


def _ref_entries(limit=True, best=False, thresholds=False) -> list[tuple]:
    out = []
    if limit:
        out.append(((Patch(color=REF_BAND, lw=0), Line2D([], [], color=REF_COLOR,
                                                         lw=0.8)),
                    "Operating limit (reference)"))  # fmt: skip
    if best:
        out.append((Line2D([], [], color=REF_COLOR, lw=0.8, ls=BEST_LS),
                    "Reference design (Hopper)"))  # fmt: skip
    if thresholds:
        out.append((Line2D([], [], color="#555555", lw=0.8, ls=THRESH_LS),
                    r"Feasible-fraction threshold $U_\varepsilon$"))  # fmt: skip
    return out


def _arm_entries(cfg, items, mode: str) -> list[tuple]:
    """Legend entries (handle, label); mode is 'line', 'marker', or 'both'."""
    out = []
    for study, lab in items:
        st = arm_style(cfg, study, lab)
        kw = {"color": st["color"], "lw": 1.1}
        kw["ls"] = st["ls"] if mode != "marker" else "none"
        if mode != "line":
            kw.update(marker=st["marker"], ms=3.8, mec="white", mew=0.3)
        out.append((Line2D([], [], **kw), display_name(st)))
    return out


def _legend(fig, entries, ncol: int, loc="outside lower center"):
    fig.legend([h for h, _ in entries], [lab for _, lab in entries], loc=loc,
               ncol=ncol, columnspacing=0.9, handletextpad=0.4, handlelength=2.0,
               handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})  # fmt: skip


def _panel(ax, letter: str, title: str):
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=8, pad=3)


def _anytime(ax, cfg, data, study, labels, cell, band_alpha=0.13) -> dict:
    """Median (line) and interquartile band of the level reported by each count.

    Quantiles are across the runs of each label at budget ``cell``.
    """
    recs = data["recs"][study]
    grid = np.geomspace(100, cell, 400)
    facts = {}
    for lab in labels:
        g = group(recs, lab, cell)
        q25, q50, q75 = stress(cfg, study, step_quantiles([r["points"] for r in g],
                                                           grid))  # fmt: skip
        st = arm_style(cfg, study, lab)
        ax.fill_between(grid, q25, q75, color=st["color"], alpha=band_alpha, lw=0)
        ax.plot(grid, q50, color=st["color"], ls=st["ls"], lw=1.1)
        facts[lab] = {"n": len(g), "final_median": float(q50[-1]),
                      "final_q25": float(q25[-1]),
                      "final_q75": float(q75[-1])}  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(100, cell)
    return facts


def _check_scatter(ax, cfg, data, study, labels, cell, pass_x=0.97,
                   pass_ha="right") -> dict:  # fmt: skip
    """Each returned design's reported level against its check value.

    The check value is the largest channel mean of the study's independent check of
    the design at that level; at most 0 meets the requirements.
    """
    recs = data["recs"][study]
    facts = {}
    for lab in labels:
        g_all = group(recs, lab, cell)
        g = [r for r in g_all if r["check_max_mean"] is not None]
        st = arm_style(cfg, study, lab)
        x = stress(cfg, study, [r["L"] for r in g])
        y = np.array([r["check_max_mean"] for r in g])
        ax.scatter(x, y, s=10, marker=st["marker"], facecolor=st["color"],
                   edgecolor="white", linewidth=0.3, alpha=0.9, zorder=3)  # fmt: skip
        facts[lab] = {"n": len(g_all), "n_design": len(g),
                      "n_pass": int(sum(r["verified"] is True for r in g)),
                      "median_level": float(np.median(x)) if g else None,
                      "median_check": float(np.median(y)) if g else None}  # fmt: skip
    ax.axhline(0.0, color="#444444", lw=0.6, zorder=1)
    y0, y1 = ax.get_ylim()
    ax.axhspan(0.0, y1, color=FAIL_FILL, lw=0, zorder=0)
    ax.set_ylim(y0, y1)
    ax.text(0.03, 0.96, "fails check", transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="#A33A00", style="italic")  # fmt: skip
    ax.text(pass_x, 0.04, "passes check", transform=ax.transAxes, ha=pass_ha,
            va="bottom", fontsize=7, color="#333333", style="italic")  # fmt: skip
    return facts


def fig_transmission(cfg: dict, data: dict):
    """S2: level reported during search, and the variance-seeking bounds per run."""
    study, recs = "s2", data["recs"]["s2"]
    top = top_budget(recs)
    labels = ["witness-focused", "greedy", "sup-bisection", "direct-score", "gp-ucb",
              "diff-evo"]  # fmt: skip
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.2), sharey=True,
                               width_ratios=[1.15, 1],
                               layout="constrained")  # fmt: skip
    facts = {"budget": top, "theta_stress": _reference(a, cfg, data, study)}
    facts["anytime"] = _anytime(a, cfg, data, study, labels, top)
    a.set_xlabel("Power flows evaluated")
    a.set_ylabel(cfg["studies"][study]["stress"])
    _panel(a, "a", "Level reported during search")

    _reference(b, cfg, data, study)
    theta, err, _ = data["truth"][study]
    st = arm_style(cfg, study, "greedy")
    x0, ticks, facts["bounds"] = 0, [], {}
    for cell in sorted({r["cell"] for r in recs if r["label"] == "greedy"}):
        g_all = group(recs, "greedy", cell)
        g = sorted((r for r in g_all if r["U"] is not None),
                   key=lambda r: (r["U"], r["L"]))  # fmt: skip
        lo = stress(cfg, study, [r["L"] for r in g])
        hi = stress(cfg, study, [r["U"] for r in g])
        xs = x0 + np.arange(len(g))
        b.vlines(xs, lo, hi, color=st["color"], lw=0.7, zorder=3)
        cover = int(sum(r["L"] <= theta + err and r["U"] >= theta - err for r in g))
        b.text(x0 + (len(g) - 1) / 2, hi.max() + 0.0012,
               f"{cover}/{len(g_all)} contain the limit", ha="center", va="bottom",
               fontsize=7)  # fmt: skip
        facts["bounds"][cell] = {
            "n": len(g_all), "n_with_U": len(g), "contain": cover,
            "median_L": float(np.median(lo)), "median_U": float(np.median(hi)),
            "median_width_c": float(np.median([r["U"] - r["L"] for r in g])),
        }  # fmt: skip
        ticks.append((x0 + (len(g) - 1) / 2, f"{cell:,} power flows"))
        x0 += len(g) + 8
    facts["thresholds"] = {}
    for eps in cfg["studies"][study]["eps"]:
        t = _threshold(data["columns"][study], eps)
        u = float(stress(cfg, study, t["U_eps"]))
        b.axhline(u, color="#555555", lw=0.8, ls=THRESH_LS, zorder=2)
        b.text(x0 - 7.5, u + 0.0008, rf"$U_{{{eps:g}}}$", fontsize=7, ha="right",
               va="bottom", color="#333333")  # fmt: skip
        facts["thresholds"][f"{eps:g}"] = {
            **t,
            "U_eps_stress": u,
            "n": len(data["columns"][study]["scores"]),
        }
    b.set_xticks([t for t, _ in ticks], [s for _, s in ticks])
    b.tick_params(axis="x", length=0)
    b.set_xlim(-3, x0 - 5)
    a.set_ylim(1.188, 1.246)
    _panel(b, "b", "Bounds per run, variance-seeking rule")
    _legend(fig, _arm_entries(cfg, [(study, lab) for lab in labels], "line")
            + _ref_entries(thresholds=True), ncol=4)  # fmt: skip
    return fig, facts


def fig_noisy_audit(cfg: dict, data: dict):
    """S3 and S5: reported level during search, and returned designs on the check."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.35),
                                  layout="constrained")  # fmt: skip
    s3 = ["witness-focused", "baseline-k16", "baseline-k64", "baseline-k256",
          "gp-ucb", "diff-evo"]  # fmt: skip
    s5 = ["witness-focused", "greedy", "config", "baseline-k1", "baseline-k4",
          "baseline-k16", "gp-ucb", "diff-evo"]  # fmt: skip
    top3, top5 = top_budget(data["recs"]["s3"]), top_budget(data["recs"]["s5"])
    facts = {"budgets": {"s3": top3, "s5": top5}}
    _reference(a, cfg, data, "s3")
    facts["s3_anytime"] = _anytime(a, cfg, data, "s3", s3, top3)
    a.set_xlabel("Pass/fail checks evaluated")
    a.set_ylabel(cfg["studies"]["s3"]["stress"])
    a.set_ylim(0, 33)
    _panel(a, "a", "Feeder, during search")

    b.set_xlim(0, 33)
    b.set_ylim(-0.04, 0.245)
    _reference(b, cfg, data, "s3", axis="x")
    facts["s3_check"] = _check_scatter(b, cfg, data, "s3", s3, top3)
    b.set_xlabel(cfg["studies"]["s3"]["stress"])
    b.set_ylabel("Largest mean constraint value")
    _panel(b, "b", "Feeder, returned designs")

    c.set_xlim(0.7, 3.45)
    c.set_ylim(-1.0, 0.5)
    _reference(c, cfg, data, "s5", axis="x")
    facts["s5_check"] = _check_scatter(c, cfg, data, "s5", s5, top5, 0.27, "left")
    c.set_xlabel(cfg["studies"]["s5"]["stress"])
    _panel(c, "c", "Covasim, returned policies")
    items = [("s3", lab) for lab in s3[:4]] + [
        ("s5", "greedy"),
        ("s5", "config"),
        ("s3", "gp-ucb"),
        ("s3", "diff-evo"),
    ]
    _legend(fig, _arm_entries(cfg, items, "both") + _ref_entries(), ncol=4)
    return fig, facts


def fig_anytime(cfg: dict, data: dict):
    """Appendix: level reported during search on Hopper and Covasim."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.25), layout="constrained")
    sets = {
        "s4": ["witness-focused", "greedy", "config", "sup-bisection", "direct-score",
               "gp-ucb", "diff-evo"],
        "s5": ["witness-focused", "greedy", "config", "baseline-k1", "baseline-k4",
               "baseline-k16", "gp-ucb", "diff-evo"],
    }  # fmt: skip
    facts = {}
    for ax, letter, study in zip(axes, "ab", ("s4", "s5"), strict=True):
        top = top_budget(data["recs"][study])
        _reference(ax, cfg, data, study)
        facts[study] = {
            "budget": top,
            **_anytime(ax, cfg, data, study, sets[study], top, 0.10),
        }
        ev = cfg["studies"][study]["evals"]
        ax.set_xlabel(f"{ev[0].upper()}{ev[1:]} evaluated")
        ax.set_ylabel(cfg["studies"][study]["stress"])
        _panel(ax, letter, f"{cfg['studies'][study]['short']}, {top:,} {ev}")
    axes[0].set_ylim(0, 13.5)
    axes[1].set_ylim(0.8, 3.0)
    items = [("s4", lab) for lab in sets["s4"]] + [
        ("s5", "baseline-k4"),
        ("s5", "baseline-k16"),
    ]
    _legend(fig, _arm_entries(cfg, items, "line") + _ref_entries(best=True), ncol=4)
    return fig, facts


def fig_feasible_fraction(cfg: dict, data: dict):
    """Appendix: fraction of sampled designs whose computed limit reaches each level."""
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH_IN, 2.1), layout="constrained")
    facts = {}
    for ax, letter, study in zip(axes, "abcd", ("s2", "s3", "s4", "s5"), strict=True):
        sc = data["columns"][study]["scores"]
        n = len(sc)
        c = np.linspace(0.0, 1.0, 2001)
        v = survival(sc, c)
        keep = v > 0
        ax.step(stress(cfg, study, c)[keep], v[keep], where="post", color="#333333",
                lw=1.0)  # fmt: skip
        _reference(ax, cfg, data, study, axis="x")
        f = {"n": n, "max_limit": float(stress(cfg, study, sc.max())),
             "fraction_at_min": float(v[0]), "thresholds": {}}  # fmt: skip
        for eps in cfg["studies"][study]["eps"]:
            t = _threshold(data["columns"][study], eps)
            u = float(stress(cfg, study, t["U_eps"]))
            ax.plot([u], [t["k"] / n], marker="o", ms=3.5, mfc="white", mec="#000000",
                    mew=0.8, zorder=4)  # fmt: skip
            ax.annotate(rf"$U_{{{eps:g}}}$", (u, t["k"] / n), xytext=(-3, 3),
                        textcoords="offset points", ha="right", fontsize=7)  # fmt: skip
            f["thresholds"][f"{eps:g}"] = {**t, "U_eps_stress": u}
        ax.set_yscale("log")
        ax.set_ylim(0.5 / n, 1.4)
        label = cfg["studies"][study]["stress"]
        ax.set_xlabel(label.replace(" multiplier", "\nmultiplier") if len(label) > 20
                      else label)  # fmt: skip
        _panel(ax, letter, cfg["studies"][study]["short"])
        facts[study] = f
    axes[0].set_ylabel(r"Fraction of sampled designs" "\n" r"with limit $\geq$ level")
    entries = _ref_entries(best=True) + [(
        Line2D([], [], ls="none", marker="o", ms=3.5, mfc="white", mec="#000000",
               mew=0.8), r"$U_\varepsilon$: $k$-th largest limit")]  # fmt: skip
    _legend(fig, entries, ncol=3)
    return fig, facts


def fig_hopper(cfg: dict, data: dict):
    """Appendix: payload of each returned Hopper design against its robustness."""
    study, recs = "s4", data["recs"]["s4"]
    top = top_budget(recs)
    labels = ["witness-focused", "greedy", "config", "sup-bisection", "direct-score",
              "gp-ucb", "diff-evo"]  # fmt: skip
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.66, 2.1), layout="constrained")
    seed, sd = 20261111, 0.012
    rng = np.random.default_rng(seed)
    facts = {"budget": top, "y_jitter_sd": sd, "jitter_seed": seed}
    _reference(ax, cfg, data, study, axis="x")
    for lab in labels:
        g_all = group(recs, lab, top)
        g = [r for r in g_all if r["has_witness"]]
        st = arm_style(cfg, study, lab)
        x = stress(cfg, study, [r["L"] for r in g])
        y = np.array([r["robust"] for r in g], dtype=float)
        y = y + rng.normal(0.0, sd, len(y))
        ok = np.array([r["verified"] is True for r in g], dtype=bool)
        ax.scatter(x[ok], y[ok], s=11, marker=st["marker"], color=st["color"],
                   edgecolor="white", linewidth=0.3, alpha=0.9, zorder=3)  # fmt: skip
        ax.scatter(x[~ok], y[~ok], s=11, marker=st["marker"], facecolor="white",
                   edgecolor=st["color"], linewidth=0.7, zorder=3)  # fmt: skip
        facts[lab] = {"n": len(g_all), "n_design": len(g), "n_pass": int(ok.sum()),
                      "n_robust_all": int(sum(r["robust"] == 1.0 for r in g)),
                      "median_payload_kg": float(np.median(x)) if g
                      else None}  # fmt: skip
    ax.set_xlabel(cfg["studies"][study]["stress"])
    ax.set_ylabel("Fraction of 8 other initial\nstates that also pass")
    ax.set_ylim(-0.07, 1.07)
    entries = _arm_entries(cfg, [(study, lab) for lab in labels], "marker")
    entries.append(
        (
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                mfc="white",
                mec="#555555",
                mew=0.7,
                ms=3.8,
            ),
            "Open: fails fixed-state check",
        )
    )
    _legend(fig, entries + _ref_entries(limit=False, best=True), ncol=1,
            loc="outside right center")  # fmt: skip
    return fig, facts


def fig_check_layer(cfg: dict, data: dict):
    """Appendix: each proposer's level before and after the S7 check layer."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN * 0.82, 2.15),
                             layout="constrained")  # fmt: skip
    facts = {}
    proposers = ["witness-focused", "sup-bisection"]
    for ax, letter, study in zip(axes, "ab", ("s3", "s4"), strict=True):
        rows = data["s7"][study]
        _reference(ax, cfg, data, study)
        f = {"budget_total": int(rows[0]["budget_total"])}
        for k, lab in enumerate(proposers):
            g = [r for r in rows if r["label"] == lab and r["tested_lower"] is not None]
            st = arm_style(cfg, study, NAIVE[study] if lab == "sup-bisection" else lab)
            before = stress(cfg, study, [r["proposer_lower"] for r in g])
            after = stress(cfg, study, [r["tested_lower"] for r in g])
            first = np.array([bool(r["adjust"]["trace"][0][1]) for r in g], dtype=bool)
            for y0, y1 in zip(before, after, strict=True):
                ax.plot([3 * k, 3 * k + 1], [y0, y1], color=st["color"], lw=0.4,
                        alpha=0.45)  # fmt: skip
            kw = {"s": 9, "marker": st["marker"], "zorder": 3}
            ax.scatter(np.full(first.sum(), 3 * k), before[first], color=st["color"],
                       **kw)  # fmt: skip
            ax.scatter(np.full((~first).sum(), 3 * k), before[~first],
                       facecolor="white", edgecolor=st["color"], linewidth=0.7,
                       **kw)  # fmt: skip
            ax.scatter(np.full(len(after), 3 * k + 1), after, color=st["color"], **kw)
            f[lab] = {"n": len(g), "report_passed_first_check": int(first.sum()),
                      "median_reported": float(np.median(before)),
                      "median_tested": float(np.median(after)),
                      "tested_pass_independent": int(sum(
                          bool((r.get("gate_check") or {}).get("feasible"))
                          for r in g))}  # fmt: skip
            ax.text(3 * k + 0.5, -0.16, display_name(st), ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=7)  # fmt: skip
        ax.set_xticks([0, 1, 3, 4], ["reported", "tested"] * 2)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylabel(cfg["studies"][study]["stress"])
        ev = cfg["studies"][study]["evals"]
        _panel(ax, letter, f"{cfg['studies'][study]['short']}, "
               f"{f['budget_total']:,} {ev} in total")  # fmt: skip
        facts[study] = f
    entries = _ref_entries(best=True) + [
        (
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                mfc="white",
                mec="#555555",
                mew=0.7,
                ms=3.8,
            ),
            "Open: report fails the first check",
        )
    ]
    _legend(fig, entries, ncol=3)
    return fig, facts


def _s6b(cfg: dict, data: dict, name: str, budget: int) -> list[dict]:
    """Double-integrator rows of one display label at one budget, seed order."""
    lab = cfg["double_integrator"]["labels"]
    return sorted((r for r in data["s6b"] if lab[r["arm"]] == name
                   and r["budget"] == budget), key=lambda r: r["seed"])  # fmt: skip


S6B_CERTIFIED = ["greedy", "config", "random"]
S6B_UNCERTIFIED = ["sup-bisection", "direct-score"]


def fig_contraction(cfg: dict, data: dict):
    """Double integrator: interval width during search and final width by budget."""
    budgets = sorted({r["budget"] for r in data["s6b"]})
    top, names = budgets[-1], S6B_CERTIFIED + S6B_UNCERTIFIED
    floor = 100 * cfg["double_integrator"]["report_grid_step"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.3), sharey=True,
                               width_ratios=[1.1, 1], layout="constrained")  # fmt: skip
    facts = {"theta_unit": THETA_EXISTS_UNIT, "budgets": budgets, "anytime": {},
             "final": {}}  # fmt: skip
    grid = np.geomspace(10, top, 400)
    for name in names:
        g = _s6b(cfg, data, name, top)
        st = arm_style(cfg, "s6b", name)
        pts = [[(p["calls"], 100 * p["width"]) for p in r["reports"]] for r in g]
        q25, q50, q75 = step_quantiles(pts, grid, before=100.0)
        if name in S6B_CERTIFIED:
            a.fill_between(grid, q25, q75, color=st["color"], alpha=0.15, lw=0)
        a.plot(grid, q50, color=st["color"], ls=st["ls"], lw=1.1)
        facts["anytime"][name] = {"n": len(g), "final_median_pct": float(q50[-1])}
    m_top = facts["anytime"]["greedy"]["final_median_pct"]
    ref = np.array([30.0, float(top)])
    a.plot(ref, 2.0 * m_top * (ref / top) ** -0.5, color="#000000", lw=0.7,
           ls=(0, (2, 2)))  # fmt: skip
    x_t = 0.06 * top
    a.text(x_t, 2.0 * m_top * (x_t / top) ** -0.5 * 1.2, r"slope $-1/2$",
           fontsize=7, ha="center", va="bottom")  # fmt: skip
    a.axhline(floor, color="#BBBBBB", lw=0.5, ls=THRESH_LS, zorder=0)
    a.text(11, floor * 1.15, "report-grid spacing", fontsize=7, color="#555555",
           va="bottom")  # fmt: skip
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(10, top)
    a.set_ylim(0.025, 60)
    a.set_xlabel("Evaluations of the analytic formula")
    a.set_ylabel("Interval width (% of stress range)")
    _panel(a, "a", f"Width during search, budget {top:,}")

    rng = np.random.default_rng(20261111)
    offs = dict(zip(names, [-0.06, -0.03, 0.0, 0.03, 0.06], strict=True))
    lb = np.log(budgets)
    for name in names:
        st = arm_style(cfg, "s6b", name)
        meds, f = [], {}
        for bud in budgets:
            g = _s6b(cfg, data, name, bud)
            w = 100 * np.array([r["final_width"] for r in g])
            ok = np.array([bool(r["covered_anytime"]) for r in g])
            x = bud * 10 ** (offs[name] + rng.uniform(-0.012, 0.012, len(g)))
            kw = {"s": 8, "marker": st["marker"], "zorder": 3}
            b.scatter(x[ok], w[ok], color=st["color"], edgecolor="white",
                      linewidth=0.25, alpha=0.85, **kw)  # fmt: skip
            b.scatter(x[~ok], w[~ok], facecolor="white", edgecolor=st["color"],
                      linewidth=0.6, **kw)  # fmt: skip
            meds.append(float(np.median(w)))
            sound = [r["witness_sound"] for r in g if r["witness_sound"] is not None]
            f[bud] = {
                "n": len(g),
                "median_width_pct": meds[-1],
                "q25_pct": float(np.quantile(w, 0.25)),
                "q75_pct": float(np.quantile(w, 0.75)),
                "contain_every_report": int(ok.sum()),
                "contain_final": int(sum(r["covered_final"] for r in g)),
                "witness_sound": int(sum(sound)),
                "witness_n": len(sound),
                "median_wall_s": float(np.median([r["wall_s"] for r in g])),
            }
        b.plot(np.array(budgets) * 10 ** offs[name], meds, color=st["color"],
               ls=st["ls"], lw=0.9, zorder=2)  # fmt: skip
        if name in S6B_CERTIFIED:
            f["median_slope"] = float(np.polyfit(lb, np.log(meds), 1)[0])
            per_seed = [np.polyfit(lb, np.log([r["final_width"] for r in rs]), 1)[0]
                        for rs in zip(*(_s6b(cfg, data, name, bud) for bud in budgets),
                                      strict=True)]  # fmt: skip
            f["per_seed_slope_median"] = float(np.median(per_seed))
        facts["final"][name] = f
    b.axhline(floor, color="#BBBBBB", lw=0.5, ls=THRESH_LS, zorder=0)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xticks(budgets, [f"{v:,}" for v in budgets])
    b.xaxis.set_minor_locator(NullLocator())
    b.set_xlim(budgets[0] / 1.6, budgets[-1] * 1.6)
    b.set_xlabel("Search budget (evaluations)")
    _panel(b, "b", "Final width per run")
    entries = _arm_entries(cfg, [("s6b", n) for n in names], "both") + [
        (Line2D([], [], ls="none", marker="o", mfc="white", mec="#555555", mew=0.6,
                ms=3.8), "Open: misses the limit at some report")]  # fmt: skip
    _legend(fig, entries, ncol=3)
    return fig, facts


# --------------------------------------------------------------------------- tables


def _fmt(cfg: dict, study: str, v: float | None, delta: bool = False) -> str:
    """Physical value (or difference) as LaTeX text."""
    if v is None:
        return "--"
    s = cfg["studies"][study]
    x = s["scale"] * v if delta else s["offset"] + s["scale"] * v
    out = f"{abs(x):.{s['digits']}f}" + (r"\%" if study == "s3" else "")
    return ("$-$" + out) if x < 0 and float(out.rstrip(r"\%")) != 0 else out


def _cells(cfg, study, recs, label, cell) -> list[str]:
    """[median reported level, designs passing / runs] of one label at one budget."""
    g = group(recs, label, cell)
    if not g:
        return ["--", ""]
    n_pass = f"{sum(r['verified'] is True for r in g)}/{len(g)}"
    if not any(r["has_witness"] for r in g):
        return ["none", n_pass]
    return [_fmt(cfg, study, float(np.median([r["L"] for r in g]))), n_pass]


def _head(text: str) -> str:
    """Two-line right-aligned column head; '|' separates the lines."""
    return r"\shortstack[r]{" + text.replace("|", r"\\") + "}"


def tab_summary(cfg: dict, data: dict):
    """Main-text summary: what each method established at each study's top budget.

    Each study has two columns: a median level and a count (designs passing the
    study's check out of runs; brackets containing the limit; or epsilon).
    """
    studies = ["s2", "s3", "s4", "s5"]
    cols = {}
    for s in studies:
        recs, top = data["recs"][s], top_budget(data["recs"][s])
        theta, err, _ = data["truth"][s]
        g_all = group(recs, "greedy", top)
        g = [r for r in g_all if r["U"] is not None]
        upper = [r"none$^{d}$", ""]
        if g:
            cov = sum(r["L"] <= theta + err and r["U"] >= theta - err for r in g)
            upper = [
                _fmt(cfg, s, float(np.median([r["U"] for r in g]))),
                f"{cov}/{len(g)}" + ("" if len(g) == len(g_all) else r"$^{c}$"),
            ]
        eps = cfg["studies"][s]["eps"]
        thr = [r"--$^{e}$", ""]
        if eps:
            t = _threshold(data["columns"][s], eps[0])
            thr = [_fmt(cfg, s, t["U_eps"]), rf"$\varepsilon{{=}}{eps[0]:g}$"]
        ref = _fmt(cfg, s, theta)
        cols[s] = {
            "budget": f"{top:,} {cfg['studies'][s]['evals'].split()[-1]}",
            "ref": rf"$\geq$\,{ref}" if s == BEST_FOUND else ref,
            "prop": _cells(cfg, s, recs, "witness-focused", top),
            "greedy": _cells(cfg, s, recs, "greedy", top),
            "naive": _cells(cfg, s, recs, NAIVE[s], top),
            "rep16": _cells(cfg, s, recs, REPLICATED_16[s], top)
            if s in REPLICATED_16 else ["--", ""],
            "gpucb": _cells(cfg, s, recs, "gp-ucb", top),
            "de": _cells(cfg, s, recs, "diff-evo", top),
            "upper": upper,
            "thr": thr,
        }  # fmt: skip
    span = [("budget", "Search budget"), ("ref", r"Operating limit (reference)$^{a}$")]
    groups = [
        (r"Returned design: median level, designs passing the check$^{b}$", [
            ("prop", r"\textbf{BRINK, proposed search}"),
            ("greedy", r"\textbf{BRINK, variance-seeking}"),
            ("naive", "Bisection"),
            ("rep16", r"Bisection, 16\ensuremath{\times} replication"),
            ("gpucb", "GP-UCB"), ("de", "Differential evolution")]),
        (r"Upper side: median value, and brackets containing the limit"
         r" or $\varepsilon$", [
            ("upper", r"\textbf{BRINK $U$, variance-seeking}"),
            ("thr", r"Feasible-fraction threshold $U_\varepsilon$")]),
    ]  # fmt: skip
    ncol = 2 * len(studies) + 1
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{cfg['studies'][s]['short']}}}"
                      for s in studies)  # fmt: skip
    unit = " & ".join(rf"\multicolumn{{2}}{{c}}{{({cfg['studies'][s]['unit']})}}"
                      for s in studies)  # fmt: skip
    rules = "".join(rf"\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}"
                    for i in range(len(studies)))  # fmt: skip
    lines = [r"\begin{tabular}{@{}l" + "rl" * len(studies) + "@{}}", r"\toprule",
             f" & {head} \\\\", f" & {unit} \\\\", rules]  # fmt: skip
    for key, name in span:
        lines.append(
            name
            + " & "
            + " & ".join(rf"\multicolumn{{2}}{{c}}{{{cols[s][key]}}}" for s in studies)
            + r" \\"
        )
    for title, rows in groups:
        lines += [
            r"\addlinespace[3pt]",
            rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\emph{{{title}}}}} \\",
        ]
        for key, name in rows:
            lines.append(name + " & " + " & ".join(
                " & ".join(cols[s][key]) for s in studies) + r" \\")  # fmt: skip
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines), cols


def tab_compare(cfg: dict, data: dict, study: str):
    """Appendix: every method on one study, from the cross-study comparison."""
    comp = data["comparison"][study]
    s4 = study == "s4"
    ref = r"T_{\mathrm{ref}}" if s4 else r"\theta"
    head = ["Budget", "Method", "Pass", _head(f"${{L\\leq {ref}}}$|always"),
            _head(f"Med.|${ref}-L$"), _head("Passing|level"),
            _head("$U$ contains|ref./runs"), _head("Med.|$U-L$"),
            _head("Reach|target"), "Core-h"]  # fmt: skip
    if s4:
        head.insert(9, _head("Robust|all 8"))
    lines = [r"\begin{tabular}{@{}rl" + "r" * (len(head) - 2) + "@{}}", r"\toprule",
             " & ".join(head) + r" \\", r"\midrule"]  # fmt: skip
    prev, facts = None, []
    order = list(cfg["arms"])

    def rank(r: dict) -> tuple:
        base = r["label"].split(" + ")[0]
        key = f"{base}@{study}" if f"{base}@{study}" in cfg["arms"] else base
        return r["cell"], " + " in r["label"], order.index(key)

    for r in sorted(comp["rows"], key=rank):
        if prev is not None and r["cell"] != prev:
            lines.append(r"\midrule")
        base = r["label"].split(" + ")[0]
        st = arm_style(cfg, study, base)
        name = tex_name(st["name"])
        name += ", tested" if r["label"].endswith("S7 layer") else ""
        name = rf"\textbf{{{name}}}" if st.get("ours") else name
        up = ["--", ""]
        if r["n_upper"]:
            mark = (
                r"\textsuperscript{\textdagger}" if r["upper_kind"] == "claimed" else ""
            )
            up = [
                f"{r['upper_covers']}/{r['n_upper']}",
                _fmt(cfg, study, r["median_width"], delta=True) + mark,
            ]
        if r["n_reach_target"] is None:
            reach = "--"
        elif r["median_raw_to_target"] is None:
            reach = f"{r['n_reach_target']}"
        else:
            reach = f"{r['n_reach_target']} ({r['median_raw_to_target']:,.0f})"
        passing = "--" if r["median_verified_L"] == 0 else _fmt(
            cfg, study, r["median_verified_L"])  # fmt: skip
        cells = [f"{r['cell']:,}" if r["cell"] != prev else "", name,
                 f"{r['verified']}/{r['n']}", f"{r['lower_ok_anytime']}/{r['n']}",
                 _fmt(cfg, study, r["median_truth_minus_L"], delta=True), passing,
                 *up, reach, f"{r['core_h']:.1f}"]  # fmt: skip
        if s4:
            cells.insert(9, "--" if r["robust_all"] is None else str(r["robust_all"]))
        lines.append(" & ".join(cells) + r" \\")
        prev = r["cell"]
        facts.append({k: r[k] for k in ("cell", "label", "n", "verified",
                                         "lower_ok_anytime", "n_upper", "upper_kind",
                                         "upper_covers", "median_width")})  # fmt: skip
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines), {"truth": comp["truth"], "target": comp["target"],
                              "rows": facts}  # fmt: skip


def tab_double_integrator(cfg: dict, data: dict):
    """Double integrator: median final width and every-report containment."""
    budgets = sorted({r["budget"] for r in data["s6b"]})
    nb = len(budgets)
    head = (rf" & \multicolumn{{{nb}}}{{c}}{{Median width (\% of range)}}"
            rf" & \multicolumn{{{nb}}}{{c}}{{Contains the limit at every report}}"
            r" & Design \\")  # fmt: skip
    sub = ("Method & " + " & ".join(f"{b:,}" for b in budgets * 2)
           + r" & passes \\")  # fmt: skip
    lines = [r"\begin{tabular}{@{}l" + "r" * (2 * nb + 1) + "@{}}", r"\toprule", head,
             rf"\cmidrule(lr){{2-{nb + 1}}}\cmidrule(lr){{{nb + 2}-{2 * nb + 1}}}",
             sub, r"\midrule"]  # fmt: skip
    facts = {}
    for name in S6B_CERTIFIED + S6B_UNCERTIFIED:
        if name == S6B_UNCERTIFIED[0]:
            lines.append(r"\midrule")
        gs = [_s6b(cfg, data, name, b) for b in budgets]
        widths = [f"{100 * np.median([r['final_width'] for r in g]):.2f}" for g in gs]
        cover = [f"{sum(bool(r['covered_anytime']) for r in g)}/{len(g)}" for g in gs]
        sound = [r["witness_sound"] for g in gs for r in g
                 if r["witness_sound"] is not None]  # fmt: skip
        passes = f"{sum(sound)}/{len(sound)}" if sound else "--"
        st = arm_style(cfg, "s6b", name)
        disp = rf"\textbf{{{st['name']}}}" if st.get("ours") else st["name"]
        lines.append(" & ".join([disp, *widths, *cover, passes]) + r" \\")
        facts[name] = {"widths_pct": widths, "contain": cover, "design_passes": passes}
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines), facts


# ------------------------------------------------------------- cross-study figures
# Five figures that put the double integrator and the four applied studies side by
# side, deterministic simulators first. Rows are grouped as the paper's methods,
# comparators that share the paper's uncertainty bounds, and uncertified baselines.

REPLICATED_4 = {"s3": "baseline-k64", "s5": "baseline-k4"}
PASS_C, FAIL_C, PARTIAL_C, NA_C = "#0072B2", "#D55E00", "#8A5A00", "#8C8C8C"
NOISY_BG = "#F4F0E6"
STATUS = {  # theory-map cell status: (face color, legend text)
    "held": ("#CFE8DC", "observed to hold"),
    "weak": ("#F6E3B4", "held, but uninformative"),
    "failed": ("#F4C7B5", "failed"),
    "na": ("#EFEFEF", "not tested"),
}
# Method rows: (key, stored label that supplies the display name and style, group).
CROSS_ROWS = [
    ("witness-focused", "witness-focused", "ours"),
    ("greedy", "greedy", "ours"),
    ("config", "config", "shared bounds"),
    ("random", "random", "shared bounds"),
    ("naive", "sup-bisection", "uncertified"),
    ("rep4", "baseline-k64", "uncertified"),
    ("rep16", "baseline-k256", "uncertified"),
    ("direct-score", "direct-score", "uncertified"),
    ("gp-ucb", "gp-ucb", "uncertified"),
    ("diff-evo", "diff-evo", "uncertified"),
]
TWO_SIDED = ("greedy", "config", "random")


def row_label(study: str, key: str) -> str | None:
    """Stored run label of a method row in one study (None: not run there)."""
    if study == "s6b":
        return {
            "greedy": "greedy",
            "config": "config",
            "random": "random",
            "naive": "sup-bisection",
            "direct-score": "direct-score",
        }.get(key)
    return {
        "witness-focused": "witness-focused",
        "greedy": "greedy",
        "config": "config",
        "naive": NAIVE[study],
        "rep4": REPLICATED_4.get(study),
        "rep16": REPLICATED_16.get(study),
        "direct-score": "direct-score",
        "gp-ucb": "gp-ucb",
        "diff-evo": "diff-evo",
    }.get(key)


def study_meta(cfg: dict, study: str) -> dict:
    """Display settings and stress units of a study, the double integrator included."""
    return cfg["double_integrator"] if study == "s6b" else cfg["studies"][study]


def cs_stress(cfg: dict, study: str, c):
    """Physical stress level of unit stress ``c`` in any study."""
    m = study_meta(cfg, study)
    return m["offset"] + m["scale"] * np.asarray(c, dtype=float)


def study_truth(data: dict, study: str) -> tuple[float, float]:
    """Reference limit and its stated error, in unit stress."""
    if study == "s6b":
        return float(THETA_EXISTS_UNIT), 0.0
    theta, err, _ = data["truth"][study]
    return float(theta), float(err)


def study_budgets(cfg: dict, data: dict, study: str) -> list[int]:
    """Search budgets of a study, ascending."""
    if study == "s6b":
        return list(cfg["double_integrator"]["budgets"])
    recs = data["recs"][study]
    return sorted({r["cell"] for r in recs if r["label"] == "witness-focused"})


def method_runs(cfg: dict, data: dict, study: str, key: str, budget: int):
    """One method row's runs at one budget in unit stress (None: not run).

    Each run has the reported level ``L``, the upper value ``U`` (None if the method
    reported none), whether it returned a design, whether that design passed the
    study's independent check, and whether [L, U] contained the limit at the final
    report (and at every report, where recorded).
    """
    label = row_label(study, key)
    if label is None:
        return None
    if study == "s6b":
        g = _s6b(cfg, data, label, budget)
        return [
            {
                "L": r["final_lower"],
                "U": r["final_upper"],
                "design": r["final_witness_x"] is not None,
                "passed": bool(r["witness_sound"]),
                "contains": bool(r["covered_final"]),
                "contains_anytime": bool(r["covered_anytime"]),
            }
            for r in g
        ] or None
    theta, err = study_truth(data, study)
    g = group(data["recs"][study], label, budget)
    return [
        {
            "L": r["L"],
            "U": r["U"],
            "design": bool(r["has_witness"]),
            "passed": r["verified"] is True,
            "contains": r["U"] is not None
            and r["L"] <= theta + err
            and r["U"] >= theta - err,
            "contains_anytime": None,
        }
        for r in g
    ] or None


def comparison_row(data: dict, study: str, key: str, budget: int) -> dict | None:
    """The cross-study comparison's summary row of one method at one budget."""
    label = row_label(study, key)
    for r in data["comparison"][study]["rows"]:
        if r["label"] == label and r["cell"] == budget:
            return r
    return None


def check_on_axis(levels: np.ndarray, lo: float, hi: float, what: str) -> None:
    """Stop if a returned design's level falls outside the drawn stress range."""
    outside = (levels < lo) | (levels > hi)
    if outside.any():
        raise SystemExit(
            f"{what}: {int(outside.sum())} returned levels outside [{lo}, {hi}]; "
            "widen cross_study.xlim in the config"
        )


def split_upper(upper: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, int]:
    """Upper bounds drawn on the axis, and the count beyond its right edge.

    An upper bound below the axis would be hidden, so it stops the script.
    """
    check_on_axis(upper[upper <= hi], lo, hi, "upper bound")
    return upper[upper <= hi], int((upper > hi).sum())


def _row_name(cfg: dict, label: str) -> str:
    return display_name(cfg["arms"][label])


def _cs_fmt(cfg: dict, study: str, v: float) -> str:
    m = study_meta(cfg, study)
    return f"{v:.{m['digits']}f}" + ("%" if study == "s3" else "")


def _rows_in(cfg: dict, data: dict, studies: list[str]) -> list[tuple]:
    """Method rows that ran in at least one of ``studies``."""
    return [
        row
        for row in CROSS_ROWS
        if any(
            row_label(s, row[0])
            and method_runs(cfg, data, s, row[0], study_budgets(cfg, data, s)[-1])
            for s in studies
        )
    ]


def _group_lines(ax, rows: list[tuple]) -> None:
    """Thin rules between the method groups (ours, shared bounds, uncertified)."""
    for i in range(1, len(rows)):
        if rows[i][2] != rows[i - 1][2]:
            ax.axhline(i - 0.5, color="#AAAAAA", lw=0.5, ls=(0, (2, 2)), zorder=1)


def _overview_panel(ax, cfg, data, study, rows, letter, rng, first) -> dict:
    cs = cfg["cross_study"]
    m = study_meta(cfg, study)
    top = study_budgets(cfg, data, study)[-1]
    lo, hi = cs["xlim"][study]
    hw = cs["jitter"]["half_width"]
    theta, err = study_truth(data, study)
    n_rows = len(rows) + 1
    ax.set_xlim(lo, hi)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for i in range(n_rows):
        ax.axhline(i, color="#EBEBEB", lw=0.5, zorder=0)
    _group_lines(ax, rows)
    ref, rlo, rhi = cs_stress(cfg, study, [theta, theta - err, theta + err])
    if err > 0:
        ax.axvspan(rlo, rhi, color=REF_BAND, lw=0, zorder=0)
    ax.axvline(
        ref,
        color=REF_COLOR,
        lw=0.8,
        zorder=1,
        ls=BEST_LS if study == BEST_FOUND else "-",
    )
    facts = {"budget": top, "reference": float(ref), "rows": {}}
    right, colors = [], []
    for i, (key, _, _) in enumerate(rows):
        rr = method_runs(cfg, data, study, key, top)
        if rr is None:
            right.append("")
            colors.append(NA_C)
            ax.text(
                lo + 0.03 * (hi - lo),
                i,
                "not run",
                va="center",
                fontsize=6,
                color="#9A9A9A",
                style="italic",
            )
            continue
        des = [r for r in rr if r["design"]]
        x = cs_stress(cfg, study, [r["L"] for r in des])
        ok = np.array([r["passed"] for r in des], dtype=bool)
        n_pass = int(ok.sum())
        check_on_axis(x, lo, hi, f"{study} {key}")
        y = i + rng.uniform(-hw, hw, size=len(x))
        ax.scatter(
            x[ok],
            y[ok],
            marker="o",
            s=6,
            facecolor=PASS_C,
            edgecolor="white",
            linewidth=0.2,
            zorder=3,
        )
        ax.scatter(
            x[~ok], y[~ok], marker="x", s=8, color=FAIL_C, linewidth=0.7, zorder=3
        )
        if len(x):
            med = float(np.median(x))
            ax.plot([med, med], [i - 0.38, i + 0.38], color="black", lw=1.0, zorder=4)
        else:
            ax.text(
                lo + 0.03 * (hi - lo),
                i,
                "no design returned",
                va="center",
                fontsize=6,
                color="#555555",
                style="italic",
            )
        right.append(f"{n_pass}/{len(rr)}")
        colors.append(
            NA_C
            if not des
            else PASS_C
            if n_pass == len(rr)
            else FAIL_C
            if n_pass == 0
            else PARTIAL_C
        )
        facts["rows"][key] = {
            "runs": len(rr),
            "designs": len(des),
            "passed": n_pass,
            "median_level": float(np.median(x)) if len(x) else None,
        }
    iu = n_rows - 1
    ax.axhline(iu - 0.5, color="#888888", lw=0.6)
    g = method_runs(cfg, data, study, "greedy", top) or []
    gu = [r for r in g if r["U"] is not None]
    right.append("")
    colors.append(NA_C)
    if gu:
        u = cs_stress(cfg, study, [r["U"] for r in gu])
        shown, beyond = split_upper(u, lo, hi)
        cov = sum(r["contains"] for r in gu)
        y = iu + rng.uniform(-hw, hw, size=len(shown))
        ax.scatter(shown, y, marker="|", s=20, color="black", linewidth=0.8, zorder=3)
        lines = [f"contains limit: {cov}/{len(gu)}"]
        if len(gu) < len(g):
            lines.insert(0, f"$U$ in {len(gu)} of {len(g)} runs")
        if beyond:
            lines.append(
                f"{beyond} beyond axis (median {_cs_fmt(cfg, study, np.median(u))})"
            )
            ax.scatter(
                [hi], [iu], marker=">", s=14, color="black", clip_on=False, zorder=4
            )
        ax.text(
            lo + 0.03 * (hi - lo),
            iu,
            "\n".join(lines),
            va="center",
            fontsize=5.5,
            color="#222222",
            linespacing=1.1,
        )
        facts["upper"] = {
            "with_U": len(gu),
            "runs": len(g),
            "contains": int(cov),
            "median_U": float(np.median(u)),
            "beyond_axis": beyond,
        }
    else:
        ax.text(
            lo + 0.03 * (hi - lo),
            iu,
            "no upper-bound test",
            va="center",
            fontsize=6,
            color="#555555",
            style="italic",
        )
    eps = cfg["studies"][study]["eps"] if study != "s6b" else []
    if eps:
        t = _threshold(data["columns"][study], eps[0])
        ue = float(cs_stress(cfg, study, t["U_eps"]))
        ax.axvline(ue, color="#444444", lw=0.7, ls=THRESH_LS, zorder=1)
        ax.text(ue, -0.6, f"$U_{{{eps[0]:g}}}$", ha="center", va="bottom", fontsize=6)
        facts["U_eps"] = {"eps": eps[0], "value": ue}
    ax.set_title(
        f"({letter}) {m['short']}\n{top:,} {m['evals']}", loc="left", fontsize=7, pad=8
    )
    ax.set_xlabel(m["stress"], fontsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    if first:
        names = [_row_name(cfg, lab) for _, lab, _ in rows] + ["BRINK upper bound $U$"]
        ax.set_yticks(range(n_rows), names)
        for tl, (_, _, grp) in zip(ax.get_yticklabels(), rows, strict=False):
            tl.set_fontweight("bold" if grp == "ours" else "normal")
        ax.get_yticklabels()[-1].set_fontweight("bold")
    else:
        ax.set_yticks(range(n_rows), [""] * n_rows)
    tw = ax.twinx()
    tw.set_ylim(ax.get_ylim())
    tw.set_yticks(range(n_rows), right)
    for lab, col in zip(tw.get_yticklabels(), colors, strict=True):
        lab.set_color(col)
        lab.set_fontweight("bold" if col == FAIL_C else "normal")
    tw.tick_params(axis="y", length=0, labelsize=6, pad=1)
    for side in ("top", "right", "left"):
        tw.spines[side].set_visible(False)
    tw.text(
        1.0,
        1.0,
        "pass",
        transform=tw.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color="#333333",
    )
    return facts


def fig_overview(cfg: dict, data: dict):
    """Every run's returned design and check outcome, and the upper bound, per study."""
    cs = cfg["cross_study"]
    rng = np.random.default_rng(cs["jitter"]["seed"])
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 6.2), layout="constrained")
    sf_det, sf_noi = fig.subfigures(2, 1, hspace=0.0)
    sf_noi.set_facecolor(NOISY_BG)
    sf_det.suptitle(
        "Deterministic simulators: a design that passes once is feasible",
        fontsize=8,
        fontweight="bold",
    )
    sf_noi.suptitle(
        "Noisy simulators: a pass can be luck, so each design is re-checked",
        fontsize=8,
        fontweight="bold",
    )
    facts = {}
    letters = iter("abcde")
    noisy_axes = None
    for sf, studies in ((sf_det, cs["deterministic"]), (sf_noi, cs["noisy"])):
        axes = sf.subplots(1, 3, gridspec_kw={"wspace": 0.05})
        rows = _rows_in(cfg, data, studies)
        for k, (ax, study) in enumerate(zip(axes, studies, strict=False)):
            facts[study] = _overview_panel(
                ax, cfg, data, study, rows, next(letters), rng, k == 0
            )
        noisy_axes = axes
    leg = noisy_axes[-1]
    leg.axis("off")
    entries = [
        (
            Line2D([], [], marker="o", ls="", color=PASS_C, markersize=4),
            "design passed the independent check",
        ),
        (
            Line2D([], [], marker="x", ls="", color=FAIL_C, markersize=4, mew=0.9),
            "design failed the check",
        ),
        (Line2D([], [], color="black", lw=1.0), "median reported level"),
        (
            Line2D([], [], marker="|", ls="", color="black", markersize=6),
            "upper bound $U$ of one run",
        ),
        (
            (Patch(color=REF_BAND, lw=0), Line2D([], [], color=REF_COLOR, lw=0.8)),
            "operating limit (band: stated error)",
        ),
        (
            Line2D([], [], color=REF_COLOR, lw=0.8, ls=BEST_LS),
            "reference design (Hopper)",
        ),
        (
            Line2D([], [], color="#444444", lw=0.7, ls=THRESH_LS),
            "feasible-fraction threshold",
        ),
        (
            Line2D([], [], color="#AAAAAA", lw=0.5, ls=(0, (2, 2))),
            "BRINK | shared bounds | uncertified",
        ),
    ]
    leg.legend(
        [h for h, _ in entries],
        [lab for _, lab in entries],
        loc="center",
        fontsize=6,
        handlelength=1.8,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
    )
    return fig, facts


def fig_accuracy(cfg: dict, data: dict):
    """Median reported level minus the limit, as a share of the stress range."""
    cs = cfg["cross_study"]
    studies = cs["deterministic"] + cs["noisy"]
    rows = CROSS_ROWS
    gap = np.full((len(rows), len(studies)), np.nan)
    cells: dict = {}
    for j, s in enumerate(studies):
        top = study_budgets(cfg, data, s)[-1]
        theta, _ = study_truth(data, s)
        for i, (key, _, _) in enumerate(rows):
            rr = method_runs(cfg, data, s, key, top)
            if rr is None:
                cells[i, j] = {"state": "not run"}
                continue
            des = [r for r in rr if r["design"]]
            if not des:
                cells[i, j] = {"state": "no design", "runs": len(rr)}
                continue
            gap[i, j] = 100.0 * (float(np.median([r["L"] for r in des])) - theta)
            cells[i, j] = {
                "state": "design",
                "gap_pct": gap[i, j],
                "runs": len(rr),
                "passed": sum(r["passed"] for r in rr),
            }
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.6), layout="constrained")
    norm = TwoSlopeNorm(vmin=-30.0, vcenter=0.0, vmax=30.0)
    cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#F2F2F2")
    im = ax.imshow(np.ma.masked_invalid(gap), cmap=cmap, norm=norm, aspect="auto")
    for (i, j), c in cells.items():
        if c["state"] == "not run":
            ax.text(
                j,
                i,
                "not run",
                ha="center",
                va="center",
                fontsize=6,
                color="#9A9A9A",
                style="italic",
            )
            continue
        if c["state"] == "no design":
            ax.text(
                j,
                i,
                f"no design\n(0 of {c['runs']})",
                ha="center",
                va="center",
                fontsize=6,
                color="#555555",
            )
            continue
        v = c["gap_pct"]
        txt = (
            "0.00%" if abs(v) < 0.005 else f"{v:+.2f}%" if abs(v) < 1 else f"{v:+.1f}%"
        ).replace("-", "\u2212")
        col = "white" if abs(v) > 18 else "black"
        ax.text(
            j,
            i - 0.17,
            txt,
            ha="center",
            va="center",
            fontsize=7,
            color=col,
            fontweight="bold",
        )
        p, n = c["passed"], c["runs"]
        ax.text(
            j,
            i + 0.23,
            f"{p}/{n} pass",
            ha="center",
            va="center",
            fontsize=6,
            color=col,
            fontweight="bold" if p < n else "normal",
        )
        if p < n:  # outline cells whose designs did not all pass the check
            ax.add_patch(
                Rectangle(
                    (j - 0.46, i - 0.44),
                    0.92,
                    0.88,
                    fill=False,
                    lw=1.3,
                    edgecolor=FAIL_C if p == 0 else PARTIAL_C,
                    ls="-" if p == 0 else (0, (3, 1.5)),
                )
            )
    nd = len(cs["deterministic"])
    ax.axvline(nd - 0.5, color="black", lw=0.9)
    for i in range(1, len(rows)):
        if rows[i][2] != rows[i - 1][2]:
            ax.axhline(i - 0.5, color="black", lw=0.6)
    heads = []
    for s in studies:
        m = study_meta(cfg, s)
        vs = "vs best design" if s == BEST_FOUND else "vs limit"
        heads.append(
            f"{m['short']}\n{study_budgets(cfg, data, s)[-1]:,} {m['evals']}\n({vs})"
        )
    ax.set_xticks(range(len(studies)), heads, fontsize=6.5)
    ax.xaxis.tick_top()
    ax.set_yticks(
        range(len(rows)), [_row_name(cfg, lab) for _, lab, _ in rows], fontsize=7
    )
    for tl, (_, _, grp) in zip(ax.get_yticklabels(), rows, strict=True):
        tl.set_fontweight("bold" if grp == "ours" else "normal")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(
        (nd - 1) / 2,
        -1.45,
        "Deterministic simulators",
        ha="center",
        fontweight="bold",
        fontsize=7.5,
    )
    ax.text(
        nd + (len(studies) - nd - 1) / 2,
        -1.45,
        "Noisy simulators",
        ha="center",
        fontweight="bold",
        fontsize=7.5,
    )
    cb = fig.colorbar(im, ax=ax, extend="both", shrink=0.75, pad=0.02, aspect=30)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label(
        "Median reported level minus the limit (% of stress range);\n"
        "red: above the limit; blue: below",
        fontsize=7,
    )
    return fig, {f"{studies[j]}:{rows[i][0]}": c for (i, j), c in cells.items()}


def fig_tightness(cfg: dict, data: dict):
    """Width of [L, U] against budget, and how often [L, U] contains the limit."""
    studies = ["s6b", "s2", "s3", "s5"]
    keys = [
        "greedy",
        "config",
        "random",
        "naive",
        "rep16",
        "direct-score",
        "gp-ucb",
        "diff-evo",
    ]
    labels = {k: lab for k, lab, _ in CROSS_ROWS}
    fig, axes = plt.subplots(
        2,
        len(studies),
        figsize=(TEXT_WIDTH_IN, 3.9),
        layout="constrained",
        sharex="col",
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )
    facts: dict = {}
    used = []
    noisy = cfg["cross_study"]["noisy"]
    for j, s in enumerate(studies):
        aw, ac = axes[0, j], axes[1, j]
        bs = study_budgets(cfg, data, s)
        facts[s] = {}
        for key in keys:
            pts = []
            for b in bs:
                rr = method_runs(cfg, data, s, key, b)
                ru = [r for r in rr or [] if r["U"] is not None]
                if not ru:
                    continue
                width = 100.0 * float(np.median([r["U"] - r["L"] for r in ru]))
                hit = [
                    r["contains_anytime"] if s == "s6b" else r["contains"] for r in ru
                ]
                pts.append((b, width, 100.0 * float(np.mean(hit)), len(ru), len(rr)))
            if not pts:
                continue
            if key not in used:
                used.append(key)
            st = cfg["arms"][labels[key]]
            two = key in TWO_SIDED
            b_, w_, c_ = (np.array([p[k] for p in pts]) for k in range(3))
            kw = {
                "color": st["color"],
                "marker": st["marker"],
                "markersize": 3.5,
                "ls": "-" if two else "--",
                "lw": 1.1 if two else 0.8,
                "markerfacecolor": st["color"] if two else "white",
                "markeredgewidth": 0.7,
            }
            aw.plot(b_, w_, **kw)
            ac.plot(b_, c_, **kw)
            for b, w, _, nu, nr in pts:
                if nu < nr:
                    aw.annotate(
                        f"{nu}/{nr}",
                        (b, w),
                        textcoords="offset points",
                        xytext=(-4, -8),
                        ha="right",
                        fontsize=5.5,
                        color=st["color"],
                    )
            facts[s][key] = [
                {
                    "budget": p[0],
                    "median_width_pct": p[1],
                    "contains_pct": p[2],
                    "with_U": p[3],
                    "runs": p[4],
                }
                for p in pts
            ]
        if s == "s6b":
            g = facts[s]["greedy"]
            b0, w0 = g[0]["budget"], g[0]["median_width_pct"]
            xs = np.array([bs[0], bs[-1]], dtype=float)
            aw.plot(xs, w0 * (xs / b0) ** -0.5, color="#555555", lw=0.7, ls=":")
            aw.text(
                xs[-1],
                w0 * (xs[-1] / b0) ** -0.5 * 0.55,
                "slope $-1/2$",
                fontsize=5.5,
                ha="right",
                color="#555555",
            )
        m = study_meta(cfg, s)
        kind = "every report" if s == "s6b" else "final report"
        aw.set_title(
            f"({'abcd'[j]}) {m['short']}\n"
            f"({'noisy' if s in noisy else 'deterministic'})",
            loc="left",
            fontsize=7,
        )
        aw.set_xscale("log")
        aw.set_yscale("log")
        aw.set_xticks(bs, [f"{b:,}" for b in bs])
        aw.xaxis.set_minor_locator(NullLocator())
        aw.set_xlim(bs[0] / 1.6, bs[-1] * 1.6)
        ac.set_ylim(-5, 105)
        ac.set_xlabel(f"Budget ({m['evals']})", fontsize=7)
        ac.set_title(
            f"contains the limit ({kind})", loc="left", fontsize=6, style="italic"
        )
        for a in (aw, ac):
            a.tick_params(labelsize=6)
            if s in noisy:
                a.set_facecolor(NOISY_BG)
    axes[0, 0].set_ylabel("Median width of $[L, U]$\n(% of stress range)", fontsize=7)
    axes[1, 0].set_ylabel("Runs (%)", fontsize=7)
    entries = []
    for key in used:
        st = cfg["arms"][labels[key]]
        two = key in TWO_SIDED
        entries.append(
            (
                Line2D(
                    [],
                    [],
                    color=st["color"],
                    marker=st["marker"],
                    ms=3.5,
                    ls="-" if two else "--",
                    markerfacecolor=st["color"] if two else "white",
                ),
                display_name(st) + ("" if two else " (claimed)"),
            )
        )
    _legend(fig, entries, ncol=4)
    return fig, facts


def fig_efficiency(cfg: dict, data: dict):
    """Evaluations until a passing design reaches the target, and compute per run."""
    studies = ["s2", "s4", "s3", "s5"]
    rows = [row for row in CROSS_ROWS if row[0] != "random"]
    noisy = cfg["cross_study"]["noisy"]
    fig, axes = plt.subplots(
        2, len(studies), figsize=(TEXT_WIDTH_IN, 4.8), layout="constrained", sharey=True
    )
    facts: dict = {}
    for j, s in enumerate(studies):
        m = study_meta(cfg, s)
        top = study_budgets(cfg, data, s)[-1]
        target = float(cs_stress(cfg, s, data["comparison"][s]["target"]))
        ar, ak = axes[0, j], axes[1, j]
        facts[s] = {"budget": top, "target": target, "rows": {}}
        for i, (key, lab, _) in enumerate(rows):
            row = comparison_row(data, s, key, top)
            if row is None:
                for a in (ar, ak):
                    a.text(
                        0.03,
                        i,
                        "not run",
                        transform=a.get_yaxis_transform(),
                        va="center",
                        fontsize=5.5,
                        color="#9A9A9A",
                        style="italic",
                    )
                continue
            st = cfg["arms"][lab]
            reach, n = row["n_reach_target"] or 0, row["n"]
            if reach:
                ar.scatter(
                    [row["median_raw_to_target"]],
                    [i],
                    marker=st["marker"],
                    s=14,
                    color=st["color"],
                    zorder=3,
                )
                ar.text(
                    row["median_raw_to_target"] * 1.2,
                    i,
                    f"{reach}/{n}",
                    va="center",
                    fontsize=5.5,
                )
            else:
                ar.text(
                    0.03,
                    i,
                    f"never (0/{n})",
                    transform=ar.get_yaxis_transform(),
                    va="center",
                    fontsize=5.5,
                    color=FAIL_C,
                )
            per_run = row["core_h"] / n
            ak.barh(i, per_run, color=st["color"], height=0.6)
            ak.text(per_run * 1.15, i, f"{per_run:.2f}", va="center", fontsize=5.5)
            facts[s]["rows"][key] = {
                "reach": reach,
                "runs": n,
                "median_evals_to_target": row["median_raw_to_target"],
                "core_h_per_run": per_run,
            }
        ar.axvline(top, color="#444444", lw=0.7, ls=":")
        ar.set_xscale("log")
        ar.set_xlim(100, top * 3)
        ak.set_xscale("log")
        ak.set_xlim(0.01, 12.0)
        ar.set_title(
            f"({'abcd'[j]}) {m['short']}\n"
            f"{'noisy' if s in noisy else 'deterministic'}; "
            f"target {_cs_fmt(cfg, s, target)}",
            loc="left",
            fontsize=7,
        )
        ar.set_xlabel(f"{m['evals'].capitalize()} to target", fontsize=6.5)
        ak.set_xlabel("Core-hours per run", fontsize=6.5)
        for a in (ar, ak):
            a.tick_params(labelsize=6)
            if s in noisy:
                a.set_facecolor(NOISY_BG)
            _group_lines(a, rows)
    for a in axes[:, 0]:
        a.set_yticks(
            range(len(rows)), [_row_name(cfg, lab) for _, lab, _ in rows], fontsize=6.5
        )
        a.set_ylim(len(rows) - 0.5, -0.5)
        for tl, (_, _, grp) in zip(a.get_yticklabels(), rows, strict=True):
            tl.set_fontweight("bold" if grp == "ours" else "normal")
    return fig, facts


# Answer categories of fig_decisions: (count key, face color, hatch, legend text).
DECISION_PARTS = (
    ("passing", PASS_C, None, "Correct: achievable, design passed"),
    ("proved", "#009E73", None, "Correct: not achievable (our upper bound)"),
    ("unbacked", "#DCEAF6", "///", "Correct: achievable, design failed or not stored"),
    ("wrong", FAIL_C, None, "Wrong answer"),
    ("undecided", "#E4E4E4", None, "Undecided"),
)
UNBACKED_HATCH = "#6BAED6"


def decision_parts(row: dict) -> dict:
    """Answers of one method and budget by fig_decisions category.

    Args:
        row: A ``t5_targets`` row of ``evaluation_dimensions.compute``.

    Returns:
        Counts per category; they partition the row's answers (runs x targets).
    """
    return {
        "passing": row["achievable_correct_passing"],
        "proved": row["not_correct"],
        "unbacked": row["achievable_correct"] - row["achievable_correct_passing"],
        "wrong": row["achievable_wrong"] + row["not_wrong"],
        "undecided": row["undecided"],
    }


def fig_decisions(cfg: dict, data: dict):
    """Answers to target stresses at the final report, scored against the limit."""
    import evaluation_dimensions as ed  # it imports this module, so not at top level

    ecfg = ed.load_config(Path(cfg["evaluation_config"]))
    res = ed.compute(cfg, ecfg, data)
    by = {(r["study"], r["cell"], r["label"]): r for r in res["t5_targets"]["rows"]}
    cs = cfg["cross_study"]
    studies, noisy, rows = cs["deterministic"] + cs["noisy"], cs["noisy"], CROSS_ROWS
    fig, axes = plt.subplots(
        1, len(studies), figsize=(TEXT_WIDTH_IN, 2.9), sharey=True, layout="constrained"
    )
    facts: dict = {}
    for j, (ax, s) in enumerate(zip(axes, studies, strict=True)):
        m, top = study_meta(cfg, s), study_budgets(cfg, data, s)[-1]
        n_targets = len(ecfg["targets"][s])
        for i, (key, _, _) in enumerate(rows):
            lab = row_label(s, key)
            r = by.get((s, top, lab)) if lab else None
            if r is None:
                ax.text(
                    0.03,
                    i,
                    "not run",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    fontsize=5.5,
                    color="#9A9A9A",
                    style="italic",
                )
                continue
            parts = decision_parts(r)
            if sum(parts.values()) != r["answers"]:
                raise SystemExit(
                    f"fig_decisions: {s} {lab} parts do not sum to answers"
                )
            n_targets = len(r["targets"])
            left = 0.0
            for k, color, hatch, _ in DECISION_PARTS:
                w = 100.0 * parts[k] / r["answers"]
                if not w:
                    continue
                ax.barh(i, w, left=left, height=0.72, color=color, hatch=hatch,
                        edgecolor=UNBACKED_HATCH if hatch else color,
                        linewidth=0)  # fmt: skip
                if w >= 14.0:
                    dark = k in ("passing", "proved", "wrong")
                    box = (
                        None
                        if not hatch
                        else {"boxstyle": "round,pad=0.12", "fc": color, "ec": "none"}
                    )
                    ax.text(left + w / 2, i, str(parts[k]), ha="center", va="center",
                            fontsize=5.5, color="white" if dark else "black",
                            bbox=box)  # fmt: skip
                left += w
            facts[f"{s}:{key}"] = {"budget": top, "runs": r["n"],
                                   "answers": r["answers"], **parts}  # fmt: skip
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("% of answers", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_title(
            f"({'abcde'[j]}) {m['short']}\n{top:,} {m['evals']}\n{n_targets} targets",
            loc="left",
            fontsize=6.5,
        )
        if s in noisy:
            ax.set_facecolor(NOISY_BG)
        _group_lines(ax, rows)
    axes[0].set_yticks(
        range(len(rows)), [_row_name(cfg, lab) for _, lab, _ in rows], fontsize=6.5
    )
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    for tl, (_, _, grp) in zip(axes[0].get_yticklabels(), rows, strict=True):
        tl.set_fontweight("bold" if grp == "ours" else "normal")
    entries = [
        (Patch(facecolor=c, hatch=h, edgecolor=UNBACKED_HATCH if h else c, lw=0), t)
        for _, c, h, t in DECISION_PARTS
    ]
    _legend(fig, entries, ncol=3)
    return fig, facts


def theory_cells(cfg: dict, data: dict) -> dict:
    """Status and plain text of each (guarantee, study) cell, from the stored rows."""
    cs = cfg["cross_study"]
    out: dict = {}
    for s in cs["deterministic"] + cs["noisy"]:
        bs = study_budgets(cfg, data, s)
        model = ("witness-focused", "greedy", "config", "random")
        runs = {
            k: [r for b in bs for r in (method_runs(cfg, data, s, k, b) or [])]
            for k in model
        }
        des = [r for k in model for r in runs[k] if r["design"]]
        ok = sum(r["passed"] for r in des)
        out["design", s] = (
            ("held", f"{ok}/{len(des)}\npassed")
            if ok == len(des)
            else ("failed", f"{len(des) - ok} of {len(des)}\nfailed")
        )
        two = [r for k in TWO_SIDED for r in runs[k]]
        ru = [r for r in two if r["U"] is not None]
        top = bs[-1]
        gu = [r for r in method_runs(cfg, data, s, "greedy", top) if r["U"] is not None]
        med_w = 100.0 * float(np.median([r["U"] - r["L"] for r in gu])) if gu else None
        if not ru:
            out["bracket", s] = ("na", "not tested")
        else:
            hit = sum(
                r["contains_anytime"] if s == "s6b" else r["contains"] for r in ru
            )
            if len(ru) == len(two) and med_w is not None and med_w < 10.0:
                out["bracket", s] = ("held", f"{hit}/{len(ru)}\ncontained")
            elif len(ru) < len(two):
                out["bracket", s] = (
                    "weak",
                    f"{hit}/{len(ru)}, but U only\nin {len(ru)} of {len(two)} runs",
                )
            else:
                out["bracket", s] = ("weak", f"{hit}/{len(ru)}, but U\nfar above")
        ws = [
            100.0
            * float(
                np.median(
                    [
                        r["U"] - r["L"]
                        for r in method_runs(cfg, data, s, "greedy", b)
                        if r["U"] is not None
                    ]
                    or [np.nan]
                )
            )
            for b in bs
        ]
        if out["bracket", s][0] == "held" and s == "s6b":
            slope = float(np.polyfit(np.log(bs), np.log(ws), 1)[0])
            slope_txt = f"{slope:.2f}".replace("-", "\u2212")
            out["rate", s] = ("held", f"slope {slope_txt}\n(theory \u22120.5)")
        elif out["bracket", s][0] == "held":
            out["rate", s] = ("weak", f"narrowed\n{ws[0]:.1f}% → {ws[-1]:.1f}%")
        else:
            out["rate", s] = ("na", "not measurable" if ru else "not tested")
        eps = cfg["studies"][s]["eps"] if s != "s6b" else []
        if s == "s6b":
            out["fraction", s] = ("na", "not run")
        elif not eps:
            out["fraction", s] = ("failed", cs["record_facts"]["s4_desk_check"])
        else:
            t = _threshold(data["columns"][s], eps[0])
            val = _cs_fmt(cfg, s, float(cs_stress(cfg, s, t["U_eps"])))
            out["fraction", s] = ("held", f"$U_{{{eps[0]:g}}}$ = {val}")
    return out


def fig_theory_map(cfg: dict, data: dict):
    """What each guarantee promises, where it was tested, and what was observed."""
    cs = cfg["cross_study"]
    studies = cs["deterministic"] + cs["noisy"]
    rows = [
        (
            "design",
            "Theorem 1, lower side",
            "A design the method accepts\ntruly meets every constraint",
        ),
        (
            "bracket",
            "Theorem 1, upper side",
            "The interval $[L, U]$ contains\nthe true operating limit",
        ),
        (
            "rate",
            "Theorem 2 (main result)",
            "$U$ closes in on the limit as\nsimulations grow, like $1/\\sqrt{N}$",
        ),
        (
            "fraction",
            "Proposition 3",
            "Above $U_\\varepsilon$, at most a share\n"
            "$\\varepsilon$ of designs can work",
        ),
    ]
    cells = theory_cells(cfg, data)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.3), layout="constrained")
    x0, tw, cw, gap = 0.0, 2.2, 0.88, 0.08
    xs = [x0 + tw + 0.35 + j * (cw + gap) + cw / 2 for j in range(len(studies))]
    for i, (key, title, body) in enumerate(rows):
        ax.add_patch(
            FancyBboxPatch(
                (x0, i - 0.42),
                tw,
                0.84,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor="#E8EEF6",
                edgecolor="#24527A",
                lw=0.7,
            )
        )
        ax.text(x0 + 0.08, i - 0.22, title, fontsize=7, fontweight="bold", va="center")
        ax.text(x0 + 0.08, i + 0.12, body, fontsize=6.5, va="center", linespacing=1.15)
        ax.annotate(
            "",
            xy=(xs[0] - cw / 2 - 0.04, i),
            xytext=(x0 + tw + 0.04, i),
            arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 0.7},
        )
        for j, s in enumerate(studies):
            status, text = cells[key, s]
            ax.add_patch(
                FancyBboxPatch(
                    (xs[j] - cw / 2, i - 0.42),
                    cw,
                    0.84,
                    boxstyle="round,pad=0,rounding_size=0.06",
                    facecolor=STATUS[status][0],
                    edgecolor="white",
                )
            )
            ax.text(
                xs[j], i, text, ha="center", va="center", fontsize=6, linespacing=1.15
            )
    for j, s in enumerate(studies):
        ax.text(
            xs[j],
            -0.62,
            study_meta(cfg, s)["short"].replace(" ", "\n", 1)
            if s == "s6b"
            else study_meta(cfg, s)["short"],
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    nd = len(cs["deterministic"])
    for grp, (a, b) in (
        ("Deterministic simulators", (0, nd - 1)),
        ("Noisy simulators", (nd, len(studies) - 1)),
    ):
        ax.plot([xs[a] - cw / 2, xs[b] + cw / 2], [-1.05, -1.05], color="black", lw=0.6)
        ax.text(
            (xs[a] + xs[b]) / 2,
            -1.12,
            grp,
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    ax.text(
        x0,
        -1.12,
        "What the theory promises",
        fontsize=7,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        x0,
        -0.62,
        "(experiments test it in each study →)",
        fontsize=6,
        va="bottom",
        color="#555555",
        style="italic",
    )
    ax.set_xlim(-0.05, xs[-1] + cw / 2 + 0.05)
    ax.set_ylim(len(rows) - 0.45, -1.55)
    ax.axis("off")
    handles = [
        Patch(facecolor=v[0], edgecolor="#BBBBBB", lw=0.4) for v in STATUS.values()
    ]
    fig.legend(
        handles,
        [v[1] for v in STATUS.values()],
        loc="outside lower center",
        ncol=4,
        fontsize=6.5,
        handlelength=1.2,
    )
    fig.text(
        0.5,
        -0.01,
        "Observed in simulation with fitted model constants; no run is "
        "formally covered by the theorems (Appendix B).",
        ha="center",
        va="top",
        fontsize=6,
        style="italic",
        color="#8A3A00",
    )
    return fig, {f"{k[0]}:{k[1]}": v for k, v in cells.items()}


# ----------------------------------------------------------------------------- main

FIGURES = {
    "fig_transmission": fig_transmission,
    "fig_noisy_audit": fig_noisy_audit,
    "fig_anytime": fig_anytime,
    "fig_feasible_fraction": fig_feasible_fraction,
    "fig_hopper": fig_hopper,
    "fig_check_layer": fig_check_layer,
    "fig_contraction": fig_contraction,
    "fig_overview": fig_overview,
    "fig_accuracy": fig_accuracy,
    "fig_tightness": fig_tightness,
    "fig_efficiency": fig_efficiency,
    "fig_decisions": fig_decisions,
    "fig_theory_map": fig_theory_map,
}
TABLES = {
    "tab_summary": tab_summary,
    "tab_double_integrator": tab_double_integrator,
} | {
    f"tab_compare_{s}": (lambda cfg, data, s=s: tab_compare(cfg, data, s))
    for s in ("s2", "s3", "s4", "s5")
}


def main(only: list[str] | None, preview: Path | None) -> None:
    """Write the selected exhibits and ``paper/figures/manifest.json``."""
    want = set(only) if only else set(FIGURES) | set(TABLES)
    unknown = want - set(FIGURES) - set(TABLES)
    if unknown:
        raise SystemExit(f"unknown exhibits: {sorted(unknown)}")
    commit, dirty = git_state()  # before this run rewrites its own outputs
    cfg = load_config()
    data = load_data(cfg)
    fig_dir, tab_dir = Path(cfg["output"]["figures"]), Path(cfg["output"]["tables"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    if preview:
        preview.mkdir(parents=True, exist_ok=True)
    man_path = fig_dir / "manifest.json"
    exhibits = json.loads(man_path.read_text())["exhibits"] if man_path.exists() else {}
    with plt.rc_context(RC):
        for name in sorted(want & set(FIGURES)):
            fig, facts = FIGURES[name](cfg, data)
            w, h = fig.get_size_inches()
            fig.savefig(fig_dir / f"{name}.pdf", metadata={"CreationDate": None})
            if preview:
                fig.savefig(preview / f"{name}.png", dpi=300)
            plt.close(fig)
            exhibits[name] = {
                "file": str(fig_dir / f"{name}.pdf"),
                "size_in": [round(w, 3), round(h, 3)],
                "facts": facts,
            }
            log.info("%s (%.2f x %.2f in)", name, w, h)
    for name in sorted(want & set(TABLES)):
        tex, facts = TABLES[name](cfg, data)
        (tab_dir / f"{name}.tex").write_text(tex)
        exhibits[name] = {"file": str(tab_dir / f"{name}.tex"), "facts": facts}
        log.info("%s", name)
    man_path.write_text(json.dumps({
        "generator": "scripts/paper_figures.py (no new runs)",
        "git_commit": commit, "git_dirty": dirty, "config_sha256": cfg["sha256"],
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(), "matplotlib": matplotlib.__version__,
        "numpy": np.__version__, "inputs": provenance(input_files(cfg, data["mcfg"])),
        "exhibits": dict(sorted(exhibits.items())),
    }, indent=1, default=str) + "\n")  # fmt: skip
    log.info("manifest -> %s", man_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="exhibit names (default: all)")
    ap.add_argument("--preview", type=Path, help="also write PNG previews here")
    args = ap.parse_args()
    main(args.only, args.preview)
