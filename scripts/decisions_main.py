"""Main-text decision figure (Figure 4) from the ``fig_decisions`` facts.

Reads the per-method answer counts that ``paper_figures.py`` records for
``fig_decisions`` in ``exhibits/figures/manifest.json``, keeps the largest
budgets of four studies and six methods (category counts unchanged), writes
``decisions_main.json``, and renders ``fig_decisions_main.pdf`` and ``.png``
beside it.

Usage:
    uv run python scripts/decisions_main.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FIG_DIR = Path("exhibits/figures")
METHODS = [
    ("witness-focused", "Proposed search"),
    ("greedy", "Variance-seeking"),
    ("config", "CONFIG adaptation"),
    ("naive", "Bisection"),
    ("rep16", "Bisection, 16x"),
    ("gp-ucb", "GP-UCB"),
]
STUDIES = [  # (id, name, unit, number of targets)
    ("s6b", "Double integrator", "evaluations", 5),
    ("s2", "Transmission", "power flows", 5),
    ("s3", "Feeder", "checks", 6),
    ("s5", "Covasim", "runs", 6),
]
PARTS = [
    ("passing", "#0072B2", "", "Achievable; design passed"),
    ("proved", "#007F5F", "..", "Not achievable; upper bound"),
    ("unbacked", "#DCEAF6", "///", "Achievable; design failed or absent"),
    ("wrong", "#D55E00", "xx", "Wrong achievable answer"),
    ("undecided", "#E4E4E4", "", "Undecided"),
]


def extract(manifest: dict) -> dict:
    """The main-text subset of the ``fig_decisions`` facts."""
    facts = manifest["exhibits"]["fig_decisions"]["facts"]
    keep = {f"{s}:{m}" for s, *_ in STUDIES for m, _ in METHODS}
    sel = {k: v for k, v in facts.items() if k in keep}
    budget = {s: next(v["budget"] for k, v in sel.items() if k.startswith(f"{s}:"))
              for s, *_ in STUDIES}  # fmt: skip
    return {
        "transformation": "Largest budgets; selected methods and four studies; "
        "category counts unchanged; percentages equal counts divided by runs "
        "times targets. No runs filtered within a selected cell.",
        "methods": [list(m) for m in METHODS],
        "studies": [
            {"id": s, "name": n, "budget": budget[s], "unit": u, "targets": k}
            for s, n, u, k in STUDIES
        ],  # fmt: skip
        "facts": sel,
    }


def main() -> None:
    """Write ``decisions_main.json`` and render the figure."""
    manifest_path = FIG_DIR / "manifest.json"
    data = extract(json.loads(manifest_path.read_text()))
    data["source_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    (FIG_DIR / "decisions_main.json").write_text(json.dumps(data, indent=2) + "\n")
    outputs = [FIG_DIR / f"fig_decisions_main.{ext}" for ext in ("pdf", "png")]
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.45,
        "axes.linewidth": 0.6,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.8))
        fig.subplots_adjust(
            left=0.187, right=0.966, top=0.91, bottom=0.23, wspace=0.88, hspace=0.66
        )
        for panel, (ax, study) in enumerate(
            zip(axes.flat, data["studies"], strict=True)
        ):
            for y, (method, _) in enumerate(data["methods"]):
                row = data["facts"].get(f"{study['id']}:{method}")
                if row is None:
                    ax.text(
                        2, y, "not run", va="center", color="#666666", style="italic"
                    )
                    continue
                if sum(row[key] for key, *_ in PARTS) != row["answers"]:
                    raise ValueError("Decision categories do not partition all answers")
                if row["answers"] != row["runs"] * study["targets"]:
                    raise ValueError("Denominator differs from seeds times targets")
                left = 0.0
                for key, color, hatch, _ in PARTS:
                    width = 100 * row[key] / row["answers"]
                    if width:
                        ax.barh(
                            y,
                            width,
                            left=left,
                            height=0.72,
                            color=color,
                            edgecolor="#404040" if hatch else "white",
                            linewidth=0.2,
                            hatch=hatch,
                        )
                        if width >= 17:
                            ink = (
                                "white"
                                if key in ("passing", "proved", "wrong")
                                else "black"
                            )
                            ax.text(
                                left + width / 2,
                                y,
                                str(row[key]),
                                ha="center",
                                va="center",
                                fontsize=8,
                                color=ink,
                                bbox=dict(facecolor=color, edgecolor="none", pad=0.1),
                            )
                        left += width
                ax.text(104, y, str(row["answers"]), va="center", ha="left", fontsize=8)
            ax.set_title(
                f"({'abcd'[panel]}) {study['name']}\n"
                f"{study['budget']:,} {study['unit']}; {study['targets']} targets",
                loc="left",
                pad=7,
            )
            ax.set_xlim(0, 120)
            ax.set_ylim(5.65, -0.65)
            ax.set_yticks(range(6), [label for _, label in data["methods"]])
            ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
            ax.tick_params(axis="y", length=0, pad=4)
            ax.tick_params(axis="x", length=2.5, pad=2)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_bounds(0, 100)
        handles = [
            Patch(
                facecolor=c,
                edgecolor="#404040" if h else "white",
                hatch=h,
                label=label,
                linewidth=0.3,
            )
            for _, c, h, label in PARTS
        ]
        fig.legend(
            handles=handles,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.51, 0.0),
            fontsize=8,
            frameon=False,
            columnspacing=1.1,
            handlelength=1.6,
            labelspacing=0.45,
            handletextpad=0.5,
        )
        for output in outputs:
            fig.savefig(output, dpi=300, bbox_inches=None, facecolor="white")
        plt.close(fig)
    print(f"rendered {', '.join(str(p) for p in outputs)}")


if __name__ == "__main__":
    main()
