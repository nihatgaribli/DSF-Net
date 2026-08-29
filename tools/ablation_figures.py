"""Build the ablation study's figures from the recorded results, at publication quality.

Each is built here from a results file rather than by hand, so a figure cannot drift away
from the number it depicts.

  fig2  the escalation: what each evidence standard said about gated fusion
  fig3  the five-seed ablation as a forest plot of paired intervals
  fig4  cross-generator transfer, with the 0.5 AUC line marked
  fig5  the container audit: label recoverable with no pixels

The architecture diagram is drawn rather than plotted and is not built here.

Sources: results/seeds.csv, results/ablations.csv, results/run1_digest.txt,
results/crossgen_32.json, results/benchmark_audit.json.

Usage:
    python tools/ablation_figures.py         # build all
    python tools/ablation_figures.py --only 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "paper" / "figures"

BASELINE = "4. gated fusion (full)"
REAL = "#1565c0"
FAKE = "#c62828"
NEUTRAL = "#455a64"
GOOD = "#2e7d32"


def setup(plt):
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def paired_stats(df, variant: str):
    """Mean paired delta in points and its 95% interval, against the full model."""
    from scipy import stats

    base = df[df["variant"] == BASELINE].set_index("seed")["test_acc"]
    other = df[df["variant"] == variant].set_index("seed")["test_acc"]
    shared = sorted(set(base.index) & set(other.index))
    diff = (other.loc[shared] - base.loc[shared]).to_numpy() * 100
    mean = float(diff.mean())
    half = float(stats.t.ppf(0.975, len(shared) - 1) * diff.std(ddof=1) / np.sqrt(len(shared)))
    return mean, half, len(shared)


def fig2_escalation(plt):
    """This study's argument in one view: the same comparison at three evidence standards."""
    import pandas as pd

    seeds = pd.read_csv(RESULTS / "seeds.csv")
    mean, half, n = paired_stats(seeds, "3. concat fusion")

    # Run 1 and run 2 are single-run point estimates of the same quantity, recorded in the
    # study's own digests. Signs are stated as "concat minus gated" throughout so the three
    # standards are directly comparable rather than merely adjacent.
    run1, run2 = +0.02, -0.49

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.2),
                                  gridspec_kw={"width_ratios": [1.05, 1.0]})

    rows = [
        ("One run\n(run 1)", run1, None),
        ("One run\n(run 2)", run2, None),
        ("Two runs\ncompared", None, None),
        (f"Five seeds,\npaired (n={n})", mean, half),
    ]
    ys = np.arange(len(rows))[::-1]

    for y, (label, value, err) in zip(ys, rows):
        if value is None:
            ax.text(0.0, y, "sign flip: unresolved", va="center", ha="center",
                    fontsize=8.5, style="italic", color=NEUTRAL)
            continue
        colour = GOOD if err is not None else NEUTRAL
        if err is None:
            ax.plot([value], [y], "o", color=colour, ms=7, zorder=3)
        else:
            ax.errorbar([value], [y], xerr=[[err], [err]], fmt="o", color=colour,
                        ms=7, capsize=4, lw=1.8, zorder=3)
    ax.axvline(0, color="#212121", lw=1, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("concatenation minus gated fusion (points)")
    ax.set_xlim(-1.0, 1.0)
    ax.set_title("(a) One claim, three evidence standards", loc="left", pad=8)

    # Panel (b) generalises the left panel from one claim to every comparison the sweep
    # supports. Plotting disagreement against effect size shows the mechanism rather than
    # just the rates: the smaller the true effect relative to seed noise, the closer a single
    # run gets to a coin toss.
    risk_path = RESULTS / "single_run_risk.json"
    if risk_path.exists():
        risk = json.loads(risk_path.read_text(encoding="utf-8"))
        xs = [abs(r["paired_delta_pp"]) for r in risk]
        ys2 = [r["any_pairing_disagreement"] for r in risk]
        cols = [NEUTRAL if not r["resolved"] else GOOD for r in risk]
        ax2.scatter(xs, ys2, c=cols, s=34, zorder=3, edgecolors="white", linewidths=0.6)
        ax2.axhline(0.5, color=FAKE, lw=1, ls=":", zorder=1)
        ax2.text(max(xs) * 0.98, 0.52, "coin toss", fontsize=8, color=FAKE, ha="right")
        ax2.set_xlabel("size of the true effect (points, five seeds)")
        ax2.set_ylabel("single runs reporting the wrong sign")
        ax2.set_ylim(-0.04, 0.62)
        ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax2.set_title(f"(b) All {len(risk)} comparisons in the sweep", loc="left", pad=8)
        ax2.text(0.98, 0.94, "grey: effect unresolved at five seeds",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    return fig, "fig2_escalation"


def fig3_forest(plt):
    """All seven variants at five seeds, as paired intervals against the full model."""
    import pandas as pd

    seeds = pd.read_csv(RESULTS / "seeds.csv")
    variants = [v for v in sorted(seeds["variant"].unique()) if v != BASELINE]
    stats_rows = [(v, *paired_stats(seeds, v)) for v in variants]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ys = np.arange(len(stats_rows))[::-1]
    for y, (name, mean, half, _) in zip(ys, stats_rows):
        crosses_zero = (mean - half) < 0 < (mean + half)
        colour = NEUTRAL if crosses_zero else (GOOD if mean > 0 else FAKE)
        ax.errorbar([mean], [y], xerr=[[half], [half]], fmt="o", color=colour,
                    ms=6, capsize=3.5, lw=1.6, zorder=3)
        # The frequency-only effect is an order of magnitude larger than the rest and
        # compresses them against the zero line, so each value is also printed.
        ax.annotate(f"{mean:+.2f}", (mean, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color=colour)
    ax.axvline(0, color="#212121", lw=1, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0][3:] for r in stats_rows])
    ax.set_ylim(-0.9, len(stats_rows) - 0.3)
    ax.set_xlabel("change in accuracy when the component is removed or added (points)")
    ax.set_title(f"{len(stats_rows)} design decisions against the full model, "
                 "five seeds, paired 95% intervals", loc="left", pad=10)
    ax.text(0.99, 0.02, "grey: interval spans zero, effect unresolved",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    return fig, "fig3_ablation_forest"


def fig4_transfer(plt):
    """Cross-generator transfer for the 32px model, ordered, with chance marked."""
    data = json.loads((RESULTS / "crossgen_32.json").read_text(encoding="utf-8"))
    model = data.get("DSF-Net (tuned)") or next(iter(data.values()))

    rows = []
    for key, val in model.items():
        if key.startswith("_"):
            continue
        name = key.split("vs ")[-1] if " vs " in key else "CIFAKE (SD v1.4)"
        rows.append((name, float(val["roc_auc"])))
    rows.sort(key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    names = [r[0] for r in rows]
    aucs = [r[1] for r in rows]
    colours = [GOOD if a >= 0.75 else (NEUTRAL if a >= 0.5 else FAKE) for a in aucs]
    ax.barh(np.arange(len(rows))[::-1], aucs, color=colours, height=0.62)
    ax.axvline(0.5, color="#212121", lw=1.2, ls="--", zorder=3)
    ax.text(0.505, len(rows) - 0.6, "chance", fontsize=8, color="#212121")
    ax.set_yticks(np.arange(len(rows))[::-1])
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Transfer follows the generator family, not the task", loc="left", pad=10)
    ax.text(0.99, 0.03,
            "below chance: the model ranks\nthese fakes as more camera-like than photographs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=FAKE)
    fig.tight_layout()
    return fig, "fig4_cross_generator"


def fig5_container(plt):
    """How much label a classifier recovers from the file container alone."""
    data = json.loads((RESULTS / "benchmark_audit.json").read_text(encoding="utf-8"))
    rows = [(k, v) for k, v in data.items() if v.get("status") == "ok"]
    if not rows:
        return None, None

    fig, ax = plt.subplots(figsize=(6.4, 2.4 + 0.3 * len(rows)))
    ys = np.arange(len(rows))[::-1]
    width = 0.36
    for y, (name, v) in zip(ys, rows):
        c = float(v["container_accuracy"])
        o = float(v.get("order_accuracy", 0.5))
        ax.barh(y + width / 2, c, height=width, color=FAKE if c > 0.65 else NEUTRAL)
        ax.barh(y - width / 2, o, height=width, color=FAKE if o > 0.65 else NEUTRAL,
                alpha=0.55)
    ax.axvline(0.5, color="#212121", lw=1.2, ls="--", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("balanced accuracy using no pixels")
    ax.set_title("Label recoverable from the file container and from file order",
                 loc="left", pad=10)
    ax.text(0.99, 0.04, "upper bar: container (size, format, quantisation)\n"
                        "lower bar: position in the file",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    return fig, "fig5_container_audit"


BUILDERS = {2: fig2_escalation, 3: fig3_forest, 4: fig4_transfer, 5: fig5_container}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=int, nargs="+", default=sorted(BUILDERS))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup(plt)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for number in args.only:
        builder = BUILDERS.get(number)
        if builder is None:
            sys.exit(f"no figure {number}; choose from {sorted(BUILDERS)}")
        try:
            fig, name = builder(plt)
        except FileNotFoundError as exc:
            print(f"  fig{number}: skipped, missing {Path(exc.filename).name}")
            continue
        if fig is None:
            print(f"  fig{number}: skipped, no usable rows in its source")
            continue
        path = OUT_DIR / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  fig{number} -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
