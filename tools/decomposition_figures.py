"""The four figures of the decomposition analysis, built from results/ rather than drawn by hand.

  sets          what the three evaluation sets contain and which difference each isolates
  decomposition the signature image: corpus and generator terms for four detectors
  transfer      per-generator ROC-AUC for every detector, with chance marked
  gate          the fusion gate under JPEG, before and after the identifiability correction

Sources: results/crossgen_seeds.csv, results/clip_probe.csv, results/gate_fix.csv,
results/gate_original.csv.

Usage:
    python tools/paper2_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "paper2" / "figures"

REFERENCE = "gen_SD15"
ORDER = ["CIFAKE-CNN", "DSF-Net", "ResNet-18", "CLIP probe"]
CORPUS_C = "#1565c0"
GEN_C = "#c62828"
INK = "#1a1a1a"
GREY = "#6b7280"
GOOD = "#2e7d32"
FAMILY = {"SD15", "Wukong"}


def setup(plt):
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def load():
    import pandas as pd

    conv = pd.read_csv(RESULTS / "crossgen_seeds.csv")
    clip = pd.read_csv(RESULTS / "clip_probe.csv").assign(arch="CLIP probe")
    return pd.concat([conv, clip], ignore_index=True)


def ci(v):
    from scipy import stats

    v = np.asarray(v, float)
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def shifts(df, arch):
    corpus, generator = [], []
    for s in sorted(df["seed"].unique()):
        d = df[(df["arch"] == arch) & (df["seed"] == s)].set_index("set")["accuracy"]
        if "A" not in d.index or REFERENCE not in d.index:
            continue
        others = [i for i in d.index if i.startswith("gen_") and i != REFERENCE]
        corpus.append((d[REFERENCE] - d["A"]) * 100)
        generator.append((d[others].mean() - d[REFERENCE]) * 100)
    return np.array(corpus), np.array(generator)


def fig_sets(plt):
    """A schematic of the three sets and the two differences they isolate."""
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")

    rows = [
        (2.5, "A", "CIFAR-10 photographs", "SD 1.4", "in distribution"),
        (1.5, "B", "ImageNet photographs", "SD 1.5", "corpus changed, generator held"),
        (0.5, "C", "ImageNet photographs", "five other generators", "both changed"),
    ]
    for y, tag, real, fake, note in rows:
        ax.text(0.15, y, tag, fontsize=11, fontweight="bold", va="center")
        ax.add_patch(plt.Rectangle((0.7, y - 0.3), 3.0, 0.6, facecolor="#e8f0fb",
                                   edgecolor=CORPUS_C, lw=1.0))
        ax.text(2.2, y, real, ha="center", va="center", fontsize=8)
        ax.text(3.85, y, "vs", ha="center", va="center", fontsize=8, color=GREY)
        ax.add_patch(plt.Rectangle((4.05, y - 0.3), 2.6, 0.6, facecolor="#fbeaea",
                                   edgecolor=GEN_C, lw=1.0))
        ax.text(5.35, y, fake, ha="center", va="center", fontsize=8)
        ax.text(6.9, y, note, va="center", fontsize=7.8, color=GREY, style="italic")

    ax.annotate("", xy=(0.45, 1.5), xytext=(0.45, 2.5),
                arrowprops=dict(arrowstyle="<->", color=CORPUS_C, lw=1.4))
    ax.text(0.30, 2.0, "corpus", rotation=90, ha="center", va="center",
            fontsize=8, color=CORPUS_C)
    ax.annotate("", xy=(0.45, 0.5), xytext=(0.45, 1.5),
                arrowprops=dict(arrowstyle="<->", color=GEN_C, lw=1.4))
    ax.text(0.30, 1.0, "generator", rotation=90, ha="center", va="center",
            fontsize=8, color=GEN_C)
    fig.savefig(OUT / "fig_sets.png")
    plt.close(fig)
    print("  fig_sets.png")


def fig_decomposition(plt):
    """The signature image: both terms, four detectors, paired intervals."""
    df = load()
    fig, ax = plt.subplots(figsize=(5.9, 3.2))
    ys = np.arange(len(ORDER))[::-1]
    off = 0.17
    for y, arch in zip(ys, ORDER):
        c, g = shifts(df, arch)
        cm, ch = ci(c)
        gm, gh = ci(g)
        ax.errorbar([cm], [y + off], xerr=[[ch], [ch]], fmt="o", color=CORPUS_C,
                    ms=5.5, capsize=3, lw=1.6, zorder=3)
        ax.errorbar([gm], [y - off], xerr=[[gh], [gh]], fmt="s", color=GEN_C,
                    ms=5, capsize=3, lw=1.6, zorder=3)
        d = np.abs(c) - np.abs(g)
        dm, _ = ci(d)
        # Fixed axis-fraction position: at max(cm, gm) the CLIP label fell off the canvas,
        # and a label that moves with the data is harder to read down the column.
        # Axis fraction must divide by the ylim span, not the row count, or every label
        # drifts upward from the row it belongs to.
        ax.text(1.02, (y + 0.7) / (len(ORDER) + 0.4), f"{dm:+.1f}",
                transform=ax.transAxes,
                fontsize=8.5, va="center", ha="left", color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(ORDER)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.set_xlabel("accuracy change (points)")
    ax.plot([], [], "o", color=CORPUS_C, label="corpus, B - A")
    ax.plot([], [], "s", color=GEN_C, label="generator, C - B")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.16), ncol=2, fontsize=8)
    ax.text(1.02, 1.03, "|corpus|", transform=ax.transAxes, fontsize=7.5,
            va="center", ha="left", color=INK)
    ax.text(1.02, 0.985, "-|gen|", transform=ax.transAxes, fontsize=7.5,
            va="center", ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_decomposition.png")
    plt.close(fig)
    print("  fig_decomposition.png")


def fig_transfer(plt):
    """Per-generator ROC-AUC for each detector, chance marked."""
    df = load()
    gens = sorted({s[4:] for s in df["set"].unique() if s.startswith("gen_")},
                  key=lambda g: (g not in FAMILY, g))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    width = 0.2
    x = np.arange(len(gens))
    colours = ["#5b8def", "#2e7d32", "#c98a00", "#8e44ad"]
    for k, arch in enumerate(ORDER):
        means, errs = [], []
        for g in gens:
            v = df[(df["arch"] == arch) & (df["set"] == f"gen_{g}")]["roc_auc"].to_numpy()
            m, h = ci(v)
            means.append(m); errs.append(h)
        ax.bar(x + (k - 1.5) * width, means, width, yerr=errs, capsize=2,
               color=colours[k], label=arch, error_kw={"lw": 0.9})
    ax.axhline(0.5, color=INK, lw=1.2, ls="--", zorder=3)
    ax.text(len(gens) - 0.45, 0.515, "chance", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([g + (" *" if g in FAMILY else "") for g in gens], fontsize=8)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.35, 0.95)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax.text(0.005, 0.02, "* Stable Diffusion family; models trained on SD 1.4",
            transform=ax.transAxes, fontsize=7.5, color=GREY)
    fig.tight_layout()
    fig.savefig(OUT / "fig_transfer.png")
    plt.close(fig)
    print("  fig_transfer.png")


def fig_gate(plt):
    """The gate under JPEG, before and after the correction, five seeds each."""
    import pandas as pd

    orig_p = RESULTS / "gate_original.csv"
    if not orig_p.exists():
        print("  fig_gate.png skipped: results/gate_original.csv not built yet")
        return
    fixed = pd.read_csv(RESULTS / "gate_fix.csv")
    orig = pd.read_csv(orig_p)
    cols = ["gate_clean", "gate_q90", "gate_q70", "gate_q50", "gate_q30"]
    xs = [100, 90, 70, 50, 30]

    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    for frame, colour, label in ((orig, GREY, "original gate"),
                                 (fixed, GOOD, "identifiable gate")):
        M = frame[cols].to_numpy()
        # Centre each seed on its own clean value: the question is travel, not level.
        M = M - M[:, [0]]
        m, sd = M.mean(0), M.std(0, ddof=1)
        ax.plot(xs, m, "o-", color=colour, lw=1.8, ms=4.5, label=label)
        ax.fill_between(xs, m - sd, m + sd, color=colour, alpha=0.16, lw=0)
    ax.axhline(0, color=INK, lw=0.9, ls="--", zorder=1)
    ax.invert_xaxis()
    ax.set_xlabel("JPEG quality")
    ax.set_ylabel("gate value minus its own clean value")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gate.png")
    plt.close(fig)
    print("  fig_gate.png")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup(plt)
    OUT.mkdir(parents=True, exist_ok=True)
    fig_sets(plt)
    fig_decomposition(plt)
    fig_transfer(plt)
    fig_gate(plt)


if __name__ == "__main__":
    main()
