"""Build the evidence figures from the data and the recorded results.

This study defines the log-magnitude spectrum in Eq. 2 and the radial profile in Eq. 3, reports
seed spreads in a table and paired intervals in another, and until now showed a reader none of
it. That is a real gap: a reviewer asked to accept a negative result about a spectral pathway
should be able to see the spectra, and one asked to accept an effect-size floor should be able
to see the spread that sets it.

Three figures, each computed here rather than drawn:

  data      what the two streams actually receive: real and generated images, their spectra,
            and the mean radial profile of each class
  spread    the five runs of each of three architectures, as deviations from that
            architecture's own mean, which is what makes the spreads comparable
  forest    every variant against the full model at five seeds, as paired 95% intervals

Sources: data/cifake_cache.npz, results/seeds.csv, results/arch_seeds.csv.

Usage:
    python tools/evidence_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures"

BASELINE = "4. gated fusion (full)"
REAL = "#1565c0"
FAKE = "#c62828"
INK = "#1a1a1a"
GREY = "#6b7280"
GOOD = "#2e7d32"


def setup(plt):
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def log_spectrum(img_u8: np.ndarray) -> np.ndarray:
    """Eq. 2 of the paper: centred log-magnitude spectrum of the greyscale image."""
    grey = img_u8.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(grey))))


def radial_profile(spec: np.ndarray, bins: int = 16) -> np.ndarray:
    """Eq. 3: azimuthal average of the spectrum over `bins` concentric bands."""
    h, w = spec.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.hypot(y - cy, x - cx)
    r = r / r.max()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, bins - 1)
    return np.array([spec[idx == k].mean() for k in range(bins)])


def fig_data(plt):
    """What the two streams receive, and where the classes differ spectrally.

    Images, their spectra from Eq. 2, and the mean radial profile from Eq. 3, in one float.
    This study defines both transforms and would otherwise never show the reader either.
    """
    cache = np.load(ROOT / "data" / "cifake_cache.npz")
    X, y = cache["X_test"], cache["y_test"]
    rng = np.random.default_rng(42)
    # CIFAKE labels generated images 1 and photographs 0.
    real_idx = rng.choice(np.flatnonzero(y == 0), 4, replace=False)
    fake_idx = rng.choice(np.flatnonzero(y == 1), 4, replace=False)

    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(3, 9, height_ratios=[1, 1, 1.55],
                          width_ratios=[1, 1, 1, 1, 0.3, 1, 1, 1, 1],
                          hspace=0.10, wspace=0.10, top=0.90, bottom=0.09)

    for col, i in enumerate(real_idx):
        for row, arr in ((0, X[i]), (1, log_spectrum(X[i]))):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(arr, cmap=None if row == 0 else "magma")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("image" if row == 0 else "spectrum", fontsize=7.5)
    for col, i in enumerate(fake_idx):
        for row, arr in ((0, X[i]), (1, log_spectrum(X[i]))):
            ax = fig.add_subplot(gs[row, 5 + col])
            ax.imshow(arr, cmap=None if row == 0 else "magma")
            ax.set_xticks([]); ax.set_yticks([])

    fig.text(0.265, 0.925, "photographs (CIFAR-10)", ha="center", fontsize=8.5, color=REAL)
    fig.text(0.745, 0.925, "generated (Stable Diffusion v1.4)", ha="center", fontsize=8.5,
             color=FAKE)

    ax = fig.add_subplot(gs[2, :])
    n = 600
    for label, colour, name in ((0, REAL, "photographs"), (1, FAKE, "generated")):
        idx = rng.choice(np.flatnonzero(y == label), n, replace=False)
        prof = np.stack([radial_profile(log_spectrum(X[i])) for i in idx])
        m, sd = prof.mean(0), prof.std(0)
        k = np.arange(1, prof.shape[1] + 1)
        ax.plot(k, m, color=colour, lw=1.8, label=f"{name} (n={n})")
        ax.fill_between(k, m - sd, m + sd, color=colour, alpha=0.16, lw=0)
    ax.set_xlabel("radial frequency band $k$ of Eq. 3")
    ax.set_ylabel(r"mean $\log(1+|F|)$")
    ax.set_xlim(1, 16)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.savefig(OUT / "fig_data.png")
    plt.close(fig)
    print("  fig_data.png")


def fig_spread(plt):
    """Five runs of each architecture, centred on their own means so spreads compare."""
    import pandas as pd

    arch = pd.read_csv(RESULTS / "arch_seeds.csv")
    seeds = pd.read_csv(RESULTS / "seeds.csv")
    groups = {
        "CIFAKE-CNN": arch[arch["arch"] == "CIFAKE-CNN"]["test_acc"].to_numpy() * 100,
        "DSF-Net": seeds[seeds["variant"] == BASELINE]["test_acc"].to_numpy() * 100,
        "ResNet-18": arch[arch["arch"] == "ResNet-18"]["test_acc"].to_numpy() * 100,
    }
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    rng = np.random.default_rng(0)
    for i, (name, vals) in enumerate(groups.items()):
        dev = vals - vals.mean()
        ax.scatter(i + rng.uniform(-0.07, 0.07, len(dev)), dev, s=34, color=INK,
                   zorder=3, alpha=0.85)
        sd = vals.std(ddof=1)
        ax.errorbar(i, 0, yerr=sd, color=GOOD, lw=2.2, capsize=7, zorder=2)
        ax.text(i, sd + 0.10, f"sd {sd:.3f}", ha="center", fontsize=8, color=GOOD)
    ax.axhline(0, color=GREY, lw=0.9, ls="--", zorder=1)
    ax.margins(y=0.22)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{n}\n{v.mean():.2f}%" for n, v in groups.items()])
    ax.set_ylabel("test accuracy minus own mean (points)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_spread.png")
    plt.close(fig)
    print("  fig_spread.png")


def fig_forest(plt):
    """Every variant against the full model at five seeds, paired 95% intervals."""
    import pandas as pd
    from scipy import stats

    df = pd.read_csv(RESULTS / "seeds.csv")
    base = df[df["variant"] == BASELINE].set_index("seed")["test_acc"]
    rows = []
    for v in sorted(df["variant"].unique()):
        if v == BASELINE:
            continue
        other = df[df["variant"] == v].set_index("seed")["test_acc"]
        shared = sorted(set(base.index) & set(other.index))
        d = (other.loc[shared] - base.loc[shared]).to_numpy() * 100
        half = float(stats.t.ppf(0.975, len(shared) - 1) * d.std(ddof=1) / np.sqrt(len(shared)))
        rows.append((v[3:], float(d.mean()), half))

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ys = np.arange(len(rows))[::-1]
    for y, (name, m, h) in zip(ys, rows):
        crosses = (m - h) < 0 < (m + h)
        colour = GREY if crosses else (GOOD if m > 0 else FAKE)
        ax.errorbar([m], [y], xerr=[[h], [h]], fmt="o", color=colour, ms=5.5,
                    capsize=3.5, lw=1.6, zorder=3)
        ax.annotate(f"{m:+.2f}", (m, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7.5, color=colour, zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.axvline(0, color=INK, lw=1, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("change when the component is removed or added (points)")
    ax.text(0.99, 0.03, "grey: interval spans zero", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=GREY)
    fig.tight_layout()
    fig.savefig(OUT / "fig_forest.png")
    plt.close(fig)
    print("  fig_forest.png")


def fig_risk(plt):
    """Single-run disagreement against effect size, over all 21 comparisons.

    This replaces the two-panel escalation figure. Its left panel plotted the four numbers of
    the escalation table and nothing else; this is the part that is not already a table.
    """
    import json

    risk = json.loads((RESULTS / "single_run_risk.json").read_text(encoding="utf-8"))
    xs = [abs(r["paired_delta_pp"]) for r in risk]
    ys = [r["any_pairing_disagreement"] for r in risk]
    cols = [GOOD if r["resolved"] else GREY for r in risk]

    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ax.scatter(xs, ys, c=cols, s=40, zorder=3, edgecolors="white", linewidths=0.7)
    ax.axhline(0.5, color=FAKE, lw=1.1, ls=":", zorder=1)
    ax.text(max(xs) * 0.99, 0.52, "coin toss", fontsize=8, color=FAKE, ha="right")
    ax.axvline(0.5, color=INK, lw=0.9, ls="--", zorder=1)
    ax.text(0.56, 0.30, "floor", fontsize=8, color=INK)
    ax.set_xlabel(r"size of the paired effect $|\bar{\delta}^{(uv)}|$ (points)")
    ax.set_ylabel(r"$R_{\mathrm{any}}$")
    ax.set_ylim(-0.04, 0.62)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.text(0.99, 0.94, "grey: unresolved at five seeds", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=GREY)
    fig.tight_layout()
    fig.savefig(OUT / "fig_risk.png")
    plt.close(fig)
    print("  fig_risk.png")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup(plt)
    OUT.mkdir(parents=True, exist_ok=True)
    fig_data(plt)
    fig_spread(plt)
    fig_forest(plt)
    fig_risk(plt)


if __name__ == "__main__":
    main()
