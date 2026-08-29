"""One figure per detector: the route from a photograph to a decision, drawn on real pixels.

This study's argument is that four detectors read four different things, and that this is why the
corpus and generator terms come out differently for each. Until now that argument was carried
entirely by aggregate accuracies, which look the same whatever the detector is doing inside.

These four figures show the inside. Each one takes the same five images, one from every
evaluation set, and follows them through one detector: what it computes, what it looks at, and
what it decides. The four are deliberately not the same figure with a different title, because
the four processes are not the same process. Under each, the same detector's score distribution
over every image of every set, so the reader sees the whole test and not a summary of it.

Requires data/per_image_predictions.npz from tools/per_image_predictions.py.

Usage:
    python tools/process_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent

def figure_dir(name: str) -> Path:
    """Where the figures go.

    Defaults into results/, so the repository is self-contained and a clone regenerates its
    own figures. Set DSF_FIGURE_OUT to send them somewhere else, which is what a build that
    typesets these figures into a document does.
    """
    override = os.environ.get("DSF_FIGURE_OUT")
    return Path(override) if override else ROOT / "results" / "figures" / name

DATA = ROOT / "data" / "per_image_predictions.npz"
SETS = ROOT / "data" / "crossgen_sets_32.npz"
OUT = figure_dir("process")

INK = "#1a1a1a"
GREY = "#6b7280"
REAL_C = "#1565c0"
FAKE_C = "#c62828"
OK_C = "#2e7d32"

# One colour per evaluation set, for the nearest-neighbour frames.
SRC_C = {"CIF-R": "#1565c0", "CIF-F": "#c62828", "ImgNet": "#00838f",
         "SD15": "#ef6c00", "ADM": "#6a1b9a"}

# One image per evaluation set: the two CIFAKE classes the detectors were trained on, the
# photographs of the corpus shift, the generator they saw and one they never saw.
SHOW = [
    ("A_real", "CIFAKE\nreal", 0, "CIF-R"),
    ("A_fake", "CIFAKE\nfake", 1, "CIF-F"),
    ("imagenet_real", "ImageNet\nreal", 0, "ImgNet"),
    ("gen_SD15", "SD 1.5\nfake", 1, "SD15"),
    ("gen_ADM", "ADM\nfake", 1, "ADM"),
]
COL = 0  # which of the cached showcase images to draw

DIST_SETS = [
    ("A_real", "CIFAKE real", REAL_C, "-"),
    ("A_fake", "CIFAKE fake", FAKE_C, "-"),
    ("imagenet_real", "ImageNet real", REAL_C, "--"),
    ("gen_SD15", "SD 1.5 fake", FAKE_C, "--"),
    ("gen_ADM", "ADM fake", FAKE_C, ":"),
]


def setup(plt):
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def verdict(ax, p, truth, gate=None):
    """The readout: a probability against the 0.5 threshold, green if the call was right."""
    correct = (p >= 0.5) == bool(truth)
    c = OK_C if correct else FAKE_C
    # Values are written past the end of the scale rather than on the bars: at this width a
    # label inside the bar lands on the threshold line as often as not.
    if gate is not None:
        ax.barh([1], [gate], height=0.5, color=GREY, alpha=0.55)
        # Just the number: "gate 0.47" runs past the axes and is overpainted by the next
        # column, and the row label already says which of the two bars this is.
        ax.text(1.06, 1, f"{gate:.2f}", va="center", fontsize=6.8, color=GREY)
    ax.barh([0], [p], height=0.5, color=c, alpha=0.9)
    ax.axvline(0.5, color=INK, lw=0.8, ls="--")
    ax.set_xlim(0, 1.62)
    ax.set_ylim(-0.55, (1.55 if gate is not None else 0.55))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(1.06, 0, f"{p:.2f}", ha="left", va="center", fontsize=7.4, color=c,
            fontweight="bold")
    for sp in ax.spines.values():
        sp.set_visible(False)


def distributions(ax, d, model):
    """Every image of every set, scored by one detector."""
    bins = np.linspace(0, 1, 41)
    for key, label, colour, style in DIST_SETS:
        p = d[f"p_{model}_{key}"]
        h, _ = np.histogram(p, bins=bins, density=True)
        ax.plot(bins[:-1] + 0.0125, h, color=colour, ls=style, lw=1.3, label=label)
    ax.axvline(0.5, color=INK, lw=0.9, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("predicted probability that the image is generated")
    ax.set_ylabel("density")
    ax.legend(fontsize=6.4, ncol=3, loc="upper center", handlelength=1.6,
              columnspacing=1.1, handletextpad=0.4)
    ax.text(0.49, 0.02, "threshold", rotation=90, va="bottom", ha="right",
            fontsize=6.2, color=GREY, transform=ax.get_xaxis_transform())
    ax.margins(y=0.30)


def header(fig, title, subtitle):
    fig.text(0.5, 0.998, title, ha="center", va="top", fontsize=9.6, fontweight="bold",
             color=INK)
    fig.text(0.5, 0.972, subtitle, ha="center", va="top", fontsize=7.0, color=GREY,
             linespacing=1.45)


def image_row(axes, d, cmapless=True):
    for ax, (key, label, truth, _) in zip(axes, SHOW):
        ax.imshow(d[f"show_images_{key}"][COL])
        blank(ax)
        ax.set_title(label, fontsize=7.4, color=REAL_C if truth == 0 else FAKE_C, pad=3)


def fig_proc_cifake(plt):
    """CIFAKE-CNN: two convolutions and a dense layer, with nothing in between to look at."""
    d = np.load(DATA)
    fig = plt.figure(figsize=(5.80, 4.04))
    gs = fig.add_gridspec(4, len(SHOW), height_ratios=[1.05, 1.05, 0.30, 1.6],
                          hspace=0.34, wspace=0.12, top=0.855, bottom=0.09)

    # Fix the filters once so the same four are shown in every column and can be compared.
    energy = np.mean([d[f"show_conv1_{k}"][COL].mean(axis=(1, 2)) for k, _, _, _ in SHOW], axis=0)
    chans = np.argsort(energy)[::-1][:4]

    top = [fig.add_subplot(gs[0, j]) for j in range(len(SHOW))]
    image_row(top, d)
    top[0].set_ylabel("input\n32 x 32", fontsize=7, color=INK, rotation=0,
                      ha="right", va="center", labelpad=8)

    for j, (key, _, truth, _code) in enumerate(SHOW):
        maps = d[f"show_conv1_{key}"][COL][chans]
        tile = np.block([[maps[0], maps[1]], [maps[2], maps[3]]])
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(tile, cmap="magma")
        ax.axhline(31.5, color="white", lw=0.7)
        ax.axvline(31.5, color="white", lw=0.7)
        blank(ax)
        if j == 0:
            ax.set_ylabel("first-layer\nresponses\n(4 of 32)", fontsize=7, color=INK,
                          rotation=0, ha="right", va="center", labelpad=8)
        vax = fig.add_subplot(gs[2, j])
        verdict(vax, float(d[f"show_p_CIFAKE-CNN_{key}"][COL]), truth)
        if j == 0:
            vax.set_ylabel("verdict", fontsize=7, color=INK, rotation=0,
                           ha="right", va="center", labelpad=8)

    ax = fig.add_subplot(gs[3, :])
    distributions(ax, d, "CIFAKE-CNN")
    header(fig, "CIFAKE-CNN reads pixels directly",
           "Two convolutions, a pooling stage and a dense layer. Nothing suppresses content, "
           "so the responses track edges and texture.")
    fig.savefig(OUT / "fig_proc_cifake.png")
    plt.close(fig)
    print("  fig_proc_cifake.png")


def fig_proc_dsfnet(plt):
    """DSF-Net: two streams that see different things, and the gate that weighs them."""
    d = np.load(DATA)
    fig = plt.figure(figsize=(5.80, 6.22))
    gs = fig.add_gridspec(6, len(SHOW), height_ratios=[0.95, 0.95, 0.95, 0.8, 0.34, 1.45],
                          hspace=0.42, wspace=0.14, top=0.885, bottom=0.06)

    top = [fig.add_subplot(gs[0, j]) for j in range(len(SHOW))]
    image_row(top, d)
    top[0].set_ylabel("input", fontsize=7, color=INK, rotation=0, ha="right",
                      va="center", labelpad=8)

    bay = np.concatenate([d[f"show_bayar_{k}"][COL].mean(axis=0) for k, _, _, _ in SHOW])
    blim = float(np.percentile(np.abs(bay), 99))
    for j, (key, _, truth, _code) in enumerate(SHOW):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(d[f"show_bayar_{key}"][COL].mean(axis=0), cmap="RdBu_r",
                  vmin=-blim, vmax=blim)
        blank(ax)
        if j == 0:
            ax.set_ylabel("spatial stream\nconstrained-conv\nresidual", fontsize=7,
                          color=INK, rotation=0, ha="right", va="center", labelpad=8)

        ax = fig.add_subplot(gs[2, j])
        ax.imshow(d[f"show_spectrum_{key}"][COL], cmap="viridis")
        blank(ax)
        if j == 0:
            ax.set_ylabel("frequency stream\nlog-magnitude\nspectrum", fontsize=7,
                          color=INK, rotation=0, ha="right", va="center", labelpad=8)

        ax = fig.add_subplot(gs[3, j])
        ax.plot(d[f"show_radial_{key}"][COL], color=INK, lw=1.2)
        ax.set_xlim(0, 15)
        ax.tick_params(labelsize=6, length=2)
        if j == 0:
            ax.set_xticks([0, 15])
            ax.set_xticklabels(["DC", "Nyquist"], fontsize=6)
            ax.set_ylabel("16-bin radial\nprofile", fontsize=7, color=INK, rotation=0,
                          ha="right", va="center", labelpad=8)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        vax = fig.add_subplot(gs[4, j])
        verdict(vax, float(d[f"show_p_DSF-Net_{key}"][COL]), truth,
                gate=float(d[f"show_gate_{key}"][COL].mean()))
        if j == 0:
            vax.set_ylabel("gate, then\nverdict", fontsize=7, color=INK,
                           rotation=0, ha="right", va="center", labelpad=8)

    ax = fig.add_subplot(gs[5, :])
    distributions(ax, d, "DSF-Net")
    header(fig, "DSF-Net splits the image into two streams and learns how to weigh them",
           "The constrained convolution suppresses content and leaves a residual; the spectrum "
           "and its radial profile carry the frequency evidence.\nThe gate is the mean over "
           "the 128 fused dimensions, and is the only stage in any of the four "
           "detectors that weighs one kind of evidence against another.")
    fig.savefig(OUT / "fig_proc_dsfnet.png")
    plt.close(fig)
    print("  fig_proc_dsfnet.png")


def fig_proc_resnet(plt):
    """ResNet-18: a general-purpose backbone fine-tuned, and where it finds its evidence."""
    d = np.load(DATA)
    fig = plt.figure(figsize=(5.80, 4.12))
    gs = fig.add_gridspec(4, len(SHOW), height_ratios=[1.05, 1.05, 0.30, 1.6],
                          hspace=0.34, wspace=0.12, top=0.855, bottom=0.09)

    top = [fig.add_subplot(gs[0, j]) for j in range(len(SHOW))]
    image_row(top, d)
    top[0].set_ylabel("input", fontsize=7, color=INK, rotation=0, ha="right",
                      va="center", labelpad=8)

    for j, (key, _, truth, _code) in enumerate(SHOW):
        img = d[f"show_images_{key}"][COL]
        cam = d[f"show_cam_resnet_{key}"][COL]
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(img.mean(axis=2), cmap="gray", alpha=0.85)
        ax.imshow(cam, cmap="inferno", alpha=0.55, extent=(0, 32, 32, 0),
                  interpolation="bilinear")
        blank(ax)
        if j == 0:
            ax.set_ylabel("Grad-CAM\nover the final\nblock (4 x 4)", fontsize=7, color=INK,
                          rotation=0, ha="right", va="center", labelpad=8)
        vax = fig.add_subplot(gs[2, j])
        verdict(vax, float(d[f"show_p_ResNet-18_{key}"][COL]), truth)
        if j == 0:
            vax.set_ylabel("verdict", fontsize=7, color=INK, rotation=0,
                           ha="right", va="center", labelpad=8)

    ax = fig.add_subplot(gs[3, :])
    distributions(ax, d, "ResNet-18")
    header(fig, "ResNet-18 brings an ImageNet backbone and localises its evidence",
           "Eleven million parameters pretrained on object recognition, then fine-tuned. The "
           "attribution is coarse because a 32 x 32 input leaves the final block only "
           "4 x 4 cells.")
    fig.savefig(OUT / "fig_proc_resnet.png")
    plt.close(fig)
    print("  fig_proc_resnet.png")


def fig_proc_clip(plt):
    """The CLIP probe: frozen semantic features, a linear decision, and what it neighbours."""
    d = np.load(DATA)
    sets = np.load(SETS)
    fig = plt.figure(figsize=(5.80, 5.89))
    gs = fig.add_gridspec(5, len(SHOW), height_ratios=[0.95, 0.85, 0.30, 0.52, 1.45],
                          hspace=0.55, wspace=0.14, top=0.885, bottom=0.06)

    top = [fig.add_subplot(gs[0, j]) for j in range(len(SHOW))]
    image_row(top, d)
    top[0].set_ylabel("input\n(upsampled\nto 224)", fontsize=7, color=INK, rotation=0,
                      ha="right", va="center", labelpad=8)

    clim = 1.05 * max(np.abs(np.sort(np.abs(d[f"show_clipcontrib_{k}"][COL]))[-12:]).max()
                      for k, _, _, _ in SHOW)
    for j, (key, _, truth, _code) in enumerate(SHOW):
        contrib = d[f"show_clipcontrib_{key}"][COL]
        order = np.argsort(np.abs(contrib))[::-1][:12]
        vals = contrib[order]
        ax = fig.add_subplot(gs[1, j])
        ax.bar(range(12), vals, color=[FAKE_C if v > 0 else REAL_C for v in vals], width=0.75)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xticks([])
        ax.set_ylim(-clim, clim)
        ax.tick_params(labelsize=6, length=2)
        if j:
            ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("12 largest of the\n512 per-dimension\ncontributions",
                          fontsize=7, color=INK, rotation=0, ha="right", va="center",
                          labelpad=8)
        vax = fig.add_subplot(gs[2, j])
        verdict(vax, float(d[f"show_p_CLIP probe_{key}"][COL]), truth)
        if j == 0:
            vax.set_ylabel("verdict", fontsize=7, color=INK, rotation=0,
                           ha="right", va="center", labelpad=8)

    # What CLIP puts next to each image, pooled across every set. If the frozen space were
    # organised by acquisition trace the neighbours would share a source; if by content, they
    # share a subject. This is the mechanism claim, drawn rather than asserted.
    pool_keys = [k for k, _, _, _ in SHOW]
    feats = np.concatenate([d[f"clipfeat_{k}"] for k in pool_keys])
    imgs = np.concatenate([sets[k] for k in pool_keys])
    origin = np.concatenate([[code] * len(d[f"clipfeat_{k}"])
                             for k, _, _, code in SHOW])
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)

    label_ax = fig.add_subplot(gs[3, 0])
    label_ax.set_ylabel("four nearest\nneighbours in the\nfrozen space,\npooled"
                        " over\nall five sets", fontsize=7, color=INK, rotation=0,
                        ha="right", va="center", labelpad=8)
    blank(label_ax)
    label_ax.patch.set_alpha(0)

    inner = gs[3, :].subgridspec(1, len(SHOW), wspace=0.22)
    for j, (key, label, _t, _code) in enumerate(SHOW):
        q = d[f"clipfeat_{key}"][int(d[f"show_idx_{key}"][COL])]
        q = q / np.linalg.norm(q)
        nn = np.argsort(feats @ q)[::-1][1:5]
        cell = inner[j].subgridspec(1, 4, wspace=0.04)
        for c, idx in enumerate(nn):
            ax = fig.add_subplot(cell[c])
            ax.imshow(imgs[idx])
            blank(ax)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(SRC_C[origin[idx]])
                sp.set_linewidth(1.6)

    box = label_ax.get_position()
    fig.legend(handles=[Patch(facecolor=SRC_C[c], label=c) for _, _, _, c in SHOW],
               loc="upper center", bbox_to_anchor=(0.55, box.y0 + 0.012), ncol=5,
               fontsize=6.4, handlelength=1.0, handleheight=0.9, columnspacing=1.2,
               handletextpad=0.4, title="frame colour gives the set a neighbour came from",
               title_fontsize=6.4)

    ax = fig.add_subplot(gs[4, :])
    distributions(ax, d, "CLIP probe")
    header(fig, "The CLIP probe never sees a pixel it was trained on",
           "A frozen ViT-B/16 encodes the image and a linear probe reads 512 numbers. "
           "Counted over all 4,000 images, 96.5% of its neighbours come from the query's"
           " own corpus\nagainst a 50% floor, the strongest corpus grouping of any "
           "representation here; with content held fixed they barely track the generator.")
    fig.savefig(OUT / "fig_proc_clip.png")
    plt.close(fig)
    print("  fig_proc_clip.png")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run tools/per_image_predictions.py first")
    setup(plt)
    OUT.mkdir(parents=True, exist_ok=True)
    fig_proc_cifake(plt)
    fig_proc_dsfnet(plt)
    fig_proc_resnet(plt)
    fig_proc_clip(plt)


if __name__ == "__main__":
    main()
