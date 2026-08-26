"""Run the trained DSF-Net on an arbitrary image, and say plainly why the answer is unreliable.

The model in this repository was trained on CIFAKE: 32x32 images, CIFAR-10 object classes,
fakes from Stable Diffusion v1.4 only. Feeding it a photograph from a phone, or an image
from a modern generator, takes it a long way outside that distribution in at least four
directions at once:

  * **Resolution.** The evidence the model reads is a high-frequency artefact. Downscaling
    is a low-pass filter, so resizing a 1024px image to 32x32 removes most of the signal
    before the model ever sees it. The study measured this: a mere 2x downscale drops
    accuracy from 95.77% to 60.31%, and a Gaussian blur of sigma 1.0 drops it to 56.02%.
  * **Generator.** Every fake in training came from Stable Diffusion v1.4. Other generators
    leave different fingerprints, or nearly none. Cross-generator transfer is the central
    open problem in this field and is untested in this project.
  * **Content.** CIFAKE covers ten CIFAR-10 categories: aeroplane, car, bird, cat, deer,
    dog, frog, horse, ship, truck. The model has never seen a human face.
  * **Post-processing.** Camera JPEG, resizing and filters each attenuate the fingerprint
    further. JPEG quality 30 alone costs 7.3 points.

So this script is not a detector for real-world images, and it does not pretend to be. It
exists to make that failure legible: it shows what the model actually receives after
downscaling, what it says, and how far outside its training distribution the input is. A
confident-looking answer here means nothing, and the script says so on every run.

Usage:
    python tools/try_image.py photo.jpg
    python tools/try_image.py real.jpg generated.png      # side by side
    python tools/try_image.py *.jpg --no-show
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from demo import (  # noqa: E402
    CKPT, FAKE_COLOUR, NOTEBOOK, REAL_COLOUR,
    load_model, normalise, require, stream_views,
)
from smoke_test import load_smoke_namespace  # noqa: E402

# Measured on the study's own test set; quoted so the warnings carry evidence rather than
# vague hedging. Source: results/robustness.csv.
ACC_CLEAN = 0.9577
ACC_RESCALE_HALF = 0.6031
ACC_BLUR_1 = 0.5602

TRAIN_CLASSES = ("aeroplane, car, bird, cat, deer, dog, frog, horse, ship, truck")


def load_image(path: Path):
    """Return (original PIL image, 32x32 uint8 array, downscale factor)."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    factor = max(img.size) / 32.0
    small = np.array(img.resize((32, 32), Image.BICUBIC), dtype=np.uint8)
    return img, small, factor


def distribution_warnings(path: Path, factor: float, aspect: float | None = None) -> list:
    """Everything about this input that puts it outside what the model was trained on."""
    notes = []
    if factor > 1.5:
        notes.append(
            f"downscaled {factor:.0f}x to reach 32x32. Downscaling is a low-pass filter and the "
            f"fingerprint lives in the high frequencies. A 2x downscale alone took this model "
            f"from {ACC_CLEAN:.1%} to {ACC_RESCALE_HALF:.1%} on its own test set."
        )
    elif factor < 1.0:
        notes.append(f"upscaled {1 / factor:.1f}x to reach 32x32; no fingerprint is created by upscaling.")

    if aspect is not None and (aspect > 1.25 or aspect < 0.8):
        notes.append(
            f"squashed to a square: the source is {aspect:.2f}:1, and resizing to 32x32 without "
            "preserving the aspect ratio resamples the two axes by different factors, which "
            "distorts the very periodicity the frequency stream measures."
        )

    if path.suffix.lower() in {".jpg", ".jpeg"}:
        notes.append("already JPEG-compressed before it reached this script, which attenuates the "
                     "artefact further; JPEG q30 alone costs 7.3 points.")

    notes.append(f"the model was trained only on Stable Diffusion v1.4 fakes. If this image came "
                 f"from any other generator, its fingerprint is one the model has never seen.")
    notes.append(f"the training classes are {TRAIN_CLASSES}. Anything else, a face above all, is "
                 f"content the model never encountered.")
    return notes


def verdict_line(prob: float) -> tuple:
    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if label == "FAKE" else 1.0 - prob
    return label, confidence


def make_figure(ns, model, device, items, out_path: Path, show: bool):
    """One row per image: what you gave it, what it received, its spectrum, what it said."""
    plt = ns["plt"]

    smalls = np.stack([it["small"] for it in items])
    _, spectrum, _ = stream_views(ns, model, smalls, device)

    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0, 1.5]})
    axes = np.atleast_2d(axes)

    titles = [
        "1.  What you gave it",
        "2.  What the model receives\n(32x32, after downscaling)",
        "3.  Log-magnitude spectrum",
        "4.  What it says, and what that is worth",
    ]

    for i, item in enumerate(items):
        prob = item["prob"]
        label, confidence = verdict_line(prob)
        colour = FAKE_COLOUR if label == "FAKE" else REAL_COLOUR

        ax = axes[i, 0]
        ax.imshow(item["original"])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{item['path'].name}\n{item['original'].size[0]}x{item['original'].size[1]}",
                      fontsize=9)

        ax = axes[i, 1]
        ax.imshow(item["small"], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"downscaled {item['factor']:.0f}x", fontsize=8, color="#c62828")

        ax = axes[i, 2]
        ax.imshow(spectrum[i], cmap="magma", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[i, 3]
        ax.axis("off")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-2.75, 0.95)
        ax.barh([0], [prob], color=colour, height=0.42)
        ax.barh([0], [1.0 - prob], left=[prob], color="#eceff1", height=0.42)
        ax.add_patch(plt.Rectangle((0, -0.21), 1.0, 0.42, fill=False, ec="#9e9e9e", lw=0.9))
        ax.plot([0.5, 0.5], [-0.34, 0.34], color="#212121", ls="--", lw=1.3)
        ax.text(0.5, 0.44, "threshold 0.5", ha="center", fontsize=7.5, color="#424242")
        ax.text(0.0, -0.34, "0", ha="center", va="top", fontsize=7.5, color="#616161")
        ax.text(0.5, -0.34, "P(FAKE)", ha="center", va="top", fontsize=8, color="#616161")
        ax.text(1.0, -0.34, "1", ha="center", va="top", fontsize=7.5, color="#616161")
        ax.text(0.0, -1.15, f"P(FAKE) = {prob:.3f}", fontsize=10.5, va="center")
        ax.text(0.0, -1.75, f"says {label}, {confidence:.1%} confident",
                fontsize=10.5, fontweight="bold", color=colour, va="center")
        ax.text(0.0, -2.40,
                "OUT OF DISTRIBUTION\nthis number is not evidence",
                fontsize=10, fontweight="bold", color=FAKE_COLOUR, va="center")

        if i == 0:
            for col in range(4):
                axes[i, col].set_title(titles[col], fontsize=10.5, pad=10)

    fig.suptitle("DSF-Net outside its training distribution", fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.005,
        "The model was trained on 32x32 CIFAKE images with Stable Diffusion v1.4 fakes. On this input "
        f"it is guessing: a 2x downscale alone already takes it from {ACC_CLEAN:.0%} to "
        f"{ACC_RESCALE_HALF:.0%}, and these images are downscaled far more than that. "
        "This figure demonstrates limitation 2 of the report, it does not detect anything.",
        ha="center", fontsize=9.5, color="#c62828",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    try:
        shown = out_path.resolve().relative_to(ROOT)
    except ValueError:
        shown = out_path
    print(f"\n  figure written to {shown}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path, help="image files to try")
    parser.add_argument("--no-show", dest="show", action="store_false", help="do not open a window")
    parser.add_argument("--no-figure", dest="figure", action="store_false", help="text output only")
    parser.add_argument("--out", type=Path, default=ROOT / "try_image_output.png")
    args = parser.parse_args()

    require(CKPT, "trained checkpoint", "run notebooks/AIGID_main.ipynb once; training writes it")
    missing = [p for p in args.images if not p.exists()]
    if missing:
        sys.exit("no such file: " + ", ".join(str(p) for p in missing))

    print("DSF-Net on out-of-distribution images")
    print("=" * 70)
    print("  READ THIS FIRST: this model was trained on 32x32 CIFAKE images whose fakes all")
    print("  came from Stable Diffusion v1.4. Ordinary photographs and images from other")
    print("  generators are outside that distribution. Treat every number below as a")
    print("  demonstration of that limitation, not as a detection result.")
    print()

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        ns = load_smoke_namespace(NOTEBOOK)
    torch = ns["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_params, _, _ = load_model(ns, device)
    print(f"  model: DSF-Net (tuned), {n_params:,} parameters, trained on CIFAKE at 32x32")
    print()

    items = []
    for path in args.images:
        original, small, factor = load_image(path)
        w, h = original.size
        items.append({"path": path, "original": original, "small": small,
                      "factor": factor, "aspect": w / h})

    smalls = np.stack([it["small"] for it in items])
    with torch.no_grad():
        x = normalise(ns, smalls).to(device)
        z, g = model.embed(x)
        probs = torch.sigmoid(model.head(z)).squeeze(1).float().cpu().numpy()
        gates = (g.mean(dim=1).float().cpu().numpy() if g is not None
                 else np.full(len(items), np.nan, dtype=np.float32))

    for item, prob, gate in zip(items, probs, gates):
        item["prob"] = float(prob)
        label, confidence = verdict_line(float(prob))
        w, h = item["original"].size
        print(f"  {item['path'].name}")
        print(f"    original {w}x{h}  ->  32x32  ({item['factor']:.0f}x downscale)")
        print(f"    P(FAKE) = {prob:.3f}  ->  says {label}, {confidence:.1%} confident, "
              f"gate {gate:.3f}")
        print("    why this answer is not evidence:")
        for note in distribution_warnings(item["path"], item["factor"], item["aspect"]):
            print(f"      - {note}")
        print()

    if args.figure:
        make_figure(ns, model, device, items, args.out, args.show)

    print()
    print("  Bottom line: on inputs like these the model is close to guessing. The honest")
    print("  demonstration is the failure itself, which is limitation 2 in report/report.md.")


if __name__ == "__main__":
    main()
