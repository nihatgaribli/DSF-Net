"""Classify a whole photograph with the 256px model, by cropping rather than resizing.

This is the tool the high-resolution track was built for. `tools/try_image.py` runs the
32x32 model and its answer on a real photograph is close to a coin flip, because reaching
32x32 means downscaling by a factor of thirty or more and downscaling is a low-pass filter
that removes the evidence before the model sees it. A controlled test made that concrete:
two 1024x1024 images identical except for a planted generator artefact scored 0.260 and
0.257, indistinguishable, while the same artefact planted directly at 32x32 moved the model
from 0.260 to 0.669.

The fix is not a better model, it is never resizing. This script takes 256x256 windows at
the image's native resolution, scores each one, and aggregates. Nothing is downscaled, so
the fingerprint that survives in the file also survives into the model.

Aggregation is the mean probability across windows, with the spread reported alongside it.
The spread matters: windows of an image disagreeing wildly means the evidence is thin and
the mean is not worth much, and that is information the single number hides.

**What this model can and cannot be trusted on.** It was trained on ImageNet photographs
against six generators: ADM, GLIDE, Midjourney, SD 1.5, VQDM and Wukong. On those it reaches
0.9514 accuracy. On BigGAN, held out of training, recall at threshold 0.5 falls to 0.507
even though ranking stays informative at 0.786 AUC. An image from a generator outside that
list, and every generator released since, is a case the model has never seen. Treat a
verdict here as evidence, not proof.

Usage:
    python tools/hires_predict.py photo.jpg
    python tools/hires_predict.py real.jpg fake.png --windows 32
    python tools/hires_predict.py *.jpg --no-figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from hires_model import load_namespace  # noqa: E402
from hires_train import CACHE_DIR, CKPT_DIR, channel_stats, load_split  # noqa: E402

CKPT = CKPT_DIR / "hires_dsfnet_best.pt"
CROP = 256
TRAINED_ON = "ADM, GLIDE, Midjourney, SD 1.5, VQDM, Wukong"
VAL_ACC = 0.9514
HELDOUT_RECALL = 0.507
HELDOUT_AUC = 0.786

REAL_COLOUR = "#1565c0"
FAKE_COLOUR = "#c62828"


def windows(image, n: int, crop: int, rng) -> np.ndarray:
    """Native-resolution windows, on a grid where the image allows one, random otherwise.

    A grid covers the frame evenly, which matters because generator artefacts are not
    uniform across an image: skies and flat regions carry far less high-frequency evidence
    than textured ones.
    """
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    if min(h, w) < crop:
        return np.empty((0, crop, crop, 3), dtype=np.uint8)

    side = int(np.floor(np.sqrt(n)))
    ys = np.linspace(0, h - crop, max(1, min(side, h // crop + 1))).astype(int)
    xs = np.linspace(0, w - crop, max(1, min(side, w // crop + 1))).astype(int)
    grid = [(y, x) for y in ys for x in xs]
    while len(grid) < n:
        grid.append((int(rng.integers(0, h - crop + 1)), int(rng.integers(0, w - crop + 1))))
    return np.stack([arr[y:y + crop, x:x + crop] for y, x in grid[:n]])


def score(ns, model, batch_uint8, mean, std, device, chunk: int = 16):
    torch = ns["torch"]
    out = []
    with torch.no_grad():
        for start in range(0, len(batch_uint8), chunk):
            part = batch_uint8[start:start + chunk].astype(np.float32) / 255.0
            x = torch.from_numpy(part).permute(0, 3, 1, 2)
            x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
            with torch.autocast(device_type=device.type, enabled=ns["AMP_ENABLED"]):
                logits = model(x.to(device))
            out.append(torch.sigmoid(logits.float().squeeze(-1)).cpu().numpy())
    return np.concatenate(out)


def make_figure(ns, items, out_path: Path, show: bool):
    plt = ns["plt"]
    n = len(items)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 3.6 * n),
                             gridspec_kw={"width_ratios": [1.2, 1.3, 1.4]})
    axes = np.atleast_2d(axes)

    for i, item in enumerate(items):
        probs = item["probs"]
        mean_p = float(probs.mean())
        label = "FAKE" if mean_p >= 0.5 else "REAL"
        colour = FAKE_COLOUR if label == "FAKE" else REAL_COLOUR

        ax = axes[i, 0]
        ax.imshow(item["image"])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{item['path'].name}\n{item['image'].size[0]}x{item['image'].size[1]}",
                      fontsize=9)

        ax = axes[i, 1]
        ax.hist(probs, bins=16, range=(0, 1), color=colour, alpha=0.75)
        ax.axvline(0.5, color="#212121", ls="--", lw=1.2)
        ax.set_xlim(0, 1)
        ax.set_xlabel("P(FAKE) per 256px window", fontsize=8.5)
        ax.set_ylabel("windows", fontsize=8.5)
        ax.tick_params(labelsize=8)

        ax = axes[i, 2]
        ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(-2.6, 1.0)
        ax.barh([0.4], [mean_p], color=colour, height=0.38)
        ax.barh([0.4], [1 - mean_p], left=[mean_p], color="#eceff1", height=0.38)
        ax.add_patch(plt.Rectangle((0, 0.21), 1.0, 0.38, fill=False, ec="#9e9e9e", lw=0.9))
        ax.plot([0.5, 0.5], [0.14, 0.66], color="#212121", ls="--", lw=1.2)
        ax.text(0.0, -0.15, f"mean P(FAKE) = {mean_p:.3f}  over {len(probs)} windows",
                fontsize=10, va="center")
        ax.text(0.0, -0.7, f"says {label}", fontsize=12, fontweight="bold",
                color=colour, va="center")
        ax.text(0.0, -1.25, f"window spread: {probs.min():.3f} to {probs.max():.3f} "
                            f"(std {probs.std():.3f})", fontsize=9.5, va="center")
        agree = float(((probs >= 0.5) == (mean_p >= 0.5)).mean())
        ax.text(0.0, -1.75, f"{agree:.0%} of windows agree with the verdict",
                fontsize=9.5, va="center",
                color="#2e7d32" if agree > 0.75 else "#ef6c00")
        ax.text(0.0, -2.30, f"trained on {TRAINED_ON};\nother generators are untested",
                fontsize=8.5, va="center", color="#616161")

        if i == 0:
            for col, title in enumerate(["The image, at native resolution",
                                         "Verdict per 256px window",
                                         "Aggregate"]):
                axes[i, col].set_title(title, fontsize=10.5, pad=10)

    fig.suptitle("DSF-Net at 256px, applied by cropping rather than resizing",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.01, 1, 0.985))
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--windows", type=int, default=25)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--no-figure", dest="figure", action="store_false")
    parser.add_argument("--out", type=Path, default=ROOT / "hires_predict_output.png")
    args = parser.parse_args()

    if not CKPT.exists():
        sys.exit(f"no high-resolution checkpoint at {CKPT}\n"
                 "  -> python tools/hires_train.py")
    missing = [p for p in args.images if not p.exists()]
    if missing:
        sys.exit("no such file: " + ", ".join(str(p) for p in missing))

    from PIL import Image

    ns = load_namespace()
    torch = ns["torch"]
    device = ns["DEVICE"]
    model = ns["DSFNet"](ns["DSFConfig"](mode="gated", width=1.5, dropout=0.1))
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.to(device).eval()

    Xtr, _, _ = load_split("train")
    mean, std = channel_stats(Xtr)

    print("DSF-Net at 256px, whole images by cropping")
    print("=" * 72)
    print(f"  trained on ImageNet photographs against {TRAINED_ON}")
    print(f"  validation accuracy {VAL_ACC:.4f} on those generators")
    print(f"  on a generator held out of training, recall falls to {HELDOUT_RECALL:.3f} "
          f"at threshold 0.5 (AUC {HELDOUT_AUC:.3f})")
    print("  nothing is resized: every window is taken at the image's native resolution\n")

    rng = np.random.default_rng(0)
    items = []
    for path in args.images:
        image = Image.open(path)
        crops = windows(image, args.windows, CROP, rng)
        w, h = image.size
        if len(crops) == 0:
            print(f"  {path.name}: {w}x{h} is smaller than {CROP}px in one dimension, so no "
                  f"window fits. Resizing it up would invent detail the model would then read "
                  f"as evidence, so this image is skipped rather than guessed at.\n")
            continue

        probs = score(ns, model, crops, mean, std, device)
        mean_p = float(probs.mean())
        label = "FAKE" if mean_p >= 0.5 else "REAL"
        agree = float(((probs >= 0.5) == (mean_p >= 0.5)).mean())

        print(f"  {path.name}  ({w}x{h}, {len(probs)} windows of {CROP}px)")
        print(f"    mean P(FAKE) = {mean_p:.3f}  ->  says {label}")
        print(f"    per-window spread {probs.min():.3f} to {probs.max():.3f} "
              f"(std {probs.std():.3f}), {agree:.0%} agree")
        if agree < 0.7:
            print("    the windows disagree substantially, so the evidence in this image is "
                  "thin and the mean should not be read as a confident verdict")
        print()
        items.append({"path": path, "image": image.convert("RGB"), "probs": probs})

    if args.figure and items:
        make_figure(ns, items, args.out, args.show)


if __name__ == "__main__":
    main()
