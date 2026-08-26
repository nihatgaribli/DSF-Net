"""Live demo: run the trained DSF-Net on real CIFAKE test images, right now.

This is the script to run on camera. It loads the *already trained* tuned DSF-Net from
`checkpoints/dsfnet_tuned_best.pt`, pulls random images out of the held-out test split,
and classifies them in front of you. Nothing is trained here, so it finishes in seconds
rather than the 87 minutes a full study run takes.

What it shows, and why each part is worth showing:

  * per-image REAL/FAKE prediction with confidence -- the model actually working;
  * the fusion gate value per image -- Section 7.3 of the report argues the gate is not
    readable as a trust signal, and you can watch it sit at ~0.36 no matter the image,
    which is exactly the negative result the report is built around;
  * optionally the accuracy over the whole 20,000-image test set, recomputed live, so the
    95.71% headline number is not something the audience has to take on trust;
  * optionally the same images after real JPEG compression, which is the degradation that
    destroys the generator fingerprint.

Usage:
    python tools/demo.py                     # 4 random test images, full test-set score on GPU
    python tools/demo.py -n 6 --seed 7       # a different, larger sample
    python tools/demo.py --jpeg 30           # degrade the sample to JPEG quality 30 first
    python tools/demo.py --no-full           # skip the whole-test-set pass
    python tools/demo.py --no-show           # write the figure, do not open a window

Requires `data/cifake_cache.npz` and `checkpoints/dsfnet_tuned_best.pt`, both produced by
running the notebook once. Neither is in git; see the README.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from smoke_test import load_smoke_namespace  # noqa: E402

CACHE = ROOT / "data" / "cifake_cache.npz"
CKPT = ROOT / "checkpoints" / "dsfnet_tuned_best.pt"
NOTEBOOK = ROOT / "notebooks" / "AIGID_main.py"

# The tuned configuration selected by the coordinate-descent sweep in Section 12
# (results/tuning.csv: lr 1e-3, dropout 0.1, width 1.5). Hard-coded rather than re-derived
# so this script cannot silently demo a differently-shaped model than the checkpoint.
BEST_DROPOUT = 0.1
BEST_WIDTH = 1.5
EXPECTED_PARAMS = 848_066

# Normalisation statistics from the executed notebook (Section 5.1), computed on the
# training split only. Repeated here to four decimals; re-deriving them would require
# reproducing the stratified split, and the rounding error is ~1e-4 of a standard
# deviation, which no prediction in this demo is sensitive to.
CHANNEL_MEAN = np.array([0.4720, 0.4630, 0.4179], dtype=np.float32)
CHANNEL_STD = np.array([0.2374, 0.2373, 0.2658], dtype=np.float32)

CLASS_NAMES = {0: "REAL", 1: "FAKE"}


def require(path: Path, what: str, how: str) -> None:
    """Fail with an instruction rather than a traceback: this script gets run live."""
    if not path.exists():
        sys.exit(f"missing {what}: {path}\n  -> {how}")


def jpeg_compress(images_uint8: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip uint8 HWC images through real JPEG at the given quality.

    Genuine JPEG encoding, not a blur approximation: the point is JPEG's quantisation of
    high-frequency DCT coefficients, which is precisely where the generator fingerprint
    lives. Same helper as the notebook's Section 15.1 degradation.
    """
    from PIL import Image

    out = np.empty_like(images_uint8)
    for i, img in enumerate(images_uint8):
        buffer = io.BytesIO()
        Image.fromarray(img).save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        out[i] = np.array(Image.open(buffer).convert("RGB"), dtype=np.uint8)
    return out


def normalise(ns: dict, images_uint8: np.ndarray):
    """uint8 [N,H,W,C] -> normalised float tensor [N,C,H,W], the way the model was trained."""
    torch = ns["torch"]
    x = ns["to_tensor01"](images_uint8)
    mean = torch.tensor(CHANNEL_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CHANNEL_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def check_stats(images_uint8: np.ndarray) -> None:
    """Warn if the cache on disk is obviously not the dataset this model was trained on.

    Deliberately loose. CHANNEL_MEAN is a *training-split* statistic, and the test split's
    own mean sits a few hundredths below it on the blue channel, which is normal and not
    worth a warning. What this is guarding against is a wholesale mismatch: a different
    dataset, a rescaled cache, images stored as BGR. A silent mismatch of that kind would
    shift every input and quietly wreck the demo on camera.
    """
    probe = (images_uint8[:2000].astype(np.float32) / 255.0).mean(axis=(0, 1, 2))
    if not np.allclose(probe, CHANNEL_MEAN, atol=0.08):
        print(
            f"  WARNING: cached images have mean {probe.round(4)}, which is far from the "
            f"training-split mean {CHANNEL_MEAN}. Is data/cifake_cache.npz really CIFAKE?"
        )


def lookup_reported(condition: str) -> float | None:
    """Accuracy this study recorded for DSF-Net (tuned) under `condition`, if it ran it.

    Read out of results/robustness.csv rather than hard-coded, so that if the study is
    re-run and the CSV changes, the demo quotes the new numbers instead of stale ones.
    """
    csv_path = ROOT / "results" / "robustness.csv"
    if not csv_path.exists():
        return None
    import csv

    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["degradation"] == condition and row["model"] == "DSF-Net (tuned)":
                return float(row["accuracy"])
    return None


def load_model(ns: dict, device):
    """Build the tuned architecture and load the trained weights into it."""
    torch = ns["torch"]
    cfg = ns["DSFConfig"](mode="gated", dropout=BEST_DROPOUT, width=BEST_WIDTH)
    model = ns["DSFNet"](cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == EXPECTED_PARAMS, (
        f"built a {n_params:,}-parameter model but the checkpoint is for "
        f"{EXPECTED_PARAMS:,}; the tuned config and the checkpoint have drifted apart"
    )

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, n_params, ckpt.get("val_auc"), ckpt.get("epoch")


def predict(ns: dict, model, images_uint8: np.ndarray, device, batch: int = 512):
    """Return (probability of FAKE, mean gate value) for each image."""
    torch = ns["torch"]
    probs, gates = [], []
    with torch.no_grad():
        for start in range(0, len(images_uint8), batch):
            chunk = images_uint8[start : start + batch]
            x = normalise(ns, chunk).to(device)
            z, g = model.embed(x)
            logit = model.head(z)
            probs.append(torch.sigmoid(logit).squeeze(1).float().cpu().numpy())
            # g is per-dimension; its mean is the "how much do I trust the pixels" summary
            # the report tracks in Section 7.3.
            gates.append(
                g.mean(dim=1).float().cpu().numpy()
                if g is not None
                else np.full(len(chunk), np.nan, dtype=np.float32)
            )
    return np.concatenate(probs), np.concatenate(gates)


def print_table(y_true: np.ndarray, probs: np.ndarray, gates: np.ndarray, idx: np.ndarray) -> None:
    print(f"  {'#':>6}  {'truth':<5}  {'predicted':<9}  {'confidence':>10}  {'gate':>6}   verdict")
    print(f"  {'-' * 6}  {'-' * 5}  {'-' * 9}  {'-' * 10}  {'-' * 6}   {'-' * 7}")
    for i in range(len(y_true)):
        pred = int(probs[i] >= 0.5)
        confidence = probs[i] if pred == 1 else 1.0 - probs[i]
        ok = pred == int(y_true[i])
        print(
            f"  {idx[i]:>6}  {CLASS_NAMES[int(y_true[i])]:<5}  {CLASS_NAMES[pred]:<9}  "
            f"{confidence:>9.1%}  {gates[i]:>6.3f}   {'correct' if ok else 'WRONG'}"
        )


REAL_COLOUR = "#1565c0"
FAKE_COLOUR = "#c62828"
OK_COLOUR = "#2e7d32"


def stream_views(ns: dict, model, images_uint8: np.ndarray, device):
    """The two things the network actually looks at, for one batch of images.

    Returns (residual, spectrum, radial), all as numpy:
      residual  [N,32,32]  output of the Bayar-Stamm constrained convolution, channel-averaged.
                           This is the spatial stream's input after content suppression.
      spectrum  [N,32,32]  centred log-magnitude spectrum, channel-averaged: the frequency
                           stream's input.
      radial    [N,16]     azimuthal average of that spectrum, the hand-designed prior.

    Computed with the model's own methods rather than reimplemented, so the figure cannot
    drift away from what the network is really fed.
    """
    torch = ns["torch"]
    with torch.no_grad():
        x = normalise(ns, images_uint8).to(device)
        residual = (
            model.spatial.front(x).mean(dim=1).cpu().numpy()
            if model.spatial is not None
            else np.zeros((len(images_uint8), 32, 32), dtype=np.float32)
        )
        spec = model.frequency.spectrum(x)
        radial = model.frequency._radial(spec).cpu().numpy()
        spectrum = spec.mean(dim=1).cpu().numpy()
    return residual, spectrum, radial


def reference_profiles(ns: dict, model, X_test, y_test, device, n_per_class: int = 2000):
    """Mean radial profile of clean REAL and clean FAKE images, as the comparison baseline.

    This is the population-level version of the figure that motivated the whole architecture
    (results/figures/03_radial_profile.png). Drawing each test image against it is what makes
    the decision legible: you can see which curve the image sits closer to.
    """
    real = X_test[y_test == 0][:n_per_class]
    fake = X_test[y_test == 1][:n_per_class]
    _, _, radial_real = stream_views(ns, model, real, device)
    _, _, radial_fake = stream_views(ns, model, fake, device)
    return radial_real.mean(axis=0), radial_fake.mean(axis=0)


def make_figure(
    ns: dict,
    model,
    device,
    images_uint8: np.ndarray,
    y_true,
    probs,
    gates,
    ref_real,
    ref_fake,
    out_path: Path,
    show: bool,
    jpeg_q=None,
):
    """One row per image, walking left to right through how the verdict is reached.

    Columns are the pipeline: the picture as it really is, what survives content
    suppression, its spectrum, where its radial energy sits relative to the two class
    averages, and the decision that comes out. The point is that a viewer who knows nothing
    about the architecture can follow a single image across the row and see why it was
    called REAL or FAKE.
    """
    plt = ns["plt"]

    residual, spectrum, radial = stream_views(ns, model, images_uint8, device)
    n = len(images_uint8)
    bins = np.arange(len(ref_real))

    # One y-scale for every radial plot: with per-row autoscaling, a curve that sits far
    # from both class averages looks identical to one that sits right on top of them.
    deltas = np.concatenate([(radial - ref_real).ravel(), ref_fake - ref_real, [0.0]])
    margin = 0.10 * (deltas.max() - deltas.min() + 1e-9)
    y_lo, y_hi = deltas.min() - margin, deltas.max() + margin

    fig, axes = plt.subplots(
        n, 5,
        figsize=(17.5, 3.15 * n),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.35, 1.5]},
    )
    axes = np.atleast_2d(axes)

    step_titles = [
        "1.  The image as it is",
        "2.  Content suppressed\n(constrained conv)",
        "3.  Log-magnitude spectrum\n(frequency stream input)",
        "4.  Spectral energy vs the two class averages",
        "5.  Verdict",
    ]

    for i in range(n):
        truth = int(y_true[i])
        pred = int(probs[i] >= 0.5)
        ok = pred == truth
        confidence = probs[i] if pred == 1 else 1.0 - probs[i]

        # --- 1. the image itself -------------------------------------------------
        ax = axes[i, 0]
        ax.imshow(images_uint8[i], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(REAL_COLOUR if truth == 0 else FAKE_COLOUR)
            spine.set_linewidth(3)
        ax.set_ylabel(
            f"ground truth\n{CLASS_NAMES[truth]}",
            fontsize=11, fontweight="bold",
            color=REAL_COLOUR if truth == 0 else FAKE_COLOUR,
        )

        # --- 2. what the spatial stream sees -------------------------------------
        ax = axes[i, 1]
        scale = np.percentile(np.abs(residual[i]), 99) + 1e-8
        ax.imshow(residual[i], cmap="RdBu_r", vmin=-scale, vmax=scale, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])

        # --- 3. what the frequency stream sees -----------------------------------
        ax = axes[i, 2]
        ax.imshow(spectrum[i], cmap="magma", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])

        # --- 4. where this image's spectrum sits relative to the two classes ------
        # Plotted as a difference from the average REAL curve. On absolute axes all three
        # curves lie on top of each other -- the class separation is two orders of
        # magnitude smaller than the spectrum's own decay -- and the panel says nothing.
        ax = axes[i, 3]
        ax.axhline(0.0, color=REAL_COLOUR, ls="--", lw=1.6, label="average REAL")
        ax.plot(bins, ref_fake - ref_real, color=FAKE_COLOUR, ls="--", lw=1.6, label="average FAKE")
        ax.plot(
            bins, radial[i] - ref_real,
            color="#212121", lw=2.4, marker="o", ms=3.5, label="this image",
        )
        ax.set_xlabel("frequency ring (centre -> Nyquist)", fontsize=8)
        ax.set_ylabel("energy - average REAL", fontsize=8)
        ax.set_ylim(y_lo, y_hi)  # shared across rows, otherwise the rows cannot be compared
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)

        # --- 5. the decision -----------------------------------------------------
        # Drawn by hand on a blank canvas rather than as a real plot: a normal axes puts
        # its grid, spines and tick labels across the text underneath the bar.
        ax = axes[i, 4]
        colour = FAKE_COLOUR if pred == 1 else REAL_COLOUR
        ax.axis("off")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-2.75, 0.95)

        ax.barh([0], [probs[i]], color=colour, height=0.42)
        ax.barh([0], [1.0 - probs[i]], left=[probs[i]], color="#eceff1", height=0.42)
        ax.add_patch(plt.Rectangle((0, -0.21), 1.0, 0.42, fill=False, ec="#9e9e9e", lw=0.9))
        ax.plot([0.5, 0.5], [-0.34, 0.34], color="#212121", ls="--", lw=1.3)
        ax.text(0.5, 0.44, "threshold 0.5", ha="center", fontsize=7.5, color="#424242")
        ax.text(0.0, -0.34, "0", ha="center", va="top", fontsize=7.5, color="#616161")
        ax.text(0.5, -0.34, "P(FAKE)", ha="center", va="top", fontsize=8, color="#616161")
        ax.text(1.0, -0.34, "1", ha="center", va="top", fontsize=7.5, color="#616161")

        ax.text(0.0, -1.15, f"P(FAKE) = {probs[i]:.3f}", fontsize=10.5, va="center")
        ax.text(
            0.0, -1.75,
            f"says {CLASS_NAMES[pred]}, {confidence:.1%} confident",
            fontsize=10.5, fontweight="bold", color=colour, va="center",
        )
        ax.text(
            0.0, -2.35,
            f"{'CORRECT' if ok else 'WRONG'}      fusion gate = {gates[i]:.3f}",
            fontsize=10.5, fontweight="bold",
            color=OK_COLOUR if ok else FAKE_COLOUR, va="center",
        )

        if i == 0:
            for col in range(5):
                axes[i, col].set_title(step_titles[col], fontsize=10.5, pad=10)

    condition = "clean test images" if jpeg_q is None else f"test images degraded to JPEG q{jpeg_q}"
    fig.suptitle(
        f"How DSF-Net decides REAL vs FAKE, on {condition}",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "Each row is one image, read left to right. Columns 2 and 3 are what the two streams "
        "actually receive. Column 4 compares this image against the class averages: they "
        "separate on average but not reliably per image, which is why the frequency stream "
        "alone scores 4.3 points below the full model. Column 5 is the gated mix and the head.",
        ha="center", fontsize=9.5, color="#424242",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  figure written to {out_path.relative_to(ROOT)}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--n-images", type=int, default=4, help="how many test images to sample; 4-6 reads best in the figure")
    parser.add_argument("--seed", type=int, default=None, help="sampling seed; omit for a different draw each run")
    parser.add_argument("--jpeg", type=int, default=None, metavar="Q", help="JPEG-compress the sample at quality Q first")
    parser.add_argument("--full", dest="full", action="store_true", default=None, help="score the whole 20k test set")
    parser.add_argument("--no-full", dest="full", action="store_false", help="skip the whole-test-set pass")
    parser.add_argument("--no-show", dest="show", action="store_false", help="do not open a figure window")
    parser.add_argument("--out", type=Path, default=ROOT / "demo_output.png", help="where to write the figure")
    args = parser.parse_args()

    require(CACHE, "dataset cache", "run notebooks/AIGID_main.ipynb once; it caches the decoded images")
    require(CKPT, "trained checkpoint", "run notebooks/AIGID_main.ipynb once; training writes it")

    print("DSF-Net live demo")
    print("=" * 62)

    # The tagged cells print shape checks and model tables of their own. Useful in the
    # smoke test, pure noise here, and this script is meant to be readable on video.
    _quiet = io.StringIO()
    with contextlib.redirect_stdout(_quiet):
        ns = load_smoke_namespace(NOTEBOOK)
    torch = ns["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Default: score the full test set on a GPU, skip it on CPU where it would stall the
    # demo for a minute or more.
    run_full = args.full if args.full is not None else (device.type == "cuda")

    data = np.load(CACHE)
    X_test, y_test = data["X_test"], data["y_test"]
    check_stats(X_test)
    print(f"  test set: {len(X_test):,} images, {int((y_test == 1).sum()):,} FAKE / {int((y_test == 0).sum()):,} REAL")

    model, n_params, val_auc, epoch = load_model(ns, device)
    print(f"  model:    DSF-Net (tuned), {n_params:,} parameters, width {BEST_WIDTH}, dropout {BEST_DROPOUT}")
    if val_auc is not None:
        print(f"  weights:  {CKPT.name}, best epoch {int(epoch) + 1}, validation AUC {val_auc:.4f}")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X_test), size=args.n_images, replace=False)
    sample, sample_y = X_test[idx], y_test[idx]

    if args.jpeg is not None:
        print(f"\n  degrading the sample: real JPEG encoding at quality {args.jpeg}")
        sample = jpeg_compress(sample, args.jpeg)

    print(f"\n{args.n_images} random held-out test images:\n")
    t0 = time.time()
    probs, gates = predict(ns, model, sample, device)
    elapsed = time.time() - t0
    print_table(sample_y, probs, gates, idx)

    correct = int(((probs >= 0.5).astype(int) == sample_y).sum())
    print(f"\n  {correct}/{args.n_images} correct on this sample, decided in {elapsed * 1000:.0f} ms")
    if not np.isnan(gates).any():
        print(
            f"  gate range across these images: {gates.min():.3f} to {gates.max():.3f} "
            f"(spread {gates.max() - gates.min():.3f})"
        )
        print("  a near-flat gate is the report's Section 7.3 negative result, visible live")

    if run_full:
        condition = "clean" if args.jpeg is None else f"JPEG q{args.jpeg}"
        print()
        print(f"scoring the full test set ({len(X_test):,} images, {condition}) ...")

        X_full = X_test
        if args.jpeg is not None:
            # Degrade the whole set too. Without this the headline accuracy below would be
            # the clean number printed underneath a degraded sample, which is simply false.
            t_jpeg = time.time()
            X_full = jpeg_compress(X_test, args.jpeg)
            print(f"  re-encoded {len(X_full):,} images as JPEG q{args.jpeg} in {time.time() - t_jpeg:.1f}s")

        t0 = time.time()
        all_probs, all_gates = predict(ns, model, X_full, device)
        elapsed = time.time() - t0
        acc = float(((all_probs >= 0.5).astype(int) == y_test).mean())
        throughput = len(X_full) / elapsed
        print(f"  accuracy: {acc:.4f}   ({acc * 100:.2f}%)")
        print(
            f"  computed in {elapsed:.1f}s, {throughput:,.0f} images/second end to end "
            "(normalisation included; the report's 21,699 img/s is the model-only benchmark)"
        )

        expected = lookup_reported(condition)
        if expected is not None:
            print(f"  reported in results/robustness.csv for this condition: {expected:.4f}")
        if args.jpeg is None:
            print("  report's headline figure: 0.9571   |   CIFAKE paper reference: 0.9298")

        if not np.isnan(all_gates).any():
            print(f"  mean gate over the whole test set: {all_gates.mean():.4f}")

    print()
    print("building the explanation figure (class-average spectral profiles first) ...")
    ref_real, ref_fake = reference_profiles(ns, model, X_test, y_test, device)
    make_figure(
        ns, model, device, sample, sample_y, probs, gates,
        ref_real, ref_fake, args.out, args.show, jpeg_q=args.jpeg,
    )


if __name__ == "__main__":
    main()
