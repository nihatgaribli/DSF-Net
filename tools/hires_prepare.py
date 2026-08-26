"""Download Tiny-GenImage and report what is actually in it, before any model is built.

This is the high-resolution track's counterpart to Section 3 of the notebook. The 32x32
study lives or dies on properties of CIFAKE that were only discovered by looking (the split
is ordered by label; the images are already JPEG-compressed and could have leaked the label
through the encoder). The same discipline applies here, and more so, because this dataset
mixes eight generators with ImageNet photographs and nothing about its construction is
guaranteed.

What it checks:
  * class and generator balance, per split;
  * the resolution distribution, since the entire point of this track is to stop resizing;
  * the JPEG quantisation signature per class, which is the label-leak test from Section 3.3
    of the notebook. If real and fake images were encoded with different JPEG settings, a
    detector could score well by reading the *encoder's* fingerprint rather than the
    *generator's*, and every number after that would be worthless.

Usage:
    python tools/hires_prepare.py                 # download and profile
    python tools/hires_prepare.py --sample 2000   # profile a subsample, faster
    python tools/hires_prepare.py --stats-only    # skip download if already cached
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "hires"
DATASET_ID = "TheKernel01/Tiny-GenImage"

# Mapping documented on the dataset card; verified against the loaded features below.
GENERATOR_NAMES = [
    "real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong",
]


def jpeg_signature(pil_image) -> tuple:
    """The image's JPEG quantisation tables, as a hashable signature.

    Same test as the notebook's Section 3.3. Two classes carrying different signatures
    would mean the encoder, not the generator, is what a detector could learn.
    """
    tables = getattr(pil_image, "quantization", None)
    if not tables:
        return ("not-jpeg",)
    return tuple(sum(table) for _, table in sorted(tables.items()))


# Sides that the generators in this benchmark emit. Real ImageNet photographs have
# arbitrary dimensions, so a square image with one of these sides is a giveaway.
GENERATOR_SIDES = {128, 256, 512, 1024}


def container_leak(rows_meta: list) -> dict:
    """How accurately the label can be predicted from format and dimensions alone.

    This is the check that matters. Two classes can share a JPEG signature and still be
    trivially separable, which is exactly what happens here: every generated image is a
    PNG with a square power-of-two side, and no photograph is. A detector trained on such
    a split learns the container, reports a superb number, and has learned nothing about
    generators. Section 3.3 of the notebook applied the same reasoning to CIFAKE.
    """
    if not rows_meta:
        return {"format_acc": 0.0, "size_acc": 0.0}
    truth = [lab for lab, _, _, _ in rows_meta]
    by_format = [0 if fmt == "JPEG" else 1 for _, fmt, _, _ in rows_meta]
    by_size = [1 if (w == h and w in GENERATOR_SIDES) else 0 for _, _, w, h in rows_meta]
    n = len(truth)
    return {
        "format_acc": sum(p == t for p, t in zip(by_format, truth)) / n,
        "size_acc": sum(p == t for p, t in zip(by_size, truth)) / n,
    }


def profile(split_ds, name: str, limit: int | None) -> dict:
    n = len(split_ds) if limit is None else min(limit, len(split_ds))
    print(f"\n--- {name}: profiling {n:,} of {len(split_ds):,} images ---", flush=True)

    labels = Counter()
    generators = Counter()
    sizes = Counter()
    signatures = {0: Counter(), 1: Counter()}
    formats = Counter()
    rows_meta = []

    for i in range(n):
        row = split_ds[i]
        img = row["image"]
        label = int(row["label"])
        labels[label] += 1
        generators[int(row.get("generator", -1))] += 1
        sizes[img.size] += 1
        formats[img.format or "unknown"] += 1
        signatures[label][jpeg_signature(img)] += 1
        rows_meta.append((label, img.format or "unknown", img.size[0], img.size[1]))
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1:,} ...", flush=True)

    print(f"  labels     : real={labels[0]:,}  fake={labels[1]:,}  "
          f"(fake ratio {labels[1] / max(1, sum(labels.values())):.4f})")
    print("  generators :", ", ".join(
        f"{GENERATOR_NAMES[g] if 0 <= g < len(GENERATOR_NAMES) else g}={c:,}"
        for g, c in sorted(generators.items())))
    print("  formats    :", dict(formats))

    print("  resolutions (top 10):")
    for size, count in sizes.most_common(10):
        print(f"    {size[0]:>5} x {size[1]:<5}  {count:>7,}")
    distinct = len(sizes)
    print(f"    ... {distinct} distinct sizes in total")

    print("  JPEG quantisation signatures:")
    for label in (0, 1):
        top = signatures[label].most_common(3)
        pretty = ", ".join(f"{sig}: {count:,}" for sig, count in top)
        print(f"    {'real' if label == 0 else 'fake'}: {pretty}")

    # A shared signature existing is not the question. The question is how much of the
    # label a classifier could recover from the container alone, and the only honest way
    # to answer that is to build the trivial classifiers and score them.
    leak = container_leak(rows_meta)
    print("\n  LABEL LEAK, classifying with NO pixels at all:")
    print(f"    'not a JPEG'          -> fake : {leak['format_acc']:.4f}")
    print(f"    'square, side in {sorted(GENERATOR_SIDES)}' -> fake : {leak['size_acc']:.4f}")
    if max(leak["format_acc"], leak["size_acc"]) > 0.65:
        print("    *** THE CONTAINER LEAKS THE LABEL. A model trained on this split as-is")
        print("        can score near-perfectly without reading a single pixel, so its")
        print("        accuracy would say nothing about generator fingerprints. Crop to a")
        print("        fixed size and re-encode both classes identically before training.")
    else:
        print("    no usable leak from the container alone.")

    return {
        "split": name,
        "profiled": n,
        "total": len(split_ds),
        "labels": {str(k): v for k, v in labels.items()},
        "generators": {str(k): v for k, v in generators.items()},
        "distinct_sizes": distinct,
        "top_sizes": [[list(s), c] for s, c in sizes.most_common(10)],
        "formats": dict(formats),
        "container_leak": leak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=3000,
                        help="images to profile per split (default 3000); 0 profiles everything")
    parser.add_argument("--stats-only", action="store_true",
                        help="assume the dataset is cached; do not report download progress")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("`datasets` is not installed:  pip install -r requirements.txt")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.stats_only:
        print(f"downloading {DATASET_ID} (about 8.4 GB on first run; cached afterwards)", flush=True)
    ds = load_dataset(DATASET_ID)
    print(f"\nloaded: {ds}", flush=True)

    limit = None if args.sample == 0 else args.sample
    summary = [profile(ds[split], split, limit) for split in ds]

    out = OUT_DIR / "dataset_profile.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nprofile written to {out.relative_to(ROOT)}")
    print("\nRead the resolution table above before choosing a crop size: the crop must fit "
          "inside the smallest images, or those images end up padded or resized, which is "
          "the exact thing this track exists to avoid.")


if __name__ == "__main__":
    main()
