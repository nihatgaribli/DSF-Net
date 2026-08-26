"""Turn Tiny-GenImage into a training cache of fixed-size crops, with the label leak removed.

Profiling this dataset (`tools/hires_prepare.py`) found that the label can be read off the
file container with no pixels involved at all:

    'not a JPEG'                     -> fake : 0.9699
    'square, side in {128,...,1024}' -> fake : 1.0000

Every generated image is a PNG with a square power-of-two side; no ImageNet photograph is
either. Training on the dataset as shipped would produce a detector that scores beautifully
and has learned nothing about generators. This script removes both cues.

**Dimensions.** Every sample is a 256x256 crop taken at the image's *native* resolution.
Nothing is ever resized, because resizing is a low-pass filter and the generator fingerprint
lives in the high frequencies; that is the whole reason this track exists. After cropping,
every sample is the same shape, so the dimension cue is gone.

**Container.** Every crop, of both classes, is re-encoded through JPEG at a quality drawn
from the same distribution. The last compression a sample went through is therefore
identically distributed across classes.

**The residual bias is real and is not fixable here.** The photographs arrived already
JPEG-compressed, the generated images did not, so after this step the real crops carry two
generations of JPEG and the fake crops one. That asymmetry survives, and any result from
this cache has to be reported with it stated. Cropping at a random offset does mitigate it:
the original 8x8 JPEG block grid ends up misaligned with the new one, which blurs the
double-compression signature rather than leaving it neatly aligned.

**BigGAN is held out of training entirely.** Its images are exactly 128x128, so a 256 crop
does not fit. Lowering the crop size to 128 to accommodate it would be worse than dropping
it: a 128 crop of a 128x128 image is the *whole image*, borders and composition included,
while every other class would contribute interior patches. The model could then separate
BigGAN on frame statistics alone, which is the same class of leak in a subtler form.
Holding it out is also the principled choice, since BigGAN is a GAN and every other
generator here is a diffusion or VQ model. It is written to its own cache and becomes a
cross-generator test case.

Usage:
    python tools/hires_build_cache.py --dry-run       # report what would be kept and dropped
    python tools/hires_build_cache.py                 # build the cache
    python tools/hires_build_cache.py --crops 3       # more crops per image
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "hires"
OUT_DIR = ROOT / "results" / "hires"
DATASET_ID = "TheKernel01/Tiny-GenImage"

CROP = 256
CROPS_PER_IMAGE = 2
JPEG_QUALITY_RANGE = (75, 95)
HELD_OUT_GENERATORS = {"BigGAN"}
# BigGAN emits 128x128, so its held-out cache is stored at that size rather than 256. The
# model accepts any size (src/hires_model.py), but the resolution difference is a confound
# in the cross-generator comparison and has to be reported alongside the number.
HELD_OUT_CROP = 128
SEED = 42


def common_recompress(crop_uint8: np.ndarray, quality: int) -> np.ndarray:
    """Push a crop through JPEG at the given quality, whatever it arrived as.

    Applied to both classes with the same quality distribution, so that the encoder of the
    *last* compression carries no information about the label.
    """
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(crop_uint8).save(buffer, format="JPEG", quality=int(quality),
                                     subsampling=0)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert("RGB"), dtype=np.uint8)


def plan_split(split_ds, gen_names: list, crop: int) -> dict:
    """Decide, per image, whether it is usable and which cache it belongs to."""
    keep, held, dropped = [], [], []
    drop_reasons = Counter()

    for i in range(len(split_ds)):
        w, h = split_ds[i]["image"].size
        label = int(split_ds[i]["label"])
        gen = int(split_ds[i]["generator"])
        name = gen_names[gen] if gen < len(gen_names) else str(gen)

        if name in HELD_OUT_GENERATORS:
            held.append((i, label, gen))
            continue
        if min(w, h) < crop:
            dropped.append((i, label, gen))
            drop_reasons[f"{'real' if label == 0 else 'fake'}: smaller than {crop}px"] += 1
            continue
        keep.append((i, label, gen))

        if (i + 1) % 5000 == 0:
            print(f"    scanned {i + 1:,} / {len(split_ds):,}", flush=True)

    return {"keep": keep, "held": held, "dropped": dropped, "reasons": drop_reasons}


def report_plan(name: str, plan: dict, gen_names: list, true_total: int | None = None) -> dict:
    keep, held, dropped = plan["keep"], plan["held"], plan["dropped"]
    total = true_total if true_total is not None else len(keep) + len(held) + len(dropped)
    kept_labels = Counter(lab for _, lab, _ in keep)
    dropped_labels = Counter(lab for _, lab, _ in dropped)

    print(f"\n  {name}: {total:,} images")
    print(f"    kept for training : {len(keep):,}  "
          f"(real {kept_labels[0]:,} / fake {kept_labels[1]:,}, "
          f"fake ratio {kept_labels[1] / max(1, len(keep)):.4f})")
    print(f"    held out          : {len(held):,}  ({', '.join(sorted(HELD_OUT_GENERATORS))})")
    print(f"    dropped, too small: {len(dropped):,}  "
          f"(real {dropped_labels[0]:,} / fake {dropped_labels[1]:,})")
    for reason, count in plan["reasons"].most_common():
        print(f"      - {reason}: {count:,}")

    by_gen = Counter(gen_names[g] if g < len(gen_names) else g for _, _, g in keep)
    print(f"    generators kept   : " + ", ".join(f"{k}={v:,}" for k, v in sorted(by_gen.items())))

    imbalance = abs(kept_labels[1] / max(1, len(keep)) - 0.5)
    if imbalance > 0.02:
        print(f"    *** the kept set is {imbalance * 100:.1f} pp away from balanced. "
              "Rebalance before training, or accuracy stops being the right metric.")
    return {
        "total": total, "kept": len(keep), "held_out": len(held), "dropped": len(dropped),
        "kept_real": kept_labels[0], "kept_fake": kept_labels[1],
        "dropped_real": dropped_labels[0], "dropped_fake": dropped_labels[1],
        "generators_kept": {str(k): v for k, v in by_gen.items()},
    }


def rebalance(keep: list, rng: np.random.Generator) -> tuple:
    """Subsample the majority class so the split is exactly balanced.

    It does not arrive balanced. Holding out BigGAN removes 2,000 generated images, and
    dropping images below the crop size removes real ones only, since every generated image
    is at least 128px square. Left alone, the validation split lands 3.1 pp from balanced,
    and at that point accuracy quietly stops meaning what it appears to mean. Discarding a
    few hundred real images is much cheaper than reasoning about a skewed prior in every
    table afterwards.
    """
    by_label = {0: [e for e in keep if e[1] == 0], 1: [e for e in keep if e[1] == 1]}
    target = min(len(by_label[0]), len(by_label[1]))
    balanced, discarded = [], 0
    for label, entries in by_label.items():
        if len(entries) > target:
            picked = rng.choice(len(entries), target, replace=False)
            balanced.extend(entries[i] for i in sorted(picked))
            discarded += len(entries) - target
        else:
            balanced.extend(entries)
    balanced.sort(key=lambda e: e[0])
    return balanced, discarded


def build_cache(split_ds, entries: list, name: str, crop: int, crops_per_image: int,
                rng: np.random.Generator) -> dict:
    """Extract crops, re-encode them identically, and write a uint8 memmap."""
    n_samples = len(entries) * crops_per_image
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    array_path = CACHE_DIR / f"{name}_crops.npy"
    meta_path = CACHE_DIR / f"{name}_meta.npz"

    print(f"\n  writing {n_samples:,} crops of {crop}x{crop} to {array_path.name} "
          f"({n_samples * crop * crop * 3 / 1e9:.1f} GB)", flush=True)

    memmap = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.uint8, shape=(n_samples, crop, crop, 3))
    labels = np.zeros(n_samples, dtype=np.uint8)
    generators = np.zeros(n_samples, dtype=np.uint8)
    sources = np.zeros(n_samples, dtype=np.int32)

    write, skipped = 0, 0
    for count, (index, label, gen) in enumerate(entries):
        image = split_ds[index]["image"].convert("RGB")
        arr = np.array(image, dtype=np.uint8)
        h, w = arr.shape[:2]
        if min(h, w) < crop:
            skipped += 1
            continue
        for _ in range(crops_per_image):
            y = int(rng.integers(0, h - crop + 1))
            x = int(rng.integers(0, w - crop + 1))
            patch = arr[y:y + crop, x:x + crop]
            quality = int(rng.integers(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1] + 1))
            memmap[write] = common_recompress(patch, quality)
            labels[write] = label
            generators[write] = gen
            sources[write] = index
            write += 1
        if (count + 1) % 2000 == 0:
            print(f"    {count + 1:,} / {len(entries):,} images", flush=True)

    memmap.flush()
    # Only the first `write` rows are real. The metadata is truncated to match, and the
    # loader sizes itself from the metadata, so the unused tail is never read.
    np.savez_compressed(meta_path, labels=labels[:write], generators=generators[:write],
                        sources=sources[:write], n_valid=np.array([write]))
    print(f"    done: {write:,} crops"
          + (f", {skipped:,} image(s) skipped as smaller than {crop}px" if skipped else ""))
    return {"samples": write, "skipped_images": skipped, "crop": crop,
            "array": str(array_path.name), "meta": str(meta_path.name)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop", type=int, default=CROP)
    parser.add_argument("--crops", type=int, default=CROPS_PER_IMAGE,
                        help="crops taken per source image")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the plan and the drops, write nothing")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("`datasets` is not installed:  pip install -r requirements.txt")

    print("Building the high-resolution crop cache")
    print("=" * 70)
    print(f"  crop {args.crop}x{args.crop}, {args.crops} per image, native resolution only")
    print(f"  both classes re-encoded to JPEG q{JPEG_QUALITY_RANGE[0]}-{JPEG_QUALITY_RANGE[1]}")
    print(f"  held out of training: {', '.join(sorted(HELD_OUT_GENERATORS))}")

    ds = load_dataset(DATASET_ID)
    gen_names = ds["train"].features["generator"].names
    print(f"  generators in the dataset: {gen_names}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"crop": args.crop, "crops_per_image": args.crops,
               "jpeg_quality_range": list(JPEG_QUALITY_RANGE),
               "held_out": sorted(HELD_OUT_GENERATORS), "splits": {}}

    rng = np.random.default_rng(SEED)
    for split_name in ds:
        print(f"\n--- scanning {split_name} ---", flush=True)
        plan = plan_split(ds[split_name], gen_names, args.crop)
        plan["keep"], balance_dropped = rebalance(plan["keep"], np.random.default_rng(SEED))
        summary["splits"][split_name] = report_plan(split_name, plan, gen_names,
                                                    true_total=len(ds[split_name]))
        summary["splits"][split_name]["dropped_for_balance"] = balance_dropped
        print(f"    dropped to balance: {balance_dropped:,}")

        if args.dry_run:
            continue
        summary["splits"][split_name]["cache"] = build_cache(
            ds[split_name], plan["keep"], split_name, args.crop, args.crops, rng)
        if plan["held"]:
            summary["splits"][split_name]["heldout_cache"] = build_cache(
                ds[split_name], plan["held"], f"{split_name}_heldout",
                HELD_OUT_CROP, args.crops, rng)

    out = OUT_DIR / "cache_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary written to {out.relative_to(ROOT)}")
    if args.dry_run:
        print("dry run: nothing was written to data/hires/")


if __name__ == "__main__":
    main()
