"""How much of the label can be read off the file container, across public benchmarks?

Auditing Tiny-GenImage before training anything found that image dimensions alone classify
it at 1.0000 and file format alone at 0.9699: every generated image is a square PNG with a
power-of-two side and no ImageNet photograph is either. A detector trained on that as
shipped reports an excellent number having learned nothing about generators.

One dataset with that property is a bug report. The question this script exists to answer is
whether it is a property of how this area builds benchmarks, which is a different and much
larger claim, and it can be answered without training anything.

**Method.** For each benchmark, stream a sample, and for every image record only what is
visible without looking at a pixel: width, height, aspect ratio, area, whether the side is a
power of two, the container format, and the JPEG quantisation-table signature. Then fit a
depth-limited decision tree on those features alone and report its cross-validated accuracy.

A tree rather than hand-written rules, because hand-written rules only find the leaks you
already suspect. The tree finds whatever is there, in whatever form that benchmark happens to
have it, and its depth limit keeps the result readable as a rule rather than a memorisation
of the sample. The single most predictive threshold is printed alongside it so the finding is
interpretable and checkable by hand.

**Reading the output.** 0.50 means the container carries no label information, which is what
a sound benchmark should show. Anything approaching 1.00 means a model can score near
perfectly on that benchmark while ignoring the images entirely, so every number reported on
it is suspect until the cue is removed.

Usage:
    python tools/benchmark_audit.py --list              # show the benchmarks to be audited
    python tools/benchmark_audit.py                     # audit all of them
    python tools/benchmark_audit.py --only CIFAKE       # one benchmark
    python tools/benchmark_audit.py --n 3000            # larger sample per benchmark
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "results" / "benchmark_audit.json"
SEED = 42

# Hard cap on rows traversed per benchmark. Some of these are millions of images and tens of
# gigabytes; the audit needs coverage across the file, not all of it, and an unbounded pass
# would spend hours on one dataset for no extra information.
STREAM_BUDGET = 40_000

# Benchmarks to audit. `label` names the column holding the real/fake label; when it is None
# the loader tries to infer one. `config` and `split` are passed straight through.
BENCHMARKS = [
    {"name": "CIFAKE", "id": "dragonintelligence/CIFAKE-image-dataset", "split": "train"},
    {"name": "Tiny-GenImage", "id": "TheKernel01/Tiny-GenImage", "split": "train"},
    {"name": "GenImage-ADM", "id": "bitmind/GenImage_ADM", "split": "train"},
    {"name": "GenImage-GLIDE", "id": "bitmind/GenImage_glide", "split": "train"},
    {"name": "GenImage-VQDM", "id": "bitmind/GenImage_VQDM", "split": "train"},
    {"name": "GenImage-Midjourney", "id": "bitmind/GenImage_MidJourney", "split": "train"},
    {"name": "GenImage-BigGAN", "id": "bitmind/GenImage_BigGAN", "split": "train"},
    {"name": "GenImage++", "id": "Lunahera/genimagepp", "split": "train"},
    {"name": "gen-image-detection", "id": "qianyuancs/gen-image-detection-datasets", "split": "train"},
    {"name": "ai-image-detection", "id": "34data/ai-image-detection-fake", "split": "train"},
]

FEATURE_NAMES = [
    "width", "height", "min_side", "max_side", "aspect", "log_area",
    "is_square", "side_is_pow2", "is_jpeg", "is_png", "quant_lo", "quant_hi",
]


def is_power_of_two(n: int) -> int:
    return int(n > 0 and (n & (n - 1)) == 0)


def container_features(raw: bytes) -> list | None:
    """Everything about an image file that does not require looking at its pixels."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        return None
    w, h = img.size
    fmt = (img.format or "").upper()
    tables = getattr(img, "quantization", None) or {}
    sums = sorted(sum(t) for t in tables.values()) if tables else []
    return [
        float(w), float(h), float(min(w, h)), float(max(w, h)),
        w / max(1.0, h), float(np.log1p(w * h)),
        float(w == h), float(is_power_of_two(w) and w == h),
        float(fmt == "JPEG"), float(fmt == "PNG"),
        float(sums[0]) if sums else 0.0,
        float(sums[-1]) if len(sums) > 1 else (float(sums[0]) if sums else 0.0),
    ]


def find_columns(features) -> tuple:
    """Locate the image column and a binary label column without assuming a schema."""
    from datasets import Image as HFImage
    from datasets import ClassLabel, Value

    image_col = next((k for k, v in features.items() if isinstance(v, HFImage)), None)

    label_col = None
    for key in ("label", "labels", "target", "class", "is_fake", "fake", "y"):
        if key in features:
            label_col = key
            break
    if label_col is None:
        label_col = next(
            (k for k, v in features.items()
             if isinstance(v, ClassLabel) or (isinstance(v, Value) and "int" in str(v.dtype))),
            None)
    return image_col, label_col


def is_cached(ds_id: str) -> bool:
    """Whether this dataset's data files are already on disk, not just its metadata folder.

    The directory alone is not enough: Hugging Face creates it as soon as anything about the
    dataset is fetched, so a folder holding a megabyte of metadata looks identical to a fully
    downloaded copy. Taking the folder as proof sent this script down the non-streaming path
    for a dataset it did not have, which then began downloading the whole thing and sat there
    for ninety minutes. Require actual data files, and enough of them to be real.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    path = Path(HF_HUB_CACHE) / ("datasets--" + ds_id.replace("/", "--"))
    if not path.exists():
        return False
    data_files = [p for p in path.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".parquet", ".arrow", ".zip", ".tar"}]
    total_mb = sum(p.stat().st_size for p in data_files) / 1e6
    return total_mb > 50


def audit(spec: dict, n: int) -> dict:
    from datasets import Image as HFImage
    from datasets import load_dataset

    name, ds_id = spec["name"], spec["id"]
    print(f"\n--- {name}  ({ds_id}) ---", flush=True)

    # A cached dataset is read from disk with random access, which is instant and gives a
    # genuinely uniform sample. Streaming does not use the local cache: it re-reads the shards
    # from the hub, and sampling across a class-ordered file then forces a full download of
    # every shard. That is what made the first attempt stall for two hours on a dataset
    # already sitting on this disk.
    cached = is_cached(ds_id)
    split_name = spec.get("split", "train")

    if cached:
        print("    cached locally; loading with random access", flush=True)
        full = load_dataset(ds_id, split=split_name)
        image_col, label_col = find_columns(full.features)
        if image_col is None or label_col is None:
            return {"name": name, "id": ds_id, "status": "skipped",
                    "reason": f"no image/label column found in {list(full.features)}"}
        full = full.cast_column(image_col, HFImage(decode=False))
        rng = np.random.default_rng(SEED)
        picks = sorted(rng.choice(len(full), min(n, len(full)), replace=False).tolist())
        rows, labels, positions = [], [], []
        for pos in picks:
            item = full[pos]
            raw = item[image_col]
            raw = raw.get("bytes") if isinstance(raw, dict) else raw
            if not raw:
                continue
            feats = container_features(raw)
            if feats is None:
                continue
            rows.append(feats)
            labels.append(int(item[label_col]))
            positions.append(pos)
        print(f"    collected {len(rows):,} images from {len(full):,} rows", flush=True)
        return finish(name, ds_id, rows, labels, positions)

    ds = load_dataset(ds_id, split=split_name, streaming=True)
    image_col, label_col = find_columns(ds.features)
    if image_col is None or label_col is None:
        return {"name": name, "id": ds_id, "status": "skipped",
                "reason": f"no image/label column found in {list(ds.features)}"}

    # decode=False hands back the original bytes, which is the only way to see the container:
    # a decoded PIL image has already lost its format and quantisation tables.
    ds = ds.cast_column(image_col, HFImage(decode=False))

    # Several of these benchmarks are stored sorted by label: CIFAKE's first half is entirely
    # fake. Reading the first n rows of such a file yields one class and no audit at all, so
    # the sample is taken from evenly spaced offsets across the split. The position of each
    # image is kept, because file order correlating with the label is itself a defect worth
    # reporting: it silently breaks any naive head/tail split.
    total = None
    try:
        info = ds.info.splits.get(spec.get("split", "train"))
        total = int(info.num_examples) if info else None
    except Exception:
        total = None

    # One strided pass, never `.skip()`. An IterableDataset re-reads from the beginning on
    # every skip, so sampling twelve offsets from a 28,000-row split costs roughly 168,000
    # image reads to collect 1,200 - which is how the first version of this script stalled
    # for two hours on a single benchmark. A single pass taking every stride-th row costs
    # one traversal and gives the same coverage across the file.
    # Sequential, stride 1. Striding across an uncached remote split would force every shard
    # to be downloaded to collect a few hundred samples, which is the expensive case this
    # path exists to avoid. Reading the head downloads the minimum. The cost is that a split
    # stored in label order yields one class, and `finish` reports that as unauditable rather
    # than guessing; the cached path above has no such limitation.
    stride, budget = 1, STREAM_BUDGET
    print(f"    not cached; streaming the first rows (cap {budget:,})"
          + (f" of {total:,}" if total else ""), flush=True)

    rows, labels, positions = [], [], []
    for i, item in enumerate(ds):
        if len(rows) >= n or i >= budget:
            break
        if i % stride:
            continue
        raw = item[image_col]
        raw = raw.get("bytes") if isinstance(raw, dict) else raw
        if not raw:
            continue
        feats = container_features(raw)
        if feats is None:
            continue
        rows.append(feats)
        labels.append(int(item[label_col]))
        positions.append(i)
        if len(rows) % 250 == 0:
            print(f"    collected {len(rows):,} of {n:,}", flush=True)
    print(f"    collected {len(rows):,} images from {min(i + 1, budget):,} rows", flush=True)

    return finish(name, ds_id, rows, labels, positions)


def finish(name: str, ds_id: str, rows: list, labels: list, positions: list) -> dict:
    """Score a collected sample. Shared by the cached and streaming paths."""
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    X, y = np.array(rows, dtype=np.float64), np.array(labels)
    if len(y) == 0:
        return {"name": name, "id": ds_id, "status": "skipped", "reason": "no images read"}
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return {"name": name, "id": ds_id, "status": "skipped",
                "reason": f"only one class in the sample ({classes.tolist()}); the split is "
                          "probably ordered by label and could not be sampled across",
                "n": int(len(y))}

    majority = counts.max() / counts.sum()
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED, class_weight="balanced")
    scores = cross_val_score(tree, X, y, cv=5, scoring="balanced_accuracy")
    tree.fit(X, y)

    # The single most useful feature on its own, for a finding that can be checked by hand.
    single = DecisionTreeClassifier(max_depth=1, random_state=SEED, class_weight="balanced")
    single_score = cross_val_score(single, X, y, cv=5, scoring="balanced_accuracy").mean()
    single.fit(X, y)
    top_feature = FEATURE_NAMES[int(single.tree_.feature[0])] if single.tree_.node_count > 1 else "none"
    threshold = float(single.tree_.threshold[0]) if single.tree_.node_count > 1 else float("nan")

    print(f"    n={len(y):,}  class balance={majority:.3f}")
    print(f"    container-only balanced accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"    best single rule: {top_feature} <= {threshold:.2f}  -> {single_score:.4f}")

    # How well file position alone predicts the label. A sorted benchmark scores near 1.0.
    pos = np.array(positions, dtype=np.float64).reshape(-1, 1)
    order_score = float(cross_val_score(
        DecisionTreeClassifier(max_depth=2, random_state=SEED, class_weight="balanced"),
        pos, y, cv=5, scoring="balanced_accuracy").mean())
    print(f"    file position alone:              {order_score:.4f}")

    return {
        "name": name, "id": ds_id, "status": "ok", "n": int(len(y)),
        "majority_class_share": float(majority),
        "order_accuracy": order_score,
        "container_accuracy": float(scores.mean()),
        "container_accuracy_std": float(scores.std()),
        "single_rule_accuracy": float(single_score),
        "single_rule_feature": top_feature,
        "single_rule_threshold": threshold,
        "tree": export_text(tree, feature_names=FEATURE_NAMES, max_depth=3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=2000, help="images sampled per benchmark")
    parser.add_argument("--only", nargs="+", default=None, help="audit only these names")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for b in BENCHMARKS:
            print(f"  {b['name']:<22} {b['id']}")
        return

    targets = [b for b in BENCHMARKS if not args.only or b["name"] in args.only]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}

    print("Container audit: how much label is visible without reading a pixel")
    print("=" * 78)

    for spec in targets:
        try:
            results[spec["name"]] = audit(spec, args.n)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=1)
            results[spec["name"]] = {"name": spec["name"], "id": spec["id"],
                                     "status": "failed", "reason": str(exc)[:300]}
        OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'benchmark':<22} {'n':>7} {'container':>10} {'order':>7} {'best single rule':>30}")
    print("-" * 80)
    for name, row in results.items():
        if row.get("status") != "ok":
            print(f"{name:<22} {'-':>7} {row.get('status', '?'):>14} {row.get('reason', '')[:34]:>34}")
            continue
        rule = f"{row['single_rule_feature']} <= {row['single_rule_threshold']:.1f}"
        print(f"{name:<22} {row['n']:>7,} {row['container_accuracy']:>10.4f} "
              f"{row.get('order_accuracy', float('nan')):>7.3f} {rule:>30}")

    print("\n0.50 means the container carries no label information. Anything near 1.00 means a")
    print("model can score on that benchmark while ignoring the images, so every number")
    print("reported on it is suspect until the cue is removed.")
    print(f"\nwritten to {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
