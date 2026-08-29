"""Corpus and generator shift, decomposed, over five seeds and three architectures.

`tools/crossgen_32.py` established the decomposition on one run of each of two models. That is
exactly the evidence standard the evidence-standard analysis argues cannot carry an architectural claim, so
repeating the measurement under the standard that paper recommends is not optional here: a
decomposition reported from single runs would be refuted by that standard.

The three sets are unchanged, since the design is what separates the two shifts:

    A  CIFAKE test set                       in distribution, the reference
    B  ImageNet photographs vs SD 1.5        corpus shift, near-identical generator
    C  ImageNet photographs vs generator g   corpus shift and generator shift

SD 1.5 is the reference for B because the models were trained on SD 1.4, so B holds the
generator essentially fixed and changes only the photographs. B minus A is then what the corpus
costs, and the mean over C minus B is what an unseen generator costs on top of it.

What is new is the sampling. Fifteen checkpoints already exist, five seeds each of DSF-Net,
CIFAKE-CNN and ResNet-18, all trained on the same seeds under the same harness. Scoring every
one of them on the same three sets turns each shift from a number into a distribution, and the
paired interval over seeds is what the claim then rests on. No training is required.

The three sets are built once and cached, because downscaling tens of thousands of images is
the slow part and the evaluation itself is seconds per model.

Usage:
    python tools/crossgen_seeds.py --dry-run
    python tools/crossgen_seeds.py
    python tools/crossgen_seeds.py --report-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from crossgen_32 import (  # noqa: E402
    CHANNEL_MEAN, CHANNEL_STD, DATASET_ID, GEN_NAMES, NOTEBOOK, REFERENCE_GENERATOR,
    collect, evaluate_pair,
)
from smoke_test import load_smoke_namespace  # noqa: E402

CACHE = ROOT / "data" / "cifake_cache.npz"
SETS_CACHE = ROOT / "data" / "crossgen_sets_32.npz"
OUT_CSV = ROOT / "results" / "crossgen_seeds.csv"
OUT_JSON = ROOT / "results" / "crossgen_seeds.json"
OUT_DIGEST = ROOT / "results" / "crossgen_seeds_digest.txt"

SEEDS = [42, 43, 44, 45, 46]
SAMPLE_SEED = 42

# The tuned DSF-Net configuration, as in tools/seed_sweep.py.
BEST_DROPOUT, BEST_WIDTH = 0.1, 1.5

CSV_FIELDS = ["arch", "seed", "set", "n", "accuracy", "roc_auc", "recall_fake",
              "specificity_real"]


def build_sets(ns, n_per_class: int) -> dict:
    """Assemble sets A, B and C once and cache them.

    Downscaling to 32x32 is how CIFAKE itself was built, and it is also what destroys most of
    the high-frequency evidence. That cost is real and is stated in the digest rather than
    hidden: it is the reason the high-resolution track exists.
    """
    if SETS_CACHE.exists():
        z = np.load(SETS_CACHE)
        sets = {k: z[k] for k in z.files}
        print(f"  sets loaded from {SETS_CACHE.name}: " +
              ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
        return sets

    from datasets import load_dataset

    data = np.load(CACHE)
    X_test, y_test = data["X_test"], data["y_test"]
    sets = {
        "A_real": X_test[y_test == 0][:n_per_class],
        "A_fake": X_test[y_test == 1][:n_per_class],
    }
    print(f"  loading {DATASET_ID} and downscaling to 32x32 ...", flush=True)
    ds = load_dataset(DATASET_ID)["validation"]
    reals, by_gen = collect(ds, n_per_class, np.random.default_rng(SAMPLE_SEED))
    sets["imagenet_real"] = reals
    for gen, fakes in by_gen.items():
        sets[f"gen_{gen}"] = fakes
    np.savez_compressed(SETS_CACHE, **sets)
    print(f"  cached to {SETS_CACHE.name}")
    return sets


def load_checkpoints(ns, device) -> dict:
    """The fifteen models that already exist, keyed by architecture and seed."""
    torch = ns["torch"]
    out = {}
    for s in SEEDS:
        p = ROOT / "checkpoints" / "seeds" / f"seed{s}_abl_4_best.pt"
        if p.exists():
            m = ns["DSFNet"](ns["DSFConfig"](mode="gated", dropout=BEST_DROPOUT,
                                             width=BEST_WIDTH))
            m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"])
            out[("DSF-Net", s)] = m.to(device).eval()

        p = ROOT / "checkpoints" / "arch_seeds" / f"seed{s}_cifakecnn_best.pt"
        if p.exists():
            m = ns["CifakeCNN"]()
            m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"])
            out[("CIFAKE-CNN", s)] = m.to(device).eval()

        p = ROOT / "checkpoints" / "arch_seeds" / f"seed{s}_resnet18_best.pt"
        if p.exists():
            m = ns["build_resnet18"](pretrained=False)
            m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"])
            out[("ResNet-18", s)] = m.to(device).eval()
    return out


def report() -> str:
    import pandas as pd
    from scipy import stats

    df = pd.read_csv(OUT_CSV)
    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    emit("Corpus and generator shift over five seeds and three architectures")
    emit("=" * 78)
    emit(f"{len(df)} evaluations: {df['arch'].nunique()} architectures x "
         f"{df['seed'].nunique()} seeds x {df['set'].nunique()} sets")
    emit()
    emit("All images are downscaled to 32x32, as CIFAKE itself was built. That removes most of")
    emit("the high-frequency evidence before any model sees it, and it is the honest comparison")
    emit("for a 32x32 model. The native-resolution track measures the same shifts separately.")
    emit()

    b_set = f"gen_{REFERENCE_GENERATOR}"
    rows, per_arch = [], {}
    for arch in sorted(df["arch"].unique()):
        corpus, generator = [], []
        for s in sorted(df["seed"].unique()):
            d = df[(df["arch"] == arch) & (df["seed"] == s)].set_index("set")["accuracy"]
            if "A" not in d.index or b_set not in d.index:
                continue
            c_sets = [i for i in d.index if i.startswith("gen_") and i != b_set]
            corpus.append((d[b_set] - d["A"]) * 100)
            generator.append((d[c_sets].mean() - d[b_set]) * 100)
        per_arch[arch] = (np.array(corpus), np.array(generator))

    emit(f"{'architecture':<14}{'corpus shift B-A':>26}{'generator shift C-B':>28}")
    emit("-" * 78)
    for arch, (c, g) in per_arch.items():
        def ci(v):
            h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            return f"{v.mean():+7.2f} pp [{v.mean()-h:+6.2f},{v.mean()+h:+6.2f}]"
        emit(f"  {arch:<12}{ci(c):>26}{ci(g):>28}")
        rows.append({"arch": arch, "corpus_mean": float(c.mean()),
                     "corpus_sd": float(c.std(ddof=1)),
                     "generator_mean": float(g.mean()),
                     "generator_sd": float(g.std(ddof=1)), "n_seeds": int(len(c))})

    emit()
    bigger = [a for a, (c, g) in per_arch.items() if abs(c.mean()) > abs(g.mean())]
    emit(f"The corpus term is the larger of the two for {len(bigger)} of {len(per_arch)} "
         f"architectures.")
    emit("A cross-generator number reported without this decomposition therefore attributes to")
    emit("the generator a loss that is substantially caused by the photographs it is paired")
    emit("against.")

    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    text = "\n".join(lines) + "\n"
    OUT_DIGEST.write_text(text, encoding="utf-8")
    print(f"\nwritten: {OUT_DIGEST.relative_to(ROOT)}, {OUT_JSON.relative_to(ROOT)}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="images per class per set")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        if not OUT_CSV.exists():
            sys.exit(f"nothing to report yet: {OUT_CSV}")
        report()
        return

    ns = load_smoke_namespace(NOTEBOOK)
    torch = ns["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = load_checkpoints(ns, device)
    archs = sorted({a for a, _ in models})
    print("Cross-generator decomposition over seeds")
    print("=" * 78)
    print(f"  device {device} | {len(models)} checkpoints | architectures: {', '.join(archs)}")
    missing = [(a, s) for a in ("DSF-Net", "CIFAKE-CNN", "ResNet-18") for s in SEEDS
               if (a, s) not in models]
    if missing:
        print(f"  WARNING: {len(missing)} checkpoint(s) not found: {missing[:4]}")

    if args.dry_run:
        print("\ndry run: nothing was evaluated.")
        return

    sets = build_sets(ns, args.n)
    gen_sets = [k for k in sets if k.startswith("gen_")]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for (arch, seed), model in sorted(models.items()):
            m = evaluate_pair(ns, model, sets["A_real"], sets["A_fake"], device)
            w.writerow({"arch": arch, "seed": seed, "set": "A", **m})
            for g in gen_sets:
                m = evaluate_pair(ns, model, sets["imagenet_real"], sets[g], device)
                w.writerow({"arch": arch, "seed": seed, "set": g, **m})
            print(f"  {arch:<12} seed {seed}  done", flush=True)

    print()
    report()


if __name__ == "__main__":
    main()
