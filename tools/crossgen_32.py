"""Cross-generator probe for the 32x32 study, with the dataset shift separated from the generator shift.

Limitation 2 of the report: every fake image in CIFAKE comes from Stable Diffusion v1.4, so
nothing in the study predicts behaviour on any other generator. Cross-generator transfer is
the central open problem in this field, and the study left it untested.

Testing it naively would confound two things at once. Images from another generator also
come from another *corpus*: Tiny-GenImage pairs its fakes with ImageNet photographs, while
CIFAKE pairs its own with CIFAR-10. A drop measured that way is a mixture of "different
generator" and "different photographs", and the two are not separable after the fact.

Three sets pull them apart, all evaluated by the same 32x32 models with the same CIFAKE
normalisation:

    A  CIFAKE test set                         in distribution, the reference
    B  ImageNet photos vs SD 1.5, at 32x32     corpus shift, near-identical generator
    C  ImageNet photos vs generator g          corpus shift and generator shift

SD 1.5 is the natural reference for B because the models were trained on SD 1.4, so B holds
the generator essentially fixed and changes only the corpus. B minus A is then what the
corpus costs, and C minus B is what each unseen generator costs on top of it.

Everything is downscaled to 32x32 with bicubic interpolation, which is how CIFAKE itself was
built. That destroys much of the high-frequency evidence, and it is the honest comparison
anyway: a 32x32 model has no other way to see a 512px image. The high-resolution track
exists precisely because of this, and `tools/hires_crossgen.py` runs the same experiment
without the downscaling.

Usage:
    python tools/crossgen_32.py
    python tools/crossgen_32.py --n 1500     # images per class per set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from demo import CHANNEL_MEAN, CHANNEL_STD, NOTEBOOK, load_model  # noqa: E402
from smoke_test import load_smoke_namespace  # noqa: E402

CACHE = ROOT / "data" / "cifake_cache.npz"
CKPT_DIR = ROOT / "checkpoints"
OUT_JSON = ROOT / "results" / "crossgen_32.json"
DATASET_ID = "TheKernel01/Tiny-GenImage"
GEN_NAMES = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]
REFERENCE_GENERATOR = "SD15"
SEED = 42


def to_32(pil_image) -> np.ndarray:
    """Bicubic down to 32x32, the way CIFAKE was constructed."""
    from PIL import Image

    return np.array(pil_image.convert("RGB").resize((32, 32), Image.BICUBIC), dtype=np.uint8)


def collect(split, n_per_class: int, rng):
    """Downscaled 32x32 images, grouped by generator, plus a pool of real photographs."""
    labels = np.array(split["label"])
    generators = np.array(split["generator"])

    real_idx = rng.choice(np.flatnonzero(labels == 0), n_per_class, replace=False)
    reals = np.stack([to_32(split[int(i)]["image"]) for i in real_idx])

    by_gen = {}
    for g in sorted(set(generators.tolist())):
        if g == 0:
            continue
        pool = np.flatnonzero(generators == g)
        if len(pool) == 0:
            continue
        take = rng.choice(pool, min(n_per_class, len(pool)), replace=False)
        name = GEN_NAMES[g] if g < len(GEN_NAMES) else str(g)
        by_gen[name] = np.stack([to_32(split[int(i)]["image"]) for i in take])
        print(f"    {name:<12} {len(by_gen[name]):>5} images", flush=True)
    return reals, by_gen


def score(ns, model, batch_uint8, device, chunk: int = 512):
    torch = ns["torch"]
    out = []
    with torch.no_grad():
        for start in range(0, len(batch_uint8), chunk):
            part = batch_uint8[start:start + chunk].astype(np.float32) / 255.0
            x = torch.from_numpy(part).permute(0, 3, 1, 2)
            x = (x - torch.tensor(CHANNEL_MEAN).view(1, 3, 1, 1)) / \
                torch.tensor(CHANNEL_STD).view(1, 3, 1, 1)
            with torch.autocast(device_type=device.type, enabled=ns["AMP_ENABLED"]):
                logits = model(x.to(device))
            out.append(torch.sigmoid(logits.float().squeeze(-1)).cpu().numpy())
    return np.concatenate(out)


def evaluate_pair(ns, model, reals, fakes, device):
    from sklearn.metrics import roc_auc_score

    n = min(len(reals), len(fakes))
    p_real = score(ns, model, reals[:n], device)
    p_fake = score(ns, model, fakes[:n], device)
    probs = np.concatenate([p_real, p_fake])
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    pred = (probs >= 0.5).astype(int)
    return {
        "n": int(2 * n),
        "accuracy": float((pred == y).mean()),
        "roc_auc": float(roc_auc_score(y, probs)),
        "recall_fake": float(pred[y == 1].mean()),
        "specificity_real": float((1 - pred[y == 0]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1000, help="images per class per set")
    args = parser.parse_args()

    if not CACHE.exists():
        sys.exit(f"missing {CACHE}; run the notebook once first")

    from datasets import load_dataset

    ns = load_smoke_namespace(NOTEBOOK)
    torch = ns["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    dsf, _, _, _ = load_model(ns, device)
    models["DSF-Net (tuned)"] = dsf
    cnn_ckpt = CKPT_DIR / "cifake_cnn_best.pt"
    if cnn_ckpt.exists():
        cnn = ns["CifakeCNN"]()
        cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device,
                                       weights_only=False)["model"])
        models["CIFAKE-CNN"] = cnn.to(device).eval()

    print("Cross-generator probe for the 32x32 models")
    print("=" * 78)
    print(f"  device {device}; {len(models)} model(s): {', '.join(models)}")

    data = np.load(CACHE)
    X_test, y_test = data["X_test"], data["y_test"]
    rng = np.random.default_rng(SEED)
    ref_real = X_test[y_test == 0][:args.n]
    ref_fake = X_test[y_test == 1][:args.n]

    print(f"\n  loading {DATASET_ID} and downscaling to 32x32 ...", flush=True)
    ds = load_dataset(DATASET_ID)["validation"]
    reals, by_gen = collect(ds, args.n, np.random.default_rng(SEED))

    results = {}
    for model_name, model in models.items():
        print(f"\n=== {model_name} ===")
        rows = {}
        rows["A  CIFAKE test (in distribution)"] = evaluate_pair(
            ns, model, ref_real, ref_fake, device)
        for gen, fakes in by_gen.items():
            tag = f"{'B' if gen == REFERENCE_GENERATOR else 'C'}  ImageNet vs {gen}"
            rows[tag] = evaluate_pair(ns, model, reals, fakes, device)

        print(f"  {'set':<34} {'acc':>7} {'AUC':>7} {'recall':>8} {'specificity':>12}")
        print("  " + "-" * 72)
        for label, m in rows.items():
            print(f"  {label:<34} {m['accuracy']:>7.4f} {m['roc_auc']:>7.4f} "
                  f"{m['recall_fake']:>8.4f} {m['specificity_real']:>12.4f}")

        a = rows["A  CIFAKE test (in distribution)"]["accuracy"]
        b_key = f"B  ImageNet vs {REFERENCE_GENERATOR}"
        if b_key in rows:
            b = rows[b_key]["accuracy"]
            print(f"\n    corpus shift      B - A = {(b - a) * 100:+.2f} pp "
                  f"(same generator family, different photographs)")
            others = [(k, v) for k, v in rows.items() if k.startswith("C ")]
            if others:
                worst = min(others, key=lambda kv: kv[1]["accuracy"])
                mean_c = float(np.mean([v["accuracy"] for _, v in others]))
                print(f"    generator shift   mean over unseen generators - B = "
                      f"{(mean_c - b) * 100:+.2f} pp")
                print(f"    worst generator   {worst[0].split('vs ')[-1]} at "
                      f"{worst[1]['accuracy']:.4f}")
            rows["_deltas_pp"] = {"corpus_shift": (b - a) * 100,
                                  "generator_shift": (mean_c - b) * 100 if others else None}
        results[model_name] = rows

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT_JSON.relative_to(ROOT)}")
    print("\nEverything above is downscaled to 32x32, which removes most of the high-frequency")
    print("evidence before the model sees it. tools/hires_crossgen.py runs the same comparison")
    print("at native resolution, and the gap between the two is the cost of the resolution.")


if __name__ == "__main__":
    main()
