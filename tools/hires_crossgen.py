"""Cross-generator evaluation, with the resolution shift separated from the generator shift.

The training script's held-out check was not a proper evaluation. Its set contained BigGAN
crops and nothing else, so there were no negatives: ROC-AUC came back `nan`, the reported
"accuracy" was recall on a single class, and F1 and ECE were meaningless. The finding
underneath it was real, but the measurement was not.

This script fixes that and separates a confound the first version could not have addressed.
BigGAN emits 128x128 images while everything else was trained at 256x256, so a bare
comparison mixes two shifts at once. Three balanced sets pull them apart:

    A  256px, real vs seen generators   the in-distribution reference
    B  128px, real vs seen generators   resolution shift alone
    C  128px, real vs BigGAN            resolution shift and generator shift together

B minus A is what the smaller crop costs. C minus B is what the unseen generator costs.
Every set is balanced, so accuracy and ROC-AUC both mean what they appear to mean, and the
real images in B and C are the same photographs as in A, sub-cropped to 128 rather than
swapped for different ones.

Usage:
    python tools/hires_crossgen.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from hires_model import load_namespace  # noqa: E402

CACHE_DIR = ROOT / "data" / "hires"
OUT_DIR = ROOT / "results" / "hires"
CKPT_DIR = ROOT / "checkpoints" / "hires"
SEED = 42


def load_split(name: str):
    arr = np.load(CACHE_DIR / f"{name}_crops.npy", mmap_mode="r")
    meta = np.load(CACHE_DIR / f"{name}_meta.npz")
    n = int(meta["n_valid"][0]) if "n_valid" in meta else len(meta["labels"])
    return arr[:n], meta["labels"][:n].astype(int), meta["generators"][:n].astype(int)


def centre_crop(batch: np.ndarray, size: int) -> np.ndarray:
    if batch.shape[1] == size:
        return batch
    off = (batch.shape[1] - size) // 2
    return batch[:, off:off + size, off:off + size]


def score(ns, model, batch_uint8: np.ndarray, mean, std, device, batch: int = 32):
    torch = ns["torch"]
    probs = []
    with torch.no_grad():
        for start in range(0, len(batch_uint8), batch):
            chunk = batch_uint8[start:start + batch].astype(np.float32) / 255.0
            x = torch.from_numpy(chunk).permute(0, 3, 1, 2)
            x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
            with torch.autocast(device_type=device.type, enabled=ns["AMP_ENABLED"]):
                logits = model(x.to(device))
            probs.append(torch.sigmoid(logits.float().squeeze(-1)).cpu().numpy())
    return np.concatenate(probs)


def build_sets(rng, Xva, yva, gva, Xho, yho, n_per_class: int = 1000):
    """Three balanced evaluation sets sharing the same real photographs."""
    real_idx = rng.choice(np.flatnonzero(yva == 0), n_per_class, replace=False)
    seen_idx = rng.choice(np.flatnonzero(yva == 1), n_per_class, replace=False)
    ho_idx = rng.choice(len(yho), min(n_per_class, len(yho)), replace=False)

    reals = np.stack([Xva[i] for i in real_idx])
    seen = np.stack([Xva[i] for i in seen_idx])
    biggan = np.stack([Xho[i] for i in ho_idx])
    # The same real crops appear in all three sets, only windowed differently, so a change
    # between sets cannot be an effect of having drawn different photographs.
    reals_128 = centre_crop(reals, 128)

    y = np.concatenate([np.zeros(n_per_class, int), np.ones(n_per_class, int)])
    return {
        "A  256px, real vs seen generators": (np.concatenate([reals, seen]), y),
        "B  128px, real vs seen generators": (
            np.concatenate([reals_128, centre_crop(seen, 128)]), y),
        "C  128px, real vs BigGAN (unseen)": (
            np.concatenate([reals_128[:len(biggan)], biggan]),
            np.concatenate([np.zeros(len(biggan), int), np.ones(len(biggan), int)])),
    }


def main() -> None:
    ns = load_namespace()
    torch = ns["torch"]
    device = ns["DEVICE"]

    Xtr, _, _ = load_split("train")
    Xva, yva, gva = load_split("validation")
    Xho, yho, gho = load_split("validation_heldout")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xtr), 4000, replace=False)
    sample = np.stack([Xtr[i] for i in idx]).astype(np.float64) / 255.0
    mean = sample.mean(axis=(0, 1, 2)).astype(np.float32)
    std = sample.std(axis=(0, 1, 2)).astype(np.float32)

    sets = build_sets(np.random.default_rng(SEED), Xva, yva, gva, Xho, yho)

    from sklearn.metrics import roc_auc_score

    specs = {
        "dsfnet": lambda: ns["DSFNet"](ns["DSFConfig"](mode="gated", width=1.5, dropout=0.1)),
        "resnet18": None,  # built below, needs torchvision
    }
    results = {}
    for name in ("dsfnet", "resnet18"):
        ckpt_path = CKPT_DIR / f"hires_{name}_best.pt"
        if not ckpt_path.exists():
            print(f"  skipping {name}: no checkpoint at {ckpt_path.name}")
            continue
        if name == "dsfnet":
            model = specs[name]()
        else:
            import torchvision
            model = torchvision.models.resnet18(weights=None)
            model.fc = ns["nn"].Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=False)["model"])
        model.to(device).eval()

        print(f"\n=== {name} ===")
        print(f"  {'set':<36} {'acc':>7} {'AUC':>7} {'recall':>7} {'specificity':>12}")
        print("  " + "-" * 72)
        rows = {}
        for label, (batch, y) in sets.items():
            p = score(ns, model, batch, mean, std, device)
            pred = (p >= 0.5).astype(int)
            acc = float((pred == y).mean())
            auc = float(roc_auc_score(y, p))
            recall = float(pred[y == 1].mean())
            spec = float((1 - pred[y == 0]).mean())
            rows[label] = {"accuracy": acc, "roc_auc": auc,
                           "recall_fake": recall, "specificity_real": spec, "n": len(y)}
            print(f"  {label:<36} {acc:>7.4f} {auc:>7.4f} {recall:>7.4f} {spec:>12.4f}")

        a = rows["A  256px, real vs seen generators"]["accuracy"]
        b = rows["B  128px, real vs seen generators"]["accuracy"]
        c = rows["C  128px, real vs BigGAN (unseen)"]["accuracy"]
        print(f"\n    resolution shift  B - A = {(b - a) * 100:+.2f} pp")
        print(f"    generator shift   C - B = {(c - b) * 100:+.2f} pp")
        rows["_deltas_pp"] = {"resolution_shift": (b - a) * 100,
                              "generator_shift": (c - b) * 100}
        results[name] = rows

    out = OUT_DIR / "crossgen.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
