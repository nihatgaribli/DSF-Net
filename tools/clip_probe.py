"""A frozen-CLIP linear probe as a fourth detector, and its corpus/generator decomposition.

The three detectors measured so far are a 141k-parameter reference CNN, an 848k-parameter
dual-stream design and an 11M-parameter fine-tuned ResNet-18. All three are convolutional and
all three are trained from scratch or fine-tuned end to end on CIFAKE, which invites the reply
that whatever varies between them is a property of small convolutional detectors rather than of
detectors. This adds the paradigm the field actually moved to: features from a frozen CLIP image
encoder with a linear probe on top, following Ojha and colleagues.

A caveat that has to be stated rather than buried. CLIP expects 224x224 input and CIFAKE is
32x32, so every image is upsampled by a factor of seven before the encoder sees it. Bicubic
upsampling smooths precisely the high-frequency content the forensic argument rests on, so this
probe cannot be reading the same evidence the convolutional detectors read. That is not a defect
of the experiment, it is the experiment: a detector working from semantic features rather than
forensic ones should show a different corpus and generator profile, and whether it does is the
question. Its absolute accuracy is not comparable to the others and is not compared.

Seeds vary the probe's initialisation and its batch order, matching how the other three
architectures were seeded. The CLIP encoder is frozen throughout and its features are extracted
once and cached, so a seed changes only what is trained.

Sets are the same three the decomposition uses, read from the cache tools/crossgen_seeds.py
built, so the comparison is against identical images.

Usage:
    python tools/clip_probe.py --dry-run
    python tools/clip_probe.py
    python tools/clip_probe.py --report-only
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cifake_cache.npz"
SETS_CACHE = DATA / "crossgen_sets_32.npz"
FEAT_CACHE = DATA / "clip_features.npz"
OUT_CSV = ROOT / "results" / "clip_probe.csv"
OUT_DIGEST = ROOT / "results" / "clip_probe_digest.txt"

SEEDS = [42, 43, 44, 45, 46]
# ViT-B/16 laion2B rather than the OpenAI weights: it is already in the local cache, so
# the experiment needs no download, and its finer 16-pixel patches are the more
# favourable choice for images that have been upsampled from 32x32.
MODEL_NAME, PRETRAINED = "ViT-B-16", "laion2b_s34b_b88k"
EPOCHS, BATCH, LR = 40, 512, 1e-3

CSV_FIELDS = ["seed", "set", "n", "accuracy", "roc_auc"]


def extract_features(device):
    """Encode every image set with a frozen CLIP encoder, once, and cache the result."""
    import open_clip
    import torch

    if FEAT_CACHE.exists():
        z = np.load(FEAT_CACHE)
        print(f"  features loaded from {FEAT_CACHE.name}: "
              + ", ".join(f"{k} {z[k].shape}" for k in z.files if not k.endswith("_y")))
        return {k: z[k] for k in z.files}

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # The preprocessing pipeline expects PIL images; doing the resize and normalisation on the
    # GPU in one batch is far faster and numerically equivalent for our purposes.
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def encode(images_u8, tag):
        out = []
        with torch.no_grad():
            for i in range(0, len(images_u8), 256):
                part = images_u8[i:i + 256].astype(np.float32) / 255.0
                x = torch.from_numpy(part).permute(0, 3, 1, 2).to(device)
                x = torch.nn.functional.interpolate(
                    x, size=224, mode="bicubic", align_corners=False, antialias=False)
                x = (x.clamp(0, 1) - mean) / std
                out.append(model.encode_image(x).float().cpu().numpy())
        f = np.concatenate(out)
        print(f"    {tag:<18} {f.shape}", flush=True)
        return f

    feats = {}
    data = np.load(CACHE)
    Xtr, ytr = data["X_trainval"], data["y_trainval"]
    feats["train_X"] = encode(Xtr, "cifake train+val")
    feats["train_y"] = ytr.astype(np.int64)
    feats["A_real"] = encode(data["X_test"][data["y_test"] == 0][:1000], "A real")
    feats["A_fake"] = encode(data["X_test"][data["y_test"] == 1][:1000], "A fake")

    if not SETS_CACHE.exists():
        sys.exit(f"missing {SETS_CACHE}; run tools/crossgen_seeds.py first")
    sets = np.load(SETS_CACHE)
    for k in sets.files:
        if k.startswith("A_"):
            continue
        feats[k] = encode(sets[k], k)

    np.savez_compressed(FEAT_CACHE, **feats)
    print(f"  cached to {FEAT_CACHE.name}")
    return feats


def train_probe(feats, seed, device):
    """A linear probe on frozen features. Seed varies initialisation and batch order."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.from_numpy(feats["train_X"]).float()
    y = torch.from_numpy(feats["train_y"]).float()
    # Standardise on the training split only; the probe sees no test statistics.
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    X = ((X - mu) / sd).to(device)
    y = y.to(device)

    probe = torch.nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(X)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = lossf(probe(X[idx]).squeeze(-1), y[idx])
            loss.backward()
            opt.step()
    return probe, mu.to(device), sd.to(device)


def score(probe, mu, sd, features, device):
    import torch

    with torch.no_grad():
        x = torch.from_numpy(features).float().to(device)
        x = (x - mu) / sd
        return torch.sigmoid(probe(x).squeeze(-1)).cpu().numpy()


def evaluate(probe, mu, sd, real_f, fake_f, device):
    from sklearn.metrics import roc_auc_score

    n = min(len(real_f), len(fake_f))
    p = np.concatenate([score(probe, mu, sd, real_f[:n], device),
                        score(probe, mu, sd, fake_f[:n], device)])
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    return {"n": int(2 * n), "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
            "roc_auc": float(roc_auc_score(y, p))}


def report() -> str:
    import pandas as pd
    from scipy import stats

    df = pd.read_csv(OUT_CSV)
    lines = []

    def emit(t=""):
        lines.append(t)
        print(t)

    emit("Frozen-CLIP linear probe: in-distribution accuracy and shift decomposition")
    emit("=" * 78)
    emit(f"{MODEL_NAME}/{PRETRAINED}, encoder frozen, {len(df['seed'].unique())} seeds.")
    emit("CIFAKE is 32x32 and CLIP takes 224x224, so every image is upsampled sevenfold before")
    emit("encoding. This probe is therefore not reading the high-frequency evidence the")
    emit("convolutional detectors read, and its absolute accuracy is not comparable to theirs.")
    emit()

    b = "gen_SD15"
    a_acc = df[df["set"] == "A"].set_index("seed")["accuracy"]
    emit(f"In distribution (set A): {a_acc.mean():.4f} mean over seeds, "
         f"sd {a_acc.std(ddof=1):.4f}")
    emit()

    corpus, generator = [], []
    for s in sorted(df["seed"].unique()):
        d = df[df["seed"] == s].set_index("set")["accuracy"]
        cs = [i for i in d.index if i.startswith("gen_") and i != b]
        corpus.append((d[b] - d["A"]) * 100)
        generator.append((d[cs].mean() - d[b]) * 100)
    corpus, generator = np.array(corpus), np.array(generator)

    def ci(v):
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return v.mean(), h

    cm, ch = ci(corpus)
    gm, gh = ci(generator)
    emit(f"corpus shift    B - A = {cm:+.2f} pp [{cm-ch:+.2f}, {cm+ch:+.2f}]")
    emit(f"generator shift C - B = {gm:+.2f} pp [{gm-gh:+.2f}, {gm+gh:+.2f}]")

    diff = np.abs(corpus) - np.abs(generator)
    dm, dh = ci(diff)
    resolved = not (dm - dh < 0 < dm + dh)
    emit(f"|corpus| - |generator| = {dm:+.2f} pp [{dm-dh:+.2f}, {dm+dh:+.2f}], "
         f"{'resolved' if resolved else 'unresolved'}")
    emit()
    emit("Compare with the convolutional detectors, where the same quantity is +2.87 on")
    emit("CIFAKE-CNN, +6.00 on DSF-Net and -0.97 on ResNet-18, all resolved.")

    text = "\n".join(lines) + "\n"
    OUT_DIGEST.write_text(text, encoding="utf-8")
    print(f"\nwritten: {OUT_DIGEST.relative_to(ROOT)}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        if not OUT_CSV.exists():
            sys.exit(f"nothing to report yet: {OUT_CSV}")
        report()
        return

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Frozen-CLIP linear probe")
    print("=" * 78)
    print(f"  device {device} | {MODEL_NAME}/{PRETRAINED} | seeds {SEEDS}")
    if args.dry_run:
        print(f"  would extract features to {FEAT_CACHE.name} and train {len(SEEDS)} probes")
        return

    feats = extract_features(device)
    gen_sets = [k for k in feats if k.startswith("gen_")]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for seed in SEEDS:
            t0 = time.time()
            probe, mu, sd = train_probe(feats, seed, device)
            m = evaluate(probe, mu, sd, feats["A_real"], feats["A_fake"], device)
            w.writerow({"seed": seed, "set": "A", **m})
            for g in gen_sets:
                mg = evaluate(probe, mu, sd, feats["imagenet_real"], feats[g], device)
                w.writerow({"seed": seed, "set": g, **mg})
            print(f"  seed {seed}: in-distribution {m['accuracy']:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print()
    report()


if __name__ == "__main__":
    main()
