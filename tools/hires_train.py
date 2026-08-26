"""Train DSF-Net at 256x256 on the leak-free crop cache, against a ResNet-18 baseline.

This is the experiment the high-resolution track exists for. The 32x32 study found the
frequency stream far weaker than the spatial one and the fusion gate useless, and one
standing explanation was that a 32x32 image simply carries too little spectral evidence
for either to work. Here the same architecture sees 64 times as many pixels per sample.

The training harness is the notebook's own `train_model`, not a reimplementation: same
optimiser, schedule, label smoothing, gradient clipping, early stopping on validation
ROC-AUC, constrained-convolution projection and resumable checkpointing. Only the data and
the input size change, so a difference in the results is attributable to those.

What it reports beyond accuracy:
  * a per-generator breakdown, since an average over six generators can hide a total
    failure on one of them;
  * BigGAN, held out of training entirely, as a cross-generator test. It is a GAN among
    diffusion and VQ models and is evaluated at its native 128x128, so the number carries
    both a generator shift and a resolution shift and cannot be read as either alone.

Usage:
    python tools/hires_train.py --bench                 # measure step time, train nothing
    python tools/hires_train.py                         # DSF-Net and ResNet-18
    python tools/hires_train.py --models dsfnet         # just DSF-Net
    python tools/hires_train.py --epochs 20 --batch 32
    python tools/hires_train.py --report-only           # rebuild the report from saved metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from hires_model import load_namespace  # noqa: E402

CACHE_DIR = ROOT / "data" / "hires"
OUT_DIR = ROOT / "results" / "hires"
CKPT_DIR = ROOT / "checkpoints" / "hires"
GEN_NAMES = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]

# The tuned DSF-Net configuration from results/tuning.csv. The architecture carries over
# unchanged, but the learning rate cannot: it was tuned at batch 256 on 32x32 CIFAKE, and a
# 256x256 crop only fits at batch 32 on 8 GB. The linear scaling rule adjusts for the batch
# change; nothing adjusts for the change of dataset and resolution, so these hyperparameters
# are inherited, not tuned, and that is a limitation of this track rather than a detail.
STUDY_BATCH = 256
BEST_LR = 1e-3
BEST_DROPOUT = 0.1
BEST_WIDTH = 1.5
SEED = 42


def load_split(name: str):
    arr = np.load(CACHE_DIR / f"{name}_crops.npy", mmap_mode="r")
    meta = np.load(CACHE_DIR / f"{name}_meta.npz")
    n = int(meta["n_valid"][0]) if "n_valid" in meta else len(meta["labels"])
    return arr[:n], meta["labels"][:n].astype(np.int64), meta["generators"][:n].astype(np.int64)


def channel_stats(arr, sample: int = 4000, seed: int = 0):
    """Normalisation statistics from the training crops themselves.

    Never reused from the 32x32 study: these are different images at a different size, and
    a shifted normalisation would quietly change what the first layer sees.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), min(sample, len(arr)), replace=False)
    batch = np.stack([arr[i] for i in idx]).astype(np.float64) / 255.0
    return batch.mean(axis=(0, 1, 2)).astype(np.float32), batch.std(axis=(0, 1, 2)).astype(np.float32)


def make_dataset(ns, arr, labels, mean, std, train: bool, crop: int = 256):
    torch = ns["torch"]
    Dataset = ns["Dataset"]

    class CropDataset(Dataset):
        """Crops held on disk as a memmap, normalised on access.

        Horizontal flip is the only augmentation, the same conservative choice the study
        made and for the same reason: a flip mirrors the Fourier magnitude spectrum about
        the vertical axis, so the fingerprint survives it, while crops and colour jitter
        attack the signal itself.
        """

        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            patch = np.asarray(arr[i])
            if crop < patch.shape[0]:
                # Random window at training time, centre window at evaluation time, so the
                # evaluation number does not move between runs.
                if train:
                    y0 = np.random.randint(0, patch.shape[0] - crop + 1)
                    x0 = np.random.randint(0, patch.shape[1] - crop + 1)
                else:
                    y0 = x0 = (patch.shape[0] - crop) // 2
                patch = patch[y0:y0 + crop, x0:x0 + crop]
            patch = patch.astype(np.float32) / 255.0
            if train and np.random.rand() < 0.5:
                patch = patch[:, ::-1]
            x = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1)
            x = (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
            return x, torch.tensor(float(labels[i]))

    return CropDataset()


def build_resnet18(ns, pretrained: bool = True):
    """ImageNet ResNet-18 with its original stem: the input is 256x256, not 32x32.

    The study replaced the 7x7 stride-2 stem for CIFAR-sized images. At this resolution
    that adaptation is not just unnecessary, it would be a handicap, so the baseline is
    given the architecture it was designed with.
    """
    import torchvision
    nn = ns["nn"]
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def evaluate(ns, model, loader, labels, generators, tag: str) -> dict:
    y_true, y_prob = ns["predict"](model, loader)
    metrics = ns["compute_metrics"](y_true, y_prob)
    per_gen = {}
    pred = (y_prob >= 0.5).astype(int)
    for g in sorted(set(generators.tolist())):
        mask = generators == g
        if mask.sum() == 0:
            continue
        name = GEN_NAMES[g] if g < len(GEN_NAMES) else str(g)
        # For a generated class this is recall; for Real it is specificity. Either way it
        # is "how often this source is called correctly", which is what a per-source
        # breakdown should show.
        per_gen[name] = {"n": int(mask.sum()),
                         "correct_rate": float((pred[mask] == y_true[mask]).mean())}
    print(f"\n  {tag}: acc {metrics['accuracy']:.4f} | AUC {metrics['roc_auc']:.4f} "
          f"| F1 {metrics['f1']:.4f} | ECE {metrics['ece']:.4f}")
    for name, row in per_gen.items():
        print(f"      {name:<12} n={row['n']:>5}  correct {row['correct_rate']:.4f}")
    return {"metrics": {k: float(v) for k, v in metrics.items() if np.isscalar(v)},
            "per_generator": per_gen}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["dsfnet", "resnet18"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=256,
                        help="sub-crop taken from the cached 256px crops; still native "
                             "resolution, never resized. Lower it to fit the GPU.")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--bench", action="store_true", help="measure step time and exit")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if not (CACHE_DIR / "train_crops.npy").exists():
        sys.exit(f"no cache at {CACHE_DIR}\n  -> python tools/hires_build_cache.py")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "hires_metrics.json"

    if args.report_only:
        if not results_path.exists():
            sys.exit(f"nothing to report yet: {results_path}")
        print(json.dumps(json.loads(results_path.read_text(encoding="utf-8")), indent=2))
        return

    ns = load_namespace()
    ns["CKPT_DIR"] = CKPT_DIR
    torch = ns["torch"]
    device = ns["DEVICE"]

    print("DSF-Net at 256x256 on Tiny-GenImage")
    print("=" * 70)
    print(f"  device {device} | batch {args.batch} | epochs {args.epochs}")

    Xtr, ytr, gtr = load_split("train")
    Xva, yva, gva = load_split("validation")
    Xho, yho, gho = load_split("validation_heldout")
    print(f"  train {len(ytr):,} crops of {Xtr.shape[1]}x{Xtr.shape[2]}, "
          f"fake ratio {ytr.mean():.4f}")
    print(f"  val   {len(yva):,} crops, fake ratio {yva.mean():.4f}")
    print(f"  held out (BigGAN, {Xho.shape[1]}x{Xho.shape[2]}): {len(yho):,} crops")

    mean, std = channel_stats(Xtr)
    print(f"  channel mean {mean.round(4)}  std {std.round(4)}")

    crop = min(args.crop, int(Xtr.shape[1]))
    print(f"  training crop {crop}x{crop} taken from the cached {Xtr.shape[1]}px crops")
    train_ds = make_dataset(ns, Xtr, ytr, mean, std, train=True, crop=crop)
    val_ds = make_dataset(ns, Xva, yva, mean, std, train=False, crop=crop)
    # BigGAN is cached at 128; never enlarge it, just take what is there.
    ho_ds = make_dataset(ns, Xho, yho, mean, std, train=False, crop=min(crop, int(Xho.shape[1])))

    DataLoader = ns["DataLoader"]
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=0)
    ho_loader = DataLoader(ho_ds, batch_size=args.batch, num_workers=0)

    if args.bench:
        model = ns["DSFNet"](ns["DSFConfig"](mode="gated", width=BEST_WIDTH,
                                             dropout=BEST_DROPOUT)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=BEST_LR)
        scaler = torch.amp.GradScaler(device.type, enabled=ns["AMP_ENABLED"])
        it = iter(train_loader)

        def one_step():
            xb, yb = next(it)
            with torch.autocast(device_type=device.type, enabled=ns["AMP_ENABLED"]):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(xb.to(device)).squeeze(-1), yb.to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        for _ in range(3):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        steps = 15
        for _ in range(steps):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_step = (time.time() - t0) / steps
        steps_per_epoch = len(train_ds) // args.batch
        print(f"\n  {per_step * 1000:.0f} ms per step, {steps_per_epoch:,} steps per epoch")
        print(f"  ~{per_step * steps_per_epoch / 60:.1f} min per epoch, "
              f"~{per_step * steps_per_epoch * args.epochs / 3600:.1f} h for {args.epochs} epochs")
        if device.type == "cuda":
            print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        return

    scaled_lr = BEST_LR * args.batch / STUDY_BATCH
    print()
    print(f"  learning rate {scaled_lr:.2e} = {BEST_LR:.0e} x {args.batch}/{STUDY_BATCH} "
          "(linear scaling for the smaller batch; inherited, not retuned)")
    cfg = ns["TrainConfig"](epochs=args.epochs, lr=scaled_lr, weight_decay=1e-4,
                            patience=args.patience, seed=SEED)
    results = {}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    specs = {
        "dsfnet": lambda: ns["DSFNet"](ns["DSFConfig"](mode="gated", width=BEST_WIDTH,
                                                       dropout=BEST_DROPOUT)),
        "resnet18": lambda: build_resnet18(ns),
    }

    for name in args.models:
        if name not in specs:
            sys.exit(f"unknown model '{name}'; choose from {sorted(specs)}")
        print(f"\n=== {name} ===", flush=True)
        model = specs[name]()
        params = ns["count_parameters"](model)
        print(f"  {params:,} trainable parameters")

        # A pretrained backbone needs a smaller learning rate or fine-tuning destroys the
        # weights; the study used 1e-4 at batch 256, scaled here the same way.
        run_cfg = cfg if name != "resnet18" else ns["TrainConfig"](
            epochs=args.epochs, lr=1e-4 * args.batch / STUDY_BATCH, weight_decay=1e-4,
            patience=args.patience, seed=SEED)

        started = time.time()
        history = ns["train_model"](model, f"hires_{name}", train_loader, val_loader,
                                    run_cfg, verbose=True)
        elapsed = time.time() - started

        entry = {"params": params, "train_time_s": round(elapsed, 1),
                 "best_val_auc": float(history["best_val_auc"]),
                 "validation": evaluate(ns, model, val_loader, yva, gva, f"{name} / validation"),
                 "heldout_biggan": evaluate(ns, model, ho_loader, yho, gho,
                                            f"{name} / BigGAN held out, 128x128")}
        results[name] = entry
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  saved to {results_path.relative_to(ROOT)}")

    print("\n" + "=" * 70)
    print(f"{'model':<12} {'params':>12} {'val acc':>9} {'val AUC':>9} {'BigGAN acc':>11}")
    print("-" * 58)
    for name, entry in results.items():
        print(f"{name:<12} {entry['params']:>12,} "
              f"{entry['validation']['metrics']['accuracy']:>9.4f} "
              f"{entry['validation']['metrics']['roc_auc']:>9.4f} "
              f"{entry['heldout_biggan']['metrics']['accuracy']:>11.4f}")
    print("\nBigGAN was never trained on and is evaluated at 128x128, so that column mixes a")
    print("generator shift with a resolution shift. It is a floor, not a clean measurement.")


if __name__ == "__main__":
    main()
