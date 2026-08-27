"""Ablate DSF-Net at 256x256, to find out which part of it earned the high-resolution result.

At 256x256 DSF-Net matched an 11M-parameter ResNet-18 with 848k parameters and lost roughly
half as much to a resolution shift and to an unseen generator. That is the first result in
the project favouring the forensic design, and on its own it does not say *why*. Three
candidate explanations, all of which were refuted at 32x32:

  * the frequency stream finally has enough spectral evidence to contribute;
  * the gate finally has something to arbitrate between;
  * neither, and the spatial stream alone simply works better on larger images.

The same seven variants as Section 14 of the study, trained under identical conditions on
the leak-free crop cache, separate these. Every variant removes exactly one design decision;
schedule, seed, data and budget are held fixed.

Variant 4 is the full gated model, which is the configuration `tools/hires_train.py` already
trained. Its checkpoint is reused rather than retrained: the config, seed, data and budget
are identical, so retraining it would spend an hour and a half reproducing a number that
already exists.

The result is one seed per variant. That is enough to see a large effect and not enough to
put an error bar on a small one, which is exactly the limitation the 32x32 study ran into
and the reason `tools/seed_sweep.py` exists. Read the deltas against that.

Usage:
    python tools/hires_ablation.py --dry-run
    python tools/hires_ablation.py                  # ~9 h for six variants
    python tools/hires_ablation.py --epochs 10
    python tools/hires_ablation.py --report-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from hires_model import load_namespace  # noqa: E402
from hires_train import (  # noqa: E402
    BEST_DROPOUT, BEST_LR, BEST_WIDTH, CACHE_DIR, CKPT_DIR, GEN_NAMES, OUT_DIR,
    STUDY_BATCH, SEED, channel_stats, evaluate, load_split, make_dataset,
)

OUT_CSV = OUT_DIR / "hires_ablation.csv"
OUT_JSON = OUT_DIR / "hires_ablation.json"
FULL_VARIANT = "4. gated fusion (full)"
MAIN_RUN_CKPT = CKPT_DIR / "hires_dsfnet_best.pt"

CSV_FIELDS = ["variant", "params", "best_val_auc", "val_acc", "val_auc", "val_f1",
              "val_ece", "biggan_recall", "train_time_s", "reused"]


def variants(ns: dict) -> dict:
    DSFConfig = ns["DSFConfig"]
    common = dict(dropout=BEST_DROPOUT, width=BEST_WIDTH)
    return {
        "1. spatial only":        dict(cfg=DSFConfig(mode="spatial", **common)),
        "2. frequency only":      dict(cfg=DSFConfig(mode="freq", **common)),
        "3. concat fusion":       dict(cfg=DSFConfig(mode="concat", **common)),
        FULL_VARIANT:             dict(cfg=DSFConfig(mode="gated", **common)),
        "5. no constrained conv": dict(cfg=DSFConfig(mode="gated", use_constrained=False, **common)),
        "6. no radial features":  dict(cfg=DSFConfig(mode="gated", use_radial=False, **common)),
        "7. heavy augmentation":  dict(cfg=DSFConfig(mode="gated", **common), heavy_aug=True),
    }


def done_variants() -> set:
    if not OUT_CSV.exists():
        return set()
    with OUT_CSV.open(encoding="utf-8", newline="") as fh:
        return {row["variant"] for row in csv.DictReader(fh)}


def append_row(row: dict) -> None:
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def report() -> None:
    import pandas as pd

    if not OUT_CSV.exists():
        sys.exit(f"nothing to report yet: {OUT_CSV}")
    df = pd.read_csv(OUT_CSV).set_index("variant").sort_index()
    if FULL_VARIANT not in df.index:
        print("the full gated model has not been recorded yet; deltas are unavailable")
        print(df.to_string())
        return

    base = df.loc[FULL_VARIANT]
    print(f"\n{'variant':<24} {'params':>10} {'val acc':>9} {'d vs full':>10} "
          f"{'val AUC':>9} {'BigGAN rec':>11}")
    print("-" * 78)
    for name, row in df.iterrows():
        delta = (row["val_acc"] - base["val_acc"]) * 100
        marker = "  <- reference" if name == FULL_VARIANT else ""
        print(f"{name:<24} {int(row['params']):>10,} {row['val_acc']:>9.4f} "
              f"{delta:>+10.2f} {row['val_auc']:>9.4f} {row['biggan_recall']:>11.4f}{marker}")

    print("\nOne seed per variant. The 32x32 study measured a run-to-run noise floor of 0.08")
    print("to 0.34 pp; a delta smaller than that is not evidence of anything. Compare these")
    print("against results/seeds_digest.txt once the seed sweep has finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if args.report_only:
        report()
        return
    if not (CACHE_DIR / "train_crops.npy").exists():
        sys.exit(f"no cache at {CACHE_DIR}\n  -> python tools/hires_build_cache.py")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    ns = load_namespace()
    ns["CKPT_DIR"] = CKPT_DIR
    torch = ns["torch"]
    device = ns["DEVICE"]

    all_variants = variants(ns)
    chosen = args.variants or list(all_variants)
    unknown = [v for v in chosen if v not in all_variants]
    if unknown:
        sys.exit("unknown variant(s): " + ", ".join(unknown)
                 + "\nchoose from:\n  " + "\n  ".join(all_variants))

    already = done_variants()
    todo = [v for v in chosen if v not in already]

    print("DSF-Net ablation at 256x256")
    print("=" * 72)
    print(f"  device {device} | crop {args.crop} | batch {args.batch} | epochs {args.epochs}")
    print(f"  {len(todo)} variant(s) to train, {len(already)} already recorded")
    reuse = FULL_VARIANT in todo and MAIN_RUN_CKPT.exists()
    if reuse:
        print(f"  '{FULL_VARIANT}' will reuse {MAIN_RUN_CKPT.name} instead of retraining")
    hours = (len(todo) - (1 if reuse else 0)) * 1.55
    print(f"  estimate: roughly {hours:.1f} h at {args.epochs} epochs")

    if args.dry_run:
        for v in todo:
            print(f"    would train  {v}")
        print("\ndry run: nothing was trained.")
        return

    Xtr, ytr, _ = load_split("train")
    Xva, yva, gva = load_split("validation")
    Xho, yho, gho = load_split("validation_heldout")
    mean, std = channel_stats(Xtr)

    crop = min(args.crop, int(Xtr.shape[1]))
    DataLoader = ns["DataLoader"]
    train_loader = DataLoader(make_dataset(ns, Xtr, ytr, mean, std, True, crop),
                              batch_size=args.batch, shuffle=True, num_workers=0)
    # Ablation 7 needs the conventional heavy stack, not the conservative flip-only one,
    # or it is not an ablation at all: it would train an identical model under an identical
    # pipeline and report a difference that is pure noise.
    heavy_loader = DataLoader(make_dataset(ns, Xtr, ytr, mean, std, True, crop, heavy=True),
                              batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(make_dataset(ns, Xva, yva, mean, std, False, crop),
                            batch_size=args.batch, num_workers=0)
    ho_loader = DataLoader(
        make_dataset(ns, Xho, yho, mean, std, False, min(crop, int(Xho.shape[1]))),
        batch_size=args.batch, num_workers=0)

    scaled_lr = BEST_LR * args.batch / STUDY_BATCH
    cfg = ns["TrainConfig"](epochs=args.epochs, lr=scaled_lr, weight_decay=1e-4,
                            patience=args.patience, seed=SEED)
    print(f"  learning rate {scaled_lr:.2e}\n")

    for index, name in enumerate(todo, start=1):
        spec = all_variants[name]
        tag = "hires_abl_" + name.split(".")[0]
        model = ns["DSFNet"](spec["cfg"])
        params = ns["count_parameters"](model)
        print(f"[{index}/{len(todo)}] {name}  ({params:,} parameters)", flush=True)

        started = time.time()
        reused = False
        if name == FULL_VARIANT and MAIN_RUN_CKPT.exists():
            # Identical configuration, seed, data and budget to the main run: retraining it
            # would spend an hour and a half reproducing a number that already exists.
            state = torch.load(MAIN_RUN_CKPT, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            model.to(device).eval()
            best_auc = float(state.get("val_auc", float("nan")))
            reused = True
            print(f"    reused {MAIN_RUN_CKPT.name} (val AUC {best_auc:.4f})", flush=True)
        else:
            loader = heavy_loader if spec.get("heavy_aug") else train_loader
            history = ns["train_model"](model, tag, loader, val_loader, cfg, verbose=True)
            best_auc = float(history["best_val_auc"])

        val = evaluate(ns, model, val_loader, yva, gva, f"{name} / validation")
        ho = evaluate(ns, model, ho_loader, yho, gho, f"{name} / BigGAN held out")

        append_row({
            "variant": name, "params": params, "best_val_auc": round(best_auc, 6),
            "val_acc": round(val["metrics"]["accuracy"], 6),
            "val_auc": round(val["metrics"]["roc_auc"], 6),
            "val_f1": round(val["metrics"]["f1"], 6),
            "val_ece": round(val["metrics"]["ece"], 6),
            # The held-out set is BigGAN only, so this is recall, not accuracy. The balanced
            # cross-generator measurement lives in tools/hires_crossgen.py.
            "biggan_recall": round(ho["metrics"]["accuracy"], 6),
            "train_time_s": round(time.time() - started, 1),
            "reused": reused,
        })
        print(f"    recorded in {OUT_CSV.name}\n", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report()
    OUT_JSON.write_text(json.dumps({"epochs": args.epochs, "crop": crop,
                                    "batch": args.batch, "lr": scaled_lr}, indent=2),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
