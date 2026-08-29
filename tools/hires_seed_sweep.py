"""Repeat the 256px architecture comparison across seeds, so it can be written up.

The 256px section rests on one seed per variant, and the analysis it feeds argues that a
single run reads as convincing and can still be wrong. Publishing a single-seed result in
support of that argument would undercut it. This closes the gap.

Only the variants the argument depends on are repeated:

  * **gated fusion (full)** - the reference, and the model compared against ResNet-18;
  * **spatial only** - the claim that the frequency stream earns nothing;
  * **concat fusion** - the claim that gating buys nothing over concatenation.

Frequency-only is deliberately left at one seed. Its effect is -9.88 pp, an order of
magnitude above any plausible run-to-run noise, so replication would change nothing about
how it is read. The two "remove one component" variants are likewise left out: at 32x32
their five-seed intervals were +0.23 and +0.06 pp, and resolving effects that small at this
resolution would cost far more GPU time than the conclusion is worth.

The seed-42 results already recorded by `tools/hires_ablation.py` are imported rather than
retrained, so only the new seeds cost anything.

Usage:
    python tools/hires_seed_sweep.py --dry-run
    python tools/hires_seed_sweep.py                    # seeds 43-46 for three variants
    python tools/hires_seed_sweep.py --seeds 43 44
    python tools/hires_seed_sweep.py --report-only
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from hires_model import load_namespace  # noqa: E402
from hires_train import (  # noqa: E402
    BEST_DROPOUT, BEST_LR, BEST_WIDTH, CACHE_DIR, CKPT_DIR, OUT_DIR, STUDY_BATCH,
    channel_stats, load_split, make_dataset,
)

OUT_CSV = OUT_DIR / "hires_seeds.csv"
OUT_DIGEST = OUT_DIR / "hires_seeds_digest.txt"
ABLATION_CSV = OUT_DIR / "hires_ablation.csv"
SEED_CKPT_DIR = CKPT_DIR / "seeds"

BASELINE = "4. gated fusion (full)"
CORE_VARIANTS = ["1. spatial only", "3. concat fusion", BASELINE]
DEFAULT_SEEDS = [43, 44, 45, 46]
IMPORTED_SEED = 42

CSV_FIELDS = ["variant", "seed", "params", "best_val_auc", "val_acc", "val_auc",
              "val_ece", "biggan_recall", "train_time_s"]


def configs(ns: dict) -> dict:
    DSFConfig = ns["DSFConfig"]
    common = dict(dropout=BEST_DROPOUT, width=BEST_WIDTH)
    return {
        "1. spatial only": DSFConfig(mode="spatial", **common),
        "3. concat fusion": DSFConfig(mode="concat", **common),
        BASELINE: DSFConfig(mode="gated", **common),
    }


def import_seed42() -> int:
    """Seed the CSV from the single-seed ablation rather than retraining those runs."""
    if OUT_CSV.exists() or not ABLATION_CSV.exists():
        return 0
    rows = []
    with ABLATION_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["variant"] not in CORE_VARIANTS:
                continue
            rows.append({
                "variant": row["variant"], "seed": IMPORTED_SEED, "params": row["params"],
                "best_val_auc": row["best_val_auc"], "val_acc": row["val_acc"],
                "val_auc": row["val_auc"], "val_ece": row["val_ece"],
                "biggan_recall": row["biggan_recall"], "train_time_s": row["train_time_s"],
            })
    if not rows:
        return 0
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def done_pairs() -> set:
    if not OUT_CSV.exists():
        return set()
    with OUT_CSV.open(encoding="utf-8", newline="") as fh:
        return {(r["variant"], int(r["seed"])) for r in csv.DictReader(fh)}


def append_row(row: dict) -> None:
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def report() -> None:
    import pandas as pd
    from scipy import stats

    if not OUT_CSV.exists():
        sys.exit(f"nothing recorded yet: {OUT_CSV}")
    df = pd.read_csv(OUT_CSV)
    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    seeds = sorted(df["seed"].unique())
    emit(f"256px architecture comparison across seeds {seeds}")
    emit()
    emit(f"{'variant':<24} {'mean acc':>9} {'std':>8} {'n':>3}   {'mean BigGAN':>12}")
    emit("-" * 62)
    for variant, group in df.groupby("variant"):
        emit(f"{variant:<24} {group['val_acc'].mean():>9.4f} "
             f"{group['val_acc'].std(ddof=1):>8.4f} {len(group):>3}   "
             f"{group['biggan_recall'].mean():>12.4f}")

    base = df[df["variant"] == BASELINE].set_index("seed")["val_acc"]
    emit()
    emit(f"Paired against '{BASELINE}', same seeds throughout:")
    emit()
    emit(f"{'variant':<24} {'delta pp':>9} {'95% CI':>19} {'p':>8}")
    emit("-" * 64)
    for variant, group in df.groupby("variant"):
        if variant == BASELINE:
            continue
        other = group.set_index("seed")["val_acc"]
        shared = sorted(set(base.index) & set(other.index))
        if len(shared) < 2:
            emit(f"{variant:<24} {'too few shared seeds':>38}")
            continue
        diff = (other.loc[shared] - base.loc[shared]).to_numpy() * 100
        mean = diff.mean()
        if diff.std(ddof=1) > 0:
            _, p = stats.ttest_rel(other.loc[shared], base.loc[shared])
            half = stats.t.ppf(0.975, len(shared) - 1) * diff.std(ddof=1) / np.sqrt(len(shared))
            emit(f"{variant:<24} {mean:>+9.2f} {f'[{mean-half:+.2f}, {mean+half:+.2f}]':>19} {p:>8.4f}")
        else:
            emit(f"{variant:<24} {mean:>+9.2f} {'degenerate':>19} {'n/a':>8}")

    emit()
    emit("Frequency-only is not in this table. Its single-seed effect is -9.88 pp, an order of")
    emit("magnitude above any plausible run-to-run noise, so repeating it would not change how")
    emit("it is read. The same applies in reverse to the two component-removal variants, whose")
    emit("32x32 five-seed intervals were +0.23 and +0.06 pp.")

    OUT_DIGEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten to {OUT_DIGEST.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", default=CORE_VARIANTS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if args.report_only:
        report()
        return
    if not (CACHE_DIR / "train_crops.npy").exists():
        sys.exit(f"no cache at {CACHE_DIR}\n  -> python tools/hires_build_cache.py")

    imported = import_seed42()
    if imported:
        print(f"imported {imported} seed-{IMPORTED_SEED} result(s) from {ABLATION_CSV.name}\n")

    ns = load_namespace()
    ns["CKPT_DIR"] = SEED_CKPT_DIR
    SEED_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch = ns["torch"]
    device = ns["DEVICE"]

    cfgs = configs(ns)
    unknown = [v for v in args.variants if v not in cfgs]
    if unknown:
        sys.exit("unknown variant(s): " + ", ".join(unknown)
                 + "\nchoose from:\n  " + "\n  ".join(cfgs))

    done = done_pairs()
    todo = [(v, s) for s in args.seeds for v in args.variants if (v, s) not in done]

    print("256px architecture comparison across seeds")
    print("=" * 72)
    print(f"  device {device} | crop {args.crop} | batch {args.batch} | epochs {args.epochs}")
    print(f"  variants: {len(args.variants)} | seeds: {args.seeds}")
    print(f"  {len(todo)} run(s) to train, {len(done)} already recorded")
    print(f"  estimate: roughly {len(todo) * 1.2:.1f} h")

    if args.dry_run:
        for v, s in todo:
            print(f"    would train  {v:<24} seed {s}")
        print("\ndry run: nothing was trained.")
        return

    Xtr, ytr, _ = load_split("train")
    Xva, yva, gva = load_split("validation")
    Xho, yho, _ = load_split("validation_heldout")
    mean, std = channel_stats(Xtr)
    crop = min(args.crop, int(Xtr.shape[1]))

    DataLoader = ns["DataLoader"]
    train_loader = DataLoader(make_dataset(ns, Xtr, ytr, mean, std, True, crop),
                              batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(make_dataset(ns, Xva, yva, mean, std, False, crop),
                            batch_size=args.batch, num_workers=0)
    ho_loader = DataLoader(
        make_dataset(ns, Xho, yho, mean, std, False, min(crop, int(Xho.shape[1]))),
        batch_size=args.batch, num_workers=0)

    scaled_lr = BEST_LR * args.batch / STUDY_BATCH

    for index, (variant, seed) in enumerate(todo, start=1):
        tag = f"hires_s{seed}_abl_{variant.split('.')[0]}"
        cfg = ns["TrainConfig"](epochs=args.epochs, lr=scaled_lr, weight_decay=1e-4,
                                patience=args.patience, seed=seed)
        model = ns["DSFNet"](cfgs[variant])
        params = ns["count_parameters"](model)
        print(f"\n[{index}/{len(todo)}] {variant}  seed {seed}  ({params:,} parameters)", flush=True)

        started = time.time()
        history = ns["train_model"](model, tag, train_loader, val_loader, cfg, verbose=True)

        y_true, y_prob = ns["predict"](model, val_loader)
        val = ns["compute_metrics"](y_true, y_prob)
        y_ho, p_ho = ns["predict"](model, ho_loader)
        ho = ns["compute_metrics"](y_ho, p_ho)

        append_row({
            "variant": variant, "seed": seed, "params": params,
            "best_val_auc": round(float(history["best_val_auc"]), 6),
            "val_acc": round(float(val["accuracy"]), 6),
            "val_auc": round(float(val["roc_auc"]), 6),
            "val_ece": round(float(val["ece"]), 6),
            # BigGAN-only set, so this is recall rather than accuracy.
            "biggan_recall": round(float(ho["accuracy"]), 6),
            "train_time_s": round(time.time() - started, 1),
        })
        print(f"    val acc {val['accuracy']:.4f} | AUC {val['roc_auc']:.4f} | "
              f"{(time.time() - started) / 60:.0f} min", flush=True)

        last = SEED_CKPT_DIR / f"{tag}_last.pt"
        if last.exists():
            last.unlink()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    report()


if __name__ == "__main__":
    main()
