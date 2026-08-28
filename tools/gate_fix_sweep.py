"""Train the corrected fusion gate over five seeds, and test whether it became readable.

Phase C of the decomposition analysis, and the one experiment that closes a limitation the evidence-standard analysis
states but does not resolve: it shows the gate is unidentifiable and proposes a fix without
training the fixed model.

The argument. The fused representation is

    z = g * P_s(z_s) + (1 - g) * P_f(z_f)

with P_s and P_f unconstrained linear maps. Scaling P_s up and moving g down leaves z unchanged,
so g is not determined by the data and cannot be read as a statement about which stream the
model trusts. The measured consequence in the evidence-standard analysis is a gate that travels less than 0.006
across the entire JPEG quality range, which looks like a model with a stable preference and is
actually a quantity with no fixed value.

The correction here is the second of the two the evidence-standard analysis proposes: normalise each branch
before the gate mixes them,

    z = g * LN(P_s(z_s)) + (1 - g) * LN(P_f(z_f))

with LayerNorm carrying no learnable affine parameters. Affine LayerNorm would reintroduce
exactly the scale freedom being removed, so `elementwise_affine=False` is not a detail.

Two questions follow, and the experiment answers both.

  1. Does the correction cost accuracy? Compared against the five seeds of the original gated
     model already in results/seeds.csv, paired on seed.
  2. Does the gate become readable? Measured as its travel across JPEG qualities 90 to 30,
     against the original's 0.006.

A gate that still does not move would mean the fix does not work. A gate that moves and costs
nothing is the useful outcome. Either is reportable; the experiment is not run to confirm.

Usage:
    python tools/gate_fix_sweep.py --dry-run
    python tools/gate_fix_sweep.py
    python tools/gate_fix_sweep.py --report-only
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from seed_sweep import (  # noqa: E402
    BASE_TRAIN_KWARGS, BEST_DROPOUT, BEST_LR, BEST_WIDTH, DEFAULT_SEEDS,
    load_notebook_namespace,
)

OUT_CSV = ROOT / "results" / "gate_fix.csv"
OUT_DIGEST = ROOT / "results" / "gate_fix_digest.txt"
SEEDS_CSV = ROOT / "results" / "seeds.csv"
CKPT_SUBDIR = ROOT / "checkpoints" / "gate_fix"
ORIGINAL_VARIANT = "4. gated fusion (full)"

JPEG_QUALITIES = [90, 70, 50, 30]
GATE_N = 2000  # images used for each gate measurement

CSV_FIELDS = ["seed", "params", "val_auc", "test_acc", "test_auc", "gate_clean",
              "gate_q90", "gate_q70", "gate_q50", "gate_q30", "gate_travel", "train_time_s"]


def make_identifiable_fusion(ns):
    """GatedFusion with each branch normalised before the convex combination."""
    nn = ns["nn"]
    torch = ns["torch"]

    class IdentifiableGatedFusion(nn.Module):
        """z = g * LN(P_s(z_s)) + (1 - g) * LN(P_f(z_f)), LayerNorm without affine.

        Fixing the scale of both branches is what makes g identifiable: the network can no
        longer compensate for a gate value by rescaling a projection, so g is forced to carry
        the mixing itself.
        """

        def __init__(self, dim_s: int, dim_f: int, dim_out: int = 128):
            super().__init__()
            self.proj_s = nn.Linear(dim_s, dim_out)
            self.proj_f = nn.Linear(dim_f, dim_out)
            self.norm_s = nn.LayerNorm(dim_out, elementwise_affine=False)
            self.norm_f = nn.LayerNorm(dim_out, elementwise_affine=False)
            self.gate = nn.Linear(dim_s + dim_f, dim_out)
            self.out_dim = dim_out

        def forward(self, z_s, z_f):
            g = torch.sigmoid(self.gate(torch.cat([z_s, z_f], dim=1)))
            s = self.norm_s(self.proj_s(z_s))
            f = self.norm_f(self.proj_f(z_f))
            return g * s + (1.0 - g) * f, g

    return IdentifiableGatedFusion


def mean_gate(ns, model, images_u8, labels) -> float:
    """Average gate value over a set of images, the same quantity the study reported."""
    torch = ns["torch"]
    loader = ns["make_loader"](ns["CifakeDataset"](images_u8, labels), ns["BATCH_SIZE"])
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, _ in loader:
            _, g = model.embed(xb.to(ns["DEVICE"], non_blocking=True))
            total += g.mean().item() * len(xb)
            count += len(xb)
    return total / count


def report() -> str:
    import pandas as pd
    from scipy import stats

    df = pd.read_csv(OUT_CSV)
    lines = []

    def emit(t=""):
        lines.append(t)
        print(t)

    emit("The corrected fusion gate over five seeds")
    emit("=" * 74)
    emit(f"{len(df)} runs. Correction: LayerNorm without affine on each branch before mixing.")
    emit()

    emit(f"{'seed':>6}{'test acc':>11}{'gate clean':>13}{'gate q30':>11}{'travel':>10}")
    emit("-" * 74)
    for _, r in df.sort_values("seed").iterrows():
        emit(f"{int(r['seed']):>6}{r['test_acc']:>11.4f}{r['gate_clean']:>13.4f}"
             f"{r['gate_q30']:>11.4f}{r['gate_travel']:>+10.4f}")

    travel = df["gate_travel"].to_numpy()
    m = float(np.mean(np.abs(travel)))
    emit()
    emit(f"Mean absolute gate travel, clean to JPEG q30: {m:.4f}")
    emit("The original gate travelled 0.0060 in run 1 and 0.0015 in run 2, and the two")
    emit("disagreed about the sign. A correction that works should exceed that clearly.")
    verdict = ("readable" if m > 0.02 else
               "still not readable" if m < 0.008 else "marginal")
    emit(f"Verdict on identifiability: {verdict}.")

    if SEEDS_CSV.exists():
        base = pd.read_csv(SEEDS_CSV)
        base = base[base["variant"] == ORIGINAL_VARIANT].set_index("seed")["test_acc"]
        this = df.set_index("seed")["test_acc"]
        shared = sorted(set(base.index) & set(this.index))
        if len(shared) >= 2:
            d = (this.loc[shared] - base.loc[shared]).to_numpy() * 100
            h = float(stats.t.ppf(0.975, len(shared) - 1) * d.std(ddof=1) / np.sqrt(len(shared)))
            resolved = not (d.mean() - h < 0 < d.mean() + h)
            emit()
            emit(f"Cost of the correction, paired against the original gate on {len(shared)} "
                 f"seeds:")
            emit(f"  {d.mean():+.2f} pp, 95% CI [{d.mean()-h:+.2f}, {d.mean()+h:+.2f}], "
                 f"{'resolved' if resolved else 'unresolved'}")
            if not resolved:
                emit("  The correction is free at this sample size: it buys identifiability")
                emit("  without a measurable change in accuracy.")

    text = "\n".join(lines) + "\n"
    OUT_DIGEST.write_text(text, encoding="utf-8")
    print(f"\nwritten: {OUT_DIGEST.relative_to(ROOT)}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        if not OUT_CSV.exists():
            sys.exit(f"nothing to report yet: {OUT_CSV}")
        report()
        return

    done = set()
    if OUT_CSV.exists():
        with OUT_CSV.open(encoding="utf-8", newline="") as fh:
            done = {int(r["seed"]) for r in csv.DictReader(fh)}
    todo = [s for s in args.seeds if s not in done]

    print("Corrected fusion gate, five seeds")
    print("=" * 74)
    print(f"  {len(todo)} run(s) to train, {len(done)} already recorded")
    print(f"  estimate: roughly {len(todo) * 4.4 / 60:.1f} h")
    if args.dry_run:
        print("\ndry run: nothing was trained.")
        return

    ns = load_notebook_namespace(quick=False)
    ns["GatedFusion"] = make_identifiable_fusion(ns)   # DSFNet resolves this at call time
    CKPT_SUBDIR.mkdir(parents=True, exist_ok=True)
    ns["CKPT_DIR"] = CKPT_SUBDIR

    torch = ns["torch"]
    DSFNet, DSFConfig = ns["DSFNet"], ns["DSFConfig"]
    X_test, y_test = ns["X_test"], ns["y_test"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_test), min(GATE_N, len(X_test)), replace=False)
    gate_imgs, gate_lbls = X_test[idx], y_test[idx]
    degraded = {q: np.stack([ns["jpeg_compress"](im, q) for im in gate_imgs])
                for q in JPEG_QUALITIES}

    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for i, seed in enumerate(todo, start=1):
            cfg = DSFConfig(mode="gated", dropout=BEST_DROPOUT, width=BEST_WIDTH)
            model = DSFNet(cfg)
            assert type(model.fusion).__name__ == "IdentifiableGatedFusion", \
                "the corrected fusion was not installed; DSFNet still built the original"
            tcfg = ns["TrainConfig"](**{**BASE_TRAIN_KWARGS, "lr": BEST_LR, "seed": seed})

            print(f"[{i}/{len(todo)}] seed {seed} ...", flush=True)
            t0 = time.time()
            hist = ns["train_model"](model, f"seed{seed}_gatefix", ns["train_loader"],
                                     ns["val_loader"], tcfg, verbose=args.verbose)
            y_true, y_prob = ns["predict"](model, ns["test_loader"])
            m = ns["compute_metrics"](y_true, y_prob)

            g_clean = mean_gate(ns, model, gate_imgs, gate_lbls)
            gates = {q: mean_gate(ns, model, degraded[q], gate_lbls) for q in JPEG_QUALITIES}
            row = {
                "seed": seed, "params": ns["count_parameters"](model),
                "val_auc": hist["best_val_auc"], "test_acc": m["accuracy"],
                "test_auc": m["roc_auc"], "gate_clean": g_clean,
                **{f"gate_q{q}": gates[q] for q in JPEG_QUALITIES},
                "gate_travel": gates[JPEG_QUALITIES[-1]] - g_clean,
                "train_time_s": round(time.time() - t0, 1),
            }
            w.writerow(row)
            fh.flush()
            print(f"      acc {m['accuracy']:.4f} | gate {g_clean:.4f} -> "
                  f"{gates[30]:.4f} ({row['gate_travel']:+.4f}) | "
                  f"{time.time() - t0:.0f}s", flush=True)

            last = CKPT_SUBDIR / f"seed{seed}_gatefix_last.pt"
            if last.exists():
                last.unlink()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    report()


if __name__ == "__main__":
    main()
