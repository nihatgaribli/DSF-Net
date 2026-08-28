"""Extend the five-seed evidence standard from one architecture to three.

`tools/seed_sweep.py` measured how much evidence an architectural claim needs by retraining
seven DSF-Net variants across five seeds. The result is strong but narrow: every one of its
21 comparisons lives inside a single architecture, so a reviewer can answer that DSF-Net is
simply an unusually noisy model and that the single-run standard is safe for better-behaved
ones. That objection is cheap to make and, until now, impossible to refute from our data.

This script removes it. The same five seeds, the same data, the same loaders and the same
training harness are applied to the study's two baselines:

  * **CIFAKE-CNN**, the small published reference architecture (141k parameters), and
  * **ResNet-18**, ImageNet-pretrained and fully fine-tuned (11.2M parameters),

which between them bracket DSF-Net by two orders of magnitude in capacity. DSF-Net itself is
*not* retrained: its five seeds already exist in `results/seeds.csv` as the full gated
variant, trained under identical conditions with the identical seeds, so re-running it would
spend twenty minutes reproducing numbers we already have.

Each architecture keeps the learning rate the study chose for it. That is deliberate. A
comparison in which a baseline is crippled by a learning rate borrowed from another model
measures the borrowing, not the architecture.

Two things come out of it. Per architecture, the seed-to-seed spread, which is the quantity
that sets the floor on what a single run can measure and which this study has so far reported
for only one model. Across architectures, the pairwise single-run disagreement rate computed
exactly as `tools/single_run_risk.py` computes it within DSF-Net, so the two are directly
comparable.

Nothing existing is overwritten. Results go to `results/arch_seeds.csv`, the digest to
`results/arch_seeds_digest.txt`, and checkpoints to `checkpoints/arch_seeds/`.

Usage:
    python tools/arch_seed_sweep.py --dry-run     # the plan and the estimate, train nothing
    python tools/arch_seed_sweep.py --quick       # 2-epoch wiring check, minutes
    python tools/arch_seed_sweep.py               # the real sweep: 2 architectures x 5 seeds
    python tools/arch_seed_sweep.py --archs CIFAKE-CNN
    python tools/arch_seed_sweep.py --report-only # recompute the digest from existing rows

The sweep is **resumable**: completed (architecture, seed) pairs are read back from
arch_seeds.csv and skipped, and `train_model` resumes from its last epoch checkpoint.
Interrupting it and re-running loses at most one epoch.

Requires `data/cifake_cache.npz` and a finished `results/seeds.csv`.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from seed_sweep import DEFAULT_SEEDS, load_notebook_namespace  # noqa: E402

SEEDS_CSV = ROOT / "results" / "seeds.csv"
OUT_CSV = ROOT / "results" / "arch_seeds.csv"
OUT_DIGEST = ROOT / "results" / "arch_seeds_digest.txt"
OUT_JSON = ROOT / "results" / "arch_single_run_risk.json"
CKPT_SUBDIR = ROOT / "checkpoints" / "arch_seeds"

# DSF-Net's five seeds already exist under this label in results/seeds.csv.
DSF_VARIANT = "4. gated fusion (full)"
DSF_LABEL = "DSF-Net (tuned)"

# BASE_CFG and RESNET_CFG from Section 11 of the notebook, repeated here for the same reason
# seed_sweep.py repeats BASE_CFG: the notebook defines them in the cell that starts training,
# which is the cell the namespace loader deliberately stops before. ResNet-18 keeps its own
# learning rate because fine-tuning pretrained weights at 3e-4 destroys them.
ARCH_CFG = {
    "CIFAKE-CNN": dict(epochs=30, lr=3e-4, weight_decay=1e-4, patience=5),
    "ResNet-18":  dict(epochs=20, lr=1e-4, weight_decay=1e-4, patience=5),
}

# Measured from the study's own run: CIFAKE-CNN is small enough to be data-bound, ResNet-18
# is roughly 2.3x slower than DSF-Net per image. Used only for the estimate printed up front.
ARCH_MINUTES = {"CIFAKE-CNN": 3.0, "ResNet-18": 8.0}

EXPECTED_PARAMS = {"CIFAKE-CNN": 141_345, "ResNet-18": 11_169_345}

CSV_FIELDS = [
    "arch", "seed", "params", "val_auc",
    "test_acc", "test_auc", "test_f1", "test_ece", "train_time_s",
]


def build_architectures(ns: dict) -> dict:
    """The study's two baselines, built exactly as Sections 8 and 9 build them."""
    return {
        "CIFAKE-CNN": lambda: ns["CifakeCNN"](),
        "ResNet-18": lambda: ns["build_resnet18"](pretrained=True),
    }


def read_done(path: Path) -> set:
    """(architecture, seed) pairs already recorded, so an interrupted sweep can resume."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as fh:
        return {(row["arch"], int(row["seed"])) for row in csv.DictReader(fh)}


def load_dsf_rows() -> dict:
    """DSF-Net's five seeds, read back from the ablation sweep rather than retrained."""
    if not SEEDS_CSV.exists():
        return {}
    with SEEDS_CSV.open(encoding="utf-8", newline="") as fh:
        return {int(r["seed"]): float(r["test_acc"])
                for r in csv.DictReader(fh) if r["variant"] == DSF_VARIANT}


def run_sweep(ns: dict, archs: dict, seeds: list, verbose: bool) -> None:
    torch = ns["torch"]
    train_model = ns["train_model"]
    predict = ns["predict"]
    compute_metrics = ns["compute_metrics"]
    count_parameters = ns["count_parameters"]
    TrainConfig = ns["TrainConfig"]

    CKPT_SUBDIR.mkdir(parents=True, exist_ok=True)
    ns["CKPT_DIR"] = CKPT_SUBDIR

    done = read_done(OUT_CSV)
    todo = [(a, s) for s in seeds for a in archs if (a, s) not in done]
    if done:
        print(f"  resuming: {len(done)} run(s) already in {OUT_CSV.name}, {len(todo)} to go\n")

    for index, (label, seed) in enumerate(todo, start=1):
        tag = f"seed{seed}_{label.replace('-', '').replace(' ', '_').lower()}"
        cfg = TrainConfig(**{**ARCH_CFG[label], "seed": seed})
        model = archs[label]()

        n = count_parameters(model)
        expected = EXPECTED_PARAMS[label]
        assert n == expected, (
            f"{label} has {n:,} parameters, expected {expected:,}; the architecture here "
            "has drifted from the one the study reports on"
        )

        print(f"[{index}/{len(todo)}] {label:12s} seed {seed} ...", flush=True)
        started = time.time()
        history = train_model(model, tag, ns["train_loader"], ns["val_loader"], cfg,
                              verbose=verbose)
        y_true, y_prob = predict(model, ns["test_loader"])
        metrics = compute_metrics(y_true, y_prob)

        write_row(OUT_CSV, {
            "arch": label,
            "seed": seed,
            "params": n,
            "val_auc": history["best_val_auc"],
            "test_acc": metrics["accuracy"],
            "test_auc": metrics["roc_auc"],
            "test_f1": metrics["f1"],
            "test_ece": metrics["ece"],
            "train_time_s": round(time.time() - started, 1),
        })
        print(f"      test acc {metrics['accuracy']:.4f} | test AUC {metrics['roc_auc']:.4f} "
              f"| {time.time() - started:.0f}s", flush=True)

        # A ResNet-18 optimiser snapshot is 134 MB. Once the run is recorded it is dead
        # weight; the best weights stay so the result can be re-evaluated.
        last = CKPT_SUBDIR / f"{tag}_last.pt"
        if last.exists():
            last.unlink()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_row(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def build_report() -> str:
    """Per-architecture spread, then the same single-run risk analysis across architectures."""
    import json

    import pandas as pd
    from scipy import stats

    df = pd.read_csv(OUT_CSV)
    by_arch = {a: g.set_index("seed")["test_acc"] for a, g in df.groupby("arch")}

    dsf = load_dsf_rows()
    if dsf:
        by_arch[DSF_LABEL] = pd.Series(dsf).sort_index()

    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    order = [a for a in ["CIFAKE-CNN", DSF_LABEL, "ResNet-18"] if a in by_arch]
    emit("Seed variance across three architectures")
    emit("=" * 78)
    emit(f"{len(order)} architectures, {sum(len(by_arch[a]) for a in order)} trained models")
    emit()
    emit(f"{'architecture':<16} {'params':>11} {'seeds':>6} {'mean acc':>10} "
         f"{'sd (pp)':>9} {'range (pp)':>11}")
    emit("-" * 78)

    spreads = {}
    for arch in order:
        acc = by_arch[arch].to_numpy() * 100
        params = (int(df[df["arch"] == arch]["params"].iloc[0]) if arch != DSF_LABEL
                  else 848_066)
        # A standard deviation needs at least two runs. A partially finished sweep is a
        # normal state for this file to be read in, so say so rather than printing nan.
        if len(acc) < 2:
            emit(f"{arch:<16} {params:>11,} {len(acc):>6} {acc.mean():>9.2f}% "
                 f"{'n/a':>9} {'n/a':>11}")
            continue
        sd = float(acc.std(ddof=1))
        spreads[arch] = sd
        emit(f"{arch:<16} {params:>11,} {len(acc):>6} {acc.mean():>9.2f}% "
             f"{sd:>9.3f} {acc.max() - acc.min():>11.3f}")

    emit()
    if len(spreads) < 2:
        emit("Fewer than two architectures have repeated runs; the spread comparison needs "
             "the sweep to finish.")
    else:
        noisiest = max(spreads, key=spreads.get)
        quietest = min(spreads, key=spreads.get)
        ratio = spreads[noisiest] / max(spreads[quietest], 1e-9)
        emit(f"The seed-to-seed spread is not a property of DSF-Net. {noisiest} varies most "
             f"({spreads[noisiest]:.3f} pp) and")
        emit(f"{quietest} least ({spreads[quietest]:.3f} pp), a factor of {ratio:.1f} across "
             f"the {len(spreads)}. Any architectural delta")
        emit("smaller than this spread is unmeasurable from a single run whatever the model.")

    # The same computation as tools/single_run_risk.py, so the two tables can be read
    # side by side: within one architecture there, between architectures here.
    emit()
    emit("How often a single run disagrees with the five-seed verdict")
    emit("-" * 78)
    emit(f"{'comparison':<34} {'paired':>8} {'CI half':>9} {'any':>7} {'same':>7} "
         f"{'resolved':>9}")
    emit("-" * 78)

    records = []
    for a, b in itertools.combinations(order, 2):
        sa, sb = by_arch[a], by_arch[b]
        shared = sorted(set(sa.index) & set(sb.index))
        if len(shared) < 2:
            continue

        diff = (sa.loc[shared].to_numpy() - sb.loc[shared].to_numpy()) * 100
        mean = float(diff.mean())
        half = float(stats.t.ppf(0.975, len(shared) - 1) *
                     diff.std(ddof=1) / np.sqrt(len(shared)))
        resolved = not (mean - half < 0 < mean + half)

        pairs = [(x, y) for x in sa.loc[shared] for y in sb.loc[shared]]
        any_wrong = float(np.mean([np.sign(x - y) != np.sign(mean) for x, y in pairs]))
        same_wrong = float(np.mean([np.sign(sa[s] - sb[s]) != np.sign(mean) for s in shared]))

        records.append({
            "a": a, "b": b, "paired_delta_pp": mean, "ci_half_pp": half,
            "resolved": bool(resolved),
            "any_pairing_disagreement": any_wrong,
            "same_seed_disagreement": same_wrong,
            "n_seeds": len(shared),
        })
        emit(f"{a + ' vs ' + b:<34} {mean:>+8.2f} {half:>9.2f} {any_wrong:>7.0%} "
             f"{same_wrong:>7.0%} {'yes' if resolved else 'NO':>9}")

    emit()
    if records:
        unresolved = [r for r in records if not r["resolved"]]
        resolved_rs = [r for r in records if r["resolved"]]
        emit(f"{len(resolved_rs)} of {len(records)} architecture comparisons are resolved at "
             f"five seeds.")
        if unresolved:
            worst = max(unresolved, key=lambda r: r["any_pairing_disagreement"])
            emit(f"Of the {len(unresolved)} that are not, the worst "
                 f"({worst['a']} vs {worst['b']}) has a single run reporting the wrong "
                 f"direction {worst['any_pairing_disagreement']:.0%} of the time.")
        if resolved_rs:
            smallest = min(resolved_rs, key=lambda r: abs(r["paired_delta_pp"]))
            emit(f"The smallest resolved effect is {abs(smallest['paired_delta_pp']):.2f} pp "
                 f"({smallest['a']} vs {smallest['b']}), which is consistent with the "
                 f"half-point floor")
            emit("the within-architecture analysis found.")

    OUT_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    text = "\n".join(lines) + "\n"
    OUT_DIGEST.write_text(text, encoding="utf-8")
    print(f"\nwritten: {OUT_DIGEST.relative_to(ROOT)}, {OUT_JSON.relative_to(ROOT)}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    global OUT_CSV, OUT_DIGEST, OUT_JSON, CKPT_SUBDIR
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--archs", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.report_only:
        if not OUT_CSV.exists():
            sys.exit(f"nothing to report yet: {OUT_CSV}")
        build_report()
        return

    chosen = args.archs or list(ARCH_CFG)
    unknown = [a for a in chosen if a not in ARCH_CFG]
    if unknown:
        sys.exit("unknown architecture(s): " + ", ".join(unknown)
                 + "\nchoose from:\n  " + "\n  ".join(ARCH_CFG))

    done = read_done(OUT_CSV)
    todo = [(a, s) for s in args.seeds for a in chosen if (a, s) not in done]
    hours = sum(ARCH_MINUTES[a] for a, _ in todo) / 60

    print("Cross-architecture seed sweep at 32x32")
    print("=" * 72)
    print(f"  seeds {args.seeds} | architectures: {', '.join(chosen)}")
    print(f"  {len(todo)} run(s) to train, {len(done)} already recorded")
    print(f"  DSF-Net is reused from {SEEDS_CSV.name}, not retrained "
          f"({len(load_dsf_rows())} seeds found)")
    print(f"  estimate: roughly {hours:.1f} h")

    if args.dry_run:
        for a, s in todo:
            print(f"    would train  {a:12s} seed {s}")
        print("\ndry run: nothing was trained.")
        return

    if args.quick:
        # A wiring check trains two epochs on a subsample. Those numbers must not reach the
        # results file: they would be indistinguishable from real rows, and the resume logic
        # would then skip the runs they impersonate.
        scratch = ROOT / "results" / "_sweep_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        OUT_CSV = scratch / "arch_seeds_quick.csv"
        OUT_DIGEST = scratch / "arch_seeds_quick_digest.txt"
        OUT_JSON = scratch / "arch_single_run_risk_quick.json"
        CKPT_SUBDIR = ROOT / "checkpoints" / "arch_seeds_quick"
        print(f"  quick mode: writing to {scratch.relative_to(ROOT)}, not to results/")

    ns = load_notebook_namespace(args.quick)
    archs = build_architectures(ns)
    run_sweep(ns, {a: archs[a] for a in chosen}, args.seeds, args.verbose)
    print()
    build_report()


if __name__ == "__main__":
    main()
