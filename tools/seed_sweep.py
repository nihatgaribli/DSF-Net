"""Retrain every ablation variant across several seeds, then test the deltas properly.

The study as submitted ran end to end twice. That is enough to detect a *non-replication*,
which is what it was used for, but it is not a variance estimate: the two runs put the
run-to-run noise floor at 0.34 pp and 0.08 pp respectively, a factor of four apart. Section
10 of the report lists this as limitation 3, and it is the first thing a reviewer will ask
about. This script closes it.

For each seed it retrains all seven ablation variants under identical conditions, records
test accuracy and ROC-AUC, and then reports, per variant, a mean with a standard deviation
and a paired test against the full gated model. A paired test is the right one here: every
variant sees the same seeds, so the seed effect cancels.

Nothing existing is overwritten. Results go to `results/seeds.csv` and
`results/seeds_digest.txt`; checkpoints go to `checkpoints/seeds/`.

Usage:
    python tools/seed_sweep.py --dry-run     # print the plan and the time estimate, train nothing
    python tools/seed_sweep.py --quick       # 2-epoch wiring check on a subsample, minutes
    python tools/seed_sweep.py               # the real sweep: 5 seeds x 7 variants
    python tools/seed_sweep.py --seeds 42 43 44 45 46 47 48
    python tools/seed_sweep.py --variants "3. concat fusion" "4. gated fusion (full)"
    python tools/seed_sweep.py --report-only # recompute the digest from an existing seeds.csv

The sweep is **resumable**: completed (variant, seed) pairs are read back from seeds.csv and
skipped, and `train_model` itself resumes from its last epoch checkpoint. Interrupting it
and re-running loses at most one epoch.

Requires `data/cifake_cache.npz` (written by the notebook's first run).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from py2ipynb import parse_cells  # noqa: E402

NOTEBOOK = ROOT / "notebooks" / "AIGID_main.py"
CACHE = ROOT / "data" / "cifake_cache.npz"
OUT_CSV = ROOT / "results" / "seeds.csv"
OUT_DIGEST = ROOT / "results" / "seeds_digest.txt"
CKPT_SUBDIR = ROOT / "checkpoints" / "seeds"

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
BASELINE_VARIANT = "4. gated fusion (full)"

# The configuration chosen by the coordinate-descent sweep in Section 12, recorded in
# results/tuning.csv (lr 1e-3, dropout 0.1, width 1.5 -> 848,066 parameters). Hard-coded so
# this script does not have to re-run the sweep, and asserted below so it cannot silently
# drift from the model the study actually reports on.
BEST_LR = 1e-3
BEST_DROPOUT = 0.1
BEST_WIDTH = 1.5
EXPECTED_FULL_PARAMS = 848_066

# BASE_CFG from Section 11, repeated here rather than read out of the namespace: the
# notebook defines it in the same cell that starts training, which is the cell this script
# deliberately stops before. The tuned run overrides only the learning rate, exactly as
# BEST_TRAIN_CFG does in Section 12.
BASE_TRAIN_KWARGS = dict(epochs=30, weight_decay=1e-4, patience=5)

CSV_FIELDS = [
    "variant", "seed", "params", "val_auc",
    "test_acc", "test_auc", "test_f1", "test_ece", "train_time_s",
]


def load_notebook_namespace(quick: bool) -> dict:
    """Execute the notebook's cells up to, but not including, its first training call.

    That point is exactly where every definition exists and the data is loaded: model
    classes, `train_model`, `predict`, `compute_metrics`, and the train/val/test loaders.
    Running the cells rather than reimplementing them is what makes these numbers
    comparable with the ones already in the report; a reimplemented training loop would be
    measuring a different thing.
    """
    # The notebook's EDA cells call plt.show(). Under an interactive backend that blocks on a
    # window nobody is there to close, and the loader hangs after writing its first figure
    # with the process alive and idle. Force a non-interactive backend before any cell runs.
    import matplotlib
    matplotlib.use("Agg", force=True)

    cells = parse_cells(NOTEBOOK.read_text(encoding="utf-8"))

    module = types.ModuleType("nb_seed_sweep")
    sys.modules["nb_seed_sweep"] = module
    ns: dict = module.__dict__

    # Section 4's EDA cells run before any training and they call save_fig, which writes
    # straight into results/figures. Loading the namespace would therefore regenerate
    # figures 01-03 of the submitted study as a side effect, and under --quick it would
    # regenerate them from a 2,000-image subsample instead of 20,000. Redirect the figure
    # and results directories to a scratch path as soon as the notebook defines them;
    # save_fig looks FIG_DIR up globally at call time, so rebinding is enough.
    scratch = ROOT / "results" / "_sweep_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    redirected = False

    executed = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # The first cell that *calls* train_model is where the notebook starts spending
        # GPU hours. Everything this script needs is defined before it.
        if "= train_model(" in source:
            break
        if quick:
            source = source.replace("QUICK_RUN = False", "QUICK_RUN = True")
        exec(compile(source, f"<cell {executed}>", "exec"), ns)
        executed += 1

        if not redirected and "FIG_DIR" in ns:
            ns["FIG_DIR"] = scratch
            ns["RESULTS_DIR"] = scratch  # DATA_DIR is left alone: the cache lives there
            redirected = True

    class _NoBar:
        def __init__(self, iterable=None, **kwargs):
            self._it = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self._it)

        def set_postfix(self, **kwargs):
            pass

    ns["tqdm"] = _NoBar
    return ns


def build_variants(ns: dict) -> dict:
    """The same seven variants as Section 14, rebuilt from the tuned configuration."""
    DSFConfig = ns["DSFConfig"]
    common = dict(dropout=BEST_DROPOUT, width=BEST_WIDTH)
    best = DSFConfig(mode="gated", **common)
    return {
        "1. spatial only":        dict(cfg=DSFConfig(mode="spatial", **common)),
        "2. frequency only":      dict(cfg=DSFConfig(mode="freq", **common)),
        "3. concat fusion":       dict(cfg=DSFConfig(mode="concat", **common)),
        "4. gated fusion (full)": dict(cfg=best),
        "5. no constrained conv": dict(cfg=DSFConfig(mode="gated", use_constrained=False, **common)),
        "6. no radial features":  dict(cfg=DSFConfig(mode="gated", use_radial=False, **common)),
        "7. heavy augmentation":  dict(cfg=best, heavy_aug=True),
    }


def read_done(path: Path) -> set:
    """(variant, seed) pairs already recorded, so an interrupted sweep can resume."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as fh:
        return {(row["variant"], int(row["seed"])) for row in csv.DictReader(fh)}


def append_row(path: Path, row: dict) -> None:
    """Write each result as soon as it exists, not at the end.

    A three-hour sweep that loses everything to a crash in hour three is worse than no
    sweep at all.
    """
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_sweep(ns: dict, variants: dict, seeds: list, verbose: bool) -> None:
    torch = ns["torch"]
    train_model = ns["train_model"]
    predict = ns["predict"]
    compute_metrics = ns["compute_metrics"]
    count_parameters = ns["count_parameters"]
    DSFNet = ns["DSFNet"]
    TrainConfig = ns["TrainConfig"]

    # Keep the sweep's checkpoints out of the main directory: 5 seeds x 7 variants is 35
    # models, and mixing them in with the study's own weights would make it impossible to
    # tell which file belongs to which result.
    CKPT_SUBDIR.mkdir(parents=True, exist_ok=True)
    ns["CKPT_DIR"] = CKPT_SUBDIR

    heavy_loader = ns["make_loader"](
        ns["CifakeDataset"](ns["X_train"], ns["y_train"], train=True, heavy_aug=True),
        ns["BATCH_SIZE"], shuffle=True,
    )

    done = read_done(OUT_CSV)
    todo = [(v, s) for s in seeds for v in variants if (v, s) not in done]
    if done:
        print(f"  resuming: {len(done)} run(s) already in {OUT_CSV.name}, {len(todo)} to go\n")

    for index, (label, seed) in enumerate(todo, start=1):
        spec = variants[label]
        tag = f"seed{seed}_abl_{label.split('.')[0]}"
        loader = heavy_loader if spec.get("heavy_aug") else ns["train_loader"]

        cfg = TrainConfig(**{**BASE_TRAIN_KWARGS, "lr": BEST_LR, "seed": seed})
        model = DSFNet(spec["cfg"])

        if label == BASELINE_VARIANT:
            n = count_parameters(model)
            assert n == EXPECTED_FULL_PARAMS, (
                f"full model has {n:,} parameters, expected {EXPECTED_FULL_PARAMS:,}; "
                "the tuned configuration here has drifted from the study's"
            )

        print(f"[{index}/{len(todo)}] {label:24s} seed {seed} ...", flush=True)
        started = time.time()
        history = train_model(model, tag, loader, ns["val_loader"], cfg, verbose=verbose)
        y_true, y_prob = predict(model, ns["test_loader"])
        metrics = compute_metrics(y_true, y_prob)

        append_row(OUT_CSV, {
            "variant": label,
            "seed": seed,
            "params": count_parameters(model),
            "val_auc": history["best_val_auc"],
            "test_acc": metrics["accuracy"],
            "test_auc": metrics["roc_auc"],
            "test_f1": metrics["f1"],
            "test_ece": metrics["ece"],
            "train_time_s": round(time.time() - started, 1),
        })
        print(f"      test acc {metrics['accuracy']:.4f} | test AUC {metrics['roc_auc']:.4f} "
              f"| {time.time() - started:.0f}s", flush=True)

        # 35 models at ~10 MB each is 350 MB of resume state that is dead weight once the
        # run is recorded. The best weights stay; only the optimiser snapshot goes.
        last = CKPT_SUBDIR / f"{tag}_last.pt"
        if last.exists():
            last.unlink()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_report() -> str:
    """Per-variant mean and spread, plus a paired test of each variant against the full model."""
    import pandas as pd
    from scipy import stats

    df = pd.read_csv(OUT_CSV)
    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    seeds = sorted(df["seed"].unique())
    emit(f"Seeds: {seeds}  ({len(seeds)} per variant)")
    emit()
    emit(f"{'variant':<24} {'mean acc':>9} {'std':>8} {'min':>8} {'max':>8} {'n':>3}")
    emit("-" * 64)
    for variant, group in df.groupby("variant"):
        acc = group["test_acc"]
        emit(f"{variant:<24} {acc.mean():>9.4f} {acc.std(ddof=1):>8.4f} "
             f"{acc.min():>8.4f} {acc.max():>8.4f} {len(acc):>3}")

    emit()
    floor = df[df["variant"] == BASELINE_VARIANT]["test_acc"].std(ddof=1) * 100
    emit(f"The full model's own spread across seeds is {floor:.3f} pp (std). That is the "
         "run-to-run noise on a single number, and it is why a lone accuracy figure quoted "
         "to four decimals means little.")
    emit("It is NOT the threshold for the table below. Every variant is trained on the same "
         "seeds as the reference, so the paired difference cancels the shared seed effect "
         "and has a much smaller spread than either variant alone. A paired interval can "
         "therefore resolve an effect well below this floor. The confidence interval is the "
         "decision rule; this number is context.")
    emit()
    emit(f"Paired comparison against '{BASELINE_VARIANT}', same seeds throughout:")
    emit()
    emit(f"{'variant':<24} {'delta pp':>9} {'95% CI':>19} {'t-test p':>10}")
    emit("-" * 66)

    base = df[df["variant"] == BASELINE_VARIANT].set_index("seed")["test_acc"]
    for variant, group in df.groupby("variant"):
        if variant == BASELINE_VARIANT:
            continue
        other = group.set_index("seed")["test_acc"]
        shared = sorted(set(base.index) & set(other.index))
        if len(shared) < 2:
            emit(f"{variant:<24} {'too few shared seeds':>40}")
            continue
        diff = (other.loc[shared] - base.loc[shared]).to_numpy() * 100.0
        mean = diff.mean()
        if len(shared) >= 2 and diff.std(ddof=1) > 0:
            t_stat, p_value = stats.ttest_rel(other.loc[shared], base.loc[shared])
            half = stats.t.ppf(0.975, len(shared) - 1) * diff.std(ddof=1) / np.sqrt(len(shared))
            ci = f"[{mean - half:+.2f}, {mean + half:+.2f}]"
            p_text = f"{p_value:.4f}"
        else:
            ci, p_text = "n/a", "n/a"
        emit(f"{variant:<24} {mean:>+9.2f} {ci:>19} {p_text:>10}")

    emit()
    emit("Reading this table: the interval is the honest output. An interval that straddles "
         "zero means the effect's sign is unresolved at five seeds; an interval clear of zero "
         "means the effect is real at this sample size, however small it looks. With five "
         "seeds the p-values are weak on their own and should not be read as thresholds.")

    report = "\n".join(lines) + "\n"
    OUT_DIGEST.write_text(report, encoding="utf-8")
    print(f"\nwritten to {OUT_DIGEST.relative_to(ROOT)}")
    return report


def main() -> None:
    global OUT_CSV, OUT_DIGEST, CKPT_SUBDIR

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="subset of variant labels; default is all seven")
    parser.add_argument("--quick", action="store_true",
                        help="2-epoch runs on a subsample: checks the wiring, produces meaningless numbers")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--report-only", action="store_true",
                        help="recompute the digest from an existing seeds.csv, train nothing")
    parser.add_argument("--verbose", action="store_true", help="per-epoch training logs")
    args = parser.parse_args()

    if args.report_only:
        if not OUT_CSV.exists():
            sys.exit(f"no results yet: {OUT_CSV}")
        build_report()
        return

    if not CACHE.exists():
        sys.exit(f"missing dataset cache: {CACHE}\n  -> run notebooks/AIGID_main.ipynb once first")

    if args.quick:
        # Quick-mode numbers are meaningless, and the resume logic reads the CSV back to
        # decide what still needs training. If a quick run were allowed to write into
        # results/seeds.csv it would mark all 35 runs as done and the real sweep would
        # silently train nothing. Keep the two completely separate.
        scratch = ROOT / "results" / "_quick_check"
        scratch.mkdir(parents=True, exist_ok=True)
        OUT_CSV = scratch / "seeds.csv"
        OUT_DIGEST = scratch / "seeds_digest.txt"
        CKPT_SUBDIR = ROOT / "checkpoints" / "seeds_quick"

    print("Seed sweep for the ablation study")
    print("=" * 66)

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        ns = load_notebook_namespace(quick=args.quick)

    variants = build_variants(ns)
    if args.variants:
        unknown = [v for v in args.variants if v not in variants]
        if unknown:
            sys.exit("unknown variant(s): " + ", ".join(unknown)
                     + "\nchoose from:\n  " + "\n  ".join(variants))
        variants = {k: v for k, v in variants.items() if k in args.variants}

    done = read_done(OUT_CSV)
    todo = [(v, s) for s in args.seeds for v in variants if (v, s) not in done]

    print(f"  device:   {ns['DEVICE']}")
    print(f"  variants: {len(variants)}")
    print(f"  seeds:    {args.seeds}")
    print(f"  runs:     {len(todo)} to train, {len(done)} already recorded")
    if args.quick:
        print("  QUICK MODE: 2 epochs on a subsample. The numbers are for wiring only.")
    else:
        # The study's own log puts seven variants at about 37 minutes on an RTX 5070.
        print(f"  estimate: roughly {len(todo) * 5.3 / 60:.1f} h on an RTX 5070, "
              "longer on a slower GPU. The sweep is resumable.")
    print(f"  output:   {OUT_CSV.relative_to(ROOT)}, {OUT_DIGEST.relative_to(ROOT)}")
    print(f"  weights:  {CKPT_SUBDIR.relative_to(ROOT)}/")
    print()

    if args.dry_run:
        for label, seed in todo:
            print(f"  would train  {label:24s} seed {seed}")
        print("\ndry run: nothing was trained.")
        return

    run_sweep(ns, variants, args.seeds, verbose=args.verbose)
    print()
    build_report()


if __name__ == "__main__":
    main()
