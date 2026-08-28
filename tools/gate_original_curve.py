"""Measure the ORIGINAL fusion gate across JPEG qualities, over the same five seeds.

The corrected gate was measured on five seeds. The original was measured twice, once in each
full run of the study, and those two runs disagreed about the sign. Comparing a five-seed mean
against two single runs would be exactly the asymmetry the evidence-standard analysis argues against, so the
original is remeasured here on the same five seeds and the same images.

No training. The five checkpoints already exist in checkpoints/seeds as the full gated variant
of the ablation sweep, and the namespace is deliberately left unpatched so that DSFNet rebuilds
the original GatedFusion rather than the corrected one.

Usage:
    python tools/gate_original_curve.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from gate_fix_sweep import GATE_N, JPEG_QUALITIES, mean_gate  # noqa: E402
from seed_sweep import BEST_DROPOUT, BEST_WIDTH, DEFAULT_SEEDS, load_notebook_namespace  # noqa: E402

OUT_CSV = ROOT / "results" / "gate_original.csv"
CKPT_DIR = ROOT / "checkpoints" / "seeds"
FIELDS = ["seed", "gate_clean"] + [f"gate_q{q}" for q in JPEG_QUALITIES] + ["gate_travel"]


def main() -> None:
    ns = load_notebook_namespace(quick=False)
    torch = ns["torch"]
    assert type(ns["GatedFusion"]).__name__ != "IdentifiableGatedFusion", \
        "the namespace was patched; this script must build the ORIGINAL fusion"

    X_test, y_test = ns["X_test"], ns["y_test"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_test), min(GATE_N, len(X_test)), replace=False)
    imgs, lbls = X_test[idx], y_test[idx]
    degraded = {q: np.stack([ns["jpeg_compress"](im, q) for im in imgs])
                for q in JPEG_QUALITIES}

    print("Original fusion gate across JPEG qualities, five seeds")
    print("=" * 70)
    rows = []
    for seed in DEFAULT_SEEDS:
        ckpt = CKPT_DIR / f"seed{seed}_abl_4_best.pt"
        if not ckpt.exists():
            print(f"  seed {seed}: checkpoint missing, skipped")
            continue
        model = ns["DSFNet"](ns["DSFConfig"](mode="gated", dropout=BEST_DROPOUT,
                                            width=BEST_WIDTH))
        model.load_state_dict(torch.load(ckpt, map_location=ns["DEVICE"],
                                         weights_only=False)["model"])
        model = model.to(ns["DEVICE"]).eval()
        assert type(model.fusion).__name__ == "GatedFusion", "wrong fusion class loaded"

        clean = mean_gate(ns, model, imgs, lbls)
        gates = {q: mean_gate(ns, model, degraded[q], lbls) for q in JPEG_QUALITIES}
        row = {"seed": seed, "gate_clean": clean,
               **{f"gate_q{q}": gates[q] for q in JPEG_QUALITIES},
               "gate_travel": gates[JPEG_QUALITIES[-1]] - clean}
        rows.append(row)
        print(f"  seed {seed}: {clean:.4f} -> {gates[30]:.4f}  ({row['gate_travel']:+.4f})")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    from scipy import stats
    t = np.array([r["gate_travel"] for r in rows])
    h = float(stats.t.ppf(0.975, len(t) - 1) * t.std(ddof=1) / np.sqrt(len(t)))
    print()
    print(f"  original gate travel: {t.mean():+.4f}, 95% CI [{t.mean()-h:+.4f}, "
          f"{t.mean()+h:+.4f}], "
          f"{'resolved' if not (t.mean()-h < 0 < t.mean()+h) else 'UNRESOLVED'}")
    print(f"  written: {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
