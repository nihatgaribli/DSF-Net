"""Per-generator transfer over five seeds and three architectures.

Phase B of the decomposition analysis. It costs nothing to run: `tools/crossgen_seeds.py` already scored
every one of the fifteen checkpoints on every generator, so this is analysis of
`results/crossgen_seeds.csv` rather than a new experiment.

Two claims from the single-run version are put under the standard the evidence-standard analysis argues for.

The first is that transfer is family-bound: the Stable Diffusion relatives keep usable signal
and everything else collapses. Reported per architecture, with a paired interval over seeds, it
either holds for all three or it does not.

The second is sharper and more fragile. Three generators were reported below chance, meaning
the detector ranks their output as more camera-like than genuine photographs. Below-chance
ranking is worse than failure and is worth a claim of its own, so it has to survive repetition:
an ROC-AUC whose interval spans 0.5 is not a below-chance result, it is an unresolved one. That
distinction is exactly what the evidence-standard analysis insists on, and it is applied here to our own
earlier number.

Usage:
    python tools/crossgen_transfer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
IN_CSV = RESULTS / "crossgen_seeds.csv"
OUT_JSON = RESULTS / "crossgen_transfer.json"
OUT_DIGEST = RESULTS / "crossgen_transfer_digest.txt"

# The generator the models were trained on is SD 1.4; SD 1.5 is its direct successor and Wukong
# is a Chinese Stable Diffusion variant, so these are the family members.
FAMILY = {"SD15", "Wukong"}
ARCHS = ["CIFAKE-CNN", "DSF-Net", "ResNet-18"]


def main() -> None:
    import pandas as pd
    from scipy import stats

    if not IN_CSV.exists():
        raise SystemExit(f"missing {IN_CSV}; run tools/crossgen_seeds.py first")

    df = pd.read_csv(IN_CSV)
    gens = sorted(g[4:] for g in df["set"].unique() if g.startswith("gen_"))
    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    def interval(v):
        v = np.asarray(v, dtype=float)
        h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
        return float(v.mean()), h

    emit("Per-generator transfer over five seeds and three architectures")
    emit("=" * 78)
    emit(f"ROC-AUC, mean over five seeds with a 95% interval. Chance is 0.500.")
    emit("An interval that spans 0.5 is unresolved, not below chance.")
    emit()
    header = f"{'generator':<12}" + "".join(f"{a:>22}" for a in ARCHS)
    emit(header)
    emit("-" * len(header))

    records = []
    for g in sorted(gens, key=lambda x: (x not in FAMILY, x)):
        cells = []
        for arch in ARCHS:
            v = df[(df["arch"] == arch) & (df["set"] == f"gen_{g}")]["roc_auc"].to_numpy()
            m, h = interval(v)
            below = (m + h) < 0.5
            above = (m - h) > 0.5
            mark = "-" if below else ("+" if above else " ")
            cells.append(f"{m:.3f} +/-{h:.3f}{mark:>2}")
            records.append({"generator": g, "arch": arch, "mean_auc": m, "ci_half": h,
                            "resolved_below_chance": bool(below),
                            "resolved_above_chance": bool(above),
                            "family": g in FAMILY})
        tag = g + (" *" if g in FAMILY else "")
        emit(f"{tag:<12}" + "".join(f"{c:>22}" for c in cells))

    emit()
    emit("* Stable Diffusion family. Models were trained on SD 1.4.")
    emit("+ interval entirely above chance    - interval entirely below chance")
    emit()

    fam = [r for r in records if r["family"]]
    out = [r for r in records if not r["family"]]
    emit(f"Family members ({len(fam)} cells): "
         f"{sum(r['resolved_above_chance'] for r in fam)} resolved above chance, "
         f"mean AUC {np.mean([r['mean_auc'] for r in fam]):.3f}.")
    emit(f"Other generators ({len(out)} cells): "
         f"{sum(r['resolved_above_chance'] for r in out)} resolved above chance, "
         f"{sum(r['resolved_below_chance'] for r in out)} resolved below chance, "
         f"{sum(not r['resolved_above_chance'] and not r['resolved_below_chance'] for r in out)}"
         f" unresolved. Mean AUC {np.mean([r['mean_auc'] for r in out]):.3f}.")
    emit()

    below = [r for r in records if r["resolved_below_chance"]]
    if below:
        emit("Resolved below chance, meaning the detector ranks generated images as more")
        emit("camera-like than photographs, and does so consistently across seeds:")
        for r in sorted(below, key=lambda r: r["mean_auc"]):
            emit(f"    {r['arch']:<12} vs {r['generator']:<11} "
                 f"AUC {r['mean_auc']:.3f} +/- {r['ci_half']:.3f}")
    else:
        emit("No architecture and generator pair is resolved below chance at five seeds.")
    emit()
    emit("The single-run version of this table reported three generators below 0.5 for")
    emit("DSF-Net. Repetition is what decides whether that was a finding or a coin toss.")

    OUT_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    OUT_DIGEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten: {OUT_DIGEST.relative_to(ROOT)}, {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
