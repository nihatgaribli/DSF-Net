"""Does an out-of-distribution detector hedge, or does it answer confidently on the wrong side?

Accuracy cannot tell these apart, and they are not the same failure. A detector at chance that
returns scores near 0.5 is one a deployment can route to a human; a detector at chance that
returns 0.02 and 0.98 with equal frequency on the wrong side offers nothing to route on.

Two counts per detector per set:

  undecided        scores in [0.4, 0.6], the band where a system could defer
  confident error  a score above 0.9 on a photograph, or below 0.1 on a generated image

Five seeds, every evaluation set this study uses, from the per-seed cache. An earlier version of
this script reported one seed and this study carried that as a limitation; the limitation is
gone, and the quantities below have intervals like every other number in this study.

Usage:
    python tools/confidence_table.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data" / "per_seed_predictions.npz"
OUT = ROOT / "results" / "confidence_table.txt"

MODELS = ["CIFAKE-CNN", "DSF-Net", "ResNet-18", "CLIP probe"]
SEEDS = [42, 43, 44, 45, 46]
IN_DIST = [("A_real", 0), ("A_fake", 1)]
OOD = [("imagenet_real", 0, "ImageNet real"), ("gen_SD15", 1, "SD 1.5"),
       ("gen_Wukong", 1, "Wukong"), ("gen_ADM", 1, "ADM"),
       ("gen_BigGAN", 1, "BigGAN"), ("gen_GLIDE", 1, "GLIDE"),
       ("gen_Midjourney", 1, "Midjourney"), ("gen_VQDM", 1, "VQDM")]


def rates(p: np.ndarray, truth: int) -> tuple[float, float]:
    undecided = float(((p > 0.4) & (p < 0.6)).mean()) * 100
    wrong = float((p >= 0.9).mean() if truth == 0 else (p <= 0.1).mean()) * 100
    return undecided, wrong


def ci(v):
    from scipy import stats

    v = np.asarray(v, float)
    return v.mean(), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def main() -> None:
    if not PRED.exists():
        raise SystemExit(f"missing {PRED}; run tools/per_seed_predictions.py first")
    d = np.load(PRED)

    lines = []

    def emit(t=""):
        lines.append(t)
        print(t)

    emit("Undecided scores and confident errors, percentages, five seeds")
    emit("=" * 74)
    emit("undecided       score in [0.4, 0.6]")
    emit("confident error score above 0.9 on a photograph, or below 0.1 on a generated image")
    emit()
    emit("Set A is the balanced in-distribution pair. The worst column is the worst of the")
    emit(f"{len(OOD)} out-of-distribution sets, chosen per seed and then averaged, so it is the")
    emit("worst a run actually saw rather than the worst of the averages.")
    emit()
    emit(f"{'detector':<13}{'undecided A':>14}{'worst OOD':>14}{'conf.err A':>14}"
         f"{'worst OOD':>14}")
    emit("-" * 74)

    table, where = {}, {}
    for m in MODELS:
        ua, wa, uo, wo, wset = [], [], [], [], []
        for s in SEEDS:
            n = sum(len(d[f"{m}|{s}|{k}"]) for k, _ in IN_DIST)
            ua.append(sum(rates(d[f"{m}|{s}|{k}"], t)[0] * len(d[f"{m}|{s}|{k}"])
                          for k, t in IN_DIST) / n)
            wa.append(sum(rates(d[f"{m}|{s}|{k}"], t)[1] * len(d[f"{m}|{s}|{k}"])
                          for k, t in IN_DIST) / n)
            per = [(rates(d[f"{m}|{s}|{k}"], t), lbl) for k, t, lbl in OOD]
            uo.append(max(x[0][0] for x in per))
            best = max(per, key=lambda x: x[0][1])
            wo.append(best[0][1])
            wset.append(best[1])
        table[m] = tuple(ci(v) for v in (ua, uo, wa, wo))
        where[m] = max(set(wset), key=wset.count)
        (uam, uah), (uom, uoh), (wam, wah), (wom, woh) = table[m]
        emit(f"{m:<13}{uam:>8.1f} +/-{uah:<4.1f}{uom:>8.1f} +/-{uoh:<4.1f}"
             f"{wam:>8.1f} +/-{wah:<4.1f}{wom:>8.1f} +/-{woh:<4.1f}")

    emit()
    emit("Worst set per detector, by confident errors: " +
         ", ".join(f"{m} on {where[m]}" for m in MODELS))
    emit()
    emit("Markdown rows for the manuscript table:")
    emit()
    for m in MODELS:
        (uam, uah), (uom, uoh), (wam, wah), (wom, woh) = table[m]
        emit(f"| {m} | {uam:.1f} +/- {uah:.1f} | {uom:.1f} +/- {uoh:.1f} | "
             f"{wam:.1f} +/- {wah:.1f} | {wom:.1f} +/- {woh:.1f} |")

    emit()
    ua = [table[m][0][0] for m in MODELS]
    uo = [table[m][1][0] for m in MODELS]
    wa = [table[m][2][0] for m in MODELS]
    wo = [table[m][3][0] for m in MODELS]
    emit(f"In distribution: undecided {min(ua):.1f} to {max(ua):.1f} per cent, confident errors "
         f"{min(wa):.1f} to {max(wa):.1f}.")
    emit(f"Out of distribution: undecided rises only to {max(uo):.1f} at worst, while confident")
    emit(f"errors reach {max(wo):.1f}.")
    emit(f"Confident errors multiply by {min(w / a for a, w in zip(wa, wo)):.0f}x to "
         f"{max(w / a for a, w in zip(wa, wo)):.0f}x; the undecided band by only "
         f"{min(o / a for a, o in zip(ua, uo)):.1f}x to "
         f"{max(o / a for a, o in zip(ua, uo)):.1f}x.")
    emit()
    best = min(MODELS, key=lambda m: table[m][3][0])
    worst = max(MODELS, key=lambda m: table[m][3][0])
    emit(f"Least confident when wrong: {best}. Most: {worst}.")
    order = sorted(MODELS, key=lambda m: table[m][3][0])
    emit("Ordering by confident errors out of distribution: " + " < ".join(order) + ".")

    OUT.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print("")
    print(f"written: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
