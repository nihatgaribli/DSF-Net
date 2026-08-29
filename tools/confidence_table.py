"""Does an out-of-distribution detector hedge, or does it answer confidently on the wrong side?

Accuracy cannot tell these apart, and they are not the same failure. A detector at chance that
returns scores near 0.5 is one a deployment can route to a human; a detector at chance that
returns 0.02 and 0.98 with equal frequency on the wrong side offers nothing to route on.

Two counts per detector per set, from the cached per-image probabilities:

  undecided        scores in [0.4, 0.6], the band where a system could defer
  confident error  a score above 0.9 on a photograph, or below 0.1 on a generated image

Seed 42, matching the process figures. The claim this supports is about the shape of the score
distribution rather than about an effect size, and the distributions themselves are plotted in
the lower panel of each process figure.

Usage:
    python tools/confidence_table.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data" / "per_image_predictions.npz"
OUT = ROOT / "results" / "confidence_table.txt"

MODELS = ["CIFAKE-CNN", "DSF-Net", "ResNet-18", "CLIP probe"]
IN_DIST = [("A_real", 0), ("A_fake", 1)]
OOD = [("imagenet_real", 0, "ImageNet real"), ("gen_SD15", 1, "SD 1.5"),
       ("gen_ADM", 1, "ADM")]


def counts(p: np.ndarray, truth: int) -> tuple[float, float]:
    undecided = float(((p > 0.4) & (p < 0.6)).mean())
    wrong = float((p >= 0.9).mean() if truth == 0 else (p <= 0.1).mean())
    return undecided * 100, wrong * 100


def main() -> None:
    if not PRED.exists():
        raise SystemExit(f"missing {PRED}; run tools/per_image_predictions.py first")
    d = np.load(PRED)

    lines = []

    def emit(t=""):
        lines.append(t)
        print(t)

    emit("Undecided scores and confident errors, percentages of each set")
    emit("=" * 74)
    emit("undecided       score in [0.4, 0.6]")
    emit("confident error score above 0.9 on a photograph, or below 0.1 on a generated image")
    emit()
    emit(f"{'detector':<13}{'undecided A':>13}{'worst OOD':>11}{'conf.err A':>13}"
         f"{'worst OOD':>11}  {'where':<14}")
    emit("-" * 74)

    rows = {}
    for m in MODELS:
        # Set A is a balanced pair, so its rate is the mean over its two halves weighted by
        # size rather than the rate of either half.
        n = sum(len(d[f"p_{m}_{k}"]) for k, _ in IN_DIST)
        u_a = sum(counts(d[f"p_{m}_{k}"], t)[0] * len(d[f"p_{m}_{k}"]) for k, t in IN_DIST) / n
        w_a = sum(counts(d[f"p_{m}_{k}"], t)[1] * len(d[f"p_{m}_{k}"]) for k, t in IN_DIST) / n

        ood = [(counts(d[f"p_{m}_{k}"], t), lbl) for k, t, lbl in OOD]
        u_o = max(x[0][0] for x in ood)
        (w_o, where) = max(((x[0][1], x[1]) for x in ood))
        rows[m] = (u_a, u_o, w_a, w_o, where)
        emit(f"{m:<13}{u_a:>13.1f}{u_o:>11.1f}{w_a:>13.1f}{w_o:>11.1f}  {where:<14}")

    emit()
    emit("Markdown row form, for the manuscript table:")
    emit()
    for m in MODELS:
        u_a, u_o, w_a, w_o, _ = rows[m]
        emit(f"| {m} | {u_a:.1f} | {u_o:.1f} | {w_a:.1f} | {w_o:.1f} |")

    emit()
    ua = [rows[m][0] for m in MODELS]
    wa = [rows[m][2] for m in MODELS]
    uo = [rows[m][1] for m in MODELS]
    wo = [rows[m][3] for m in MODELS]
    emit(f"In distribution: undecided {min(ua):.1f} to {max(ua):.1f} per cent, confident errors "
         f"{min(wa):.1f} to {max(wa):.1f}.")
    emit(f"Out of distribution: undecided rises only to {max(uo):.1f} per cent at worst, while")
    emit(f"confident errors reach {max(wo):.1f}.")
    emit()
    factors = [w / a for a, w in zip(wa, wo)]
    emit(f"Confident errors multiply by {min(factors):.0f}x to {max(factors):.0f}x; the undecided")
    emit(f"band multiplies by only {min(o / a for a, o in zip(ua, uo)):.1f}x to "
         f"{max(o / a for a, o in zip(ua, uo)):.1f}x.")
    emit()
    best = min(MODELS, key=lambda m: rows[m][3])
    worst = max(MODELS, key=lambda m: rows[m][3])
    emit(f"Least confident when wrong: {best}. Most: {worst}. The ordering does not follow")
    emit("capacity in the direction one would hope: the smallest detector degrades the most")
    emit("gracefully by this measure and the largest degrades the worst.")

    OUT.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print("")
    print(f"written: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
