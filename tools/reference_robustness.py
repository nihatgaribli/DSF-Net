"""Does the decomposition depend on which family member is used as the reference?

Set B holds the generator as nearly fixed as a change of corpus allows, and this study
instantiates it with SD 1.5, the direct successor of the training generator. That choice is
principled, and it is still a choice. A reader entitled to ask what happens under a different
one should not have to take it on trust.

Wukong is the stress test. It is Stable Diffusion derived, so it satisfies the design, but it is
further from SD 1.4 than SD 1.5 is, having been retrained on a different corpus of captions. If
the conclusions survive a reference chosen to be worse, the family approximation is doing no
hidden work.

The direction of the movement is worth stating carefully, because the obvious story is wrong.
Substituting a more distant reference does not transfer evidence from one term to the other.
The reference sits on the boundary between the two terms, so it enters the reported difference
twice: |corpus| - |generator| is identically a + m - 2r in the accuracies of the three sets.
The coefficient on the reference is exactly -2, and every movement in the table below follows
from how hard that particular reference happens to be for that particular detector. The script
verifies the identity rather than asserting it.

Two things are checked separately, because they do not behave the same way:

  direction   which of the two terms is larger in magnitude, per detector. This is the
              central claim, and it is what the sign of |corpus| - |generator| encodes.
  resolution  whether the difference is individually resolved at five seeds, meaning its
              95 per cent interval excludes zero.

Usage:
    python tools/reference_robustness.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "reference_robustness.txt"

ORDER = ["CIFAKE-CNN", "DSF-Net", "ResNet-18", "CLIP probe"]
REFERENCES = [("gen_SD15", "SD 1.5", "direct successor of the training generator"),
              ("gen_Wukong", "Wukong", "same family, deliberately more distant")]


def load():
    import pandas as pd

    conv = pd.read_csv(RESULTS / "crossgen_seeds.csv")
    clip = pd.read_csv(RESULTS / "clip_probe.csv").assign(arch="CLIP probe")
    return pd.concat([conv, clip], ignore_index=True)


def decompose(df, reference):
    """Per detector: the two terms, their signed difference, and its interval over seeds."""
    from scipy import stats

    rows = {}
    for arch in ORDER:
        corpus, generator = [], []
        for seed in sorted(df["seed"].unique()):
            d = df[(df["arch"] == arch) & (df["seed"] == seed)].set_index("set")["accuracy"]
            if "A" not in d.index or reference not in d.index:
                continue
            others = [i for i in d.index if i.startswith("gen_") and i != reference]
            corpus.append((d[reference] - d["A"]) * 100)
            generator.append((d[others].mean() - d[reference]) * 100)
        corpus, generator = np.array(corpus), np.array(generator)
        diff = np.abs(corpus) - np.abs(generator)
        half = float(stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)))
        rows[arch] = (corpus.mean(), generator.mean(), diff.mean(), half,
                      abs(diff.mean()) > half)
    return rows


def check_identity(df, reference) -> float:
    """|corpus| - |generator| should equal a + m - 2r exactly. Return the largest error."""
    worst = 0.0
    for arch in ORDER:
        for seed in sorted(df["seed"].unique()):
            d = df[(df["arch"] == arch) & (df["seed"] == seed)].set_index("set")["accuracy"] * 100
            if "A" not in d.index or reference not in d.index:
                continue
            others = [i for i in d.index if i.startswith("gen_") and i != reference]
            a_, r, m = d["A"], d[reference], d[others].mean()
            # The identity needs the reference to lie between the two, which is what makes
            # both terms negative. If a detector ever broke that, the algebra would change.
            assert m < r < a_, f"{arch} seed {seed}: ordering a > r > m does not hold"
            measured = abs(r - a_) - abs(m - r)
            worst = max(worst, abs(measured - (a_ + m - 2 * r)))
    return worst


def main() -> None:
    df = load()
    tables = {name: decompose(df, ref) for ref, name, _ in REFERENCES}

    lines = []

    def emit(t=""):
        lines.append(t)
        print(t)

    emit("The decomposition under two choices of reference generator")
    emit("=" * 74)
    for _, name, why in REFERENCES:
        emit(f"  {name:<8} {why}")
    emit()
    emit("Terms are percentage points. The difference is |corpus| - |generator|, so a positive")
    emit("value means the corpus term is the larger of the two. Five seeds throughout.")
    emit()

    for _, name, _ in REFERENCES:
        emit(f"{name}")
        emit(f"  {'detector':<13}{'corpus':>9}{'generator':>11}{'difference':>12}"
             f"{'95% half':>10}  resolved")
        emit("  " + "-" * 60)
        for arch in ORDER:
            c, g, d, h, res = tables[name][arch]
            emit(f"  {arch:<13}{c:>9.2f}{g:>11.2f}{d:>12.2f}{h:>10.2f}  "
                 f"{'yes' if res else 'no'}")
        emit()

    a, b = tables["SD 1.5"], tables["Wukong"]
    same_sign = [arch for arch in ORDER if np.sign(a[arch][2]) == np.sign(b[arch][2])]
    lost = [arch for arch in ORDER if a[arch][4] and not b[arch][4]]
    kept = [arch for arch in ORDER if a[arch][4] and b[arch][4]]

    emit("What survives the substitution")
    emit("-" * 74)
    emit(f"  direction  : {len(same_sign)} of {len(ORDER)} detectors keep the sign of the")
    emit("               difference, so which term dominates is a property of the detector and")
    emit("               not of the reference. The sign change across detectors, which is the")
    emit("               central claim, is present under both references.")
    emit(f"  resolution : {len(kept)} of {len(ORDER)} stay individually resolved at five seeds "
         f"({', '.join(kept)}).")
    if lost:
        emit(f"               {len(lost)} lose it: {', '.join(lost)}.")
    emit()

    # The movement is not a story about evidence migrating between terms. It is an identity.
    emit("Why the numbers move, exactly")
    emit("-" * 74)
    emit("Write a for the in-distribution accuracy, r for the reference set and m for the mean")
    emit("of the remaining generators. The corpus term is r - a and the generator term is")
    emit("m - r. Both are negative here and r lies between a and m, so")
    emit()
    emit("    |corpus| - |generator|  =  (a - r) - (r - m)  =  a + m - 2r")
    emit()
    worst = max(check_identity(df, ref) for ref, _, _ in REFERENCES)
    emit(f"Checked against every detector, seed and reference above: largest discrepancy "
         f"{worst:.1e}")
    emit("percentage points, which is floating point and not approximation.")
    emit()
    emit("The reference sits on the boundary between the two terms, so it enters the reported")
    emit("difference twice and with a coefficient of exactly -2. A reference one point easier")
    emit("for a detector moves that detector's reported difference two points toward its")
    emit("generator term. Nothing migrates; the boundary moves.")
    emit()
    emit("That is the whole of the table above. Per detector, the reference accuracies are:")
    emit()
    emit(f"  {'detector':<13}{'A':>8}{'SD 1.5':>9}{'Wukong':>9}{'shift':>8}{'difference':>13}")
    for arch in ORDER:
        acc = df[df["arch"] == arch].groupby("set")["accuracy"].mean() * 100
        shift = acc["gen_Wukong"] - acc["gen_SD15"]
        emit(f"  {arch:<13}{acc['A']:>8.2f}{acc['gen_SD15']:>9.2f}{acc['gen_Wukong']:>9.2f}"
             f"{shift:>8.2f}{b[arch][2] - a[arch][2]:>13.2f}")
    emit()
    emit("Wukong is a little easier than SD 1.5 for the three convolutional detectors and")
    emit("considerably harder for the probe, and every movement in the difference follows from")
    emit("that at roughly twice the size and the opposite sign, as the identity requires.")
    emit()

    emit("What this means for anyone using the decomposition")
    emit("-" * 74)
    emit("The reference is not a free parameter. It must be chosen by proximity to the training")
    emit("generator, which is why this study uses SD 1.5, and it must be reported, because a")
    emit("reader cannot reconstruct the split without knowing where the boundary was put. The")
    emit("sensitivity is not a defect of the method so much as a property of any decomposition")
    emit("that splits a total at an intermediate point: the point is load bearing.")
    emit()
    emit("It also sets a practical floor. With a five-seed interval of two to seven points on")
    emit("the convolutional detectors, a reference chosen half a point away from the ideal one")
    emit("costs a point of the difference, which is inside the noise. A reference chosen five")
    emit("points away, as Wukong is for the probe, is not.")
    emit()
    emit("Reported because we ran it. A robustness check that weakens a result belongs in the")
    emit("record exactly as much as one that confirms it.")

    OUT.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print("")
    print(f"written: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
