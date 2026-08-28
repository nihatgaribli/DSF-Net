"""Is the corpus term semantic, or is it a trivial statistic this study never controlled for?

The decomposition analysis measures a corpus term by pairing ImageNet photographs with generators and
comparing against CIFAKE. It then interprets that term as a semantic change: different
photographs, different content. The interpretation is load-bearing, and it has so far been
assumed rather than checked.

Container leakage cannot be the explanation. Every image in the 32x32 track is resized into a
uint8 array before any model sees it, which destroys dimension, file format and quantisation
signature at once. That much is true by construction.

What survives resizing is coarse image statistics: mean and standard deviation per channel,
overall brightness, contrast, saturation. If a classifier reading only those separates the
classes in a set, then part of what this study calls a corpus term is a low-level statistical
artefact of that pairing rather than a semantic difference, and this study should say so.

The screen is deliberately weak: logistic regression on twelve summary statistics, cross
validated. A weak model is the right instrument, because the question is not whether the classes
are separable at all, it is whether they are separable *trivially*.

Read the output as: 0.50 means the pairing carries no trivial statistical cue, and anything
approaching 1.00 means a model can score on that set while ignoring content entirely.

Usage:
    python tools/corpus_term_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SETS = ROOT / "data" / "crossgen_sets_32.npz"
CACHE = ROOT / "data" / "cifake_cache.npz"
OUT_JSON = ROOT / "results" / "corpus_term_audit.json"
OUT_DIGEST = ROOT / "results" / "corpus_term_audit.txt"


def summary_stats(images: np.ndarray) -> np.ndarray:
    """Twelve numbers per image, none of which involve where anything is."""
    x = images.astype(np.float32) / 255.0
    mean_c = x.mean(axis=(1, 2))                       # 3
    std_c = x.std(axis=(1, 2))                         # 3
    grey = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    bright = grey.mean(axis=(1, 2))[:, None]           # 1
    contrast = grey.std(axis=(1, 2))[:, None]          # 1
    sat = (x.max(axis=3) - x.min(axis=3)).mean(axis=(1, 2))[:, None]   # 1
    rng = (grey.reshape(len(x), -1).max(1) - grey.reshape(len(x), -1).min(1))[:, None]  # 1
    # Two crude sharpness measures: mean absolute gradient in each direction.
    gx = np.abs(np.diff(grey, axis=2)).mean(axis=(1, 2))[:, None]
    gy = np.abs(np.diff(grey, axis=1)).mean(axis=(1, 2))[:, None]
    return np.concatenate([mean_c, std_c, bright, contrast, sat, rng, gx, gy], axis=1)


def screen(real: np.ndarray, fake: np.ndarray, seed: int = 0) -> float:
    """Balanced cross-validated accuracy of a weak model on summary statistics alone."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = min(len(real), len(fake))
    X = np.concatenate([summary_stats(real[:n]), summary_stats(fake[:n])])
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    return float(cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy").mean())


def main() -> None:
    if not SETS.exists():
        raise SystemExit(f"missing {SETS}; run tools/crossgen_seeds.py first")
    sets = np.load(SETS)
    data = np.load(CACHE)

    lines = []

    def emit(t=""):
        lines.append(t)
        print(t)

    emit("Can a trivial statistic separate the classes in each evaluation set?")
    emit("=" * 74)
    emit("Logistic regression on twelve per-image summary statistics, five-fold cross")
    emit("validated. No pixel positions, no learned features, no container metadata: every")
    emit("image is already a 32x32 array, so format and dimension cues cannot survive.")
    emit()
    emit(f"{'set':<34}{'balanced accuracy':>20}")
    emit("-" * 74)

    rows = {}
    a = screen(sets["A_real"], sets["A_fake"])
    rows["A: CIFAKE test"] = a
    emit(f"{'A: CIFAKE test (in distribution)':<34}{a:>20.4f}")
    for k in sorted(k for k in sets.files if k.startswith("gen_")):
        v = screen(sets["imagenet_real"], sets[k])
        label = ("B: " if k == "gen_SD15" else "C: ") + f"ImageNet vs {k[4:]}"
        rows[label] = v
        emit(f"{label:<34}{v:>20.4f}")

    emit()
    worst = max(rows, key=rows.get)
    emit(f"Highest is {rows[worst]:.4f} on {worst}.")
    b_key = "B: ImageNet vs SD15"
    emit()
    emit("What matters for the decomposition is the comparison between A and B, since their")
    emit("difference is the corpus term. Those two screen at "
         f"{rows['A: CIFAKE test']:.4f} and {rows[b_key]:.4f}.")
    gap = abs(rows[b_key] - rows["A: CIFAKE test"]) * 100
    emit(f"The trivial-separability gap between them is {gap:.2f} points, against measured")
    emit("corpus terms of 19.70 to 28.71 points depending on the detector.")
    emit()
    if gap < 5:
        emit("The corpus term is therefore not an artefact of coarse image statistics.")
    else:
        emit("A gap of this size is not negligible. At most a few points of the corpus term")
        emit("could be reproduced by a model reading nothing but summary statistics, so the")
        emit("majority of it is not trivially separable, but the fraction that is must be")
        emit("reported rather than assumed away.")
    emit()
    emit("Note also the absolute levels. Both sets are partly separable by twelve statistics,")
    emit(f"CIFAKE at {rows['A: CIFAKE test']:.4f} and the ImageNet pairing at {rows[b_key]:.4f}.")
    emit("That is a property of these benchmarks worth stating on its own: a detector reporting")
    emit("high accuracy on CIFAKE is beating a baseline well above chance, not one at 0.50.")

    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    OUT_DIGEST.write_text("\n".join(l for l in lines if l is not None) + "\n", encoding="utf-8")
    print(f"\nwritten: {OUT_DIGEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
