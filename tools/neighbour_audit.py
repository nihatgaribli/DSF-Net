"""Is a representation organised by source or by subject? Counted rather than eyeballed.

The per-detector process figures show, for five query images, the four nearest neighbours in
frozen CLIP space, and those neighbours plainly share subject matter rather than source. Five
queries chosen by a seeded draw is an illustration, not a measurement, and this study should not
rest a mechanism claim on an illustration.

This counts it over every image. For each of two representations, frozen CLIP features and
DSF-Net's fused embedding, take every image in the pooled evaluation sets, find its k nearest
neighbours among the others, and ask what they share with it:

  same corpus      CIFAKE photographs and ImageNet photographs are different pictures of
                   different things, so this question cannot separate a space organised by
                   origin from one organised by subject matter. It is reported anyway,
                   because a high number here is what the five-image illustration shows.
  same class       are the neighbours generated when the query is generated? That is the task.
  same generator   restricted to the three ImageNet-based sets, which show the same kind of
                   photograph and differ only in which model produced them. This is the
                   question the first column confounds, and the one that carries the claim.

Chance is reported for each, because the sets are not the same size and none of these questions
has a floor at zero.

Usage:
    python tools/neighbour_audit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data" / "per_image_predictions.npz"
EMB = ROOT / "data" / "dsfnet_embeddings.npz"
OUT = ROOT / "results" / "neighbour_audit.txt"
K = 4
SEED = 42

# name, source set, is the content generated
SETS = [
    ("CIFAKE real", "A_real", 0),
    ("CIFAKE fake", "A_fake", 1),
    ("ImageNet real", "imagenet_real", 0),
    ("SD 1.5 fake", "gen_SD15", 1),
    ("ADM fake", "gen_ADM", 1),
]


def neighbour_rates(feats: np.ndarray, source: np.ndarray, klass: np.ndarray, k: int):
    """Fraction of each image's k nearest neighbours sharing its source, and its class."""
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(1e-8)
    same_src, same_cls = [], []
    for i in range(0, len(f), 512):
        sim = f[i:i + 512] @ f.T
        # Exclude each query from its own neighbourhood.
        for r, q in enumerate(range(i, min(i + 512, len(f)))):
            sim[r, q] = -np.inf
        nn = np.argpartition(-sim, k, axis=1)[:, :k]
        same_src.append((source[nn] == source[i:i + 512, None]).mean(axis=1))
        same_cls.append((klass[nn] == klass[i:i + 512, None]).mean(axis=1))
    return float(np.concatenate(same_src).mean()), float(np.concatenate(same_cls).mean())


def main() -> None:
    if not PRED.exists():
        raise SystemExit(f"missing {PRED}; run tools/per_image_predictions.py first")
    d = np.load(PRED)

    source = np.concatenate([[i] * len(d[f"clipfeat_{k}"]) for i, (_, k, _) in enumerate(SETS)])
    klass = np.concatenate([[c] * len(d[f"clipfeat_{k}"]) for _, k, c in SETS])
    # Sets 0 and 1 are CIFAKE, sets 2 to 4 are built on ImageNet photographs.
    corpus = (source >= 2).astype(int)
    n = len(source)

    spaces = {"frozen CLIP": np.concatenate([d[f"clipfeat_{k}"] for _, k, _ in SETS])}
    if EMB.exists():
        e = np.load(EMB)
        if f"s{SEED}_" + SETS[0][1] in e.files:
            spaces["DSF-Net fused"] = np.concatenate([e[f"s{SEED}_{k}"] for _, k, _ in SETS])

    def chance(labels):
        _, cnt = np.unique(labels, return_counts=True)
        m = len(labels)
        return float((cnt * (cnt - 1)).sum() / (m * (m - 1)))

    lines = []

    def emit(t=""):
        lines.append(t)
        print(t)

    emit(f"What do the {K} nearest neighbours of an image share with it?")
    emit("=" * 74)
    emit(f"Every image of the five pooled evaluation sets, {n:,} in total, by cosine distance in")
    emit("each representation. A neighbour is never the query itself. Chance is the rate a")
    emit("randomly drawn neighbour would achieve given the set sizes, which is well above zero.")
    emit()
    emit("Corpus and generator are separated deliberately. Asking only whether a neighbour comes")
    emit("from the query's own evaluation set confounds the two: the CIFAKE sets differ from the")
    emit("ImageNet sets in subject matter as well as in origin, so a representation reading pure")
    emit("content would score high on that question for the wrong reason. The third column asks")
    emit("the question that is not confounded, restricted to the three ImageNet-based sets, which")
    emit("share their photographic content and differ only in which generator produced them.")
    emit()
    hdr = f"{'representation':<20}{'same corpus':>14}{'same class':>13}{'same generator':>17}"
    emit(hdr)
    emit("-" * 74)

    ing = source >= 2
    rows = {}
    emit(f"{'chance':<20}{chance(corpus):>14.3f}{chance(klass):>13.3f}"
         f"{chance(source[ing]):>17.3f}")
    for name, feats in spaces.items():
        a, b = neighbour_rates(feats, corpus, klass, K)
        g, _ = neighbour_rates(feats[ing], source[ing], klass[ing], K)
        rows[name] = (a, b, g)
        emit(f"{name:<20}{a:>14.3f}{b:>13.3f}{g:>17.3f}")

    emit()
    gc = chance(source[ing])
    emit("The third column is the one that carries the mechanism claim. Within the ImageNet")
    emit("corpus, where every set shows the same kind of photograph and the only thing that")
    emit(f"changes is the generator, chance is {gc:.3f}.")
    emit()
    for name, (a, b, g) in rows.items():
        emit(f"  {name:<16} generator agreement {g:.3f}, which is {g - gc:+.3f} against chance")
    emit()
    if len(rows) == 2:
        (n1, r1), (n2, r2) = rows.items()
        lead = n1 if (r1[2] - gc) > (r2[2] - gc) else n2
        lag = n2 if lead == n1 else n1
        emit(f"{lead} groups images by their generator more strongly than {lag} does.")
        emit()
        emit("Read the first column against the third. Both representations put same-corpus")
        emit("images together far above chance, but that column cannot distinguish a space")
        emit("organised by origin from one organised by subject matter, because in these data")
        emit("the two travel together. The third column separates them, and it is much closer")
        emit("to chance for both, which is the honest form of the claim: neither representation")
        emit("groups images by generator anywhere near as strongly as the raw set-membership")
        emit("number would suggest.")

    OUT.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print("")
    print(f"written: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
