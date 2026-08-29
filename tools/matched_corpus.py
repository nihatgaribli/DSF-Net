"""Does the corpus term survive when the two sets are matched on low-level statistics?

Section 5.2 screens both evaluation sets with a weak model on twelve per-image summary
statistics and finds that neither is at chance: set A separates at 0.6730 and set B at 0.6180.
From that this study derives a bound, that at most about five points of a corpus term between
19.70 and 28.71 could be low-level rather than semantic.

A bound is weaker than a removal, and the removal is available. If each set is reduced to
real-fake pairs whose twelve statistics are nearly identical, then a model reading only those
statistics is left at chance inside both sets, and any accuracy difference that remains between
them cannot be attributed to the statistics. The corpus term measured on the matched subsets is
the part of it the screen cannot explain.

Matching is one to one within each set on the propensity score fitted from those twelve
statistics, under a caliper. Pairs that cannot be matched inside the caliper are dropped rather
than stretched, so the price is sample size and the price is visible: the matched subsets are
smaller and their intervals wider. That is the honest trade and it is reported alongside.

The detectors are not retrained and not re-run. Matching selects rows, and the per-image scores
for every row are already cached by tools/per_seed_predictions.py, so the matched accuracies are
a re-indexing of numbers this study already stands on.

Usage:
    python tools/matched_corpus.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SETS = ROOT / "data" / "crossgen_sets_32.npz"
PRED = ROOT / "data" / "per_seed_predictions.npz"
OUT = ROOT / "results" / "matched_corpus.txt"

ORDER = ["CIFAKE-CNN", "DSF-Net", "ResNet-18", "CLIP probe"]
SEEDS = [42, 43, 44, 45, 46]
# set name -> (real half, fake half)
PAIRS = {"A": ("A_real", "A_fake"), "B": ("imagenet_real", "gen_SD15")}
CALIPER = 0.20  # standard deviations of the logit propensity, the usual choice


def summary_stats(images: np.ndarray) -> np.ndarray:
    """The same twelve numbers section 5.2 screens on, and for the same reason."""
    x = images.astype(np.float32) / 255.0
    mean_c = x.mean(axis=(1, 2))
    std_c = x.std(axis=(1, 2))
    grey = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    bright = grey.mean(axis=(1, 2))[:, None]
    contrast = grey.std(axis=(1, 2))[:, None]
    sat = (x.max(axis=3) - x.min(axis=3)).mean(axis=(1, 2))[:, None]
    rng = (grey.reshape(len(x), -1).max(1) - grey.reshape(len(x), -1).min(1))[:, None]
    gx = np.abs(np.diff(grey, axis=2)).mean(axis=(1, 2))[:, None]
    gy = np.abs(np.diff(grey, axis=1)).mean(axis=(1, 2))[:, None]
    return np.concatenate([mean_c, std_c, bright, contrast, sat, rng, gx, gy], axis=1)


def screen(fr: np.ndarray, ff: np.ndarray, seed: int = 0) -> float:
    """Balanced cross-validated accuracy of a weak model on the statistics alone."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = min(len(fr), len(ff))
    X = np.concatenate([fr[:n], ff[:n]])
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    return float(cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy").mean())


def match(fr: np.ndarray, ff: np.ndarray, caliper: float):
    """One-to-one nearest-neighbour matching on the propensity score, under a caliper.

    Matching in the raw twelve-dimensional space keeps too little: at a caliper tight enough
    to balance the covariates, fewer than a hundred pairs survive out of a thousand, and the
    intervals that come out the other side are wider than the effect. The propensity score is
    the standard answer. It is the fitted probability that an image is generated given only
    the twelve statistics, which is exactly the quantity section 5.2 screens on, and matching
    on it balances everything that quantity depends on while collapsing the search to one
    dimension.

    The caliper is in standard deviations of the logit propensity, the usual convention.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([fr, ff])
    y = np.concatenate([np.zeros(len(fr), int), np.ones(len(ff), int)])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    model.fit(X, y)
    logit = model.decision_function(X)
    width = caliper * logit.std()
    a, b = logit[:len(fr)], logit[len(fr):]

    # Take the real images in order of how isolated their best partner is, so a scarce match
    # is not spent on an image that had alternatives.
    d = np.abs(a[:, None] - b[None, :])
    taken = np.zeros(len(b), bool)
    ia, ib = [], []
    for i in np.argsort(d.min(axis=1)):
        row = np.where(taken, np.inf, d[i])
        j = int(np.argmin(row))
        if row[j] <= width:
            taken[j] = True
            ia.append(int(i))
            ib.append(j)
    return np.array(ia), np.array(ib)


def matched_accuracy(pred, model, seed, real_key, fake_key, ir, jf) -> float:
    """Accuracy on a matched subset. The two index arrays are equal length by construction."""
    pr = pred[f"{model}|{seed}|{real_key}"][ir]
    pf = pred[f"{model}|{seed}|{fake_key}"][jf]
    return float((np.concatenate([pr < 0.5, pf >= 0.5])).mean())


def published_accuracy(pred, model, seed, real_key, fake_key) -> float:
    """The full-set accuracy exactly as results/crossgen_seeds.csv computed it.

    evaluate_pair in tools/crossgen_32.py truncates both halves to the shorter one and takes
    the first n of each rather than resampling. Set B pairs 1000 photographs against 500
    generated images, so its published accuracy uses only the first 500 photographs. Computing
    a balanced accuracy over all 1500 instead would give a baseline that appears nowhere in
    this study, and the comparison below would be against a number no reader can find.
    """
    pr = pred[f"{model}|{seed}|{real_key}"]
    pf = pred[f"{model}|{seed}|{fake_key}"]
    n = min(len(pr), len(pf))
    return float(np.concatenate([pr[:n] < 0.5, pf[:n] >= 0.5]).mean())


def main() -> None:
    if not PRED.exists():
        raise SystemExit(f"missing {PRED}; run tools/per_seed_predictions.py first")
    from scipy import stats as st

    sets = np.load(SETS)
    pred = np.load(PRED)
    lines = []

    def emit(t=""):
        lines.append(t)
        print(t)

    emit("The corpus term before and after matching on twelve summary statistics")
    emit("=" * 74)
    emit(f"One-to-one nearest-neighbour matching on the propensity score fitted from "
         f"the twelve")
    emit(f"statistics, caliper {CALIPER} standard deviations of its logit. Unmatched "
         f"images are")
    emit("dropped rather than stretched.")
    emit()

    idx, sizes, screens = {}, {}, {}
    emit(f"{'set':<6}{'pair':<28}{'n before':>10}{'n after':>9}"
         f"{'screen before':>15}{'screen after':>14}")
    emit("-" * 74)
    for name, (rk, fk) in PAIRS.items():
        sr, sf = summary_stats(sets[rk]), summary_stats(sets[fk])
        before = screen(sr, sf)
        ir, jf = match(sr, sf, CALIPER)
        after = screen(sr[ir], sf[jf])
        idx[name] = (rk, fk, ir, jf)
        sizes[name] = (min(len(sr), len(sf)), len(ir))
        screens[name] = (before, after)
        emit(f"{name:<6}{rk + ' vs ' + fk:<28}{sizes[name][0]:>10}{sizes[name][1]:>9}"
             f"{before:>15.4f}{after:>14.4f}")

    emit()
    highest = max(v[1] for v in screens.values())
    if highest < 0.53:
        emit(f"Neither matched set screens above {highest:.4f}, so a model reading only these")
        emit("twelve statistics can no longer tell the classes apart in either of them, and")
        emit("what remains of the corpus term is not something they can explain. One of the")
        emit("two screens below 0.5, which is a weak model generalising worse than guessing")
        emit("once the signal it had is gone, not a signal in the other direction.")
    else:
        emit(f"A matched set still screens at {highest:.4f}. Matching has reduced the trivial")
        emit("signal without removing it, so what follows is a tightened bound rather than a")
        emit("clean removal, and should be described that way.")
    emit()

    emit("Corpus term, full sets against matched subsets, five seeds")
    emit("-" * 74)
    emit(f"  {'detector':<13}{'full':>9}{'95% half':>10}{'matched':>10}{'95% half':>10}"
         f"{'change':>9}")
    rows = {}
    for model in ORDER:
        full, matched = [], []
        for seed in SEEDS:
            fa, fb, ma, mb = None, None, None, None
            for name, (rk, fk, ir, jf) in idx.items():
                f = published_accuracy(pred, model, seed, rk, fk)
                m = matched_accuracy(pred, model, seed, rk, fk, ir, jf)
                if name == "A":
                    fa, ma = f, m
                else:
                    fb, mb = f, m
            # The corpus term is B minus A, in percentage points.
            full.append((fb - fa) * 100)
            matched.append((mb - ma) * 100)

        def ci(v):
            v = np.asarray(v, float)
            return v.mean(), float(st.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))

        fm, fh = ci(full)
        mm, mh = ci(matched)
        rows[model] = (fm, fh, mm, mh)
        emit(f"  {model:<13}{fm:>9.2f}{fh:>10.2f}{mm:>10.2f}{mh:>10.2f}{mm - fm:>9.2f}")

    emit()
    kept = [m for m in ORDER if abs(rows[m][2]) > rows[m][3]]
    emit(f"{len(kept)} of {len(ORDER)} corpus terms remain resolved after matching "
         f"({', '.join(kept)}).")
    shrink = [rows[m][2] - rows[m][0] for m in ORDER]
    emit(f"The term moves by {min(shrink):+.2f} to {max(shrink):+.2f} points. Section 5.2 bounds")
    emit("the trivially separable part at about five points, and matching removes whatever part")
    emit("of it the twelve statistics could have carried.")

    OUT.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print("")
    print(f"written: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
