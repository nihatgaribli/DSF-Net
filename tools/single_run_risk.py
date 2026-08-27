"""How often would a single run have reported the wrong answer?

This study argues that architectural claims in this area are established at an evidence
standard that cannot support them. Section 6.4 shows this happening once, to our own central
claim. This script turns the anecdote into a rate.

The five-seed sweep left 35 trained models: seven variants at five seeds each. For any pair of
variants there are 25 ways to pick one run of each, and each of those pairings is exactly the
comparison a single-run paper would have published. Comparing the sign of each against the
paired five-seed verdict gives the probability that a single run reports the wrong direction.

Two rates are reported because they answer different questions:

  **any-pairing** takes one run of A and one of B independently, which is what happens when
  two numbers are compared across papers, or across a table assembled at different times.

  **same-seed** compares the two runs that shared a seed, which is the best case: it is what a
  careful single-run study does, holding initialisation fixed across the comparison. If even
  this disagrees often, seed control is not a substitute for repetition.

Nothing is trained. This reads results/seeds.csv and costs seconds.

Usage:
    python tools/single_run_risk.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS_CSV = ROOT / "results" / "seeds.csv"
OUT_JSON = ROOT / "results" / "single_run_risk.json"
OUT_TXT = ROOT / "results" / "single_run_risk.txt"
BASELINE = "4. gated fusion (full)"


def main() -> None:
    import pandas as pd
    from scipy import stats

    if not SEEDS_CSV.exists():
        raise SystemExit(f"missing {SEEDS_CSV}; run tools/seed_sweep.py first")

    df = pd.read_csv(SEEDS_CSV)
    by_variant = {v: g.set_index("seed")["test_acc"] for v, g in df.groupby("variant")}
    variants = sorted(by_variant)
    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    emit("How often a single run disagrees with the five-seed verdict")
    emit("=" * 78)
    emit(f"{len(variants)} variants, {len(df)} trained models, "
         f"{len(df['seed'].unique())} seeds each")
    emit()
    emit(f"{'comparison':<44} {'paired':>8} {'any':>7} {'same':>7} {'resolved':>9}")
    emit(f"{'':<44} {'delta':>8} {'pair':>7} {'seed':>7} {'':>9}")
    emit("-" * 80)

    records = []
    for a, b in itertools.combinations(variants, 2):
        sa, sb = by_variant[a], by_variant[b]
        shared = sorted(set(sa.index) & set(sb.index))
        if len(shared) < 2:
            continue

        diff = (sa.loc[shared] - sb.loc[shared]).to_numpy() * 100
        mean = float(diff.mean())
        half = float(stats.t.ppf(0.975, len(shared) - 1) *
                     diff.std(ddof=1) / np.sqrt(len(shared)))
        resolved = not (mean - half < 0 < mean + half)

        # Every way of picking one run of each: the comparison a single-run paper publishes.
        pairs = [(x, y) for x in sa.loc[shared] for y in sb.loc[shared]]
        any_wrong = np.mean([np.sign(x - y) != np.sign(mean) for x, y in pairs])
        same_wrong = np.mean([np.sign(sa[s] - sb[s]) != np.sign(mean) for s in shared])

        records.append({
            "a": a, "b": b, "paired_delta_pp": mean, "ci_half_pp": half,
            "resolved": bool(resolved),
            "any_pairing_disagreement": float(any_wrong),
            "same_seed_disagreement": float(same_wrong),
        })
        label = f"{a[3:]} vs {b[3:]}"
        emit(f"{label:<44} {mean:>+8.2f} {any_wrong:>7.0%} {same_wrong:>7.0%} "
             f"{'yes' if resolved else 'NO':>9}")

    emit()
    unresolved = [r for r in records if not r["resolved"]]
    resolved = [r for r in records if r["resolved"]]

    if unresolved:
        worst = max(unresolved, key=lambda r: r["any_pairing_disagreement"])
        emit(f"Among the {len(unresolved)} comparisons the five-seed test leaves unresolved, a "
             f"single run picks a direction anyway, and picks the one opposite to the small "
             f"measured mean up to {worst['any_pairing_disagreement']:.0%} of the time. Those "
             "are the comparisons where a published single-run result is a coin toss reported "
             "as a finding.")
    emit()
    if resolved:
        worst_resolved = max(resolved, key=lambda r: r["any_pairing_disagreement"])
        emit(f"Among the {len(resolved)} comparisons that are resolved, the highest single-run "
             f"disagreement is {worst_resolved['any_pairing_disagreement']:.0%} "
             f"({worst_resolved['a'][3:]} vs {worst_resolved['b'][3:]}). A real effect can "
             "still be reported backwards by one run when it is small relative to seed noise.")

    emit()
    emit("Same-seed columns are the best case a single-run study can achieve: the same "
         "initialisation on both sides of the comparison. Where that column is also non-zero, "
         "controlling the seed did not rescue the comparison, and only repetition would have.")

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT_TXT.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
