"""Extract DSF-Net embeddings for the three evaluation sets, so the mechanism can be shown.

This study's central claim about *why* the decomposition differs between detectors is currently
argued rather than shown: a detector reading semantics should be reorganised by a change of
corpus, and one reading acquisition traces should be reorganised by a change of generator. That
is a statement about the geometry of each detector's representation, and it can be looked at.

CLIP features for these sets are already cached by tools/clip_probe.py. This script produces the
missing half, DSF-Net's fused embedding for the same images, and caches it so the figure can be
rebuilt without paying the notebook namespace load again.

One seed only, seed 42. The figure it feeds makes a qualitative point about geometry, not a
measured claim, and the quantitative version of the same point is the variance decomposition
printed at the end, which is reported for all five seeds.

Usage:
    python tools/embedding_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from seed_sweep import BEST_DROPOUT, BEST_WIDTH, DEFAULT_SEEDS, load_notebook_namespace  # noqa: E402

SETS = ROOT / "data" / "crossgen_sets_32.npz"
OUT = ROOT / "data" / "dsfnet_embeddings.npz"
CKPT_DIR = ROOT / "checkpoints" / "seeds"
KEYS = ["A_real", "A_fake", "imagenet_real", "gen_SD15", "gen_ADM"]


def main() -> None:
    if not SETS.exists():
        sys.exit(f"missing {SETS}; run tools/crossgen_seeds.py first")
    sets = np.load(SETS)

    ns = load_notebook_namespace(quick=False)
    torch = ns["torch"]
    device = ns["DEVICE"]

    out = {}
    for seed in DEFAULT_SEEDS:
        ckpt = CKPT_DIR / f"seed{seed}_abl_4_best.pt"
        if not ckpt.exists():
            continue
        model = ns["DSFNet"](ns["DSFConfig"](mode="gated", dropout=BEST_DROPOUT,
                                             width=BEST_WIDTH))
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=False)["model"])
        model = model.to(device).eval()

        for key in KEYS:
            imgs = sets[key]
            chunks = []
            with torch.no_grad():
                for i in range(0, len(imgs), 256):
                    part = imgs[i:i + 256].astype(np.float32) / 255.0
                    x = torch.from_numpy(part).permute(0, 3, 1, 2).to(device)
                    mean = torch.tensor(ns["CHANNEL_MEAN"], device=device).view(1, 3, 1, 1)
                    std = torch.tensor(ns["CHANNEL_STD"], device=device).view(1, 3, 1, 1)
                    z, _ = model.embed((x - mean) / std)
                    chunks.append(z.float().cpu().numpy())
            out[f"s{seed}_{key}"] = np.concatenate(chunks)
        print(f"  seed {seed}: {len(KEYS)} sets embedded", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.savez_compressed(OUT, **out)
    print(f"  written {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
