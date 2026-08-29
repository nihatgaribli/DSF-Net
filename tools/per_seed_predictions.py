"""Per-image probabilities for every detector, every seed, every evaluation set.

results/crossgen_seeds.csv holds one accuracy per cell, which is enough for the decomposition
and for nothing else. Two of this study's open questions need the scores underneath it:

  the confidence result   section 6.5 currently reports the shape of the score distribution on
                          one seed, and says so as a limitation. Five seeds removes it.
  the matched corpus term section 5.2 bounds how much of the corpus term a trivial statistic
                          could reproduce. Bounding it is weaker than removing it, and it can
                          be removed by re-scoring a subsample of A and B matched on those
                          statistics. Matching selects rows, so it needs per-image scores.

Caching once serves both, and any later question about the score distribution rather than its
mean, without paying the notebook namespace load again.

The CLIP probe is retrained per seed from cached features, which is what it is: a linear layer,
seconds per seed. The other three load their seed checkpoints.

Usage:
    python tools/per_seed_predictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from seed_sweep import BEST_DROPOUT, BEST_WIDTH, DEFAULT_SEEDS, load_notebook_namespace  # noqa: E402

SETS = ROOT / "data" / "crossgen_sets_32.npz"
CLIP_FEATS = ROOT / "data" / "clip_features.npz"
OUT = ROOT / "data" / "per_seed_predictions.npz"


def main() -> None:
    # Namespace first: importing torch before pandas corrupts the heap on this machine, and
    # the notebook imports pandas in its second cell.
    sets = np.load(SETS)
    keys = list(sets.files)
    ns = load_notebook_namespace(quick=False)
    torch = ns["torch"]
    device = ns["DEVICE"]
    mean = torch.tensor(ns["CHANNEL_MEAN"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(ns["CHANNEL_STD"], device=device).view(1, 3, 1, 1)

    def score(model, imgs):
        """Identical to score() in tools/crossgen_32.py, autocast included.

        The published table was produced under mixed precision. Scoring in float32 here
        moves a handful of images across the threshold, and every number derived from this
        cache would then disagree with this study by a few thousandths for no reason.
        """
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), 512):
                part = imgs[i:i + 512].astype(np.float32) / 255.0
                x = torch.from_numpy(part).permute(0, 3, 1, 2)
                x = (x - mean.cpu()) / std.cpu()
                with torch.autocast(device_type=device.type, enabled=ns["AMP_ENABLED"]):
                    logits = model(x.to(device))
                out.append(torch.sigmoid(logits.float().squeeze(-1)).cpu().numpy())
        return np.concatenate(out)

    feats = np.load(CLIP_FEATS)
    X = torch.from_numpy(feats["train_X"]).float()
    y = torch.from_numpy(feats["train_y"]).float().to(device)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    Xn = ((X - mu) / sd).to(device)

    out = {}
    for seed in DEFAULT_SEEDS:
        builders = {
            "DSF-Net": (ROOT / "checkpoints" / "seeds" / f"seed{seed}_abl_4_best.pt",
                        lambda: ns["DSFNet"](ns["DSFConfig"](mode="gated", dropout=BEST_DROPOUT,
                                                             width=BEST_WIDTH))),
            "CIFAKE-CNN": (ROOT / "checkpoints" / "arch_seeds" / f"seed{seed}_cifakecnn_best.pt",
                           lambda: ns["CifakeCNN"]()),
            "ResNet-18": (ROOT / "checkpoints" / "arch_seeds" / f"seed{seed}_resnet18_best.pt",
                          lambda: ns["build_resnet18"](pretrained=False)),
        }
        for name, (ckpt, build) in builders.items():
            if not ckpt.exists():
                sys.exit(f"missing {ckpt}")
            model = build()
            model.load_state_dict(torch.load(ckpt, map_location=device,
                                             weights_only=False)["model"])
            model = model.to(device).eval()
            for k in keys:
                out[f"{name}|{seed}|{k}"] = score(model, sets[k])
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.manual_seed(seed)
        probe = torch.nn.Linear(X.shape[1], 1).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = torch.nn.BCEWithLogitsLoss()
        for _ in range(40):
            perm = torch.randperm(len(Xn), device=device)
            for i in range(0, len(Xn), 512):
                j = perm[i:i + 512]
                opt.zero_grad(set_to_none=True)
                lossf(probe(Xn[j]).squeeze(-1), y[j]).backward()
                opt.step()
        for k in keys:
            fn = ((torch.from_numpy(feats[k]).float() - mu) / sd).to(device)
            with torch.no_grad():
                out[f"CLIP probe|{seed}|{k}"] = torch.sigmoid(
                    probe(fn).squeeze(-1)).cpu().numpy()

        print(f"  seed {seed}: 4 detectors x {len(keys)} sets", flush=True)

    np.savez_compressed(OUT, **out)
    print(f"  written {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1e6:.1f} MB, "
          f"{len(out)} arrays)")
    verify(out)


def verify(out: dict) -> None:
    """Check the cache against the accuracies already published in results/.

    Anything derived from this cache has to sit beside this study's own table, so the two must
    agree. They do not agree exactly, and the reason is worth recording rather than hiding:
    the convolutional detectors run under mixed precision, and re-running the same model on
    the same images moves an occasional borderline image across the 0.5 threshold. The probe
    is linear algebra on cached features and reproduces exactly.
    """
    import pandas as pd

    ref = pd.concat([pd.read_csv(ROOT / "results" / "crossgen_seeds.csv"),
                     pd.read_csv(ROOT / "results" / "clip_probe.csv").assign(arch="CLIP probe")],
                    ignore_index=True)
    gens = [k for k in np.load(SETS).files if k.startswith("gen_")]
    pairs = [("A", "A_real", "A_fake")] + [(g, "imagenet_real", g) for g in gens]

    diffs = []
    for model in ["CIFAKE-CNN", "DSF-Net", "ResNet-18", "CLIP probe"]:
        for seed in DEFAULT_SEEDS:
            for name, real, fake in pairs:
                row = ref[(ref["arch"] == model) & (ref["seed"] == seed)
                          & (ref["set"] == name)]
                if row.empty:
                    continue
                pr, pf = out[f"{model}|{seed}|{real}"], out[f"{model}|{seed}|{fake}"]
                # evaluate_pair truncates both halves to the shorter one and does not resample.
                n = min(len(pr), len(pf))
                pred = (np.concatenate([pr[:n], pf[:n]]) >= 0.5).astype(int)
                y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
                diffs.append(abs(float((pred == y).mean()) - float(row["accuracy"].iloc[0])))

    diffs = np.array(diffs)
    exact = int((diffs < 1e-9).sum())
    print(f"  checked against results/: {exact}/{len(diffs)} cells identical, "
          f"mean {diffs.mean() * 100:.4f} points, max {diffs.max() * 100:.2f}")
    assert diffs.max() < 0.005, (
        f"cache disagrees with the published table by {diffs.max() * 100:.2f} points, which is "
        "too large to be mixed-precision jitter. Something about the scoring path changed.")


if __name__ == "__main__":
    main()
