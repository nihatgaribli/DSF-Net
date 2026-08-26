"""Audit the high-resolution crop cache before anything is trained on it.

Two questions, both of which have to be answered before a single number from this cache
means anything.

**Did the normalisation actually remove the leak?** The raw dataset let a classifier reach
1.0000 from image dimensions alone and 0.9699 from file format alone. After fixed-size
cropping and a common re-encode, trivial classifiers built on low-level statistics should
fall back to chance. If they do not, the cache still leaks and every result from it would
be about the container rather than the generator.

**Is there more spectral signal at 256x256 than at 32x32?** The 32x32 study found the
frequency stream to be far weaker than the spatial one, and one plausible explanation was
simply that a 32x32 image carries very little spectral evidence. The same 16-bin radial
profile plus logistic regression that scored 0.7846 on CIFAKE is run here, at eight times
the resolution, as a first read on whether that explanation holds.

The comparison is not controlled: this is a different dataset, with seven generators
instead of one and ImageNet content instead of CIFAR-10. It is a signal, not a conclusion.

Usage:
    python tools/hires_audit.py
"""

import sys, contextlib, io as _io
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from hires_model import load_namespace
ns = load_namespace()
build_radial_bins = ns["build_radial_bins"]

def load(split, n=6000, seed=0):
    arr = np.load(f"data/hires/{split}_crops.npy", mmap_mode="r")
    meta = np.load(f"data/hires/{split}_meta.npz")
    y = meta["labels"].astype(int)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), n // 2, replace=False) for c in (0, 1)])
    rng.shuffle(idx)
    return arr, idx, y[idx]

def radial(batch_uint8, nbins=16):
    x = torch.from_numpy(np.ascontiguousarray(batch_uint8)).float().div_(255.).permute(0, 3, 1, 2)
    spec = torch.log1p(torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1)).abs())
    size = spec.shape[-1]
    idx, counts = build_radial_bins(size, nbins)
    flat = spec.mean(dim=1).reshape(spec.shape[0], -1)
    out = torch.zeros(spec.shape[0], nbins)
    out.index_add_(1, idx, flat)
    return (out / counts).numpy()

def score(X, y, name):
    ntr = int(0.7 * len(y))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(X[:ntr], y[:ntr])
    acc = clf.score(X[ntr:], y[ntr:])
    print(f"  {name:<46} {acc:.4f}")
    return acc

arr, idx, y = load("train", 6000)
print(f"audit on {len(idx):,} cached 256x256 crops (balanced)\n")

batch = np.stack([arr[i] for i in idx])

print("LEAK CHECKS  (should sit near 0.50 if normalisation worked):")
stats = np.concatenate([batch.reshape(len(batch), -1, 3).mean(1),
                        batch.reshape(len(batch), -1, 3).std(1)], axis=1)
score(stats, y, "per-channel mean and std")
score(batch[:, ::32, ::32, :].reshape(len(batch), -1).astype(np.float32), y, "raw subsampled pixels (8x8 grid)")

print("\nLEGITIMATE SPECTRAL SIGNAL  (the fingerprint we actually want):")
rad = np.concatenate([radial(batch[i:i+256]) for i in range(0, len(batch), 256)])
acc = score(rad, y, "16-bin radial spectrum profile, 256x256")
print(f"\n  same classifier on CIFAKE at 32x32 (the study): 0.7846")
print(f"  here at 256x256:                                {acc:.4f}   ({(acc-0.7846)*100:+.1f} pp)")
