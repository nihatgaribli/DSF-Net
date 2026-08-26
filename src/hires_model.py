"""DSF-Net at arbitrary input resolutions, without touching the submitted notebook.

The 32x32 study in `notebooks/AIGID_main.ipynb` is a finished, twice-replicated deliverable.
This module leaves it alone and adapts its architecture for the high-resolution track by
patching a single method in a *local copy* of the notebook's namespace.

Only one thing in DSF-Net is tied to 32x32. The spatial stream is convolutions followed by
global average pooling, so it already accepts any size. `FrequencyStream.spectrum` is an
FFT over whatever it is handed, and `spec_cnn` also ends in a global average pool. But
`FrequencyStream.__init__` builds its radial-profile bins with a hard-coded
`build_radial_bins(32, n_bins)` and registers them as buffers, so `_radial` produces
nonsense the moment the spectrum is not 32x32.

The fix here computes the bins for whatever spatial size actually arrives and caches them
per size. At 32x32 it returns exactly what the original buffers held, so every number in
the existing study is reproducible bit for bit; `verify_unchanged_at_32` asserts that
against the shipped checkpoint rather than asking anyone to take it on trust.

Why this matters for the track: the whole point of going to high resolution is to stop
resizing images, because resizing is a low-pass filter and the generator fingerprint lives
in the high frequencies. A model that silently mangles its own spectral features at any
size other than 32x32 cannot be part of that.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from smoke_test import load_smoke_namespace  # noqa: E402

NOTEBOOK = ROOT / "notebooks" / "AIGID_main.py"
TUNED_CKPT = ROOT / "checkpoints" / "dsfnet_tuned_best.pt"

# The tuned configuration from results/tuning.csv, as used everywhere else in the project.
BEST_DROPOUT = 0.1
BEST_WIDTH = 1.5
EXPECTED_FULL_PARAMS = 848_066


def _radial_any_size(self, spec):
    """Azimuthal average of a centred spectrum, for a spectrum of any spatial size.

    Replaces the notebook's `_radial`, which indexes into buffers built once for 32x32.
    Bins depend only on the spatial size, so they are computed on demand and cached per
    (size, device); a training run touches one or two sizes, so the cache stays tiny.
    """
    import torch

    size = spec.shape[-1]
    assert spec.shape[-2] == size, (
        f"radial binning assumes a square spectrum, got {tuple(spec.shape[-2:])}. "
        "Crop to a square before the FFT rather than letting this average over rings "
        "that are not circular."
    )

    cache = getattr(self, "_radial_cache", None)
    if cache is None:
        cache = {}
        # Plain attribute, not a buffer: these are derived constants, they must not enter
        # the state_dict or checkpoints from the 32x32 study would stop loading.
        object.__setattr__(self, "_radial_cache", cache)

    key = (int(size), str(spec.device))
    bins = cache.get(key)
    if bins is None:
        idx, counts = _BUILD_RADIAL_BINS(int(size), self.n_bins)
        bins = (idx.to(spec.device), counts.to(spec.device))
        cache[key] = bins
    idx, counts = bins

    b = spec.shape[0]
    flat = spec.mean(dim=1).reshape(b, -1)
    out = torch.zeros(b, self.n_bins, device=spec.device, dtype=flat.dtype)
    out.index_add_(1, idx, flat)
    return out / counts


_BUILD_RADIAL_BINS = None


def load_namespace(quiet: bool = True) -> dict:
    """The notebook's model definitions, patched for arbitrary input sizes."""
    global _BUILD_RADIAL_BINS

    if quiet:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            ns = load_smoke_namespace(NOTEBOOK)
    else:
        ns = load_smoke_namespace(NOTEBOOK)

    _BUILD_RADIAL_BINS = ns["build_radial_bins"]
    # Patch the class in this local namespace only. The notebook file is untouched.
    ns["FrequencyStream"]._radial = _radial_any_size
    return ns


def build_model(ns: dict, mode: str = "gated", width: float = BEST_WIDTH,
                dropout: float = BEST_DROPOUT, **kwargs):
    cfg = ns["DSFConfig"](mode=mode, width=width, dropout=dropout, **kwargs)
    return ns["DSFNet"](cfg)


def verify_any_size(ns: dict, sizes=(32, 64, 128, 224, 256)) -> None:
    """Every variant must build, run and back-propagate at every size we intend to use."""
    import torch

    print(f"{'size':>6}  {'output':>12}  {'radial bins':>12}  gradients")
    print("-" * 52)
    for size in sizes:
        x = torch.randn(2, 3, size, size)
        for mode in ("gated", "concat", "spatial", "freq"):
            model = build_model(ns, mode=mode)
            out = model(x)
            assert out.shape == (2, 1), f"{mode} at {size}: bad output {tuple(out.shape)}"
            out.sum().backward()
        model = build_model(ns, mode="gated")
        spec = model.frequency.spectrum(x)
        radial = model.frequency._radial(spec)
        assert radial.shape == (2, model.frequency.n_bins)
        assert torch.isfinite(radial).all(), f"radial profile is not finite at {size}"
        print(f"{size:>6}  {str(tuple(out.shape)):>12}  {str(tuple(radial.shape)):>12}  ok")


def verify_unchanged_at_32(ns: dict) -> None:
    """The patch must not move a single decimal of the finished 32x32 study.

    Loads the shipped tuned checkpoint and scores the real test set. Anything other than
    the reported 0.9571 means this module has quietly changed the model, and the high
    resolution track would no longer be an extension of the study but a different one.
    """
    import numpy as np
    import torch

    cache = ROOT / "data" / "cifake_cache.npz"
    if not cache.exists() or not TUNED_CKPT.exists():
        print("skipping the 32x32 regression check: cache or checkpoint missing")
        return

    sys.path.insert(0, str(ROOT / "tools"))
    from demo import normalise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ns)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == EXPECTED_FULL_PARAMS, f"{n_params:,} parameters, expected {EXPECTED_FULL_PARAMS:,}"

    ckpt = torch.load(TUNED_CKPT, map_location=device, weights_only=False)
    # strict=True is the point of the check: if the patch had added or renamed a parameter,
    # the study's own checkpoint would refuse to load here rather than load silently wrong.
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    data = np.load(cache)
    X_test, y_test = data["X_test"], data["y_test"]
    probs = []
    with torch.no_grad():
        for start in range(0, len(X_test), 512):
            x = normalise(ns, X_test[start:start + 512]).to(device)
            z, _ = model.embed(x)
            probs.append(torch.sigmoid(model.head(z)).squeeze(1).float().cpu().numpy())
    acc = float(((np.concatenate(probs) >= 0.5).astype(int) == y_test).mean())

    print(f"\n32x32 regression check: accuracy {acc:.4f} (study reports 0.9571)")
    assert abs(acc - 0.9571) < 1e-4, (
        f"patched model scores {acc:.4f} but the study reports 0.9571; the resolution "
        "patch has changed the model's behaviour at 32x32"
    )
    print("  unchanged: the patch is transparent at the study's own resolution")


if __name__ == "__main__":
    ns = load_namespace()
    print("DSF-Net at arbitrary resolutions\n")
    verify_any_size(ns)
    verify_unchanged_at_32(ns)
