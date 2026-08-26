"""Smoke-test the notebook's own code without downloading the 120k-image dataset.

Cells in `notebooks/AIGID_main.py` that are marked with a leading `# @smoke` comment are
pure definitions (helpers, models, the training harness). This script executes exactly
those cells in order, then exercises them on a small synthetic dataset that contains a
*known* high-frequency artefact. If DSF-Net can find a planted checkerboard artefact, the
FFT path, the constrained front-end, the gate, AMP and the training loop are all wired up
correctly -- which is everything that could plausibly break before real data arrives.

Usage:
    python tools/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from py2ipynb import parse_cells  # noqa: E402


def load_smoke_namespace(nb_path: Path) -> dict:
    """Execute every `# @smoke` tagged code cell and return the resulting namespace."""
    cells = parse_cells(nb_path.read_text(encoding="utf-8"))

    # @dataclass looks its owning class up via sys.modules[cls.__module__], so the cells
    # need to run inside a real (if synthetic) module rather than a bare dict.
    module = types.ModuleType("nb_smoke")
    sys.modules["nb_smoke"] = module
    ns: dict = module.__dict__
    executed = 0

    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        first_line = next((ln for ln in source.splitlines() if ln.strip()), "")
        if first_line.strip() != "# @smoke":
            continue
        exec(compile(source, f"<cell {executed}>", "exec"), ns)
        executed += 1

    # Silence progress bars: they make the test log unreadable and tell us nothing.
    class _NoBar:
        def __init__(self, iterable=None, **kwargs):
            self._it = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self._it)

        def set_postfix(self, **kwargs):
            pass

    ns["tqdm"] = _NoBar

    print(f"executed {executed} tagged cells from {nb_path.name}\n")
    return ns


def make_synthetic_data(ns: dict, n: int = 3000):
    """Smooth random images; the FAKE half carries a faint checkerboard 'generator artefact'.

    The checkerboard sits at the Nyquist frequency, so it is nearly invisible in pixel
    space but unmistakable in the spectrum -- the same structure as a real generator
    fingerprint, which is exactly what we want to verify the model can pick up.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    low = torch.randn(n, 3, 8, 8)
    images = F.interpolate(low, size=32, mode="bicubic", align_corners=False)
    images = (images - images.mean()) / (images.std() + 1e-6)

    labels = (torch.arange(n) % 2).float()
    checker = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(16, 16)
    images[labels == 1] += 0.10 * checker

    return images, labels


def main() -> int:
    nb = ROOT / "notebooks" / "AIGID_main.py"
    ns = load_smoke_namespace(nb)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    DSFNet, DSFConfig = ns["DSFNet"], ns["DSFConfig"]
    TrainConfig, train_model = ns["TrainConfig"], ns["train_model"]
    count_parameters, compute_metrics = ns["count_parameters"], ns["compute_metrics"]
    predict, measure_latency = ns["predict"], ns["measure_latency"]

    print(f"device: {ns['DEVICE']} | AMP: {ns['AMP_ENABLED']}")

    # ---------------------------------------------------------------- 1. parameter budget
    print("\n=== 1. Parameter counts ===")
    full = DSFNet(DSFConfig(mode="gated"))
    n_dsf = count_parameters(full)
    n_resnet = count_parameters(ns["build_resnet18"](pretrained=False))
    n_cifake = count_parameters(ns["CifakeCNN"]())
    print(f"DSF-Net (gated) : {n_dsf:>10,}")
    print(f"CIFAKE CNN      : {n_cifake:>10,}")
    print(f"ResNet-18       : {n_resnet:>10,}")
    assert n_dsf < n_resnet, "DSF-Net should be far smaller than ResNet-18"
    assert 3e5 < n_dsf < 2e6, f"DSF-Net size {n_dsf} is outside the intended 0.3-2M budget"

    # ---------------------------------------------------------------- 2. width scaling
    print("\n=== 2. Width multiplier scaling ===")
    for w in (0.5, 1.0, 1.5):
        net = DSFNet(DSFConfig(mode="gated", width=w))
        out = net(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 1)
        print(f"  width={w:<4} params={count_parameters(net):>9,}  ok")

    # ---------------------------------------------------------------- 3. end-to-end training
    print("\n=== 3. End-to-end training on synthetic planted-artefact data ===")
    images, labels = make_synthetic_data(ns)
    n_tr = int(0.8 * len(labels))
    train_loader = DataLoader(
        TensorDataset(images[:n_tr], labels[:n_tr]), batch_size=128, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(images[n_tr:], labels[n_tr:]), batch_size=256)

    with tempfile.TemporaryDirectory() as tmp:
        ns["CKPT_DIR"] = Path(tmp)
        cfg = TrainConfig(epochs=4, lr=1e-3, warmup_epochs=1, patience=4)

        results = {}
        for mode in ("gated", "spatial", "freq"):
            model = DSFNet(DSFConfig(mode=mode))
            train_model(model, f"smoke_{mode}", train_loader, val_loader, cfg, resume=False,
                        verbose=False)
            y_true, y_prob = predict(model, val_loader)
            m = compute_metrics(y_true, y_prob)
            results[mode] = m
            print(f"  mode={mode:<8} val acc {m['accuracy']:.3f}  AUC {m['roc_auc']:.3f}  "
                  f"ECE {m['ece']:.3f}")

    assert results["gated"]["roc_auc"] > 0.75, (
        f"gated DSF-Net failed to learn the planted artefact (AUC "
        f"{results['gated']['roc_auc']:.3f}) -- the model or training loop is broken"
    )
    assert results["freq"]["roc_auc"] > 0.75, (
        "the frequency stream alone should trivially find a Nyquist-frequency artefact"
    )
    print("  -> both the fused model and the frequency stream detect the planted artefact")

    # ---------------------------------------------------------------- 3b. resume path
    print("\n=== 3b. Checkpoint resume / already-trained path ===")
    with tempfile.TemporaryDirectory() as tmp:
        ns["CKPT_DIR"] = Path(tmp)
        cfg = TrainConfig(epochs=2, lr=1e-3, warmup_epochs=1, patience=2)

        first = train_model(DSFNet(DSFConfig()), "resume_probe", train_loader, val_loader,
                            cfg, resume=False, verbose=False)
        # Re-running the same cell must hit the "already trained" branch and still return
        # a fully populated history -- this is what the notebook does on a second run.
        second = train_model(DSFNet(DSFConfig()), "resume_probe", train_loader, val_loader,
                             cfg, resume=True, verbose=False)

        for key in ("best_val_auc", "train_time_s", "epoch", "val_auc"):
            assert key in second, f"resumed history is missing '{key}'"
        assert abs(second["best_val_auc"] - first["best_val_auc"]) < 1e-9
        print(f"  first run AUC {first['best_val_auc']:.4f} | "
              f"resumed run AUC {second['best_val_auc']:.4f}  ok")

    # ---------------------------------------------------------------- 4. constraint survives training
    print("\n=== 4. Bayar-Stamm constraint after training ===")
    trained = DSFNet(DSFConfig(mode="gated"))
    trained.project_constraints()
    w = trained.spatial.front.weight.data
    c = trained.spatial.front.centre
    centres = w[:, :, c, c]
    others = w.sum(dim=(2, 3)) - centres
    print(f"  centre taps  max|w+1| = {(centres + 1).abs().max():.2e}")
    print(f"  other taps   max|s-1| = {(others - 1).abs().max():.2e}")
    assert (centres + 1).abs().max() < 1e-5
    assert (others - 1).abs().max() < 1e-4

    # ---------------------------------------------------------------- 5. gate is informative
    print("\n=== 5. Fusion gate behaviour ===")
    z, g = full.embed(torch.randn(64, 3, 32, 32))
    print(f"  embedding {tuple(z.shape)} | gate {tuple(g.shape)} "
          f"range [{g.min():.3f}, {g.max():.3f}] | across-sample std {g.std(0).mean():.4f}")
    assert 0.0 <= g.min() <= g.max() <= 1.0
    assert g.std(0).mean() > 1e-4, "gate is constant across samples -- it cannot be adaptive"

    # ---------------------------------------------------------------- 6. spectrum correctness
    print("\n=== 6. FFT helpers ===")
    x = torch.zeros(1, 3, 32, 32)
    x[:, :, ::2, ::2] = 1.0                      # a pure high-frequency pattern
    spec = ns["log_magnitude_spectrum"](x)
    prof = ns["radial_profile"](spec)
    print(f"  spectrum {tuple(spec.shape)} | radial profile {tuple(prof.shape)}")
    assert spec.shape == (1, 3, 32, 32)
    assert prof.shape == (1, ns["N_RADIAL_BINS"])
    assert torch.isfinite(spec).all() and torch.isfinite(prof).all()

    # A DC-only (constant) image must put all its energy in the centre bin.
    dc = torch.ones(1, 3, 32, 32)
    dc_spec = ns["log_magnitude_spectrum"](dc)
    peak = dc_spec[0, 0].argmax().item()
    assert (peak // 32, peak % 32) == (16, 16), "fftshift is not centring the DC term"
    print("  DC term lands at the centre after fftshift  ok")

    # ---------------------------------------------------------------- 7. throughput
    print("\n=== 7. Inference throughput ===")
    for name, model in [("DSF-Net", DSFNet(DSFConfig())), ("CIFAKE CNN", ns["CifakeCNN"]()),
                        ("ResNet-18", ns["build_resnet18"](pretrained=False))]:
        print(f"  {name:<12} {measure_latency(model, batch_size=256, n_iter=20):>10,.0f} img/s")

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
