"""What each detector actually does to a photograph, recorded rather than described.

This study argues that four detectors reading four different things is why the corpus and
generator terms come out differently for each of them. Every word of that argument is currently
carried by aggregate accuracies. A reader has no way to see an image enter a detector and a
number come out, and no way to see that the four routes through an image are genuinely different
rather than four labels on the same convolutional stack.

This records the route. For a small showcase set it caches every intermediate that makes each
detector's process visible on real pixels:

  CIFAKE-CNN   first-layer responses, the only thing between raw pixels and the classifier
  DSF-Net      the constrained-convolution residual, the log-magnitude spectrum, the 16-bin
               radial profile and the fusion gate, one per stream and then the gate that
               weighs them
  ResNet-18    a Grad-CAM attribution over the final block, showing where in the frame the
               fine-tuned backbone finds its evidence
  CLIP probe   the frozen 512-d embedding and the probe's signed per-dimension contribution

and, for every image of every evaluation set, all four predicted probabilities, so score
distributions can be drawn per detector per set rather than collapsed to an accuracy.

Seed 42 throughout. These figures show a process, not a measured claim; every measured claim in
this study uses five seeds and lives in crossgen_seeds.csv.

Usage:
    python tools/per_image_predictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from seed_sweep import BEST_DROPOUT, BEST_WIDTH, load_notebook_namespace  # noqa: E402

SETS = ROOT / "data" / "crossgen_sets_32.npz"
CLIP_FEATS = ROOT / "data" / "clip_features.npz"
OUT = ROOT / "data" / "per_image_predictions.npz"
SEED = 42
KEYS = ["A_real", "A_fake", "imagenet_real", "gen_SD15", "gen_ADM"]
N_SHOWCASE = 3


def grad_cam(torch, model, feature_fn, x):
    """Attribution over a chosen feature map: activations weighted by their gradients."""
    store = {}

    def keep(_m, _i, out):
        store["a"] = out
        out.retain_grad()

    handle = feature_fn(model).register_forward_hook(keep)
    model.zero_grad(set_to_none=True)
    model(x).squeeze(-1).sum().backward()
    handle.remove()

    a = store["a"]
    w = a.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * a).sum(dim=1))
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    return (cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)).detach().cpu().numpy()


def main() -> None:
    # The namespace has to be built before torch is imported here. Loading torch first and
    # pandas second corrupts the heap on this machine (pyarrow and torch disagree about a
    # shared runtime), and the notebook imports pandas in its second cell.
    sets = np.load(SETS)
    ns = load_notebook_namespace(quick=False)
    torch = ns["torch"]
    device = ns["DEVICE"]
    logmag = ns["log_magnitude_spectrum"]
    radial = ns["radial_profile"]
    mean = torch.tensor(ns["CHANNEL_MEAN"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(ns["CHANNEL_STD"], device=device).view(1, 3, 1, 1)

    def to_batch(imgs):
        x = torch.from_numpy(imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        return (x - mean) / std

    models = {}
    d = ns["DSFNet"](ns["DSFConfig"](mode="gated", dropout=BEST_DROPOUT, width=BEST_WIDTH))
    d.load_state_dict(torch.load(ROOT / "checkpoints" / "seeds" / f"seed{SEED}_abl_4_best.pt",
                                 map_location=device, weights_only=False)["model"])
    models["DSF-Net"] = d.to(device).eval()

    c = ns["CifakeCNN"]()
    c.load_state_dict(torch.load(ROOT / "checkpoints" / "arch_seeds" /
                                 f"seed{SEED}_cifakecnn_best.pt",
                                 map_location=device, weights_only=False)["model"])
    models["CIFAKE-CNN"] = c.to(device).eval()

    r = ns["build_resnet18"](pretrained=False)
    r.load_state_dict(torch.load(ROOT / "checkpoints" / "arch_seeds" /
                                 f"seed{SEED}_resnet18_best.pt",
                                 map_location=device, weights_only=False)["model"])
    models["ResNet-18"] = r.to(device).eval()

    out = {}

    # Every image of every set, all three convolutional detectors.
    for key in KEYS:
        imgs = sets[key]
        for name, model in models.items():
            probs, gates = [], []
            with torch.no_grad():
                for i in range(0, len(imgs), 256):
                    x = to_batch(imgs[i:i + 256])
                    if name == "DSF-Net":
                        z, g = model.embed(x)
                        gates.append(g.mean(dim=1).float().cpu().numpy())
                        logits = model.head(z)
                    else:
                        logits = model(x)
                    probs.append(torch.sigmoid(logits.float().squeeze(-1)).cpu().numpy())
            out[f"p_{name}_{key}"] = np.concatenate(probs)
            if gates:
                out[f"gatemean_{key}"] = np.concatenate(gates)
        print(f"  {key:<16} {len(imgs):>5} images, 3 convolutional detectors", flush=True)

    # The showcase: a handful of images, every intermediate each detector computes.
    rng = np.random.default_rng(7)
    show_idx = {k: rng.choice(len(sets[k]), N_SHOWCASE, replace=False) for k in KEYS}
    for key in KEYS:
        imgs = sets[key][show_idx[key]]
        out[f"show_images_{key}"] = imgs
        out[f"show_idx_{key}"] = show_idx[key]
        x = to_batch(imgs)

        with torch.no_grad():
            raw = torch.from_numpy(imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
            spec = logmag(raw)
            out[f"show_spectrum_{key}"] = spec.mean(dim=1).float().cpu().numpy()
            out[f"show_radial_{key}"] = radial(spec).float().cpu().numpy()

            dsf = models["DSF-Net"]
            out[f"show_bayar_{key}"] = dsf.spatial.front(x).float().cpu().numpy()
            z, g = dsf.embed(x)
            out[f"show_gate_{key}"] = g.float().cpu().numpy()
            out[f"show_p_DSF-Net_{key}"] = torch.sigmoid(
                dsf.head(z).float().squeeze(-1)).cpu().numpy()

            cif = models["CIFAKE-CNN"]
            out[f"show_conv1_{key}"] = torch.relu(
                cif.features[0](x)).float().cpu().numpy()
            out[f"show_p_CIFAKE-CNN_{key}"] = torch.sigmoid(
                cif(x).float().squeeze(-1)).cpu().numpy()

            out[f"show_p_ResNet-18_{key}"] = torch.sigmoid(
                models["ResNet-18"](x).float().squeeze(-1)).cpu().numpy()

        out[f"show_cam_resnet_{key}"] = grad_cam(torch, models["ResNet-18"],
                                                 lambda m: m.layer4, x.clone())
        out[f"show_cam_dsf_{key}"] = grad_cam(torch, models["DSF-Net"],
                                              lambda m: m.spatial.block3, x.clone())
        print(f"  {key:<16} showcase intermediates for {N_SHOWCASE} images", flush=True)

    # The CLIP probe, which sees cached features rather than pixels.
    feats = np.load(CLIP_FEATS)
    torch.manual_seed(SEED)
    X = torch.from_numpy(feats["train_X"]).float()
    y = torch.from_numpy(feats["train_y"]).float().to(device)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    Xn = ((X - mu) / sd).to(device)
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

    w = probe.weight.detach().cpu().numpy()
    for key in KEYS:
        fn = ((torch.from_numpy(feats[key]).float() - mu) / sd).to(device)
        with torch.no_grad():
            out[f"p_CLIP probe_{key}"] = torch.sigmoid(probe(fn).squeeze(-1)).cpu().numpy()
        out[f"clipfeat_{key}"] = feats[key]
        # The probe is linear, so a decision splits exactly into 512 per-dimension
        # contributions. Keep them for the showcase; they are the whole of what it computes.
        out[f"show_clipcontrib_{key}"] = fn[show_idx[key]].cpu().numpy() * w
        out[f"show_p_CLIP probe_{key}"] = out[f"p_CLIP probe_{key}"][show_idx[key]]
    out["clip_bias"] = probe.bias.detach().cpu().numpy()
    print("  CLIP probe scored on all sets, showcase contributions kept")

    np.savez_compressed(OUT, **out)
    print(f"  written {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
