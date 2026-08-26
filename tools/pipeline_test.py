"""Execute the ENTIRE notebook end to end against a synthetic stand-in for CIFAKE.

`smoke_test.py` only runs the cells tagged `# @smoke` -- the model and training definitions.
This script runs *every* cell: EDA, the classical baselines, all three neural models, the
hyperparameter sweep, the test comparison, all seven ablations, the ten-condition robustness
sweep, Grad-CAM, t-SNE, calibration and the error analysis. It is the check that answers
"does Run All actually work", without downloading 120,000 images or spending GPU hours.

Two substitutions make that affordable:

* `datasets.load_dataset` is replaced by a generator producing CIFAKE-shaped data, where the
  FAKE half carries a planted Nyquist-frequency artefact standing in for a real generator
  fingerprint.
* `QUICK_RUN = False` is rewritten to `True` in the notebook source, which is the notebook's
  own switch for subsampled data and 2-epoch training runs.

Nothing else is patched, so a failure here is a genuine failure in the notebook.

Usage:
    python tools/pipeline_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from py2ipynb import parse_cells  # noqa: E402


# --------------------------------------------------------------------------- fake dataset
class _LabelFeature:
    def __init__(self, names):
        self.names = names


class FakeSplit:
    """Minimal stand-in for a Hugging Face image dataset split."""

    def __init__(self, images, labels, names):
        self._images = images
        self._labels = labels
        self.features = {"label": _LabelFeature(names)}

    def __len__(self):
        return len(self._labels)

    def __iter__(self):
        from PIL import Image

        for arr, lab in zip(self._images, self._labels):
            yield {"image": Image.fromarray(arr), "label": int(lab)}

    def __getitem__(self, i):
        from PIL import Image

        return {"image": Image.fromarray(self._images[i]), "label": int(self._labels[i])}

    def __repr__(self):
        return f"FakeSplit(n={len(self)})"


def _make_split(n: int, seed: int):
    """CIFAKE-shaped uint8 images; the FAKE half carries a faint checkerboard artefact."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(seed)
    low = torch.randn(n, 3, 8, 8, generator=g)
    imgs = F.interpolate(low, size=32, mode="bicubic", align_corners=False)
    imgs = (imgs - imgs.amin()) / (imgs.amax() - imgs.amin() + 1e-8)

    labels = torch.randint(0, 2, (n,), generator=g)          # 0 = FAKE index, see names below
    checker = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(16, 16)

    # A Nyquist checkerboard is trivially separable in the spectrum, so a plain planted
    # artefact gives 100% accuracy and the error-analysis cells never run. Marking only
    # 70% of the FAKE images makes the task genuinely unsolvable at ~85% accuracy, which
    # guarantees misclassifications and exercises those cells too.
    # Marking 10% of the REAL images too makes both error directions occur, so the
    # confident-mistakes grid (which needs both) is exercised as well.
    is_fake = labels == 0
    draw = torch.rand(n, generator=g)
    marked = (is_fake & (draw < 0.70)) | (~is_fake & (draw < 0.10))
    imgs[marked] = (imgs[marked] + 0.05 * checker).clamp(0, 1)

    arr = (imgs.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).byte().numpy()
    return arr, labels.numpy().astype(np.uint8)


def fake_load_dataset(dataset_id, *args, **kwargs):
    """Stand-in for datasets.load_dataset with CIFAKE's split sizes scaled down."""
    names = ["FAKE", "REAL"]
    tr_x, tr_y = _make_split(6500, seed=1)
    te_x, te_y = _make_split(2400, seed=2)
    return {
        "train": FakeSplit(tr_x, tr_y, names),
        "test": FakeSplit(te_x, te_y, names),
    }


# --------------------------------------------------------------------------- runner
def main() -> int:
    import matplotlib

    matplotlib.use("Agg")  # never try to open a window

    import datasets

    datasets.load_dataset = fake_load_dataset

    source = (ROOT / "notebooks" / "AIGID_main.py").read_text(encoding="utf-8")
    assert "QUICK_RUN = False" in source, "notebook no longer defines QUICK_RUN = False"
    source = source.replace("QUICK_RUN = False", "QUICK_RUN = True", 1)

    cells = [c for c in parse_cells(source) if c["cell_type"] == "code"]
    print(f"running {len(cells)} code cells against synthetic data\n")

    module = types.ModuleType("nb_pipeline")
    sys.modules["nb_pipeline"] = module
    ns = module.__dict__

    workdir = tempfile.mkdtemp(prefix="aigid_pipeline_")
    original_cwd = os.getcwd()
    os.chdir(workdir)
    print(f"working directory: {workdir}\n")

    failures = []
    try:
        for i, cell in enumerate(cells):
            code = "".join(cell["source"])
            preview = next((ln for ln in code.splitlines()
                            if ln.strip() and not ln.strip().startswith("#")), "(comments only)")
            label = f"cell {i:02d}: {preview[:68]}"
            try:
                exec(compile(code, f"<cell {i}>", "exec"), ns)
                print(f"  ok    {label}")
            except Exception:
                print(f"  FAIL  {label}")
                traceback.print_exc()
                failures.append((i, preview))
                break  # later cells depend on earlier ones; stop at the first real failure
    finally:
        os.chdir(original_cwd)

    print()
    if failures:
        print(f"PIPELINE TEST FAILED at cell {failures[0][0]}: {failures[0][1]}")
        return 1

    # Confirm the notebook actually produced its outputs rather than silently skipping them.
    results = Path(workdir) / "results"
    expected_files = ["metrics.csv", "ablations.csv", "robustness.csv", "tuning.csv", "digest.txt"]
    missing = [f for f in expected_files if not (results / f).exists()]
    figures = sorted((results / "figures").glob("*.png"))

    print(f"result files : {len(expected_files) - len(missing)}/{len(expected_files)} present")
    print(f"figures saved: {len(figures)}")
    for fig in figures:
        print(f"   {fig.name}")

    if missing:
        print(f"\nMISSING OUTPUTS: {missing}")
        return 1
    # 14 figures always; the two error-analysis figures only exist if the model made mistakes.
    if len(figures) < 14:
        print(f"\nExpected at least 14 figures, found {len(figures)}")
        return 1

    print("\n--- results/digest.txt ---")
    print((results / "digest.txt").read_text(encoding="utf-8"))
    print("\nPIPELINE TEST PASSED: every cell ran and every output was produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
