"""Execute AIGID_main.ipynb in place, with all outputs and figures embedded.

The `jupyter nbconvert` CLI entry point is not registered in this environment, so this
uses the nbconvert Python API directly. Every executed cell is logged with a running
elapsed time, so a long run can be followed with `tail -f`.

The notebook is written back to disk whether the run succeeds or fails, so a failure
leaves a notebook containing the traceback rather than nothing.

Usage:
    python tools/run_notebook.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "AIGID_main.ipynb"

START = time.time()


def _elapsed() -> str:
    seconds = int(time.time() - START)
    return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


class LoggingExecutor(ExecutePreprocessor):
    """ExecutePreprocessor that announces each cell before running it."""

    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type == "code":
            first = next(
                (ln for ln in cell.source.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")),
                "(comments only)",
            )
            print(f"[{_elapsed()}] cell {index:3d} | {first[:78]}", flush=True)
        return super().preprocess_cell(cell, resources, index)


def main() -> int:
    print(f"executing {NOTEBOOK}")
    print(f"working directory for the kernel: {NOTEBOOK.parent}\n", flush=True)

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    executor = LoggingExecutor(timeout=-1, kernel_name="python3", allow_errors=False)

    status = 0
    try:
        executor.preprocess(notebook, {"metadata": {"path": str(NOTEBOOK.parent)}})
        print(f"\n[{_elapsed()}] ALL CELLS EXECUTED SUCCESSFULLY", flush=True)
    except CellExecutionError as exc:
        print(f"\n[{_elapsed()}] CELL FAILED\n{exc}", flush=True)
        status = 1
    finally:
        nbformat.write(notebook, NOTEBOOK)
        print(f"[{_elapsed()}] notebook written back to {NOTEBOOK}", flush=True)

    results = ROOT / "results"
    figures = sorted((results / "figures").glob("*.png"))
    print(f"\nfigures produced: {len(figures)}")
    for fig in figures:
        print(f"   {fig.name}")

    digest = results / "digest.txt"
    if digest.exists():
        print("\n--- results/digest.txt ---")
        print(digest.read_text(encoding="utf-8"))

    return status


if __name__ == "__main__":
    raise SystemExit(main())
