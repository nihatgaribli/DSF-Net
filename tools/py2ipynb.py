"""Convert a jupytext-style "percent format" .py file into a .ipynb notebook.

The notebook for this project is authored as a plain Python file using the
standard percent-format cell markers:

    # %% [markdown]
    # Markdown text goes here, one `# ` prefixed line per line.

    # %%
    print("code cell")

Keeping the master copy as .py means the notebook is diff-able, easy to edit,
and impossible to corrupt by hand-editing JSON. Run this script to regenerate
the .ipynb whenever the .py changes.

Usage:
    python tools/py2ipynb.py notebooks/AIGID_main.py notebooks/AIGID_main.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CELL_MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"


def _strip_markdown(lines: list[str]) -> list[str]:
    """Turn `# some text` comment lines back into raw markdown lines."""
    out = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return out


def _trim_blank_edges(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def parse_cells(text: str) -> list[dict]:
    """Split percent-format source into a list of notebook cell dicts."""
    cells: list[dict] = []
    cell_type = "code"
    buffer: list[str] = []

    def flush() -> None:
        body = _trim_blank_edges(list(buffer))
        if not body:
            return
        if cell_type == "markdown":
            body = _strip_markdown(body)
        # nbformat stores source as a list of lines that each keep their newline,
        # except the final line which has none.
        source = [ln + "\n" for ln in body[:-1]] + [body[-1]]
        # nbformat >= 4.5 requires a unique cell id. Derive it from the cell index so the
        # ids stay stable across rebuilds and the .ipynb diffs cleanly.
        cell = {
            "cell_type": cell_type,
            "id": f"cell-{len(cells):03d}",
            "metadata": {},
            "source": source,
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith(CELL_MARKER):
            flush()
            buffer = []
            cell_type = "markdown" if line.startswith(MARKDOWN_MARKER) else "code"
        else:
            buffer.append(line)
    flush()
    return cells


def build_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    cells = parse_cells(src.read_text(encoding="utf-8"))
    dst.write_text(
        json.dumps(build_notebook(cells), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    n_md = sum(c["cell_type"] == "markdown" for c in cells)
    print(f"{src} -> {dst}: {len(cells)} cells ({n_md} markdown, {len(cells) - n_md} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
