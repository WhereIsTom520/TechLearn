#!/usr/bin/env python3
"""Execute the canonical fail-closed launcher.

This file exists only as a stable entry point for start.sh. The canonical
launcher source is valid Python and is compiled directly by CI and the package
builder; no runtime source rewriting is performed.
"""
from __future__ import annotations

import runpy
from pathlib import Path

SOURCE = Path(__file__).with_name("easy_launcher.py")


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec")
    runpy.run_path(str(SOURCE), run_name="__main__")


if __name__ == "__main__":
    main()
