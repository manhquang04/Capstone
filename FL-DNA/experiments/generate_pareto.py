"""Compatibility wrapper for the root-level generate_pareto.py script."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "generate_pareto.py"), run_name="__main__")
