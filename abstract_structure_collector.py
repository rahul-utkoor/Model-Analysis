#!/usr/bin/env python3
"""Compatibility wrapper for tools/abstract_structure_collector.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "tools" / "abstract_structure_collector.py"), run_name="__main__")
