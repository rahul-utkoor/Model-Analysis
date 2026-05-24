#!/usr/bin/env python3
"""Compatibility wrapper for tools/region_structure_api_server.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "tools" / "region_structure_api_server.py"), run_name="__main__")
