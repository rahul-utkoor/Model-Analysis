#!/usr/bin/env python
"""Generate the final static pruning propagation research report bundle."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experimental.final_report.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
