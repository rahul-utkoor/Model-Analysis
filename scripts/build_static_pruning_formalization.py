#!/usr/bin/env python
"""Generate static pruning propagation formalization reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experimental.formalization.build_formalization import build_formalization


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/formalization")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    written = build_formalization(args.output_dir)
    print(f"[formalization] generated={len(written)} output_dir={args.output_dir}")
    if args.verbose:
        for path in written:
            print(f"[formalization] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
