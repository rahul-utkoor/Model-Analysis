#!/usr/bin/env python
"""Explain selected static deadbranch propagation records."""

from __future__ import annotations

import argparse
import json
import sys

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--contains")
    parser.add_argument("--blocked-only", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def _matches(row: dict, needle: str) -> bool:
    return not needle or needle in json.dumps(row, sort_keys=True).lower()


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        model = get_model_config(args.model)["hf_id"]
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    path = root / "reports" / "deadbranch_propagation" / f"{safe_model_name(model)}.json"
    if not path.exists():
        print(f"[missing] Run: python scripts/analyze_deadbranch_propagation.py --model {args.model}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("blocked_pairs", []) if args.blocked_only else data.get("pairs", [])
    needle = str(args.contains or "").lower()
    rows = [row for row in rows if _matches(row, needle)][: args.limit]
    print(f"# Deadbranch Propagation Selection: {data.get('model_name')}")
    print(f"matches: {len(rows)}")
    for row in rows:
        if args.blocked_only:
            print(f"- layer {row.get('layer_index')} {row.get('pair_kind')} [{row.get('status')}] blocker={row.get('blocker_type')}")
        else:
            print(f"- layer {row.get('layer_index')} {row.get('pair_kind')} [{row.get('status')}]")
            print(f"  {row.get('producer_op_name')} -> {row.get('consumer_op_name')}")
            print(f"  mapping={row.get('required_mapping')}:{row.get('mapping_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
