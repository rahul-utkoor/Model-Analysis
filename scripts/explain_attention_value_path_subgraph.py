#!/usr/bin/env python
"""Explain selected attention value-path artifact records."""

from __future__ import annotations

import argparse
import json
import sys

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--contains")
    parser.add_argument("--status", choices=["seedable", "partial", "blocked", "unknown"])
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    root = get_project_root()
    try:
        model = get_model_config(args.model)["hf_id"]
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    path = root / "reports/attention_value_path_subgraphs" / safe_model_name(model) / "summary.json"
    if not path.exists():
        print(f"[missing] Run: python scripts/build_attention_value_path_subgraphs.py --model {args.model} --layers all --export-onnx", file=sys.stderr)
        return 1
    rows = json.loads(path.read_text(encoding="utf-8")).get("paths", [])
    needle = str(args.contains or "").lower()
    rows = [
        row for row in rows
        if (args.layer is None or row.get("layer_index") == args.layer)
        and (not args.status or row.get("analysis_status") == args.status)
        and (not needle or needle in json.dumps(row, sort_keys=True).lower())
    ][: args.limit]
    print(f"# Attention Value-Path Selection: {model}")
    print(f"matches: {len(rows)}")
    for row in rows:
        print(f"- layer {row.get('layer_index')} {row.get('path_name')} [{row.get('analysis_status')}]")
        print(f"  mapping={row.get('axis_mapping', {}).get('mapping_status')}")
        print(f"  onnx={row.get('artifact_paths', {}).get('onnx', '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
