#!/usr/bin/env python
"""Compare layer subgraph validation packs."""

from __future__ import annotations

import argparse
import json
import sys

from model_analysis.layer_subgraph_validation_compare import compare_layer_subgraph_validation_packs, comparison_to_markdown
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare layer subgraph validation packs.")
    parser.add_argument("--models", required=True, help="Comma-separated model names.")
    parser.add_argument("--layer", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = get_project_root()
    packs = []
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        try:
            config = get_model_config(name)
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            continue
        safe = safe_model_name(config["hf_id"])
        path = root / "reports" / "layer_subgraph_validation" / safe / f"layer_{args.layer}" / "index.json"
        if not path.exists():
            print(f"[missing] {path}", file=sys.stderr)
            continue
        packs.append(json.loads(path.read_text(encoding="utf-8")))
    if not packs:
        print("[error] no layer subgraph validation packs found", file=sys.stderr)
        return 1
    comparison = compare_layer_subgraph_validation_packs(packs)
    out_json = root / "reports" / "layer_subgraph_validation_compare" / f"layer_{args.layer}_summary.json"
    out_md = root / "reports" / "layer_subgraph_validation_compare" / f"layer_{args.layer}_summary.md"
    write_json(comparison, out_json)
    write_markdown(comparison_to_markdown(comparison), out_md)
    print(f"[layer-subgraph-compare] packs={comparison['num_packs']}")
    print(f"[layer-subgraph-compare] json={out_json}")
    print(f"[layer-subgraph-compare] markdown={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
