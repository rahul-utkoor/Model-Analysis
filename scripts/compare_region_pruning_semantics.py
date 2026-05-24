#!/usr/bin/env python
"""Compare Region Pruning Semantics reports across models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.region_pruning_semantics_compare import compare_region_pruning_semantics, comparison_to_markdown
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare region pruning semantics reports.")
    parser.add_argument("--models", required=True, help="'all' or comma-separated configured model names.")
    return parser.parse_args()


def _configs(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    root = get_project_root()
    reports = []
    for config in _configs(args.models):
        safe = safe_model_name(config["hf_id"])
        path = root / "reports" / "region_pruning_semantics" / f"{safe}.json"
        if not path.exists():
            print(f"[missing] {path}", file=sys.stderr)
            continue
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:
        print("[error] no region pruning semantics reports found", file=sys.stderr)
        return 1
    comparison = compare_region_pruning_semantics(reports)
    out_json = root / "reports" / "region_pruning_semantics_compare" / "summary.json"
    out_md = root / "reports" / "region_pruning_semantics_compare" / "summary.md"
    write_json(comparison, out_json)
    write_markdown(comparison_to_markdown(comparison), out_md)
    print(f"[region-pruning-semantics-compare] models={comparison['num_models']}")
    print(f"[region-pruning-semantics-compare] json={out_json}")
    print(f"[region-pruning-semantics-compare] markdown={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
