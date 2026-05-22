#!/usr/bin/env python
"""Compare symbolic Dimension IRs across models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.dimension_ir_compare import compare_dimension_irs, dimension_ir_comparison_to_markdown
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare symbolic Dimension IR reports.")
    parser.add_argument("--models", required=True, help="'all' or comma-separated configured model names/HF IDs.")
    return parser.parse_args()


def _selected_models(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        configs = _selected_models(args.models)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    irs = []
    missing = []
    for config in configs:
        safe_name = safe_model_name(config["hf_id"])
        path = root / "reports" / "dimension_ir" / f"{safe_name}.json"
        if path.exists():
            irs.append(_load_json(path))
        else:
            missing.append(config["name"])
    if missing:
        print(f"[missing] Dimension IRs missing for: {', '.join(missing)}. Run: python scripts/build_dimension_ir.py --model all", file=sys.stderr)
        return 1

    comparison = compare_dimension_irs(irs)
    write_json(comparison, root / "reports" / "dimension_ir" / "comparison.json")
    write_markdown(dimension_ir_comparison_to_markdown(comparison), root / "reports" / "dimension_ir" / "comparison.md")
    print(f"[dimension-ir-comparison] models: {comparison['num_models']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
