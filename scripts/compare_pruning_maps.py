#!/usr/bin/env python
"""Compare compiler-style pruning maps across models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.pruning_map_compare import compare_model_pruning_maps, pruning_map_comparison_to_markdown
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model pruning maps.")
    parser.add_argument("--models", required=True, help="'all' or comma-separated configured model names/HF IDs.")
    return parser.parse_args()


def _selected_models(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(item.strip()) for item in value.split(",") if item.strip()]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        configs = _selected_models(args.models)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    maps = []
    missing = []
    for config in configs:
        safe_name = safe_model_name(config["hf_id"])
        path = root / "reports" / "model_pruning_maps" / f"{safe_name}.json"
        if path.exists():
            maps.append(_load_json(path))
        else:
            missing.append(config["name"])
    if missing:
        print(f"[missing] Pruning maps missing for: {', '.join(missing)}. Run: python scripts/build_pruning_map.py --model all", file=sys.stderr)
        return 1

    comparison = compare_model_pruning_maps(maps)
    write_json(comparison, root / "reports" / "model_pruning_maps" / "comparison.json")
    write_markdown(pruning_map_comparison_to_markdown(comparison), root / "reports" / "model_pruning_maps" / "comparison.md")
    print(f"[comparison] models: {comparison['num_models']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
