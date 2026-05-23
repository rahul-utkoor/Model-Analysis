#!/usr/bin/env python
"""Compare Tensor Graph IR summaries imported for multiple models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown
from model_analysis.tensor_ir_compare import compare_tensor_graphs, tensor_ir_comparison_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare frontend-independent Tensor IR reports.")
    parser.add_argument("--models", required=True, help="'all' or comma-separated model names/HF IDs.")
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
    graphs = []
    missing = []
    for config in configs:
        safe_name = safe_model_name(config["hf_id"])
        path = root / "reports" / "tensor_ir" / f"{safe_name}.json"
        if path.exists():
            graphs.append(_load_json(path))
        else:
            missing.append(config["name"])
    if missing:
        print(
            f"[missing] Tensor IR reports missing for: {', '.join(missing)}. Run: python scripts/build_tensor_ir.py --model all",
            file=sys.stderr,
        )
        return 1
    comparison = compare_tensor_graphs(graphs)
    write_json(comparison, root / "reports" / "tensor_ir" / "comparison.json")
    write_markdown(
        tensor_ir_comparison_to_markdown(comparison),
        root / "reports" / "tensor_ir" / "comparison.md",
    )
    print(f"[tensor-ir-comparison] models: {comparison['num_models']}")
    print(f"  operations: {comparison['summary']['total_ops']}")
    print(f"  forks: {comparison['summary']['total_fork_ops']}")
    print(f"  joins: {comparison['summary']['total_join_ops']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
