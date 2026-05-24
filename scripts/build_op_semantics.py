#!/usr/bin/env python
"""Build pruning-relevant primitive op semantics over Tensor IR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.op_semantics import build_op_semantics_ir, op_semantics_ir_to_markdown, write_op_semantics_json
from model_analysis.op_semantics_text import write_op_semantics_text
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pruning-relevant Op Semantics IR.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe(path: Path) -> dict | None:
    return _load_json(path) if path.exists() else None


def _configs(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def build_one(root: Path, config: dict, verbose: bool) -> bool:
    safe = safe_model_name(config["hf_id"])
    tensor_path = root / "reports" / "tensor_ir" / f"{safe}.json"
    tree_path = root / "reports" / "structural_region_trees" / f"{safe}.json"
    region_semantics_path = root / "reports" / "region_pruning_semantics" / f"{safe}.json"
    expansion_path = root / "reports" / "abstract_node_expansions" / safe / "abstract_node_expansions_main.json"
    if not tensor_path.exists():
        print(f"[missing] Tensor IR missing. Run: python scripts/build_tensor_ir.py --model {config['name']}", file=sys.stderr)
        return False

    semantics = build_op_semantics_ir(
        _load_json(tensor_path),
        structural_region_tree=_maybe(tree_path),
        region_pruning_semantics=_maybe(region_semantics_path),
        abstract_expansion_report=_maybe(expansion_path),
        source_tensor_ir_path=str(tensor_path),
        source_region_tree_path=str(tree_path) if tree_path.exists() else None,
        source_region_pruning_semantics_path=str(region_semantics_path) if region_semantics_path.exists() else None,
    )
    json_path = root / "reports" / "op_semantics" / f"{safe}.json"
    text_path = root / "reports" / "op_semantics_dumps" / f"{safe}.opsem"
    md_path = root / "reports" / "op_semantics_explanations" / f"{safe}.md"
    write_op_semantics_json(semantics, json_path)
    write_op_semantics_text(semantics, text_path)
    write_markdown(op_semantics_ir_to_markdown(semantics), md_path)

    if verbose:
        summary = semantics.summary
        print(f"[op-semantics] {semantics.model_name}")
        print(f"  ops: {summary.get('num_ops', 0)}")
        print(f"  parameterized_ops: {summary.get('parameterized_ops', 0)}")
        print(f"  unknown_ops: {summary.get('unknown_ops', 0)}")
        print(f"  json: {json_path}")
        print(f"  text: {text_path}")
        print(f"  markdown: {md_path}")
    return True


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        configs = _configs(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    ok = True
    for config in configs:
        ok = build_one(root, config, args.verbose) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

