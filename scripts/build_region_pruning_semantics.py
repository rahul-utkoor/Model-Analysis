#!/usr/bin/env python
"""Build static pruning propagation semantics over Structural Region Tree reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.region_pruning_semantics import (
    build_region_pruning_semantics,
    region_pruning_semantics_to_markdown,
    write_region_pruning_semantics_json,
)
from model_analysis.region_pruning_semantics_text import write_region_pruning_semantics_text
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build region-level pruning propagation semantics.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_models(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def _maybe(path: Path) -> dict | None:
    return _load_json(path) if path.exists() else None


def build_one(root: Path, config: dict, verbose: bool) -> bool:
    safe = safe_model_name(config["hf_id"])
    tree_path = root / "reports" / "structural_region_trees" / f"{safe}.json"
    tensor_path = root / "reports" / "tensor_ir" / f"{safe}.json"
    rdim_path = root / "reports" / "region_dimension_ir" / f"{safe}.json"
    expansion_path = root / "reports" / "abstract_node_expansions" / safe / "abstract_node_expansions_main.json"

    if not tree_path.exists():
        print(f"[missing] Structural Region Tree missing. Run: python scripts/build_structural_region_tree.py --model {config['name']}", file=sys.stderr)
        return False
    if not tensor_path.exists():
        print(f"[missing] Tensor IR missing. Run: python scripts/build_tensor_ir.py --model {config['name']}", file=sys.stderr)
        return False

    semantics = build_region_pruning_semantics(
        _load_json(tree_path),
        _load_json(tensor_path),
        region_dimension_ir=_maybe(rdim_path),
        abstract_expansion_report=_maybe(expansion_path),
        source_region_tree_path=str(tree_path),
        source_region_dimension_ir_path=str(rdim_path) if rdim_path.exists() else None,
    )

    json_path = root / "reports" / "region_pruning_semantics" / f"{safe}.json"
    text_path = root / "reports" / "region_pruning_semantics_dumps" / f"{safe}.rpsem"
    md_path = root / "reports" / "region_pruning_semantics_explanations" / f"{safe}.md"
    write_region_pruning_semantics_json(semantics, json_path)
    write_region_pruning_semantics_text(semantics, text_path)
    write_markdown(region_pruning_semantics_to_markdown(semantics), md_path)

    if verbose:
        summary = semantics.summary
        print(f"[region-pruning-semantics] {semantics.model_name}")
        print(f"  regions: {summary.get('num_regions', 0)}")
        print(f"  directly_prunable: {summary.get('directly_prunable_regions', 0)}")
        print(f"  mlp_opportunities: {summary.get('mlp_pruning_opportunities', 0)}")
        print(f"  attention_blocked: {summary.get('attention_blocked_regions', 0)}")
        print(f"  residual_blocked: {summary.get('residual_blocked_regions', 0)}")
        print(f"  json: {json_path}")
        print(f"  text: {text_path}")
        print(f"  markdown: {md_path}")
    return True


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        configs = _selected_models(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    ok = True
    for config in configs:
        ok = build_one(root, config, args.verbose) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
