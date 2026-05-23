#!/usr/bin/env python
"""Build compiler-inspired Structural Region Trees from Tensor IR reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown
from model_analysis.structural_region_detection import build_structural_region_tree, load_tensor_graph_dict
from model_analysis.structural_region_tree import structural_region_tree_to_dict, structural_region_tree_to_markdown, write_structural_region_tree_json
from model_analysis.structural_region_tree_text import write_structural_region_tree_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Structural Region Tree over Tensor IR.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--format", choices=["json", "md", "text", "all"], default="all")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable-semantic-fusion", action="store_true", help="Do not recover decomposed activation/feed-forward regions.")
    return parser.parse_args()


def _selected_models(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def _table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def _interfaces_markdown(model_name: str, interfaces: list[dict]) -> str:
    return "\n".join(
        [
            f"# Structural Region Interfaces: {model_name}",
            "",
            _table(interfaces, ["region_id", "region_type", "pruning_role", "prunable_dimensions", "protected_dimensions", "blocked_dimensions", "constraints"]),
            "",
            "Interfaces are preliminary symbolic propagation descriptions over Tensor IR regions; they are not pruning transformations.",
            "",
        ]
    )


def _patterns_markdown(model_name: str, patterns: dict[str, int]) -> str:
    rows = [{"region_type": key, "count": value} for key, value in sorted(patterns.items())]
    return "\n".join([f"# Structural Region Patterns: {model_name}", "", _table(rows, ["region_type", "count"]), ""])


def _write_outputs(root: Path, safe_name: str, tree, output_format: str) -> None:
    data = structural_region_tree_to_dict(tree)
    interfaces = {"model_name": tree.model_name, "interfaces": data["interfaces"], "summary": data["summary"]}
    patterns = {"model_name": tree.model_name, "region_type_counts": data["summary"].get("region_type_counts", {}), "summary": data["summary"]}
    if output_format in {"json", "all"}:
        write_structural_region_tree_json(tree, root / "reports" / "structural_region_trees" / f"{safe_name}.json")
        write_json(interfaces, root / "reports" / "structural_region_interfaces" / f"{safe_name}.json")
        write_json(patterns, root / "reports" / "structural_region_patterns" / f"{safe_name}.json")
    if output_format in {"md", "all"}:
        write_markdown(structural_region_tree_to_markdown(tree), root / "reports" / "structural_region_trees" / f"{safe_name}.md")
        write_markdown(_interfaces_markdown(tree.model_name, data["interfaces"]), root / "reports" / "structural_region_interfaces" / f"{safe_name}.md")
        write_markdown(_patterns_markdown(tree.model_name, patterns["region_type_counts"]), root / "reports" / "structural_region_patterns" / f"{safe_name}.md")
    if output_format in {"text", "all"}:
        write_structural_region_tree_text(tree, root / "reports" / "structural_region_dumps" / f"{safe_name}.srtree")


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        configs = _selected_models(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    failed = False
    for config in configs:
        safe_name = safe_model_name(config["hf_id"])
        source_path = root / "reports" / "tensor_ir" / f"{safe_name}.json"
        if not source_path.exists():
            print(f"[missing] Tensor IR missing. Run: python scripts/build_tensor_ir.py --model {config['name']}", file=sys.stderr)
            failed = True
            continue
        tree = build_structural_region_tree(
            load_tensor_graph_dict(source_path),
            enable_semantic_fusion=not args.disable_semantic_fusion,
        )
        _write_outputs(root, safe_name, tree, args.format)
        if args.verbose:
            summary = tree.summary
            print(f"[structural-region-tree] {tree.model_name}")
            print(f"  regions: {summary['num_regions']}")
            print(f"  primitive leaves: {summary['num_primitive_regions']}")
            print(f"  gelu_fusions: {summary['num_gelu_fusions']}")
            print(f"  fused_feedforward: {summary['num_feedforward_fusions']}")
            print(f"  feedforward: {summary['num_feedforward_regions']}")
            print(f"  attention_skeleton: {summary['num_attention_skeleton_regions']}")
            print(f"  residual_merges: {summary['num_residual_merge_regions']}")
            print(f"  blocked: {summary['num_blocked_regions']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
