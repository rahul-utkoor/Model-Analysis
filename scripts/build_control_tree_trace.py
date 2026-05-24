#!/usr/bin/env python
"""Build stepwise dataflow control-tree construction traces from Tensor IR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.control_tree_trace import (
    build_control_tree_trace,
    control_tree_trace_to_dict,
    control_tree_trace_to_markdown,
    write_control_tree_trace_json,
)
from model_analysis.control_tree_trace_text import write_control_tree_trace_text
from model_analysis.control_tree_trace_viz import write_control_tree_step_dot_files
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stepwise Structural Region Tree construction trace.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--format", choices=["json", "md", "text", "dot", "all"], default="all")
    parser.add_argument("--max-snapshot-nodes", type=int, default=500)
    parser.add_argument("--max-dot-steps", type=int, default=50)
    parser.add_argument("--render-svg", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _selected_models(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_markdown(model_name: str, trace_data: dict) -> str:
    summary = trace_data.get("summary", {})
    rows = [
        {"metric": key, "value": value}
        for key, value in sorted(summary.items())
        if not isinstance(value, dict)
    ]
    lines = [
        f"# Control Tree Step Summary: {model_name}",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['value']} |")
    lines.extend(
        [
            "",
            "## Created Region Types",
            "",
            "| region_type | count |",
            "| --- | --- |",
        ]
    )
    for key, value in sorted(summary.get("created_region_type_counts", {}).items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "This is an explanatory construction summary over Tensor IR; it does not modify models.", ""])
    return "\n".join(lines)


def _write_outputs(root: Path, safe_name: str, trace, output_format: str, max_dot_steps: int, render_svg: bool) -> None:
    trace_data = control_tree_trace_to_dict(trace)
    if output_format in {"json", "all"}:
        write_control_tree_trace_json(trace, root / "reports" / "control_tree_steps" / f"{safe_name}.json")
        write_json(
            {"model_name": trace.model_name, "summary": trace_data.get("summary", {})},
            root / "reports" / "control_tree_step_summaries" / f"{safe_name}.json",
        )
    if output_format in {"md", "all"}:
        write_markdown(control_tree_trace_to_markdown(trace), root / "reports" / "control_tree_steps" / f"{safe_name}.md")
        write_markdown(
            _summary_markdown(trace.model_name, trace_data),
            root / "reports" / "control_tree_step_summaries" / f"{safe_name}.md",
        )
    if output_format in {"text", "all"}:
        write_control_tree_trace_text(trace, root / "reports" / "control_tree_step_dumps" / f"{safe_name}.ctrace")
    if output_format in {"dot", "all"}:
        write_control_tree_step_dot_files(
            trace_data,
            root / "reports" / "control_tree_step_graphs" / safe_name,
            max_steps=max_dot_steps,
            render_svg=render_svg,
        )


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
        tensor_path = root / "reports" / "tensor_ir" / f"{safe_name}.json"
        if not tensor_path.exists():
            print(f"[missing] Tensor IR missing. Run: python scripts/build_tensor_ir.py --model {config['name']}", file=sys.stderr)
            failed = True
            continue
        tensor_graph = json.loads(tensor_path.read_text(encoding="utf-8"))
        tree_path = root / "reports" / "structural_region_trees" / f"{safe_name}.json"
        fusion_path = root / "reports" / "semantic_fusion" / f"{safe_name}.json"
        structural_tree = _load_json(tree_path)
        semantic_fusion = _load_json(fusion_path)
        if args.verbose and structural_tree is None:
            print(f"[warn] Structural Region Tree not found for {config['name']}; using detector candidates.")
        if args.verbose and semantic_fusion is None:
            print(f"[warn] Semantic fusion report not found for {config['name']}; continuing without fusion report.")
        trace = build_control_tree_trace(
            tensor_graph,
            structural_region_tree=structural_tree,
            semantic_fusion_report=semantic_fusion,
            max_snapshot_nodes=args.max_snapshot_nodes,
        )
        if structural_tree is not None:
            trace.final_region_tree_path = str(tree_path)
        _write_outputs(root, safe_name, trace, args.format, args.max_dot_steps, args.render_svg)
        if args.verbose:
            summary = trace.summary
            print(f"[control-tree-trace] {trace.model_name}")
            print(f"  steps: {summary.get('num_steps', 0)}")
            print(f"  collapses: {summary.get('num_collapse_steps', 0)}")
            print(f"  skips: {summary.get('num_skip_steps', 0)}")
            print(f"  final_active_nodes: {summary.get('final_active_node_count', 0)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
