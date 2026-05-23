#!/usr/bin/env python
"""Import available frontend summaries into frontend-independent Tensor Graph IR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.onnx_to_tensor_ir import build_tensor_graph_from_onnx_summary
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown
from model_analysis.tensor_ir import tensor_graph_to_dict, tensor_graph_to_markdown, write_tensor_graph_json
from model_analysis.tensor_ir_text import write_tensor_graph_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frontend-independent Tensor Graph IR.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--format", choices=["json", "md", "text", "all"], default="all")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _selected_models(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats_to_markdown(graph_data: dict) -> str:
    summary = graph_data.get("summary", {})
    counts = summary.get("canonical_op_type_counts", {})
    lines = [
        f"# Tensor IR Statistics: {graph_data.get('model_name', '')}",
        "",
        f"- Frontend: `{graph_data.get('source_frontend', 'unknown')}`",
        f"- Operations: `{summary.get('num_ops', 0)}`",
        f"- Values: `{summary.get('num_values', 0)}`",
        f"- Fork operations: `{summary.get('num_fork_ops', 0)}`",
        f"- Join operations: `{summary.get('num_join_ops', 0)}`",
        "",
        "## Canonical Operations",
        "",
        "| canonical_op_type | count |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(counts.items()))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Statistics summarize the canonical tensor-dataflow graph imported through the ONNX frontend. Tensor IR, not ONNX, is the intended substrate for future structural-region analysis.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(root: Path, safe_name: str, graph, output_format: str) -> None:
    data = tensor_graph_to_dict(graph)
    stats = {
        "model_name": graph.model_name,
        "source_frontend": graph.source_frontend,
        "summary": graph.summary,
        "metadata": graph.metadata,
    }
    if output_format in {"json", "all"}:
        write_tensor_graph_json(graph, root / "reports" / "tensor_ir" / f"{safe_name}.json")
        write_json(stats, root / "reports" / "tensor_ir_stats" / f"{safe_name}.json")
    if output_format in {"md", "all"}:
        write_markdown(tensor_graph_to_markdown(graph), root / "reports" / "tensor_ir" / f"{safe_name}.md")
        write_markdown(_stats_to_markdown(data), root / "reports" / "tensor_ir_stats" / f"{safe_name}.md")
    if output_format in {"text", "all"}:
        write_tensor_graph_text(graph, root / "reports" / "tensor_ir_dumps" / f"{safe_name}.tir")


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
        source_path = root / "reports" / "onnx_graphs" / f"{safe_name}.json"
        if not source_path.exists():
            print(
                f"[missing] ONNX graph summary missing. Run: python scripts/generate_structural_inventory.py --model {config['name']} --require-onnx",
                file=sys.stderr,
            )
            failed = True
            continue
        graph = build_tensor_graph_from_onnx_summary(_load_json(source_path), config)
        _write_outputs(root, safe_name, graph, args.format)
        if args.verbose:
            summary = graph.summary
            print(f"[tensor-ir] {config['name']} frontend={graph.source_frontend}")
            print(f"  ops: {summary['num_ops']}")
            print(f"  values: {summary['num_values']}")
            print(f"  forks: {summary['num_fork_ops']}")
            print(f"  joins: {summary['num_join_ops']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
