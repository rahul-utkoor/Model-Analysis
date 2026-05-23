#!/usr/bin/env python
"""Analyze directed and join-centered local ONNX subgraphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown
from model_analysis.subgraph_analysis import (
    build_subgraph_analysis_report,
    join_subgraphs_to_markdown,
    pattern_summaries_to_markdown,
    pruning_evidence_to_markdown,
    subgraph_analysis_report_to_dict,
    subgraph_analysis_report_to_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze k-node and join-aware ONNX subgraphs.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--max-nodes", type=int, default=5)
    parser.add_argument("--max-subgraphs-per-size", type=int)
    parser.add_argument("--branch-depth", type=int, default=2)
    parser.add_argument("--post-join-depth", type=int, default=2)
    parser.add_argument("--max-join-subgraphs", type=int)
    parser.add_argument("--format", choices=["json", "md", "both"], default="both")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _selected_models(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_outputs(root: Path, safe_name: str, report, output_format: str) -> None:
    data = subgraph_analysis_report_to_dict(report)
    residual = [item for item in data["join_subgraphs"] if item.get("is_residual_like")]
    focused_patterns = {
        "model_name": report.model_name,
        "pattern_summaries": data["pattern_summaries"],
        "summary": data["summary"],
    }
    focused_evidence = {
        "model_name": report.model_name,
        "pruning_evidence": data["pruning_evidence"],
        "summary": {"evidence_type_counts": data["summary"].get("evidence_type_counts", {})},
    }
    dimension_evidence = {
        "model_name": report.model_name,
        "dimension_evidence": [
            item for item in data["pruning_evidence"] if item.get("suggested_constraint_type")
        ],
    }
    joins = {"model_name": report.model_name, "join_subgraphs": data["join_subgraphs"]}
    residuals = {"model_name": report.model_name, "residual_subgraphs": residual}
    if output_format in {"json", "both"}:
        write_json(data, root / "reports" / "subgraphs" / f"{safe_name}.json")
        write_json(focused_patterns, root / "reports" / "subgraph_patterns" / f"{safe_name}.json")
        write_json(focused_evidence, root / "reports" / "subgraph_pruning_analysis" / f"{safe_name}.json")
        write_json(dimension_evidence, root / "reports" / "subgraph_dimension_evidence" / f"{safe_name}.json")
        write_json(joins, root / "reports" / "join_subgraphs" / f"{safe_name}.json")
        write_json(residuals, root / "reports" / "residual_subgraphs" / f"{safe_name}.json")
    if output_format in {"md", "both"}:
        write_markdown(subgraph_analysis_report_to_markdown(report), root / "reports" / "subgraphs" / f"{safe_name}.md")
        write_markdown(pattern_summaries_to_markdown(report), root / "reports" / "subgraph_patterns" / f"{safe_name}.md")
        write_markdown(pruning_evidence_to_markdown(report), root / "reports" / "subgraph_pruning_analysis" / f"{safe_name}.md")
        write_markdown(
            pruning_evidence_to_markdown({**data, "pruning_evidence": dimension_evidence["dimension_evidence"]}),
            root / "reports" / "subgraph_dimension_evidence" / f"{safe_name}.md",
        )
        write_markdown(join_subgraphs_to_markdown(report), root / "reports" / "join_subgraphs" / f"{safe_name}.md")
        write_markdown(join_subgraphs_to_markdown(report, residual_only=True), root / "reports" / "residual_subgraphs" / f"{safe_name}.md")


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
        summary_path = root / "reports" / "onnx_graphs" / f"{safe_name}.json"
        if not summary_path.exists():
            print(
                f"[missing] ONNX graph summary missing. Run: python scripts/generate_structural_inventory.py --model {config['name']} --require-onnx",
                file=sys.stderr,
            )
            failed = True
            continue
        report = build_subgraph_analysis_report(
            _load_json(summary_path),
            config,
            max_nodes=args.max_nodes,
            max_subgraphs_per_size=args.max_subgraphs_per_size,
            branch_depth=args.branch_depth,
            post_join_depth=args.post_join_depth,
            max_join_subgraphs=args.max_join_subgraphs,
        )
        _write_outputs(root, safe_name, report, args.format)
        if args.verbose:
            summary = report.summary
            print(f"[subgraphs] {config['name']}")
            print(f"  paths: {summary['num_path_subgraphs']}")
            print(f"  joins: {summary['num_join_subgraphs']}")
            print(f"  residual-like joins: {summary['num_residual_like_join_subgraphs']}")
            print(f"  patterns: {summary['num_patterns']}")
            print(f"  evidence: {sum(summary['evidence_type_counts'].values())}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
