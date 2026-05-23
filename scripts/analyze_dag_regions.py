#!/usr/bin/env python
"""Analyze bounded DAG motifs and multi-join ONNX regions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.dag_region_analysis import (
    build_dag_region_analysis_report,
    dag_region_evidence_to_markdown,
    dag_region_patterns_to_markdown,
    dag_region_report_to_dict,
    dag_region_report_to_markdown,
)
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DAG motif and multi-join ONNX regions.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--max-branch-depth", type=int, default=4)
    parser.add_argument("--max-regions", type=int)
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
    data = dag_region_report_to_dict(report)
    patterns = {
        "model_name": report.model_name,
        "pattern_summaries": data["pattern_summaries"],
        "summary": data["summary"],
    }
    evidence = {
        "model_name": report.model_name,
        "pruning_evidence": data["pruning_evidence"],
        "summary": {"suggested_constraint_counts": data["summary"].get("suggested_constraint_counts", {})},
    }
    if output_format in {"json", "both"}:
        write_json(data, root / "reports" / "dag_regions" / f"{safe_name}.json")
        write_json(patterns, root / "reports" / "dag_region_patterns" / f"{safe_name}.json")
        write_json(evidence, root / "reports" / "dag_region_pruning_evidence" / f"{safe_name}.json")
    if output_format in {"md", "both"}:
        write_markdown(dag_region_report_to_markdown(report), root / "reports" / "dag_regions" / f"{safe_name}.md")
        write_markdown(
            dag_region_patterns_to_markdown(report),
            root / "reports" / "dag_region_patterns" / f"{safe_name}.md",
        )
        write_markdown(
            dag_region_evidence_to_markdown(report),
            root / "reports" / "dag_region_pruning_evidence" / f"{safe_name}.md",
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
        summary_path = root / "reports" / "onnx_graphs" / f"{safe_name}.json"
        if not summary_path.exists():
            print(
                f"[missing] ONNX graph summary missing. Run: python scripts/generate_structural_inventory.py --model {config['name']} --require-onnx",
                file=sys.stderr,
            )
            failed = True
            continue
        report = build_dag_region_analysis_report(
            _load_json(summary_path),
            config,
            max_branch_depth=args.max_branch_depth,
            max_regions=args.max_regions,
        )
        _write_outputs(root, safe_name, report, args.format)
        if args.verbose:
            summary = report.summary
            print(f"[dag-regions] {config['name']}")
            print(f"  regions: {summary['num_regions']}")
            print(f"  forks: {summary['num_fork_regions']}")
            print(f"  diamonds: {summary['num_diamond_regions']}")
            print(f"  join-fork-join: {summary['num_join_fork_join_regions']}")
            print(f"  residual-like: {summary['num_residual_like_regions']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

