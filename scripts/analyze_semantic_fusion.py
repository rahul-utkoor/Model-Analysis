#!/usr/bin/env python
"""Detect decomposed activation and feed-forward semantic fusions in Tensor IR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown
from model_analysis.semantic_fusion import (
    build_semantic_fusion_report,
    semantic_fusion_report_to_dict,
    semantic_fusion_report_to_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze decomposed semantic fusions over Tensor IR.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--verbose", action="store_true")
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


def _patterns_markdown(model_name: str, fusions: list[dict]) -> str:
    rows: dict[tuple[str, str, str], int] = {}
    for fusion in fusions:
        key = (fusion.get("fusion_type", ""), fusion.get("pattern", ""), fusion.get("confidence", ""))
        rows[key] = rows.get(key, 0) + 1
    items = [
        {"fusion_type": key[0], "pattern": key[1], "confidence": key[2], "count": count}
        for key, count in sorted(rows.items())
    ]
    return "\n".join(
        [
            f"# Fused Region Patterns: {model_name}",
            "",
            _table(items, ["fusion_type", "pattern", "confidence", "count"]),
            "",
            "Patterns are recovered from Tensor IR for semantic region construction; they do not modify model artifacts.",
            "",
        ]
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
        source_path = root / "reports" / "tensor_ir" / f"{safe_name}.json"
        if not source_path.exists():
            print(f"[missing] Tensor IR missing. Run: python scripts/build_tensor_ir.py --model {config['name']}", file=sys.stderr)
            failed = True
            continue
        tensor_graph = json.loads(source_path.read_text(encoding="utf-8"))
        report = build_semantic_fusion_report(tensor_graph)
        data = semantic_fusion_report_to_dict(report)
        write_json(data, root / "reports" / "semantic_fusion" / f"{safe_name}.json")
        write_markdown(semantic_fusion_report_to_markdown(report), root / "reports" / "semantic_fusion" / f"{safe_name}.md")
        patterns = {"model_name": report.model_name, "fusions": data["fusions"], "summary": data["summary"]}
        write_json(patterns, root / "reports" / "fused_region_patterns" / f"{safe_name}.json")
        write_markdown(_patterns_markdown(report.model_name, data["fusions"]), root / "reports" / "fused_region_patterns" / f"{safe_name}.md")
        if args.verbose:
            print(f"[semantic-fusion] {report.model_name}")
            print(f"  gelu_fusions: {report.summary['num_gelu_fusions']}")
            print(f"  feedforward_fusions: {report.summary['num_feedforward_fusions']}")
            print(f"  confidence_counts: {report.summary['confidence_counts']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
