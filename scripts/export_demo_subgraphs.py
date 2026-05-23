#!/usr/bin/env python
"""Export a curated small set of analyzed ONNX subgraphs for Netron."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx

from model_analysis.onnx_subgraph_extractor import (
    extract_onnx_subgraph_model,
    load_subgraph_records,
    make_subgraph_export_report,
    netron_index_to_markdown,
    safe_artifact_name,
    subgraph_export_report_to_markdown,
    write_subgraph_export_report_json,
)
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a curated Netron subgraph demo set.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-per-category", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _take(records: list[dict], predicate, limit: int) -> list[dict]:
    selected = [record for record in records if predicate(record)]
    selected.sort(key=lambda record: (record.get("subgraph_kind", ""), record.get("subgraph_id", "")))
    return selected[:limit]


def _curated_records(records: list[dict], limit: int) -> list[dict]:
    paths = [record for record in records if record["subgraph_kind"] == "path"]
    joins = [record for record in records if record["subgraph_kind"] == "join"]
    dags = [record for record in records if record["subgraph_kind"] == "dag_region"]
    categories = [
        _take(paths, lambda item: any(op in {"MatMul", "Gemm"} for op in item["op_types"]) and item.get("pruning_class") == "directly_prunable", limit),
        _take(paths, lambda item: "Softmax" in item["op_types"], limit),
        _take(joins, lambda item: "Join(Add)" in item["pattern"] and item["metadata"].get("is_residual_like", False), limit),
        sorted(
            dags,
            key=lambda item: (
                0 if item["metadata"].get("region_kind") == "join_fork_join" else 1 if item["metadata"].get("region_kind") == "diamond" else 2,
                item["subgraph_id"],
            ),
        )[:limit],
        _take(paths, lambda item: any(op in {"Reshape", "Transpose"} for op in item["op_types"]), limit),
    ]
    selected: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for category in categories:
        for record in category:
            key = (record["subgraph_kind"], record["subgraph_id"])
            if key not in seen:
                seen.add(key)
                selected.append(record)
    return selected


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        config = get_model_config(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    safe_name = safe_model_name(config["hf_id"])
    source_path = root / "data" / "models" / "onnx" / safe_name / "model.onnx"
    if not source_path.exists():
        print(
            f"[missing] ONNX model missing. Run: python scripts/export_to_onnx.py --model {config['name']}",
            file=sys.stderr,
        )
        return 1
    records = load_subgraph_records(config["name"], safe_name, ["path", "join", "dag_region"])
    selected = _curated_records(records, args.max_per_category)
    if not selected:
        print(
            f"[missing] No analyzed subgraphs found. Run: python scripts/analyze_subgraphs.py --model {config['name']} and python scripts/analyze_dag_regions.py --model {config['name']}",
            file=sys.stderr,
        )
        return 1
    source_model = onnx.load(source_path)
    output_root = root / "artifacts" / "subgraph_onnx" / safe_name / "demo"
    results = []
    for record in selected:
        record = dict(record)
        record["source_onnx_path"] = str(source_path)
        filename = f"{record['subgraph_kind']}__{safe_artifact_name(record['subgraph_id'])}.onnx"
        result = extract_onnx_subgraph_model(source_model, record, output_root / filename, config["name"])
        results.append(result)
        if args.verbose:
            print(f"[{result.status}] {result.subgraph_kind}:{result.subgraph_id} -> {result.output_onnx_path}")
    report = make_subgraph_export_report(
        config["name"],
        source_path,
        output_root,
        results,
        metadata={"curated_demo": True, "max_per_category": args.max_per_category},
    )
    report_name = f"{safe_name}__demo"
    write_subgraph_export_report_json(report, root / "reports" / "subgraph_exports" / f"{report_name}.json")
    write_markdown(
        subgraph_export_report_to_markdown(report),
        root / "reports" / "subgraph_exports" / f"{report_name}.md",
    )
    write_markdown(
        netron_index_to_markdown(report),
        root / "reports" / "netron_subgraph_index" / f"{report_name}.md",
    )
    print(f"[demo-export] successful: {report.summary['num_successful_exports']}")
    print(f"[demo-export] failed: {report.summary['num_failed_exports']}")
    print(f"[index] reports/netron_subgraph_index/{report_name}.md")
    return 1 if report.failed_exports else 0


if __name__ == "__main__":
    raise SystemExit(main())

