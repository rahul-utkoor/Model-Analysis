#!/usr/bin/env python
"""Export selected analyzed subgraphs as Netron-visualizable ONNX artifacts."""

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
    select_subgraphs_for_export,
    subgraph_export_report_to_markdown,
    write_subgraph_export_report_json,
)
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected analysis subgraphs for Netron.")
    parser.add_argument("--model", required=True, help="Configured model name or Hugging Face ID.")
    parser.add_argument("--kind", choices=["path", "join", "dag_region", "all"], default="all")
    parser.add_argument("--subgraph-id", action="append", help="Exact subgraph ID; repeat or comma-separate.")
    parser.add_argument("--pattern-contains")
    parser.add_argument("--pruning-class")
    parser.add_argument("--risk-level")
    parser.add_argument("--max-exports", type=int)
    parser.add_argument("--output-root", default="artifacts/subgraph_onnx")
    parser.add_argument("--check-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _kinds(value: str) -> list[str]:
    return ["path", "join", "dag_region"] if value == "all" else [value]


def _ids(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    result = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _output_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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
    kinds = _kinds(args.kind)
    records = load_subgraph_records(config["name"], safe_name, kinds)
    requested_ids = _ids(args.subgraph_id)
    selected = select_subgraphs_for_export(
        records,
        subgraph_ids=requested_ids,
        kinds=kinds,
        pattern_contains=args.pattern_contains,
        pruning_class=args.pruning_class,
        risk_level=args.risk_level,
        max_exports=args.max_exports,
    )
    if not selected:
        print("[missing] No subgraph records matched the requested selection.", file=sys.stderr)
        return 1
    output_root = _output_root(root, args.output_root)
    source_model = onnx.load(source_path)
    results = []
    for record in selected:
        record = dict(record)
        record["source_onnx_path"] = str(source_path)
        kind = record["subgraph_kind"]
        output_path = output_root / safe_name / kind / f"{safe_artifact_name(record['subgraph_id'])}.onnx"
        result = extract_onnx_subgraph_model(
            source_model,
            record,
            output_path,
            config["name"],
            check_model=args.check_model,
        )
        results.append(result)
        if args.verbose:
            print(f"[{result.status}] {result.subgraph_kind}:{result.subgraph_id} -> {result.output_onnx_path}")
            if result.status != "success":
                print(f"  reason: {result.reason}")
    report = make_subgraph_export_report(
        config["name"],
        source_path,
        output_root,
        results,
        metadata={"selection_kind": args.kind, "check_model": args.check_model},
    )
    write_subgraph_export_report_json(report, root / "reports" / "subgraph_exports" / f"{safe_name}.json")
    write_markdown(
        subgraph_export_report_to_markdown(report),
        root / "reports" / "subgraph_exports" / f"{safe_name}.md",
    )
    write_markdown(
        netron_index_to_markdown(report),
        root / "reports" / "netron_subgraph_index" / f"{safe_name}.md",
    )
    print(f"[export] successful: {report.summary['num_successful_exports']}")
    print(f"[export] failed: {report.summary['num_failed_exports']}")
    print(f"[baseline] Original ONNX model: {source_path}")
    print(f"[baseline] Netron command: netron {source_path}")
    print(f"[index] reports/netron_subgraph_index/{safe_name}.md")
    return 1 if report.failed_exports else 0


if __name__ == "__main__":
    raise SystemExit(main())
