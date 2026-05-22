#!/usr/bin/env python
"""Build PyTorch-to-ONNX correspondence and shape evidence reports."""

from __future__ import annotations

import argparse
import json
import sys

from model_analysis.correspondence import (
    build_module_node_correspondence,
    build_parameter_evidence,
    correspondence_report_to_markdown,
    write_correspondence_json,
)
from model_analysis.dependency_validation import dependency_validation_to_markdown, validate_dependency_graph_with_evidence
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, load_model_registry
from model_analysis.reporting import write_json, write_markdown
from model_analysis.shape_evidence import build_shape_evidence, shape_evidence_report_to_markdown, write_shape_evidence_json


def select_models(model_arg: str) -> list[dict]:
    if model_arg == "all":
        return load_model_registry()
    return [get_model_config(model_arg)]


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _should_write_json(format_arg: str) -> bool:
    return format_arg in {"json", "both"}


def _should_write_markdown(format_arg: str) -> bool:
    return format_arg in {"md", "both"}


def build_one(config: dict, require_dependency_graph: bool, format_arg: str, verbose: bool) -> None:
    root = get_project_root()
    safe_name = safe_model_name(config["hf_id"])
    structural_path = root / "reports" / "structural_inventory" / f"{safe_name}.json"
    onnx_path = root / "reports" / "onnx_graphs" / f"{safe_name}.json"
    dependency_path = root / "reports" / "dependency_graphs" / f"{safe_name}.json"

    if not structural_path.exists():
        raise FileNotFoundError(f"Structural inventory missing. Run: python scripts/generate_structural_inventory.py --model {config['name']}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX graph summary missing. Run: python scripts/generate_structural_inventory.py --model {config['name']} --require-onnx")
    if require_dependency_graph and not dependency_path.exists():
        raise FileNotFoundError(f"Dependency graph missing. Run: python scripts/build_dependency_graph.py --model {config['name']} --require-onnx")

    torch_summary = _load_json(structural_path)
    onnx_summary = _load_json(onnx_path)
    dependency_graph = _load_json(dependency_path) if dependency_path.exists() else None

    parameter_evidence = build_parameter_evidence(torch_summary, onnx_summary)
    correspondence = build_module_node_correspondence(torch_summary, onnx_summary, parameter_evidence, dependency_graph)
    shape_report = build_shape_evidence(onnx_summary)

    correspondence_dir = root / "reports" / "correspondence"
    shape_dir = root / "reports" / "shape_evidence"
    validation_dir = root / "reports" / "validated_dependency_graphs"
    if _should_write_json(format_arg):
        write_correspondence_json(correspondence, correspondence_dir / f"{safe_name}.json")
        write_shape_evidence_json(shape_report, shape_dir / f"{safe_name}.json")
    if _should_write_markdown(format_arg):
        write_markdown(correspondence_report_to_markdown(correspondence), correspondence_dir / f"{safe_name}.md")
        write_markdown(shape_evidence_report_to_markdown(shape_report), shape_dir / f"{safe_name}.md")

    validation = None
    if dependency_graph:
        validation = validate_dependency_graph_with_evidence(dependency_graph, correspondence, shape_report)
        if _should_write_json(format_arg):
            write_json(validation, validation_dir / f"{safe_name}.json")
        if _should_write_markdown(format_arg):
            write_markdown(dependency_validation_to_markdown(validation), validation_dir / f"{safe_name}.md")

    if verbose:
        print(f"[correspondence] {config['name']}")
        print(f"  module correspondences: {correspondence.summary.get('num_module_correspondences', 0)}")
        print(f"  matched parameters: {correspondence.summary.get('matched_parameters', 0)}")
        print(f"  tensor shapes: {shape_report.summary.get('num_tensor_shapes', 0)}")
        if validation:
            print(f"  validated units: {validation['summary']['num_validated_units']}")
            print(f"  validated edges: {validation['summary']['num_validated_edges']}")
    print(f"[ok] built correspondence and shape evidence for {config['name']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PyTorch-to-ONNX correspondence and shape evidence reports.")
    parser.add_argument("--model", required=True, help="Configured model name, Hugging Face ID, or 'all'.")
    parser.add_argument("--require-dependency-graph", action="store_true", help="Fail if dependency graph JSON is missing.")
    parser.add_argument("--format", choices=["json", "md", "both"], default="both", help="Report formats to write. Defaults to both.")
    parser.add_argument("--verbose", action="store_true", help="Print summary stats.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = 0
    try:
        configs = select_models(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    for config in configs:
        try:
            build_one(config, args.require_dependency_graph, args.format, args.verbose)
        except FileNotFoundError as exc:
            failures += 1
            print(f"[missing] {exc}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            print(f"[error] failed for {config.get('name', '<unknown>')}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
