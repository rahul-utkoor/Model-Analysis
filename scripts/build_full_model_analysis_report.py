#!/usr/bin/env python
"""Build a structured full-model static analysis report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model_analysis.full_model_analysis_report import (
    build_full_model_analysis_report,
    detect_layers,
    discover_model_artifacts,
    parse_layer_selection,
)
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--export-onnx-subgraphs", dest="export_onnx", action="store_true", default=True)
    parser.add_argument("--no-export-onnx-subgraphs", dest="export_onnx", action="store_false")
    parser.add_argument("--render-svg", action="store_true")
    parser.add_argument("--include-auxiliary", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-root", default="reports/model_analysis_reports")
    parser.add_argument("--artifact-root", default="artifacts/model_analysis_subgraphs")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        config = get_model_config(args.model)
        model_name = config["hf_id"]
        _, loaded, _ = discover_model_artifacts(root, model_name)
        available_layers = detect_layers(
            loaded.get("region_pruning_semantics"), loaded.get("abstract_expansion")
        )
        if not available_layers:
            available_layers = [0]
        layers = parse_layer_selection(args.layers, available_layers, args.max_layers)
        report = build_full_model_analysis_report(
            root=root,
            model_name=model_name,
            layers=layers,
            output_root=root / Path(args.output_root),
            artifact_root=root / Path(args.artifact_root),
            export_onnx_subgraphs=args.export_onnx,
            render_svg=args.render_svg,
            include_auxiliary=args.include_auxiliary,
            strict=args.strict,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    if args.verbose:
        safe = safe_model_name(report["model_name"])
        summary = report["model_summary"]
        print(f"[full-model-report] {report['model_name']}")
        print(f"  layers: {summary.get('layers_generated', 0)}")
        print(f"  subgraphs: {summary.get('total_subgraphs', 0)}")
        print(f"  onnx exported/skipped/failed: {summary.get('onnx_exported', 0)}/{summary.get('onnx_skipped', 0)}/{summary.get('onnx_failed', 0)}")
        print(f"  safe/constrained/blocked/auxiliary/unknown: {summary.get('safe', 0)}/{summary.get('constrained', 0)}/{summary.get('blocked', 0)}/{summary.get('auxiliary', 0)}/{summary.get('unknown', 0)}")
        print(f"  valid plans: {summary.get('plan_validation', {}).get('valid', 0)}")
        print(f"  report: {root / args.output_root / safe / 'index.md'}")
        print(f"  artifacts: {root / args.artifact_root / safe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
