#!/usr/bin/env python
"""Build structured static analysis reports for configured models."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_analysis.cross_model_analysis_report import build_cross_model_analysis_report, write_cross_model_analysis_report
from model_analysis.cross_model_analysis_report_text import cross_model_report_to_markdown
from model_analysis.full_model_analysis_report import (
    build_full_model_analysis_report,
    detect_layers,
    discover_model_artifacts,
    missing_required_artifacts,
    parse_layer_selection,
)
from model_analysis.paths import get_project_root
from model_analysis.registry import get_model_config, list_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--export-onnx-subgraphs", dest="export_onnx", action="store_true", default=True)
    parser.add_argument("--no-export-onnx-subgraphs", dest="export_onnx", action="store_false")
    parser.add_argument("--render-svg", action="store_true")
    parser.add_argument("--include-auxiliary", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-root", default="reports/model_analysis_reports")
    parser.add_argument("--artifact-root", default="artifacts/model_analysis_subgraphs")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _configs(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    root = get_project_root()
    output_root = root / Path(args.output_root)
    artifact_root = root / Path(args.artifact_root)
    succeeded: list[str] = []
    skipped: list[dict] = []
    for config in _configs(args.models):
        model_name = config["hf_id"]
        _, loaded, missing = discover_model_artifacts(root, model_name)
        required_missing = missing_required_artifacts(missing)
        if required_missing:
            skipped.append({"model": model_name, "missing_artifacts": required_missing})
            if args.verbose:
                print(f"[skip] {model_name}: missing required artifacts")
            continue
        available_layers = detect_layers(
            loaded.get("region_pruning_semantics"), loaded.get("abstract_expansion")
        )
        if not available_layers:
            available_layers = [0]
        layers = parse_layer_selection(args.layers, available_layers, args.max_layers)
        try:
            build_full_model_analysis_report(
                root=root,
                model_name=model_name,
                layers=layers,
                output_root=output_root,
                artifact_root=artifact_root,
                export_onnx_subgraphs=args.export_onnx,
                render_svg=args.render_svg,
                include_auxiliary=args.include_auxiliary,
                strict=args.strict,
            )
        except Exception as exc:
            skipped.append({"model": model_name, "error": str(exc)})
            if args.strict:
                raise
            if args.verbose:
                print(f"[skip] {model_name}: {exc}")
            continue
        succeeded.append(model_name)
        if args.verbose:
            print(f"[ok] {model_name}")
    cross = build_cross_model_analysis_report(root, succeeded + [item["model"] for item in skipped], output_root)
    cross["build_summary"] = {"models_succeeded": succeeded, "models_skipped": skipped}
    write_cross_model_analysis_report(cross, output_root, cross_model_report_to_markdown)
    if args.verbose:
        print(f"[all-model-reports] succeeded={len(succeeded)} skipped={len(skipped)}")
        print(f"[all-model-reports] cross_model={output_root / 'cross_model' / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
