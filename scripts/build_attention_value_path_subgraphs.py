#!/usr/bin/env python
"""Build complete attention value-path evidence artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx

from model_analysis.attention_value_path_subgraph import (
    attention_value_path_report_to_dict,
    bind_path_to_onnx,
    detect_attention_value_paths,
    export_attention_value_path,
    make_attention_value_path_report,
    write_attention_value_path_report,
)
from model_analysis.attention_value_path_subgraph_text import attention_value_path_report_to_markdown
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--layers", choices=["all"])
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--render-svg", action="store_true")
    parser.add_argument("--output-dir", default="reports/attention_value_path_subgraphs")
    parser.add_argument("--artifact-dir", default="artifacts/attention_value_path_subgraphs")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _source_onnx(root: Path, safe: str) -> Path | None:
    static = root / "data/models/onnx_static" / safe / "model.static.onnx"
    dynamic = root / "data/models/onnx" / safe / "model.onnx"
    return static if static.exists() else dynamic if dynamic.exists() else None


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        config = get_model_config(args.model)
        model = config["hf_id"]
        safe = safe_model_name(model)
        deadbranch_path = root / "reports/deadbranch_propagation" / f"{safe}.json"
        if not deadbranch_path.exists():
            raise FileNotFoundError(f"deadbranch report missing: {deadbranch_path}; run scripts/analyze_deadbranch_propagation.py --model {args.model}")
        source_path = _source_onnx(root, safe)
        if source_path is None:
            raise FileNotFoundError(f"local ONNX source model missing for {model}")
        source_model = onnx.load(source_path)
        paths = detect_attention_value_paths(model, json.loads(deadbranch_path.read_text(encoding="utf-8")))
        selected_layer = args.layer if args.layer is not None else None
        if selected_layer is not None:
            paths = [path for path in paths if path.layer_index == selected_layer]
        elif args.layers != "all":
            paths = [path for path in paths if path.layer_index == 0]
        artifact_root = root / args.artifact_dir
        for path in paths:
            bind_path_to_onnx(path, source_model)
            if args.export_onnx:
                export_attention_value_path(path, source_model, source_path, artifact_root, args.render_svg)
        report = make_attention_value_path_report(model, paths)
        report_data = attention_value_path_report_to_dict(report)
        report_root = root / args.output_dir / safe
        for layer in sorted({path.layer_index for path in paths}):
            layer_paths = [path for path in paths if path.layer_index == layer]
            layer_report = make_attention_value_path_report(model, layer_paths)
            layer_root = report_root / f"layer_{layer}"
            write_attention_value_path_report(layer_report, layer_root / "index.json")
            write_markdown(attention_value_path_report_to_markdown(attention_value_path_report_to_dict(layer_report)), layer_root / "index.md")
        write_attention_value_path_report(report, report_root / "summary.json")
        write_markdown(attention_value_path_report_to_markdown(report_data), report_root / "summary.md")
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    if args.verbose:
        print(f"[attention-value-path] {model}")
        print(f"  paths: {report.total_paths}")
        print(f"  seedable/partial/blocked/unknown: {report.seedable}/{report.partial}/{report.blocked}/{report.unknown}")
        print(f"  exported/skipped/failed: {report.exported}/{report.skipped}/{report.failed}")
        print(f"  report: {report_root / 'summary.md'}")
        print(f"  artifacts: {artifact_root / safe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
