#!/usr/bin/env python3
"""Annotate ONNX nodes with strict MLIR-derived axis-semantics metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_analysis.onnx_axis_semantics_export import annotate_onnx_axis_semantics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input ONNX graph or local subgraph.")
    parser.add_argument("--output", required=True, help="Annotated ONNX output path.")
    parser.add_argument("--sidecar-json", required=True, help="Strict axis-semantics sidecar JSON path.")
    parser.add_argument("--dot", help="Annotated DOT output path.")
    parser.add_argument("--svg", help="Annotated SVG output path.")
    parser.add_argument("--mlir-output-dir", help="Directory for evidence-unit and MLIR artifacts.")
    parser.add_argument("--onnx-mlir", help="Path to onnx-mlir executable.")
    parser.add_argument("--mlir-opt", help="Reserved compatibility option for MLIR tooling.")
    parser.add_argument("--native-pass-tool", help="Path to pruning-axis-dependence native tool.")
    parser.add_argument("--run-native-pass", action="store_true", help="Run native MLIR dependence extraction when available.")
    parser.add_argument("--allow-no-mlir", action="store_true", help="Permit output with UNKNOWN semantics if MLIR is unavailable.")
    parser.add_argument("--annotation-mode", choices=["attributes", "doc_string", "both"], default="doc_string")
    parser.add_argument("--doc-string-format", choices=["compact", "verbose", "minimal"], default="compact")
    parser.add_argument("--include-verbose-onnx-attributes", action="store_true", help="Include full relations/evidence JSON as ONNX attributes.")
    parser.add_argument("--leader-report", help="Optional Markdown report for MLIR-derived leader candidates.")
    parser.add_argument("--fallback-doc-string", action="store_true", help="Fallback to doc_string annotations if custom attributes fail ONNX checker.")
    parser.add_argument("--check-onnx", action="store_true", help="Run onnx.checker on the annotated ONNX output.")
    parser.add_argument("--model-name", help="Optional display model name for the sidecar.")
    parser.add_argument("--max-nodes", type=int, help="Limit nodes annotated with MLIR evidence; skipped nodes are unknown.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = annotate_onnx_axis_semantics(
            input_path=args.input,
            output_path=args.output,
            sidecar_json=args.sidecar_json,
            dot_path=args.dot,
            svg_path=args.svg,
            mlir_output_dir=args.mlir_output_dir,
            onnx_mlir_path=args.onnx_mlir,
            mlir_opt_path=args.mlir_opt,
            native_pass_tool=args.native_pass_tool,
            run_native_pass=args.run_native_pass,
            allow_no_mlir=args.allow_no_mlir,
            annotation_mode=args.annotation_mode,
            doc_string_format=args.doc_string_format,
            include_verbose_onnx_attributes=args.include_verbose_onnx_attributes,
            leader_report=args.leader_report,
            fallback_doc_string=args.fallback_doc_string,
            check_onnx=args.check_onnx,
            model_name=args.model_name,
            max_nodes=args.max_nodes,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.verbose:
        print(json.dumps({
            "output_onnx": payload["output_onnx"],
            "sidecar_json": str(args.sidecar_json),
            "semantic_counts": payload["semantic_counts"],
            "evidence_tier_counts": payload["evidence_tier_counts"],
            "blocker_counts": payload["blocker_counts"],
            "leader_candidate_counts": payload["leader_candidate_counts"],
            "checker": payload["checker"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
