"""Analyze one selected ONNX subgraph using local ONNX-MLIR evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from experimental.mlir_axis_bridge.bridge_runner import analyze_onnx_with_mlir_bridge
from experimental.mlir_axis_bridge.report import render_json, render_markdown, render_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Path to one local ONNX subgraph artifact.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated local MLIR evidence and report output.")
    parser.add_argument("--onnx-mlir")
    parser.add_argument("--mlir-opt")
    parser.add_argument("--native-dependence-json", help="Optional externally generated native MLIR dependence JSON report.")
    parser.add_argument("--prefer-native-dependence", action="store_true", help="Prefer imported native dependence relations when they prove a supported pattern.")
    parser.add_argument("--emit-python-dependence-json", help="Write the strongest Python affine-extractor dependence report for inspection.")
    parser.add_argument("--native-pass-tool", help="Path to the optional standalone pruning-axis dependence tool.")
    parser.add_argument("--run-native-pass", action="store_true", help="Run the optional native dependence tool on the richest lowered MLIR artifact.")
    parser.add_argument("--native-output-dir", help="Directory for JSON emitted by the optional native dependence tool.")
    parser.add_argument("--hint", choices=["auto", "ffn", "qk-score", "attention-context", "attention-value-path", "residual", "layernorm"], default="auto")
    parser.add_argument("--format", choices=["markdown", "json", "text"], default="text")
    parser.add_argument("--show-toolchain", action="store_true")
    parser.add_argument("--show-artifacts", action="store_true")
    parser.add_argument("--show-accesses", action="store_true")
    parser.add_argument("--show-axis", action="store_true")
    parser.add_argument("--show-dfa", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    try:
        result = analyze_onnx_with_mlir_bridge(
            args.onnx,
            output_dir / "mlir_artifacts",
            args.onnx_mlir,
            args.mlir_opt,
            args.hint,
            args.native_dependence_json,
            args.prefer_native_dependence,
            args.emit_python_dependence_json,
            args.run_native_pass,
            args.native_pass_tool,
            args.native_output_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    if args.format == "markdown":
        text, report_name = render_markdown(result), "report.md"
    elif args.format == "json":
        text, report_name = render_json(result), "report.json"
    else:
        show_all = args.show_all
        text, report_name = render_text(
            result,
            show_toolchain=args.show_toolchain or show_all,
            show_artifacts=args.show_artifacts or show_all,
            show_accesses=args.show_accesses or show_all,
            show_axis=args.show_axis or show_all,
            show_dfa=args.show_dfa or show_all,
        ), "report.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / report_name
    report_path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
    if args.verbose:
        print(f"[mlir-axis-bridge] wrote {report_path}")
        print(
            f"[mlir-axis-bridge] artifacts={result.summary['num_artifacts']} "
            f"patterns={result.summary['axis_patterns']} evidence={result.summary['evidence_source']}"
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
