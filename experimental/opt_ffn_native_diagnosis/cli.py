"""Diagnose and repair OPT FFN native MLIR local evidence artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from experimental.opt_ffn_native_diagnosis.report import write_report_bundle
from experimental.opt_ffn_native_diagnosis.runner import MODEL_NAME, run_diagnosis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME, choices=[MODEL_NAME])
    parser.add_argument("--layers", choices=["layer0", "all"], default="layer0")
    parser.add_argument("--output-dir", default="reports/opt_ffn_native_diagnosis")
    parser.add_argument("--run-native-pass", action="store_true")
    parser.add_argument("--native-pass-tool")
    parser.add_argument("--onnx-mlir")
    parser.add_argument("--mlir-opt")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_diagnosis(
        args.layers,
        args.output_dir,
        run_native_pass=args.run_native_pass,
        native_pass_tool=args.native_pass_tool,
        onnx_mlir=args.onnx_mlir,
        mlir_opt=args.mlir_opt,
    )
    written = write_report_bundle(Path(args.output_dir), report)
    if args.verbose:
        print(
            f"[opt-ffn-native-diagnosis] native={report.native_proven}/{report.total_layers} "
            f"fallback_only={report.fallback_only} failed={report.failed} blockers={report.blockers_by_kind}"
        )
        for path in written:
            print(f"[opt-ffn-native-diagnosis] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
