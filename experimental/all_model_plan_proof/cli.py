"""Generate all-model static pruning propagation plan proof reports."""

from __future__ import annotations

import argparse

from experimental.all_model_plan_proof.report import write_report_bundle
from experimental.all_model_plan_proof.runner import AllModelRunOptions, run_all_model_proof


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="all, default, or a comma-separated model list.")
    parser.add_argument("--layers", choices=["all", "layer0"], default="layer0")
    parser.add_argument("--output-dir", default="reports/all_model_plan_proof")
    parser.add_argument("--format", choices=["markdown", "json", "both"], default="both")
    parser.add_argument("--build-missing-value-paths", action="store_true")
    parser.add_argument("--no-native-pass", action="store_true")
    parser.add_argument("--native-pass-tool")
    parser.add_argument("--onnx-mlir")
    parser.add_argument("--mlir-opt")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        proof = run_all_model_proof(
            args.models,
            AllModelRunOptions(
                layers=args.layers,
                output_root=f"{args.output_dir}/artifacts",
                build_missing_value_paths=args.build_missing_value_paths,
                run_native_pass=not args.no_native_pass,
                native_pass_tool=args.native_pass_tool,
                onnx_mlir=args.onnx_mlir,
                mlir_opt=args.mlir_opt,
                verbose=args.verbose,
            ),
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    written = write_report_bundle(args.output_dir, proof, args.format)
    aggregate = proof.aggregate
    print(
        f"[all-model-proof] expected={aggregate.total_expected} proven={aggregate.total_proven} "
        f"native={aggregate.native_evidence_count} access={aggregate.access_evidence_count} "
        f"fallback={aggregate.fallback_count} partial={aggregate.partial_count} missing={aggregate.missing_count} "
        f"unsupported={aggregate.unsupported_count} failed={aggregate.failed_count}"
    )
    for path in written:
        print(f"[all-model-proof] wrote {path}")
    return 0 if aggregate.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
