"""Run an MLIR evidence coverage study over local model-analysis subgraphs."""

from __future__ import annotations

import argparse

from experimental.mlir_evidence_coverage.aggregate import aggregate_coverage
from experimental.mlir_evidence_coverage.discovery import build_default_coverage_cases
from experimental.mlir_evidence_coverage.report import write_report_bundle
from experimental.mlir_evidence_coverage.runner import CoverageRunOptions, run_coverage_case


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="default", help="default, all, or a comma-separated model list.")
    parser.add_argument("--layers", choices=["layer0", "all"], default="layer0")
    parser.add_argument("--patterns", default="all", help="all or comma-separated CoveragePatternKind values.")
    parser.add_argument("--output-dir", default="reports/mlir_evidence_coverage")
    parser.add_argument("--format", choices=["markdown", "json", "both"], default="both")
    parser.add_argument("--no-native-pass", action="store_true")
    parser.add_argument("--native-pass-tool")
    parser.add_argument("--onnx-mlir")
    parser.add_argument("--mlir-opt")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cases = build_default_coverage_cases(args.models, args.layers, args.patterns)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    options = CoverageRunOptions(
        output_root=f"{args.output_dir}/artifacts",
        run_native_pass=not args.no_native_pass,
        native_pass_tool=args.native_pass_tool,
        onnx_mlir=args.onnx_mlir,
        mlir_opt=args.mlir_opt,
        verbose=args.verbose,
    )
    results = []
    for case in cases:
        if args.verbose:
            print(f"[mlir-coverage] analyze {case.case_id}: {case.onnx_path}")
        result = run_coverage_case(case, options)
        results.append(result)
        if args.verbose:
            print(f"[mlir-coverage] {case.case_id}: tier={result.evidence_tier.value} verdict={result.verdict.value}")
    aggregate = aggregate_coverage(results)
    written = write_report_bundle(args.output_dir, results, aggregate, args.format)
    print(
        f"[mlir-coverage] cases={aggregate.total_cases} found={aggregate.found_cases} missing={aggregate.missing_cases} "
        f"native={aggregate.native_proven} access={aggregate.access_proven} fallback={aggregate.fallback_proven} "
        f"blocked={aggregate.blocked_as_expected} partial={aggregate.partial} unknown={aggregate.unknown} failed={aggregate.failed}"
    )
    for path in written:
        print(f"[mlir-coverage] wrote {path}")
    return 0 if aggregate.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
