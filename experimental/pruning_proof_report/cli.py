"""Generate a cross-evidence pruning proof report for selected local subgraphs."""

from __future__ import annotations

import argparse

from experimental.pruning_proof_report.aggregate import aggregate_evidence
from experimental.pruning_proof_report.config import proof_cases, select_case
from experimental.pruning_proof_report.report import write_report_bundle
from experimental.pruning_proof_report.runner import ProofRunOptions, run_proof_case


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=["default", "all"], default="default")
    parser.add_argument("--case", help="Run one configured proof case by ID.")
    parser.add_argument("--output-dir", default="reports/pruning_proof_report")
    parser.add_argument("--format", choices=["markdown", "json", "both"], default="both")
    parser.add_argument("--no-mlir", action="store_true", help="Use the ONNX-local bridge without ONNX-MLIR lowering.")
    parser.add_argument("--no-native-pass", action="store_true", help="Skip the optional local native dependence executable.")
    parser.add_argument("--native-pass-tool")
    parser.add_argument("--onnx-mlir")
    parser.add_argument("--mlir-opt")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cases = select_case(proof_cases(args.models), args.case)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    options = ProofRunOptions(
        use_mlir=not args.no_mlir,
        run_native_pass=not args.no_native_pass,
        native_pass_tool=args.native_pass_tool,
        onnx_mlir=args.onnx_mlir,
        mlir_opt=args.mlir_opt,
        output_root=f"{args.output_dir}/artifacts",
        verbose=args.verbose,
    )
    evidence = []
    for case in cases:
        if args.verbose:
            print(f"[pruning-proof] analyze {case.case_id}: {case.onnx_path}")
        item = run_proof_case(case, options)
        evidence.append(item)
        if args.verbose:
            print(f"[pruning-proof] {case.case_id}: source={item.evidence_source} verdict={item.verdict}")
    aggregate = aggregate_evidence(evidence)
    written = write_report_bundle(args.output_dir, evidence, aggregate, args.format)
    print(
        f"[pruning-proof] cases={aggregate.cases_total} found={aggregate.cases_found} "
        f"missing={aggregate.cases_missing} proven={aggregate.proven} fallback={aggregate.fallback_proven} "
        f"blocked={aggregate.blocked} partial={aggregate.partial} unknown={aggregate.unknown} failed={aggregate.failed}"
    )
    for path in written[:2]:
        print(f"[pruning-proof] wrote {path}")
    return 0 if aggregate.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
