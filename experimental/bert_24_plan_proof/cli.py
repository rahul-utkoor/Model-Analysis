"""Generate the BERT 24-plan static propagation proof report."""

from __future__ import annotations

import argparse

from experimental.bert_24_plan_proof.report import write_report_bundle
from experimental.bert_24_plan_proof.runner import run_bert_24_plan_proof


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/bert_24_plan_proof")
    parser.add_argument("--plans", default="reports/pruning_plans/bert-base-uncased.json")
    parser.add_argument("--validations", default="reports/pruning_plan_validation/bert-base-uncased.json")
    parser.add_argument("--value-paths", default="reports/attention_value_path_subgraphs/bert-base-uncased/summary.json")
    parser.add_argument("--coverage", default="reports/mlir_evidence_coverage_bert_24_plan/index.json")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        proof = run_bert_24_plan_proof(
            plans_path=args.plans,
            validations_path=args.validations,
            value_paths_path=args.value_paths,
            coverage_path=args.coverage,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    json_path, md_path = write_report_bundle(args.output_dir, proof)
    summary = proof.summary
    print(
        f"[bert-24-plan-proof] expected={summary.expected_plans} proven={summary.total_proven} "
        f"missing={summary.missing} partial={summary.partial} failed={summary.failed} "
        f"verdict={summary.final_verdict}"
    )
    if args.verbose:
        print(f"[bert-24-plan-proof] wrote {json_path}")
        print(f"[bert-24-plan-proof] wrote {md_path}")
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
