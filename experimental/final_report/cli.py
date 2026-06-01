"""Generate the final static pruning propagation research report."""

from __future__ import annotations

import argparse

from experimental.final_report.collector import collect_final_report_data
from experimental.final_report.report import write_report_bundle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/final")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        data = collect_final_report_data(strict=args.strict)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2
    written = write_report_bundle(args.output_dir, data)
    aggregate = data.aggregate_summary
    print(
        f"[final-report] proven={aggregate.proven_plans}/{aggregate.expected_plans} "
        f"native={aggregate.native_mlir_evidence} fallback={aggregate.high_level_mlir_fallback} "
        f"unsupported={aggregate.unsupported} partial={aggregate.partial} missing={aggregate.missing} failed={aggregate.failed}"
    )
    if args.verbose:
        for path in written:
            print(f"[final-report] wrote {path}")
        for warning in data.warnings:
            print(f"[final-report] warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
