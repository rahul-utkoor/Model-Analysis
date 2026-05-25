#!/usr/bin/env python
"""Build/check static pipeline status for configured models."""

from __future__ import annotations

import argparse

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.static_coverage_report import build_static_coverage_report, write_static_coverage_report
from model_analysis.static_coverage_report_text import static_coverage_report_to_markdown
from model_analysis.static_pipeline_orchestrator import build_static_pipeline_for_models, configured_model_names
from model_analysis.static_pipeline_status import write_model_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--build-missing-analysis", action="store_true")
    parser.add_argument("--build-layer-packs", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = get_project_root()
    names = configured_model_names(args.models)
    statuses = build_static_pipeline_for_models(
        root=root,
        models=names,
        build_missing_analysis=args.build_missing_analysis,
        build_layer_packs=args.build_layer_packs,
        strict=args.strict,
        verbose=args.verbose,
    )
    out = root / "reports" / "static_pipeline_status"
    for status in statuses:
        safe = safe_model_name(status["model_name"])
        write_model_status(status, out / f"{safe}.json", out / f"{safe}.md")
    report = build_static_coverage_report(root, statuses)
    write_static_coverage_report(
        report,
        root / "reports" / "static_coverage_study",
        static_coverage_report_to_markdown,
    )
    if args.verbose:
        summary = report["summary"]
        print(
            "[static-pipeline-all] complete/partial/skipped/failed: "
            f"{summary['complete_models']}/{summary['partial_models']}/{summary['skipped_models']}/{summary['failed_models']}"
        )
        print(f"[static-pipeline-all] coverage: {root / 'reports' / 'static_coverage_study' / 'index.md'}")
        for status in statuses:
            config = get_model_config(status["configured_model"]["name"])
            del config
            print(f"  {status['model_name']}: {status['final_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
