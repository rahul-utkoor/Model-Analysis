#!/usr/bin/env python
"""Generate the cross-model static coverage study from status manifests."""

from __future__ import annotations

import argparse
import json

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.static_coverage_report import build_static_coverage_report, write_static_coverage_report
from model_analysis.static_coverage_report_text import static_coverage_report_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _models(value: str) -> list[str]:
    if value == "all":
        return [get_model_config(name)["hf_id"] for name in list_models()]
    return [get_model_config(item.strip())["hf_id"] for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    root = get_project_root()
    statuses = []
    for model in _models(args.models):
        path = root / "reports" / "static_pipeline_status" / f"{safe_model_name(model)}.json"
        if path.exists():
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            statuses.append(
                {
                    "model_name": model,
                    "final_status": "skipped",
                    "summary": {
                        "completed_stages": 0,
                        "skipped_stages": 0,
                        "failed_stages": 0,
                        "missing_artifacts": [str(path)],
                    },
                    "stages": [],
                    "artifacts": {},
                    "notes": ["Static pipeline status manifest missing."],
                }
            )
    report = build_static_coverage_report(root, statuses)
    write_static_coverage_report(
        report,
        root / "reports" / "static_coverage_study",
        static_coverage_report_to_markdown,
    )
    if args.verbose:
        summary = report["summary"]
        print(
            "[static-coverage] complete/partial/skipped/failed: "
            f"{summary['complete_models']}/{summary['partial_models']}/{summary['skipped_models']}/{summary['failed_models']}"
        )
        print(f"[static-coverage] report={root / 'reports' / 'static_coverage_study' / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
