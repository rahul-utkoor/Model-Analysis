#!/usr/bin/env python
"""Compare generated full-model static analysis reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_analysis.cross_model_analysis_report import build_cross_model_analysis_report, write_cross_model_analysis_report
from model_analysis.cross_model_analysis_report_text import cross_model_report_to_markdown
from model_analysis.paths import get_project_root
from model_analysis.registry import get_model_config, list_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--output-root", default="reports/model_analysis_reports")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _model_names(value: str) -> list[str]:
    if value == "all":
        return [get_model_config(name)["hf_id"] for name in list_models()]
    return [get_model_config(item.strip())["hf_id"] for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    root = get_project_root()
    output_root = root / Path(args.output_root)
    report = build_cross_model_analysis_report(root, _model_names(args.models), output_root)
    write_cross_model_analysis_report(report, output_root, cross_model_report_to_markdown)
    if args.verbose:
        print(f"[model-analysis-compare] models={len(report.get('models', []))}")
        print(f"[model-analysis-compare] report={output_root / 'cross_model' / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
