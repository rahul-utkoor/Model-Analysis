#!/usr/bin/env python
"""Compare static deadbranch propagation reports across models."""

from __future__ import annotations

import argparse
import json
import sys

from model_analysis.deadbranch_propagation_compare import compare_deadbranch_reports, deadbranch_compare_to_markdown
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown


def _models(value: str) -> list[str]:
    if value == "all":
        return [get_model_config(name)["hf_id"] for name in list_models()]
    return [get_model_config(item.strip())["hf_id"] for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    root = get_project_root()
    reports = []
    for model in _models(args.models):
        path = root / "reports" / "deadbranch_propagation" / f"{safe_model_name(model)}.json"
        if not path.exists():
            print(f"[missing] {path}", file=sys.stderr)
            continue
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:
        print("[error] no deadbranch reports found", file=sys.stderr)
        return 1
    comparison = compare_deadbranch_reports(reports)
    out = root / "reports" / "deadbranch_propagation_compare"
    write_json(comparison, out / "index.json")
    write_markdown(deadbranch_compare_to_markdown(comparison), out / "index.md")
    if args.verbose:
        print(f"[deadbranch-compare] models={comparison['num_models']}")
        print(f"[deadbranch-compare] pairs={comparison['summary']['total_pairs']}")
        print(f"[deadbranch-compare] report={out / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
