#!/usr/bin/env python
"""Compare generated attention value-path artifact reports."""

from __future__ import annotations

import argparse
import json
import sys

from model_analysis.attention_value_path_subgraph_compare import attention_value_path_compare_to_markdown, compare_attention_value_path_reports
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
    parser.add_argument("--output-dir", default="reports/attention_value_path_subgraph_compare")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    root = get_project_root()
    reports = []
    for model in _models(args.models):
        path = root / "reports/attention_value_path_subgraphs" / safe_model_name(model) / "summary.json"
        if path.exists():
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:
        print("[error] no attention value-path reports found", file=sys.stderr)
        return 1
    data = compare_attention_value_path_reports(reports)
    output = root / args.output_dir
    write_json(data, output / "index.json")
    write_markdown(attention_value_path_compare_to_markdown(data), output / "index.md")
    if args.verbose:
        print(f"[attention-value-path-compare] models={data['num_models']}")
        print(f"[attention-value-path-compare] paths={data['summary']['total_paths']}")
        print(f"[attention-value-path-compare] report={output / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
