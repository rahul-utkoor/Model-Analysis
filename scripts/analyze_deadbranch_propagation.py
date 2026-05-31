#!/usr/bin/env python
"""Analyze static deadbranch propagation opportunities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.deadbranch_propagation import analyze_deadbranch_propagation, deadbranch_report_to_dict, write_deadbranch_report
from model_analysis.deadbranch_propagation_text import deadbranch_report_to_markdown, deadbranch_report_to_text
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--output-dir", default="reports/deadbranch_propagation")
    parser.add_argument("--dump-dir", default="reports/deadbranch_propagation_dumps")
    parser.add_argument("--explain-dir", default="reports/deadbranch_propagation_explanations")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _configs(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_one(root: Path, config: dict, args: argparse.Namespace) -> bool:
    model = config["hf_id"]
    safe = safe_model_name(model)
    source = root / "reports" / "op_semantics" / f"{safe}.json"
    if not source.exists():
        print(f"[missing] Op Semantics missing. Run: python scripts/build_op_semantics.py --model {config['name']}", file=sys.stderr)
        return False
    report = deadbranch_report_to_dict(analyze_deadbranch_propagation(model, _load(source)))
    json_path = root / args.output_dir / f"{safe}.json"
    dump_path = root / args.dump_dir / f"{safe}.deadbranch"
    md_path = root / args.explain_dir / f"{safe}.md"
    write_deadbranch_report(report, json_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text(deadbranch_report_to_text(report), encoding="utf-8")
    write_markdown(deadbranch_report_to_markdown(report), md_path)
    if args.verbose:
        summary = report["summary"]
        print(f"[deadbranch] {model}")
        print(f"  pairs: {summary['total_pairs']}")
        print(f"  ffn_pairs: {summary['ffn_pairs']}")
        print(f"  attention_value_pairs: {summary['attention_value_pairs']}")
        print(f"  blocked_qk: {summary['query_key_blocked_pairs']}")
        print(f"  sparsegpt_alignment: {summary['sparsegpt_alignment_status']}")
        print(f"  json: {json_path}")
        print(f"  markdown: {md_path}")
    return True


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        configs = _configs(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    ok = True
    for config in configs:
        built = build_one(root, config, args)
        ok = built and ok
        if not built and args.strict:
            return 1
    return 0 if ok or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
