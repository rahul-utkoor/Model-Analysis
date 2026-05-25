#!/usr/bin/env python
"""Explain selected pruning plan validation records."""

from __future__ import annotations

import argparse
import json
import sys

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain pruning plan validation records.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--contains")
    parser.add_argument("--status", choices=["valid", "warning", "invalid", "unknown"])
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def _matches(item: dict, args: argparse.Namespace) -> bool:
    if args.status and item.get("validation_status") != args.status:
        return False
    if args.failed_only and item.get("validation_status") not in {"invalid", "warning"}:
        return False
    if args.contains:
        needle = args.contains.lower()
        haystack = " ".join(
            [
                str(item.get("candidate_region_name", "")),
                str(item.get("plan_kind", "")),
                str(item.get("plan_id", "")),
                str(item.get("candidate_id", "")),
            ]
        ).lower()
        if needle not in haystack:
            return False
    return True


def _markdown(model: str, rows: list[dict]) -> str:
    lines = ["# Pruning Plan Validation Selection: " + model, "", "| plan | status | score | failed checks |", "| --- | --- | --- | --- |"]
    for item in rows:
        failed = ", ".join(check.get("check_type", "") for check in item.get("checks", []) if check.get("status") == "fail")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("candidate_region_name", "")).replace("|", "\\|"),
                    str(item.get("validation_status", "")),
                    str(item.get("validation_score", "")),
                    failed,
                ]
            )
            + " |"
        )
    lines.extend(["", "This is a static pruning-plan validation explanation. It does not modify models.", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        config = get_model_config(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    safe = safe_model_name(config["hf_id"])
    path = root / "reports" / "pruning_plan_validation" / f"{safe}.json"
    if not path.exists():
        print(f"[missing] Pruning plan validation missing. Run: python scripts/validate_pruning_plans.py --model {config['name']}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [item for item in data.get("validations", []) if _matches(item, args)][: args.limit]
    suffix_parts = []
    if args.status:
        suffix_parts.append(f"status_{args.status}")
    if args.failed_only:
        suffix_parts.append("failed_only")
    if args.contains:
        suffix_parts.append("contains_" + "_".join(args.contains.lower().split())[:60])
    suffix = "__selection__" + "__".join(suffix_parts) if suffix_parts else "__selection"
    out_json = root / "reports" / "pruning_plan_validation_explanations" / f"{safe}{suffix}.json"
    out_md = root / "reports" / "pruning_plan_validation_explanations" / f"{safe}{suffix}.md"
    write_json({"model_name": data.get("model_name"), "matches": rows}, out_json)
    write_markdown(_markdown(data.get("model_name", safe), rows), out_md)
    print(f"[pruning-plan-validation-explain] matches={len(rows)}")
    for item in rows:
        failed = ", ".join(check.get("check_type", "") for check in item.get("checks", []) if check.get("status") == "fail")
        print(f"- {item.get('candidate_region_name')} [{item.get('validation_status')}] score={item.get('validation_score')} failed={failed}")
    print(f"[pruning-plan-validation-explain] json={out_json}")
    print(f"[pruning-plan-validation-explain] markdown={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
