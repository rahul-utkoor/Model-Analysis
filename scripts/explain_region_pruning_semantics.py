#!/usr/bin/env python
"""Explain selected region pruning semantics records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain region pruning semantics.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--contains", default=None)
    parser.add_argument("--region-type", default=None)
    parser.add_argument("--blocked-only", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _matches(region: dict, args: argparse.Namespace) -> bool:
    if args.contains:
        haystack = " ".join([region.get("region_name", ""), region.get("region_id", ""), region.get("section", "")]).lower()
        if args.contains.lower() not in haystack:
            return False
    if args.region_type and region.get("region_type") != args.region_type:
        return False
    if args.blocked_only:
        has_hard_blocker = any(blocker.get("severity") == "blocker" for blocker in region.get("blockers", []))
        if not (has_hard_blocker or region.get("pruning_role") in {"blocked", "protected", "constraint_carrier"}):
            return False
    return True


def _markdown(model: str, rows: list[dict], args: argparse.Namespace) -> str:
    lines = [f"# Region Pruning Semantics Explanation: {model}", "", f"- Matches: `{len(rows)}`", ""]
    for region in rows:
        lines.extend(
            [
                f"## {region.get('region_name')}",
                "",
                f"- Region id: `{region.get('region_id')}`",
                f"- Type: `{region.get('region_type')}`",
                f"- Role: `{region.get('pruning_role')}`",
                f"- Section: `{region.get('section')}`",
                f"- Op range: `{region.get('op_range')}`",
                "",
                "### Dimensions",
                "",
            ]
        )
        for dim in region.get("dimensions", []):
            lines.append(f"- `{dim['dim_name']}`: `{dim['status']}` ({dim['symbolic_role']}) - {dim.get('reason', '')}")
        lines.extend(["", "### Repairs", ""])
        for repair in region.get("repair_obligations", []):
            lines.append(f"- `{repair['obligation_type']}` required={repair.get('required')} - {repair.get('explanation', '')}")
        lines.extend(["", "### Blockers", ""])
        for blocker in region.get("blockers", []):
            lines.append(f"- `{blocker['blocker_type']}` ({blocker['severity']}) - {blocker.get('explanation', '')}")
        lines.append("")
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
    path = root / "reports" / "region_pruning_semantics" / f"{safe}.json"
    if not path.exists():
        print(f"[missing] Region pruning semantics missing. Run: python scripts/build_region_pruning_semantics.py --model {config['name']}", file=sys.stderr)
        return 1
    report = _load(path)
    rows = [region for region in report.get("regions", []) if _matches(region, args)]
    rows = rows[: args.limit]
    stem = "__blocked" if args.blocked_only else "__selection"
    if args.contains:
        stem += "__contains_" + "".join(ch if ch.isalnum() else "_" for ch in args.contains.lower()).strip("_")
    if args.region_type:
        stem += "__" + args.region_type
    out_json = root / "reports" / "region_pruning_semantics_explanations" / f"{safe}{stem}.json"
    out_md = root / "reports" / "region_pruning_semantics_explanations" / f"{safe}{stem}.md"
    write_json({"model_name": report.get("model_name"), "matches": rows}, out_json)
    write_markdown(_markdown(report.get("model_name", args.model), rows, args), out_md)
    print(f"[region-pruning-semantics-explain] matches={len(rows)}")
    for region in rows:
        print(f"- {region.get('region_name')} [{region.get('region_type')}] role={region.get('pruning_role')}")
    print(f"[region-pruning-semantics-explain] json={out_json}")
    print(f"[region-pruning-semantics-explain] markdown={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
