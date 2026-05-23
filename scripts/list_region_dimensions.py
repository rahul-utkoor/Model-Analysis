#!/usr/bin/env python
"""List RegionDimensionIR variables for region-aware legality analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List region-aware symbolic dimensions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--only-prunable", action="store_true")
    parser.add_argument("--only-blocked", action="store_true")
    parser.add_argument("--only-protected", action="store_true")
    parser.add_argument("--contains")
    parser.add_argument("--axis-role")
    parser.add_argument("--region-type")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _columns() -> list[str]:
    return ["var_id", "region_type", "region_name", "dim_name", "axis_role", "prunable", "protected", "blocked", "confidence"]


def _table(rows: list[dict]) -> str:
    columns = _columns()
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |" for row in rows)
    return "\n".join(lines)


def _markdown(model_name: str, rows: list[dict]) -> str:
    return "\n".join(
        [
            f"# Region Dimensions: {model_name}",
            "",
            _table(rows),
            "",
            "These variables belong to semantic structural regions and are listed for static legality analysis only.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    try:
        config = get_model_config(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    root = get_project_root()
    safe_name = safe_model_name(config["hf_id"])
    path = root / "reports" / "region_dimension_ir" / f"{safe_name}.json"
    if not path.exists():
        print(f"[missing] Region Dimension IR missing. Run: python scripts/build_region_dimension_ir.py --model {config['name']}", file=sys.stderr)
        return 1
    ir = _load_json(path)
    blocked = set(ir.get("blocked_dimensions", []))
    rows = sorted(ir.get("dimension_variables", []), key=lambda item: item.get("var_id", ""))
    if args.only_prunable:
        rows = [item for item in rows if item.get("prunable")]
    if args.only_blocked:
        rows = [item for item in rows if item.get("blocked") or item.get("var_id") in blocked]
    if args.only_protected:
        rows = [item for item in rows if item.get("protected")]
    if args.contains:
        needle = args.contains.lower()
        rows = [
            item for item in rows
            if any(needle in str(item.get(key, "")).lower() for key in ("var_id", "region_type", "region_name", "dim_name"))
        ]
    if args.axis_role:
        rows = [item for item in rows if item.get("axis_role") == args.axis_role]
    if args.region_type:
        rows = [item for item in rows if item.get("region_type") == args.region_type]
    if args.limit is not None:
        rows = rows[: args.limit]
    print("\t".join(_columns()))
    for item in rows:
        print("\t".join(str(item.get(column, "")) for column in _columns()))
    payload = {"model_name": ir.get("model_name"), "dimensions": rows}
    write_json(payload, root / "reports" / "region_blocked_analysis" / f"{safe_name}__dimension_list.json")
    write_markdown(_markdown(ir.get("model_name", config["name"]), rows), root / "reports" / "region_blocked_analysis" / f"{safe_name}__dimension_list.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
