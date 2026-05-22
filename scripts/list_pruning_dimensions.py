#!/usr/bin/env python
"""List Dimension IR variables for pruning legality checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List pruning Dimension IR variables.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--only-prunable", action="store_true")
    parser.add_argument("--only-blocked", action="store_true")
    parser.add_argument("--contains", help="Substring filter over var_id/owner/dim.")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _markdown(model_name: str, rows: list[dict]) -> str:
    return "\n".join(
        [
            f"# Pruning Dimensions: {model_name}",
            "",
            _table(rows, ["var_id", "owner_name", "owner_type", "dim_name", "size", "semantic_role", "prunable", "confidence"]),
            "",
        ]
    )


def _print_rows(rows: list[dict]) -> None:
    columns = ["var_id", "owner_name", "owner_type", "dim_name", "size", "semantic_role", "prunable", "confidence"]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row.get(column, "")) for column in columns))


def main() -> int:
    args = parse_args()
    try:
        config = get_model_config(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    root = get_project_root()
    safe_name = safe_model_name(config["hf_id"])
    ir_path = root / "reports" / "dimension_ir" / f"{safe_name}.json"
    if not ir_path.exists():
        print(f"[missing] Dimension IR missing. Run: python scripts/build_dimension_ir.py --model {config['name']}", file=sys.stderr)
        return 1
    ir = _load_json(ir_path)
    rows = list(ir.get("dimension_variables", []))
    if args.only_prunable:
        rows = [row for row in rows if row.get("prunable")]
    if args.only_blocked:
        rows = [row for row in rows if row.get("semantic_role") == "blocked" or row.get("var_id") in set(ir.get("blocked_dimensions", []))]
    if args.contains:
        needle = args.contains.lower()
        rows = [
            row
            for row in rows
            if needle in row.get("var_id", "").lower()
            or needle in row.get("owner_name", "").lower()
            or needle in row.get("dim_name", "").lower()
        ]
    if args.limit is not None:
        rows = rows[: args.limit]
    _print_rows(rows)
    payload = {"model_name": ir.get("model_name"), "dimensions": rows}
    write_json(payload, root / "reports" / "ir_analysis" / f"{safe_name}__dimension_list.json")
    write_markdown(_markdown(ir.get("model_name"), rows), root / "reports" / "ir_analysis" / f"{safe_name}__dimension_list.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
