#!/usr/bin/env python
"""Explain blocked pruning regions from Dimension IR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.ir_analysis import explain_blocked_regions
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain Dimension IR blocked regions.")
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def _selected(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


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
            f"# Blocked Pruning Regions: {model_name}",
            "",
            "## Summary",
            "",
            f"- Blocked items: `{len(rows)}`",
            "",
            "## Blocked Dimensions",
            "",
            _table(rows, ["blocked_id", "dimension_var_id", "constraint_id", "block_type", "severity", "explanation"]),
            "",
            "## Blocking Constraints",
            "",
            _table(rows, ["constraint_id", "block_type", "severity", "explanation"]),
            "",
            "## Mitigations",
            "",
            _table(rows, ["blocked_id", "mitigation"]),
            "",
            "## Interpretation",
            "",
            "Blocked regions are static Dimension-IR evidence. They explain why pruning cannot be considered structurally legal without stronger mapping, repair, or equality evidence.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        configs = _selected(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    failed = False
    for config in configs:
        safe_name = safe_model_name(config["hf_id"])
        ir_path = root / "reports" / "dimension_ir" / f"{safe_name}.json"
        if not ir_path.exists():
            print(f"[missing] Dimension IR missing. Run: python scripts/build_dimension_ir.py --model {config['name']}", file=sys.stderr)
            failed = True
            continue
        ir = _load_json(ir_path)
        rows = explain_blocked_regions(ir)
        write_json({"model_name": ir.get("model_name"), "blocked_regions": rows}, root / "reports" / "ir_analysis" / f"{safe_name}__blocked_regions.json")
        write_markdown(_markdown(ir.get("model_name"), rows), root / "reports" / "ir_analysis" / f"{safe_name}__blocked_regions.md")
        print(f"[blocked-regions] {config['name']}: {len(rows)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
