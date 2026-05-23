#!/usr/bin/env python
"""Explain protected and unresolved constraints in RegionDimensionIR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.region_ir_analysis import explain_region_blocked_dimensions
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain blocked RegionDimensionIR variables.")
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def _selected(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(item.get(column, "")).replace("|", "\\|") for column in columns) + " |" for item in rows)
    return "\n".join(lines)


def _markdown(model_name: str, rows: list[dict]) -> str:
    return "\n".join(
        [
            f"# Blocked Region Dimensions: {model_name}",
            "",
            "## Summary",
            "",
            f"- Blocked/protected or unresolved dimension obligations: `{len(rows)}`",
            "",
            "## Dimensions",
            "",
            _table(rows, ["blocked_id", "region_type", "dimension_var_id", "dim_name", "axis_role", "block_type", "severity", "explanation", "mitigation"]),
            "",
            "## Interpretation",
            "",
            "These explanations come from semantic region constraints. They are static diagnostics and do not modify models.",
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
        path = root / "reports" / "region_dimension_ir" / f"{safe_name}.json"
        if not path.exists():
            print(f"[missing] Region Dimension IR missing. Run: python scripts/build_region_dimension_ir.py --model {config['name']}", file=sys.stderr)
            failed = True
            continue
        ir = _load_json(path)
        rows = explain_region_blocked_dimensions(ir)
        write_json({"model_name": ir.get("model_name"), "blocked_dimensions": rows}, root / "reports" / "region_blocked_analysis" / f"{safe_name}__blocked_dimensions.json")
        write_markdown(_markdown(ir.get("model_name", config["name"]), rows), root / "reports" / "region_blocked_analysis" / f"{safe_name}__blocked_dimensions.md")
        print(f"[region-blocked-dimensions] {config['name']}: {len(rows)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
