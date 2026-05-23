#!/usr/bin/env python
"""Build region-aware Dimension IR from Structural Region Tree reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.region_dimension_ir import (
    build_region_dimension_ir,
    region_dimension_ir_to_dict,
    region_dimension_ir_to_markdown,
    write_region_dimension_ir_json,
)
from model_analysis.region_dimension_ir_text import write_region_dimension_ir_text
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build region-aware symbolic Dimension IR.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--format", choices=["json", "md", "text", "all"], default="all")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-on-unresolved", action="store_true")
    return parser.parse_args()


def _selected_models(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _table(rows: list[dict], columns: list[str], limit: int = 300) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def _constraints_markdown(model_name: str, equations: list[dict]) -> str:
    return "\n".join(
        [
            f"# Region Constraint Equations: {model_name}",
            "",
            _table(equations, ["constraint_id", "region_id", "region_type", "lhs", "relation", "rhs", "constraint_type", "blocking", "confidence"]),
            "",
            "These equations are derived from structural region interfaces and remain conservative static evidence.",
            "",
        ]
    )


def _equivalence_markdown(model_name: str, classes: list[dict]) -> str:
    return "\n".join(
        [
            f"# Region Dimension Equivalence Classes: {model_name}",
            "",
            _table(classes, ["class_id", "class_type", "representative", "members", "size", "confidence", "constraints"]),
            "",
        ]
    )


def _write_outputs(root: Path, safe_name: str, ir, output_format: str) -> None:
    data = region_dimension_ir_to_dict(ir)
    if output_format in {"json", "all"}:
        write_region_dimension_ir_json(ir, root / "reports" / "region_dimension_ir" / f"{safe_name}.json")
        write_json({"model_name": ir.model_name, "constraint_equations": data["constraint_equations"], "summary": data["summary"]}, root / "reports" / "region_constraint_equations" / f"{safe_name}.json")
        write_json({"model_name": ir.model_name, "equivalence_classes": data["equivalence_classes"], "summary": data["summary"]}, root / "reports" / "region_dimension_equivalence" / f"{safe_name}.json")
    if output_format in {"md", "all"}:
        write_markdown(region_dimension_ir_to_markdown(ir), root / "reports" / "region_dimension_ir" / f"{safe_name}.md")
        write_markdown(_constraints_markdown(ir.model_name, data["constraint_equations"]), root / "reports" / "region_constraint_equations" / f"{safe_name}.md")
        write_markdown(_equivalence_markdown(ir.model_name, data["equivalence_classes"]), root / "reports" / "region_dimension_equivalence" / f"{safe_name}.md")
    if output_format in {"text", "all"}:
        write_region_dimension_ir_text(ir, root / "reports" / "region_pruning_ir_dumps" / f"{safe_name}.rdim")


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        configs = _selected_models(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    failed = False
    for config in configs:
        safe_name = safe_model_name(config["hf_id"])
        source_path = root / "reports" / "structural_region_trees" / f"{safe_name}.json"
        if not source_path.exists():
            print(f"[missing] Structural Region Tree missing. Run: python scripts/build_structural_region_tree.py --model {config['name']}", file=sys.stderr)
            failed = True
            continue
        ir = build_region_dimension_ir(_load_json(source_path))
        _write_outputs(root, safe_name, ir, args.format)
        if args.verbose:
            summary = ir.summary
            print(f"[region-dimension-ir] {ir.model_name}")
            print(f"  dimensions: {summary['num_dimension_variables']}")
            print(f"  equations: {summary['num_constraint_equations']}")
            print(f"  equivalence_classes: {summary['num_equivalence_classes']}")
            print(f"  blocked_dimensions: {summary['num_blocked_dimensions']}")
            print(f"  unresolved: {summary['num_unresolved_constraints']}")
        if args.fail_on_unresolved and ir.unresolved_constraints:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
