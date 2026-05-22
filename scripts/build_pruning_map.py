#!/usr/bin/env python
"""Build compiler-style pruning opportunity maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.pruning_opportunity import (
    build_model_pruning_map,
    model_pruning_map_to_dict,
    model_pruning_map_to_markdown,
)
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_json, write_markdown


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compiler-style pruning opportunity maps.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--require-validation", action="store_true", help="Fail if validated dependency graph report is missing.")
    parser.add_argument("--format", choices=["json", "md", "both"], default="both")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _selected_models(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def _write_outputs(root: Path, safe_name: str, model_map, output_format: str) -> None:
    data = model_pruning_map_to_dict(model_map)
    if output_format in {"json", "both"}:
        write_json(data, root / "reports" / "model_pruning_maps" / f"{safe_name}.json")
        write_json({"model_name": model_map.model_name, "opportunities": data["opportunities"]}, root / "reports" / "pruning_opportunities" / f"{safe_name}.json")
        write_json({"model_name": model_map.model_name, "constraints": data["propagation_constraints"]}, root / "reports" / "propagation_constraints" / f"{safe_name}.json")
        write_json({"model_name": model_map.model_name, "structural_risks": data["structural_risks"]}, root / "reports" / "structural_risk_maps" / f"{safe_name}.json")
    if output_format in {"md", "both"}:
        write_markdown(model_pruning_map_to_markdown(model_map), root / "reports" / "model_pruning_maps" / f"{safe_name}.md")
        write_markdown(_opportunities_markdown(model_map), root / "reports" / "pruning_opportunities" / f"{safe_name}.md")
        write_markdown(_constraints_markdown(model_map), root / "reports" / "propagation_constraints" / f"{safe_name}.md")
        write_markdown(_risks_markdown(model_map), root / "reports" / "structural_risk_maps" / f"{safe_name}.md")


def _table(rows: list[dict], columns: list[str], limit: int = 250) -> str:
    if not rows:
        return "_None._"
    selected = rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _opportunities_markdown(model_map) -> str:
    rows = model_pruning_map_to_dict(model_map)["opportunities"]
    return "\n".join(
        [
            f"# Pruning Opportunities: {model_map.model_name}",
            "",
            _table(rows, ["opportunity_id", "opportunity_type", "root_unit_name", "risk_level", "executability", "confidence", "reason"]),
            "",
        ]
    )


def _constraints_markdown(model_map) -> str:
    rows = model_pruning_map_to_dict(model_map)["propagation_constraints"]
    return "\n".join(
        [
            f"# Propagation Constraints: {model_map.model_name}",
            "",
            _table(rows, ["constraint_id", "src_dim_id", "dst_dim_id", "constraint_type", "direction", "edge_type", "confidence", "reason"]),
            "",
        ]
    )


def _risks_markdown(model_map) -> str:
    rows = model_pruning_map_to_dict(model_map)["structural_risks"]
    return "\n".join(
        [
            f"# Structural Risk Map: {model_map.model_name}",
            "",
            _table(rows, ["risk_id", "risk_type", "affected_units", "severity", "reason", "mitigation"]),
            "",
        ]
    )


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
        graph_path = root / "reports" / "dependency_graphs" / f"{safe_name}.json"
        validation_path = root / "reports" / "validated_dependency_graphs" / f"{safe_name}.json"
        if not graph_path.exists():
            print(f"[missing] Dependency graph missing. Run: python scripts/build_dependency_graph.py --model {config['name']}", file=sys.stderr)
            failed = True
            continue
        if args.require_validation and not validation_path.exists():
            print(f"[missing] Validation report missing. Run: python scripts/build_correspondence.py --model {config['name']}", file=sys.stderr)
            failed = True
            continue
        validation = _load_json(validation_path) if validation_path.exists() else None
        model_map = build_model_pruning_map(_load_json(graph_path), validation)
        _write_outputs(root, safe_name, model_map, args.format)
        if args.verbose:
            summary = model_map.summary
            print(f"[pruning-map] {config['name']}")
            print(f"  dimensions: {summary['num_pruning_dimensions']}")
            print(f"  constraints: {summary['num_constraints']}")
            print(f"  opportunities: {summary['num_opportunities']}")
            print(f"  blocked: {summary['num_blocked_opportunities']}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
