"""Readable textual dump for Region Pruning Semantics IR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir
from model_analysis.region_pruning_semantics import RegionPruningSemantics, region_pruning_semantics_to_dict


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def region_pruning_semantics_to_text(value: RegionPruningSemantics | dict[str, Any]) -> str:
    data = region_pruning_semantics_to_dict(value) if isinstance(value, RegionPruningSemantics) else value
    lines = [f'region_pruning_semantics @{_escape(data.get("model_name", "model"))} {{']
    for region in data.get("regions", []):
        lines.append(f'  region "{_escape(region.get("region_name", region.get("region_id")))}" [{_escape(region.get("region_type"))}] {{')
        lines.append(f'    role = {region.get("pruning_role", "unknown")}')
        if region.get("dimensions"):
            lines.append("    dims {")
            for dim in region["dimensions"]:
                lines.append(f'      {dim["dim_name"]} : {dim["status"]} // {dim["symbolic_role"]}')
            lines.append("    }")
        if region.get("propagation_rules"):
            lines.append("    rules {")
            for rule in region["propagation_rules"]:
                targets = ", ".join(rule.get("target_dimensions", []))
                lines.append(f'      {rule["index_mapping"]} {rule["source_dimension"]} -> {targets}')
            lines.append("    }")
        if region.get("repair_obligations"):
            lines.append("    repairs {")
            for repair in region["repair_obligations"]:
                required = "required" if repair.get("required") else "conditional"
                lines.append(f'      {repair["obligation_type"]} {required}')
            lines.append("    }")
        if region.get("blockers"):
            lines.append("    blockers {")
            for blocker in region["blockers"]:
                lines.append(f'      {blocker["blocker_type"]} {blocker["severity"]}')
            lines.append("    }")
        else:
            lines.append("    blockers { none }")
        lines.append("  }")
        lines.append("")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_region_pruning_semantics_text(value: RegionPruningSemantics | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(region_pruning_semantics_to_text(value), encoding="utf-8")
