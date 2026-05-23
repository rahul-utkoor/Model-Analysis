"""Readable textual dump for region-aware Dimension IR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir
from model_analysis.region_dimension_ir import RegionDimensionIR, region_dimension_ir_to_dict


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def region_dimension_ir_to_text(ir: RegionDimensionIR | dict) -> str:
    data = region_dimension_ir_to_dict(ir) if isinstance(ir, RegionDimensionIR) else ir
    lines = [
        f'region_dim.module @{_escape(data.get("model_name", "model"))} frontend("{_escape(data.get("source_frontend", "unknown"))}") {{'
    ]
    for dimension in sorted(data.get("dimension_variables", []), key=lambda item: item["var_id"]):
        lines.extend(
            [
                f'  region_dim %{dimension["var_id"]} region("%{dimension["region_id"]}") type("{_escape(dimension["region_type"])}")',
                f'      name("{_escape(dimension["dim_name"])}") axis("{_escape(dimension["axis_role"])}") size("{_escape(dimension.get("size"))}")',
                f'      prunable({str(dimension["prunable"]).lower()}) protected({str(dimension["protected"]).lower()}) propagated({str(dimension["propagated"]).lower()}) blocked({str(dimension["blocked"]).lower()})',
            ]
        )
    for constraint in sorted(data.get("constraint_equations", []), key=lambda item: item["constraint_id"]):
        lines.extend(
            [
                "",
                f'  region_constraint %{constraint["constraint_id"]} {constraint["relation"]}(%{constraint["lhs"]}, %{constraint["rhs"]})',
                f'      region("%{constraint["region_id"]}") type("{_escape(constraint["constraint_type"])}")',
                f'      blocking({str(constraint["blocking"]).lower()}) confidence("{_escape(constraint["confidence"])}")',
            ]
        )
    for equivalent in sorted(data.get("equivalence_classes", []), key=lambda item: item["class_id"]):
        members = ", ".join(f"%{item}" for item in equivalent["members"])
        lines.extend(
            [
                "",
                f"  region_eq_class %{equivalent['class_id']} members({members})",
                f'      type("{_escape(equivalent["class_type"])}") size("{_escape(equivalent.get("size"))}") confidence("{_escape(equivalent["confidence"])}")',
            ]
        )
    for dimension in data.get("blocked_dimensions", []):
        lines.append(f"  // blocked: %{dimension}")
    for constraint in data.get("unresolved_constraints", []):
        lines.append(f"  // unresolved: %{constraint}")
    lines.extend(["}", ""])
    return "\n".join(lines)


def write_region_dimension_ir_text(ir: RegionDimensionIR | dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(region_dimension_ir_to_text(ir), encoding="utf-8")
