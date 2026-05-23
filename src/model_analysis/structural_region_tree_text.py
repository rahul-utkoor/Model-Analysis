"""Readable compiler-style dump for Structural Region Trees."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir
from model_analysis.structural_region_tree import StructuralRegionTree, structural_region_tree_to_dict


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def structural_region_tree_to_text(tree: StructuralRegionTree | dict) -> str:
    data = structural_region_tree_to_dict(tree) if isinstance(tree, StructuralRegionTree) else tree
    regions = {item["region_id"]: item for item in data.get("regions", [])}
    interfaces = {item["region_id"]: item for item in data.get("interfaces", [])}
    lines = [
        f'region.module @{_escape(data.get("model_name", "model"))} frontend("{_escape(data.get("source_frontend", "unknown"))}") {{'
    ]

    def emit(region_id: str, indent: int) -> None:
        region = regions[region_id]
        interface = interfaces.get(region_id, {})
        pad = " " * indent
        op_list = ", ".join(f"%{item}" for item in region.get("op_ids", []))
        children = region.get("children", [])
        header = (
            f'{pad}region %{region_id} {region.get("region_type")} ops({op_list}) '
            f'role("{_escape(interface.get("pruning_role", "unknown"))}") '
            f'confidence("{_escape(region.get("confidence", "low"))}")'
        )
        if not children:
            lines.append(header)
            return
        lines.append(header + " {")
        for constraint in interface.get("constraints", []):
            lines.append(f'{pad}  constraint("{_escape(constraint.get("type", "unknown"))}")')
        for child in children:
            emit(child, indent + 2)
        lines.append(f"{pad}}}")

    emit(data["root_region_id"], 2)
    lines.extend(["}", ""])
    return "\n".join(lines)


def write_structural_region_tree_text(tree: StructuralRegionTree | dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(structural_region_tree_to_text(tree), encoding="utf-8")
