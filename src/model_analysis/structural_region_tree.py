"""Hierarchical structural-region tree data model over Tensor Graph IR."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class StructuralRegion:
    region_id: str
    region_type: str
    name: str
    op_ids: list[str]
    value_ids: list[str]
    children: list[str]
    parent: str | None
    entry_ops: list[str]
    exit_ops: list[str]
    input_values: list[str]
    output_values: list[str]
    internal_values: list[str]
    boundary_input_values: list[str]
    boundary_output_values: list[str]
    contains_fork: bool
    contains_join: bool
    single_entry: bool
    single_exit: bool
    region_depth: int
    confidence: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralRegionInterface:
    region_id: str
    region_type: str
    prunable_dimensions: list[dict[str, Any]]
    protected_dimensions: list[dict[str, Any]]
    propagated_dimensions: list[dict[str, Any]]
    blocked_dimensions: list[dict[str, Any]]
    constraints: list[dict[str, Any]]
    pruning_role: str
    reason: str


@dataclass
class StructuralRegionTree:
    model_name: str
    source_frontend: str
    root_region_id: str
    regions: list[StructuralRegion]
    interfaces: list[StructuralRegionInterface]
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def structural_region_to_dict(region: StructuralRegion) -> dict[str, Any]:
    return asdict(region)


def structural_region_interface_to_dict(interface: StructuralRegionInterface) -> dict[str, Any]:
    return asdict(interface)


def structural_region_tree_to_dict(tree: StructuralRegionTree) -> dict[str, Any]:
    return asdict(tree)


def write_structural_region_tree_json(tree: StructuralRegionTree, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(structural_region_tree_to_dict(tree), indent=2), encoding="utf-8")


def load_structural_region_tree_json(path: Path) -> StructuralRegionTree:
    data = json.loads(path.read_text(encoding="utf-8"))
    return StructuralRegionTree(
        model_name=data["model_name"],
        source_frontend=data.get("source_frontend", "unknown"),
        root_region_id=data["root_region_id"],
        regions=[StructuralRegion(**item) for item in data.get("regions", [])],
        interfaces=[StructuralRegionInterface(**item) for item in data.get("interfaces", [])],
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 400) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(_cell(row.get(column, "")) for column in columns) + " |")
    if len(rows) > limit:
        lines.append(f"| ... | {len(rows) - limit} more rows omitted |" + " |" * (len(columns) - 2))
    return "\n".join(lines)


def structural_region_tree_to_markdown(tree: StructuralRegionTree | dict) -> str:
    data = structural_region_tree_to_dict(tree) if isinstance(tree, StructuralRegionTree) else tree
    summary = data.get("summary", {})
    regions = [
        {
            "region_id": item.get("region_id"),
            "region_type": item.get("region_type"),
            "ops": len(item.get("op_ids", [])),
            "children": len(item.get("children", [])),
            "depth": item.get("region_depth"),
            "fork": item.get("contains_fork"),
            "join": item.get("contains_join"),
            "confidence": item.get("confidence"),
        }
        for item in data.get("regions", [])
    ]
    interfaces = [
        {
            "region_id": item.get("region_id"),
            "region_type": item.get("region_type"),
            "pruning_role": item.get("pruning_role"),
            "prunable_dimensions": item.get("prunable_dimensions"),
            "blocked_dimensions": item.get("blocked_dimensions"),
            "constraints": item.get("constraints"),
        }
        for item in data.get("interfaces", [])
    ]
    return "\n".join(
        [
            f"# Structural Region Tree: {data.get('model_name', '')}",
            "",
            "## Summary",
            "",
            f"- Tensor IR frontend provenance: `{data.get('source_frontend', 'unknown')}`",
            f"- Regions: `{summary.get('num_regions', 0)}`",
            f"- Primitive regions: `{summary.get('num_primitive_regions', 0)}`",
            f"- Feed-forward regions: `{summary.get('num_feedforward_regions', 0)}`",
            f"- GELU semantic fusions: `{summary.get('num_gelu_fusions', 0)}`",
            f"- Fused feed-forward regions: `{summary.get('num_feedforward_fusions', 0)}`",
            f"- Attention skeleton regions: `{summary.get('num_attention_skeleton_regions', 0)}`",
            f"- Residual merge regions: `{summary.get('num_residual_merge_regions', 0)}`",
            f"- Fork regions: `{summary.get('num_fork_regions', 0)}`",
            f"- Join regions: `{summary.get('num_join_regions', 0)}`",
            "",
            "## Regions",
            "",
            _table(regions, ["region_id", "region_type", "ops", "children", "depth", "fork", "join", "confidence"]),
            "",
            "## Region Interfaces",
            "",
            _table(interfaces, ["region_id", "region_type", "pruning_role", "prunable_dimensions", "blocked_dimensions", "constraints"]),
            "",
            "## Interpretation",
            "",
            "This tree is a compiler-inspired hierarchy over frontend-independent Tensor IR. Primitive TensorOps remain leaves, while semantic regions summarize local tensor dataflow and preliminary propagation constraints. It is analysis only and does not modify models.",
            "",
        ]
    )
