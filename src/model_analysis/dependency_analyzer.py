"""Analysis helpers for pruning dependency graphs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from model_analysis.dependency_graph import DependencyGraph


HIGH_VALUE_TYPES = {"attention_qkv", "mlp_expansion", "mlp_projection", "conv"}


def _unit_to_review_item(unit) -> dict[str, Any]:
    return {
        "item": unit.unit_id,
        "source": unit.source,
        "reason": unit.reason,
        "confidence": unit.confidence,
    }


def analyze_dependency_graph(graph: DependencyGraph) -> dict[str, Any]:
    """Summarize a dependency graph for pruning planning."""
    unit_type_counts = Counter(unit.unit_type for unit in graph.prunable_units)
    edge_type_counts = Counter(edge.edge_type for edge in graph.dependency_edges)
    confidence_counts = Counter(unit.confidence for unit in graph.prunable_units)
    confidence_counts.update(edge.confidence for edge in graph.dependency_edges)

    sorted_linear_units = sorted(
        [unit for unit in graph.prunable_units if unit.unit_type in {"linear", "gemm", "matmul"}],
        key=lambda unit: unit.parameter_count or 0,
        reverse=True,
    )
    large_linear_cutoff = sorted_linear_units[:10]

    high_value_targets = []
    for unit in graph.prunable_units:
        if unit.unit_type in HIGH_VALUE_TYPES:
            high_value_targets.append(
                {
                    "unit_id": unit.unit_id,
                    "unit_type": unit.unit_type,
                    "name": unit.name,
                    "confidence": unit.confidence,
                    "reason": unit.reason,
                }
            )
        elif unit in large_linear_cutoff and (unit.parameter_count or 0) > 0:
            high_value_targets.append(
                {
                    "unit_id": unit.unit_id,
                    "unit_type": unit.unit_type,
                    "name": unit.name,
                    "confidence": unit.confidence,
                    "reason": "Large linear/projection unit by parameter count.",
                }
            )
        elif unit.unit_type == "embedding":
            high_value_targets.append(
                {
                    "unit_id": unit.unit_id,
                    "unit_type": unit.unit_type,
                    "name": unit.name,
                    "confidence": unit.confidence,
                    "reason": "Large embeddings may be valuable, but require output tying and vocabulary caveat checks.",
                }
            )

    manual_review_items = list(graph.ambiguous_units)
    manual_review_items.extend(_unit_to_review_item(unit) for unit in graph.prunable_units if unit.confidence == "low")
    manual_review_items.extend(
        {
            "item": f"{edge.src} -> {edge.dst}",
            "source": "dependency_edge",
            "reason": edge.reason,
            "confidence": edge.confidence,
        }
        for edge in graph.dependency_edges
        if edge.edge_type == "residual_coupling" or edge.confidence == "low"
    )

    forward_paths = [
        {
            "src": edge.src,
            "dst": edge.dst,
            "edge_type": edge.edge_type,
            "affected_dims": edge.affected_dims,
            "confidence": edge.confidence,
            "reason": edge.reason,
        }
        for edge in graph.dependency_edges
        if edge.direction in {"forward", "bidirectional"}
    ]
    backward_constraints = [
        {
            "src": edge.src,
            "dst": edge.dst,
            "edge_type": edge.edge_type,
            "affected_dims": edge.affected_dims,
            "confidence": edge.confidence,
            "reason": edge.reason,
        }
        for edge in graph.dependency_edges
        if edge.direction in {"backward", "bidirectional"}
    ]

    return {
        "model_name": graph.model_name,
        "num_prunable_units": len(graph.prunable_units),
        "num_dependency_edges": len(graph.dependency_edges),
        "num_coupled_groups": len(graph.coupled_groups),
        "num_independent_units": len(graph.independent_units),
        "num_ambiguous_units": len(graph.ambiguous_units),
        "unit_type_counts": dict(unit_type_counts),
        "edge_type_counts": dict(edge_type_counts),
        "confidence_counts": dict(confidence_counts),
        "high_value_pruning_targets": high_value_targets,
        "manual_review_items": manual_review_items,
        "forward_propagation_paths": forward_paths,
        "backward_propagation_constraints": backward_constraints,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if not rows:
        return "_None detected._"
    selected = rows[:limit] if limit else rows
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if limit and len(rows) > limit:
        omitted = {columns[0]: "..."}
        if len(columns) > 1:
            omitted[columns[1]] = f"{len(rows) - limit} more rows omitted"
        lines.append("| " + " | ".join(str(omitted.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def dependency_analysis_to_markdown(summary: dict[str, Any]) -> str:
    """Render dependency analyzer output as Markdown."""
    lines = [
        f"# Dependency Summary: {summary['model_name']}",
        "",
        "## High-Value Pruning Targets",
        "",
        _markdown_table(summary["high_value_pruning_targets"], ["unit_id", "unit_type", "name", "confidence", "reason"], limit=100),
        "",
        "## Forward Propagation Paths",
        "",
        _markdown_table(summary["forward_propagation_paths"], ["src", "dst", "edge_type", "affected_dims", "confidence", "reason"], limit=150),
        "",
        "## Backward Propagation Constraints",
        "",
        _markdown_table(summary["backward_propagation_constraints"], ["src", "dst", "edge_type", "affected_dims", "confidence", "reason"], limit=150),
        "",
        "## Manual Review Items",
        "",
        _markdown_table(summary["manual_review_items"], ["item", "name", "unit_id", "source", "confidence", "reason"], limit=150),
        "",
        "## Caveats",
        "",
        "This summary is a conservative static pruning-dependency IR. It is not an executable pruning transform and does not prove functional correctness after pruning.",
        "",
        "## Counts",
        "",
        f"- Prunable units: `{summary['num_prunable_units']}`",
        f"- Dependency edges: `{summary['num_dependency_edges']}`",
        f"- Coupled groups: `{summary['num_coupled_groups']}`",
        f"- Independent units: `{summary['num_independent_units']}`",
        f"- Ambiguous units: `{summary['num_ambiguous_units']}`",
        "",
    ]
    return "\n".join(lines)
