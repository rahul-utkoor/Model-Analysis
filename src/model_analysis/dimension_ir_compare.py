"""Comparison helpers for pruning Dimension IRs."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _matrix(irs: list[dict[str, Any]], summary_key: str) -> dict[str, dict[str, int]]:
    return {
        ir.get("model_name", f"model_{index}"): dict(ir.get("summary", {}).get(summary_key, {}))
        for index, ir in enumerate(irs)
    }


def compare_dimension_irs(irs: list[dict]) -> dict:
    """Compare Dimension IR summaries across models."""
    models = [ir.get("model_name", f"model_{index}") for index, ir in enumerate(irs)]
    class_matrix = {
        ir.get("model_name", f"model_{index}"): dict(Counter(item.get("class_type") for item in ir.get("equivalence_classes", []) if item.get("class_type")))
        for index, ir in enumerate(irs)
    }
    blocked_matrix = {
        ir.get("model_name", f"model_{index}"): {"blocked_dimensions": len(ir.get("blocked_dimensions", []))}
        for index, ir in enumerate(irs)
    }
    unresolved_matrix = {
        ir.get("model_name", f"model_{index}"): {"unresolved_constraints": len(ir.get("unresolved_constraints", []))}
        for index, ir in enumerate(irs)
    }
    return {
        "num_models": len(irs),
        "models": models,
        "dimension_role_matrix": _matrix(irs, "semantic_role_counts"),
        "constraint_type_matrix": _matrix(irs, "constraint_type_counts"),
        "relation_matrix": _matrix(irs, "relation_counts"),
        "equivalence_class_matrix": class_matrix,
        "blocked_dimension_matrix": blocked_matrix,
        "unresolved_constraint_matrix": unresolved_matrix,
        "summary": {
            "total_dimension_variables": sum(ir.get("summary", {}).get("num_dimension_variables", 0) for ir in irs),
            "total_constraint_equations": sum(ir.get("summary", {}).get("num_constraint_equations", 0) for ir in irs),
            "total_blocked_dimensions": sum(ir.get("summary", {}).get("num_blocked_dimensions", 0) for ir in irs),
            "total_unresolved_constraints": sum(ir.get("summary", {}).get("num_unresolved_constraints", 0) for ir in irs),
        },
    }


def _matrix_to_markdown(matrix: dict[str, dict[str, int]]) -> str:
    columns = sorted({key for row in matrix.values() for key in row})
    if not columns:
        return "_None._"
    lines = [
        "| model | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---" for _ in columns) + " |",
    ]
    for model, row in sorted(matrix.items()):
        lines.append("| " + model + " | " + " | ".join(str(row.get(column, 0)) for column in columns) + " |")
    return "\n".join(lines)


def dimension_ir_comparison_to_markdown(comparison: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dimension IR Comparison",
            "",
            "## Summary",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Model names: `{comparison.get('models', [])}`",
            f"- Total dimension variables: `{comparison.get('summary', {}).get('total_dimension_variables', 0)}`",
            f"- Total constraint equations: `{comparison.get('summary', {}).get('total_constraint_equations', 0)}`",
            f"- Total blocked dimensions: `{comparison.get('summary', {}).get('total_blocked_dimensions', 0)}`",
            f"- Total unresolved constraints: `{comparison.get('summary', {}).get('total_unresolved_constraints', 0)}`",
            "",
            "## Dimension Role Matrix",
            "",
            _matrix_to_markdown(comparison.get("dimension_role_matrix", {})),
            "",
            "## Constraint Type Matrix",
            "",
            _matrix_to_markdown(comparison.get("constraint_type_matrix", {})),
            "",
            "## Relation Matrix",
            "",
            _matrix_to_markdown(comparison.get("relation_matrix", {})),
            "",
            "## Equivalence Class Matrix",
            "",
            _matrix_to_markdown(comparison.get("equivalence_class_matrix", {})),
            "",
            "## Blocked Dimension Matrix",
            "",
            _matrix_to_markdown(comparison.get("blocked_dimension_matrix", {})),
            "",
            "## Unresolved Constraint Matrix",
            "",
            _matrix_to_markdown(comparison.get("unresolved_constraint_matrix", {})),
            "",
            "## Interpretation",
            "",
            "This comparison summarizes symbolic Dimension IR structure across models. It compares roles, relations, constraint classes, blocked dimensions, and unresolved constraints; it does not execute pruning or evaluate accuracy.",
            "",
        ]
    )
