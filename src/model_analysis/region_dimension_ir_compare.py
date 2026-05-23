"""Cross-model comparison helpers for region-aware Dimension IR."""

from __future__ import annotations

from typing import Any


def _matrix(irs: list[dict], key: str) -> dict[str, dict[str, int]]:
    return {
        ir.get("model_name", f"model_{index}"): dict(ir.get("summary", {}).get(key, {}))
        for index, ir in enumerate(irs)
    }


def compare_region_dimension_irs(irs: list[dict]) -> dict:
    models = [ir.get("model_name", f"model_{index}") for index, ir in enumerate(irs)]
    return {
        "num_models": len(irs),
        "models": models,
        "axis_role_matrix": _matrix(irs, "axis_role_counts"),
        "region_type_matrix": _matrix(irs, "region_type_counts"),
        "constraint_type_matrix": _matrix(irs, "constraint_type_counts"),
        "relation_matrix": _matrix(irs, "relation_counts"),
        "blocked_dimension_matrix": {
            model: {"blocked_dimensions": len(ir.get("blocked_dimensions", []))}
            for model, ir in zip(models, irs)
        },
        "unresolved_constraint_matrix": {
            model: {"unresolved_constraints": len(ir.get("unresolved_constraints", []))}
            for model, ir in zip(models, irs)
        },
        "summary": {
            "total_dimension_variables": sum(ir.get("summary", {}).get("num_dimension_variables", 0) for ir in irs),
            "total_constraint_equations": sum(ir.get("summary", {}).get("num_constraint_equations", 0) for ir in irs),
            "total_equivalence_classes": sum(ir.get("summary", {}).get("num_equivalence_classes", 0) for ir in irs),
            "total_blocked_dimensions": sum(ir.get("summary", {}).get("num_blocked_dimensions", 0) for ir in irs),
            "total_unresolved_constraints": sum(ir.get("summary", {}).get("num_unresolved_constraints", 0) for ir in irs),
        },
    }


def _matrix_to_markdown(matrix: dict[str, dict[str, int]]) -> str:
    columns = sorted({key for row in matrix.values() for key in row})
    if not columns:
        return "_None._"
    lines = ["| model | " + " | ".join(columns) + " |", "| --- | " + " | ".join("---" for _ in columns) + " |"]
    for model, row in sorted(matrix.items()):
        lines.append("| " + model + " | " + " | ".join(str(row.get(column, 0)) for column in columns) + " |")
    return "\n".join(lines)


def region_dimension_ir_comparison_to_markdown(comparison: dict[str, Any]) -> str:
    summary = comparison.get("summary", {})
    return "\n".join(
        [
            "# Region-Aware Dimension IR Comparison",
            "",
            "## Summary",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Dimension variables: `{summary.get('total_dimension_variables', 0)}`",
            f"- Constraint equations: `{summary.get('total_constraint_equations', 0)}`",
            f"- Equivalence classes: `{summary.get('total_equivalence_classes', 0)}`",
            f"- Blocked dimensions: `{summary.get('total_blocked_dimensions', 0)}`",
            f"- Unresolved constraints: `{summary.get('total_unresolved_constraints', 0)}`",
            "",
            "## Axis Role Matrix",
            "",
            _matrix_to_markdown(comparison.get("axis_role_matrix", {})),
            "",
            "## Region Type Matrix",
            "",
            _matrix_to_markdown(comparison.get("region_type_matrix", {})),
            "",
            "## Constraint Type Matrix",
            "",
            _matrix_to_markdown(comparison.get("constraint_type_matrix", {})),
            "",
            "## Relation Matrix",
            "",
            _matrix_to_markdown(comparison.get("relation_matrix", {})),
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
            "This comparison summarizes dimensions and constraints derived from semantic structural regions. It is static analysis only and does not modify models.",
            "",
        ]
    )
