"""Cross-model comparison helpers for Structural Region Trees."""

from __future__ import annotations

from typing import Any


def _matrix(trees: list[dict], key: str) -> dict[str, dict[str, int]]:
    return {
        tree.get("model_name", f"model_{index}"): dict(tree.get("summary", {}).get(key, {}))
        for index, tree in enumerate(trees)
    }


def compare_structural_region_trees(trees: list[dict]) -> dict:
    models = [tree.get("model_name", f"model_{index}") for index, tree in enumerate(trees)]
    type_sets = {
        model: {kind for kind, count in tree.get("summary", {}).get("region_type_counts", {}).items() if count}
        for model, tree in zip(models, trees)
    }
    common = set.intersection(*type_sets.values()) if type_sets else set()
    specific = {
        model: sorted(kinds - set().union(*(other for name, other in type_sets.items() if name != model)))
        for model, kinds in type_sets.items()
    }
    return {
        "num_models": len(trees),
        "models": models,
        "region_type_matrix": _matrix(trees, "region_type_counts"),
        "pruning_role_matrix": _matrix(trees, "pruning_role_counts"),
        "confidence_matrix": _matrix(trees, "confidence_counts"),
        "common_region_types": sorted(common),
        "model_specific_region_types": specific,
        "summary": {
            "total_regions": sum(tree.get("summary", {}).get("num_regions", 0) for tree in trees),
            "total_primitives": sum(tree.get("summary", {}).get("num_primitive_regions", 0) for tree in trees),
            "total_feedforward_regions": sum(tree.get("summary", {}).get("num_feedforward_regions", 0) for tree in trees),
            "total_residual_merge_regions": sum(tree.get("summary", {}).get("num_residual_merge_regions", 0) for tree in trees),
            "total_blocked_regions": sum(tree.get("summary", {}).get("num_blocked_regions", 0) for tree in trees),
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


def structural_region_tree_comparison_to_markdown(comparison: dict[str, Any]) -> str:
    summary = comparison.get("summary", {})
    specific = [
        f"- `{model}`: {', '.join(types) if types else 'None'}"
        for model, types in sorted(comparison.get("model_specific_region_types", {}).items())
    ]
    return "\n".join(
        [
            "# Structural Region Tree Comparison",
            "",
            "## Summary",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Regions: `{summary.get('total_regions', 0)}`",
            f"- Primitive leaves: `{summary.get('total_primitives', 0)}`",
            f"- Feed-forward regions: `{summary.get('total_feedforward_regions', 0)}`",
            f"- Residual merge regions: `{summary.get('total_residual_merge_regions', 0)}`",
            f"- Blocked regions: `{summary.get('total_blocked_regions', 0)}`",
            "",
            "## Region Type Matrix",
            "",
            _matrix_to_markdown(comparison.get("region_type_matrix", {})),
            "",
            "## Pruning Role Matrix",
            "",
            _matrix_to_markdown(comparison.get("pruning_role_matrix", {})),
            "",
            "## Confidence Matrix",
            "",
            _matrix_to_markdown(comparison.get("confidence_matrix", {})),
            "",
            "## Common Region Types",
            "",
            "\n".join(f"- `{item}`" for item in comparison.get("common_region_types", [])) or "_None._",
            "",
            "## Model-Specific Region Types",
            "",
            "\n".join(specific) or "_None._",
            "",
            "## Interpretation",
            "",
            "The Structural Region Tree comparison operates over Tensor IR and compares compiler-style semantic tensor-dataflow regions. It reports preliminary propagation structure only and does not modify models.",
            "",
        ]
    )
