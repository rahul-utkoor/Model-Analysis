"""Comparison helpers for DAG motif and multi-join region reports."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _matrix(reports: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    return {
        report.get("model_name", f"model_{index}"): dict(report.get("summary", {}).get(key, {}))
        for index, report in enumerate(reports)
    }


def compare_dag_region_reports(reports: list[dict]) -> dict:
    """Compare DAG region patterns and constraint evidence across models."""
    models = [report.get("model_name", f"model_{index}") for index, report in enumerate(reports)]
    pattern_matrix: dict[str, dict[str, int]] = {}
    pattern_sets: dict[str, set[str]] = {}
    for model, report in zip(models, reports):
        counts = Counter()
        for pattern in report.get("pattern_summaries", []):
            if pattern.get("pattern"):
                counts[pattern["pattern"]] += pattern.get("count", 0)
        pattern_matrix[model] = dict(counts)
        pattern_sets[model] = set(counts)
    common = set.intersection(*pattern_sets.values()) if pattern_sets else set()
    model_specific = {
        model: sorted(patterns - set().union(*(other for name, other in pattern_sets.items() if name != model)))
        for model, patterns in pattern_sets.items()
    }
    return {
        "num_models": len(reports),
        "models": models,
        "region_kind_matrix": _matrix(reports, "region_kind_counts"),
        "pattern_matrix": pattern_matrix,
        "pruning_class_matrix": _matrix(reports, "pruning_class_counts"),
        "risk_level_matrix": _matrix(reports, "risk_level_counts"),
        "suggested_constraint_matrix": _matrix(reports, "suggested_constraint_counts"),
        "common_region_patterns": sorted(common),
        "model_specific_region_patterns": model_specific,
        "summary": {
            "total_regions": sum(report.get("summary", {}).get("num_regions", 0) for report in reports),
            "total_join_fork_join_regions": sum(
                report.get("summary", {}).get("num_join_fork_join_regions", 0) for report in reports
            ),
            "total_residual_like_regions": sum(
                report.get("summary", {}).get("num_residual_like_regions", 0) for report in reports
            ),
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


def dag_region_comparison_to_markdown(comparison: dict[str, Any]) -> str:
    specific = [
        f"- `{model}`: {', '.join(patterns) if patterns else 'None'}"
        for model, patterns in sorted(comparison.get("model_specific_region_patterns", {}).items())
    ]
    return "\n".join(
        [
            "# DAG Region Comparison",
            "",
            "## Summary",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Regions: `{comparison.get('summary', {}).get('total_regions', 0)}`",
            f"- Join-fork-join regions: `{comparison.get('summary', {}).get('total_join_fork_join_regions', 0)}`",
            f"- Residual-like regions: `{comparison.get('summary', {}).get('total_residual_like_regions', 0)}`",
            "",
            "## Region Kind Matrix",
            "",
            _matrix_to_markdown(comparison.get("region_kind_matrix", {})),
            "",
            "## Pattern Matrix",
            "",
            _matrix_to_markdown(comparison.get("pattern_matrix", {})),
            "",
            "## Pruning Class Matrix",
            "",
            _matrix_to_markdown(comparison.get("pruning_class_matrix", {})),
            "",
            "## Risk Level Matrix",
            "",
            _matrix_to_markdown(comparison.get("risk_level_matrix", {})),
            "",
            "## Suggested Constraint Matrix",
            "",
            _matrix_to_markdown(comparison.get("suggested_constraint_matrix", {})),
            "",
            "## Common Region Patterns",
            "",
            "\n".join(f"- `{item}`" for item in comparison.get("common_region_patterns", [])) or "_None._",
            "",
            "## Model-Specific Region Patterns",
            "",
            "\n".join(specific) or "_None._",
            "",
            "## Interpretation",
            "",
            "This comparison captures fork, join, diamond, and join-fork-join structural evidence across saved ONNX summaries. It does not modify models.",
            "",
        ]
    )

