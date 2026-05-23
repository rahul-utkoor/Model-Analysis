"""Comparison helpers for local ONNX subgraph structural reports."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _summary_matrix(reports: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    return {
        report.get("model_name", f"model_{index}"): dict(report.get("summary", {}).get(key, {}))
        for index, report in enumerate(reports)
    }


def compare_subgraph_reports(reports: list[dict]) -> dict:
    """Compare pattern, risk, evidence, and join summaries across models."""
    models = [report.get("model_name", f"model_{index}") for index, report in enumerate(reports)]
    pattern_matrix: dict[str, dict[str, int]] = {}
    pattern_sets: dict[str, set[str]] = {}
    join_matrix: dict[str, dict[str, int]] = {}
    residual_matrix: dict[str, dict[str, int]] = {}
    for model, report in zip(models, reports):
        patterns = Counter()
        for item in report.get("pattern_summaries", []):
            if item.get("pattern"):
                patterns[item["pattern"]] += item.get("count", 0)
        pattern_matrix[model] = dict(patterns)
        pattern_sets[model] = set(patterns)
        summary = report.get("summary", {})
        join_matrix[model] = {
            "join_subgraphs": summary.get("num_join_subgraphs", 0),
            "bias_add": summary.get("bias_add_count", 0),
            "residual_add": summary.get("residual_add_count", 0),
            "elementwise_add": summary.get("elementwise_add_count", 0),
            "unknown_add": summary.get("unknown_add_count", 0),
        }
        residual_matrix[model] = {
            "residual_like_join_subgraphs": summary.get("num_residual_like_join_subgraphs", 0),
            "residual_like_patterns": summary.get("residual_like_pattern_count", 0),
        }
    common_patterns = set.intersection(*pattern_sets.values()) if pattern_sets else set()
    model_specific_patterns = {
        model: sorted(patterns - set().union(*(other for name, other in pattern_sets.items() if name != model)))
        for model, patterns in pattern_sets.items()
    }
    return {
        "num_models": len(reports),
        "models": models,
        "pattern_matrix": pattern_matrix,
        "pruning_class_matrix": _summary_matrix(reports, "pruning_class_counts"),
        "risk_level_matrix": _summary_matrix(reports, "risk_level_counts"),
        "evidence_type_matrix": _summary_matrix(reports, "evidence_type_counts"),
        "join_summary_matrix": join_matrix,
        "residual_summary_matrix": residual_matrix,
        "common_patterns": sorted(common_patterns),
        "model_specific_patterns": model_specific_patterns,
        "summary": {
            "total_path_subgraphs": sum(report.get("summary", {}).get("num_path_subgraphs", 0) for report in reports),
            "total_join_subgraphs": sum(report.get("summary", {}).get("num_join_subgraphs", 0) for report in reports),
            "total_residual_like_joins": sum(
                report.get("summary", {}).get("num_residual_like_join_subgraphs", 0) for report in reports
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


def subgraph_comparison_to_markdown(comparison: dict[str, Any]) -> str:
    specific = [
        f"- `{model}`: {', '.join(patterns) if patterns else 'None'}"
        for model, patterns in sorted(comparison.get("model_specific_patterns", {}).items())
    ]
    return "\n".join(
        [
            "# Subgraph Pattern Comparison",
            "",
            "## Summary",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Directed path subgraphs: `{comparison.get('summary', {}).get('total_path_subgraphs', 0)}`",
            f"- Join-centered subgraphs: `{comparison.get('summary', {}).get('total_join_subgraphs', 0)}`",
            f"- Residual-like joins: `{comparison.get('summary', {}).get('total_residual_like_joins', 0)}`",
            "",
            "## Path Pattern Comparison",
            "",
            _matrix_to_markdown(comparison.get("pattern_matrix", {})),
            "",
            "## Join Pattern Comparison",
            "",
            _matrix_to_markdown(comparison.get("join_summary_matrix", {})),
            "",
            "## Residual-Like Join Comparison",
            "",
            _matrix_to_markdown(comparison.get("residual_summary_matrix", {})),
            "",
            "## Pruning Class Matrix",
            "",
            _matrix_to_markdown(comparison.get("pruning_class_matrix", {})),
            "",
            "## Risk Level Matrix",
            "",
            _matrix_to_markdown(comparison.get("risk_level_matrix", {})),
            "",
            "## Evidence Comparison",
            "",
            _matrix_to_markdown(comparison.get("evidence_type_matrix", {})),
            "",
            "## Common Patterns",
            "",
            "\n".join(f"- `{item}`" for item in comparison.get("common_patterns", [])) or "_None._",
            "",
            "## Model-Specific Patterns",
            "",
            "\n".join(specific) or "_None._",
            "",
            "## Interpretation",
            "",
            "This comparison separates directed local paths from join-centered and residual-like structures. It is structural evidence for later pruning-map and Dimension-IR refinement, not executable pruning.",
            "",
        ]
    )
