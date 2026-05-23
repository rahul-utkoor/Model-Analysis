"""Cross-model comparison of frontend-independent Tensor Graph IR reports."""

from __future__ import annotations

from typing import Any


def _summary_matrix(graphs: list[dict], key: str) -> dict[str, dict[str, int]]:
    return {
        graph.get("model_name", f"model_{index}"): dict(graph.get("summary", {}).get(key, {}))
        for index, graph in enumerate(graphs)
    }


def compare_tensor_graphs(graphs: list[dict]) -> dict:
    models = [graph.get("model_name", f"model_{index}") for index, graph in enumerate(graphs)]
    fork_join_matrix = {
        graph.get("model_name", f"model_{index}"): {
            "fork_ops": graph.get("summary", {}).get("num_fork_ops", 0),
            "join_ops": graph.get("summary", {}).get("num_join_ops", 0),
        }
        for index, graph in enumerate(graphs)
    }
    return {
        "num_models": len(graphs),
        "models": models,
        "canonical_op_type_matrix": _summary_matrix(graphs, "canonical_op_type_counts"),
        "semantic_role_matrix": _summary_matrix(graphs, "semantic_role_counts"),
        "region_hint_matrix": _summary_matrix(graphs, "region_hint_counts"),
        "fork_join_matrix": fork_join_matrix,
        "summary": {
            "total_ops": sum(graph.get("summary", {}).get("num_ops", 0) for graph in graphs),
            "total_values": sum(graph.get("summary", {}).get("num_values", 0) for graph in graphs),
            "total_fork_ops": sum(graph.get("summary", {}).get("num_fork_ops", 0) for graph in graphs),
            "total_join_ops": sum(graph.get("summary", {}).get("num_join_ops", 0) for graph in graphs),
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


def tensor_ir_comparison_to_markdown(comparison: dict[str, Any]) -> str:
    summary = comparison.get("summary", {})
    return "\n".join(
        [
            "# Tensor IR Comparison",
            "",
            "## Summary",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Total operations: `{summary.get('total_ops', 0)}`",
            f"- Total tensor values: `{summary.get('total_values', 0)}`",
            f"- Total fork operations: `{summary.get('total_fork_ops', 0)}`",
            f"- Total join operations: `{summary.get('total_join_ops', 0)}`",
            "",
            "## Canonical Operation Type Matrix",
            "",
            _matrix_to_markdown(comparison.get("canonical_op_type_matrix", {})),
            "",
            "## Semantic Role Matrix",
            "",
            _matrix_to_markdown(comparison.get("semantic_role_matrix", {})),
            "",
            "## Region Hint Matrix",
            "",
            _matrix_to_markdown(comparison.get("region_hint_matrix", {})),
            "",
            "## Fork / Join Matrix",
            "",
            _matrix_to_markdown(comparison.get("fork_join_matrix", {})),
            "",
            "## Interpretation",
            "",
            "This comparison operates on frontend-independent Tensor IR. The current graphs were imported from ONNX summaries; future frontends can populate the same canonical operations, values, forks, joins, and region hints.",
            "",
        ]
    )
