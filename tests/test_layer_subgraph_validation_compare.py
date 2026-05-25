from __future__ import annotations

from copy import deepcopy

from model_analysis.layer_subgraph_validation_compare import compare_layer_subgraph_validation_packs, comparison_to_markdown
from test_layer_subgraph_validation_pack import build_pack


def test_compare_summary_aggregates_class_counts() -> None:
    first = build_pack()
    second = deepcopy(first)
    second["model_name"] = "other"

    comparison = compare_layer_subgraph_validation_packs([first, second])

    assert comparison["summary"]["total_subgraphs"] == 2 * len(first["subgraphs"])
    assert "safe" in comparison["pruning_class_matrix"]
    assert "Layer Subgraph Validation Comparison" in comparison_to_markdown(comparison)
