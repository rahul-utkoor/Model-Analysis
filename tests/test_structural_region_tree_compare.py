from __future__ import annotations

from model_analysis.structural_region_tree_compare import (
    compare_structural_region_trees,
    structural_region_tree_comparison_to_markdown,
)


def tree_report(model: str, feedforward: int, residual: int) -> dict:
    return {
        "model_name": model,
        "summary": {
            "num_regions": 10 + feedforward + residual,
            "num_primitive_regions": 10,
            "num_feedforward_regions": feedforward,
            "num_residual_merge_regions": residual,
            "num_blocked_regions": residual,
            "region_type_counts": {
                "ModelRegion": 1,
                "PrimitiveRegion": 10,
                "FeedForwardRegion": feedforward,
                "ResidualMergeRegion": residual,
            },
            "pruning_role_counts": {"directly_prunable": feedforward, "blocked": residual},
            "confidence_counts": {"high": feedforward + 10, "medium": residual},
        },
    }


def test_compare_structural_region_trees_builds_matrices() -> None:
    comparison = compare_structural_region_trees([tree_report("bert", 2, 1), tree_report("vit", 1, 1)])

    assert comparison["num_models"] == 2
    assert comparison["region_type_matrix"]["bert"]["FeedForwardRegion"] == 2
    assert comparison["pruning_role_matrix"]["vit"]["blocked"] == 1
    assert "ResidualMergeRegion" in comparison["common_region_types"]
    assert comparison["summary"]["total_blocked_regions"] == 2
    assert "Structural Region Tree Comparison" in structural_region_tree_comparison_to_markdown(comparison)
