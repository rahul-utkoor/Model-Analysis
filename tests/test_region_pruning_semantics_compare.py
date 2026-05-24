from __future__ import annotations

from model_analysis.region_pruning_semantics import build_region_pruning_semantics, region_pruning_semantics_to_dict
from model_analysis.region_pruning_semantics_compare import compare_region_pruning_semantics
from test_region_pruning_semantics import synthetic_inputs


def test_compare_region_pruning_semantics_summarizes_role_counts() -> None:
    tree, tensor_ir, rdim = synthetic_inputs()
    one = region_pruning_semantics_to_dict(build_region_pruning_semantics(tree, tensor_ir, region_dimension_ir=rdim))
    two = dict(one)
    two["model_name"] = "synthetic-two"

    comparison = compare_region_pruning_semantics([one, two])

    assert comparison["num_models"] == 2
    assert "directly_prunable" in comparison["pruning_role_matrix"]
    assert "FeedForwardRegion" in comparison["region_type_matrix"]
    assert "feed_forward_block" in comparison["semantic_category_matrix"]
    assert comparison["repair_obligation_matrix"]["same_indices_across_mlp"]["synthetic"] == 1
