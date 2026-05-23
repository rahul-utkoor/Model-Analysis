from __future__ import annotations

from model_analysis.region_dimension_ir import build_region_dimension_ir, region_dimension_ir_to_dict
from model_analysis.region_ir_analysis import check_region_pruning_legality, make_region_pruning_request
from model_analysis.structural_region_detection import build_structural_region_tree
from model_analysis.structural_region_tree import structural_region_tree_to_dict
from test_semantic_fusion import gelu_tensor_graph


def test_fused_feedforward_region_is_inserted_in_tree_without_false_residual_add() -> None:
    tree = build_structural_region_tree(gelu_tensor_graph())
    feedforward = [region for region in tree.regions if region.region_type == "FeedForwardRegion"]
    residual = [region for region in tree.regions if region.region_type == "ResidualMergeRegion"]

    assert len(feedforward) == 1
    assert feedforward[0].metadata["semantic_fusion"] is True
    assert feedforward[0].metadata["activation_kind"] == "gelu"
    assert not any("add" in region.op_ids for region in residual)
    assert tree.summary["num_gelu_fusions"] == 1
    assert tree.summary["num_feedforward_fusions"] == 1


def test_fused_feedforward_exposes_intermediate_dimension_and_legality_repairs() -> None:
    tree = build_structural_region_tree(gelu_tensor_graph())
    ir = build_region_dimension_ir(structural_region_tree_to_dict(tree))
    dimensions = [
        item for item in ir.dimension_variables
        if item.region_type == "FeedForwardRegion" and item.dim_name == "intermediate_dim"
    ]
    root = next(item for item in dimensions if item.prunable)
    result = check_region_pruning_legality(
        region_dimension_ir_to_dict(ir),
        make_region_pruning_request(ir.model_name, root.var_id),
    )

    assert len(dimensions) == 2
    assert any(item.evidence[-1]["source"] == "semantic_fusion" for item in dimensions)
    assert result.status == "legal_with_repairs"
    assert any(item.repair_type == "same_indices" for item in result.minimal_repair_set)
