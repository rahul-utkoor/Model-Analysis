from __future__ import annotations

from model_analysis.structural_region_detection import build_structural_region_tree
from test_structural_region_detection import synthetic_tensor_graph


def test_tree_contains_root_regions_and_primitive_leaves() -> None:
    tree = build_structural_region_tree(synthetic_tensor_graph())
    regions = {region.region_id: region for region in tree.regions}
    types = [region.region_type for region in tree.regions]

    assert regions[tree.root_region_id].region_type == "ModelRegion"
    assert types.count("PrimitiveRegion") == len(synthetic_tensor_graph()["ops"])
    assert "FeedForwardRegion" in types
    assert "ResidualMergeRegion" in types
    assert all(not region.children for region in tree.regions if region.region_type == "PrimitiveRegion")
    assert any(region.parent != tree.root_region_id for region in tree.regions if region.region_type == "PrimitiveRegion")


def test_region_interfaces_expose_pruning_roles_and_constraints() -> None:
    tree = build_structural_region_tree(synthetic_tensor_graph())
    by_type = {}
    for interface in tree.interfaces:
        by_type.setdefault(interface.region_type, []).append(interface)

    assert by_type["LinearProjectionRegion"][0].pruning_role == "directly_prunable"
    assert by_type["FeedForwardRegion"][0].prunable_dimensions[0]["dim_name"] == "intermediate_dim"
    assert by_type["ResidualMergeRegion"][0].pruning_role == "blocked"
    assert by_type["ResidualMergeRegion"][0].blocked_dimensions[0]["dim_name"] == "hidden_dim"
    assert tree.summary["num_residual_merge_regions"] >= 1
