from __future__ import annotations

from model_analysis.structural_region_detection import build_structural_region_tree
from model_analysis.structural_region_tree_text import structural_region_tree_to_text
from test_structural_region_detection import synthetic_tensor_graph


def test_text_dump_contains_root_semantic_regions_and_blocking_constraint() -> None:
    text = structural_region_tree_to_text(build_structural_region_tree(synthetic_tensor_graph()))

    assert "region.module @tiny" in text
    assert "ModelRegion" in text
    assert "LinearProjectionRegion" in text
    assert "ResidualMergeRegion" in text
    assert 'constraint("branch_hidden_equality")' in text
