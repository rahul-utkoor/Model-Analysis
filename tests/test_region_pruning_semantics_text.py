from __future__ import annotations

from model_analysis.region_pruning_semantics import build_region_pruning_semantics
from model_analysis.region_pruning_semantics_text import region_pruning_semantics_to_text
from test_region_pruning_semantics import synthetic_inputs


def test_text_dump_contains_readable_region_entries() -> None:
    tree, tensor_ir, rdim = synthetic_inputs()
    text = region_pruning_semantics_to_text(build_region_pruning_semantics(tree, tensor_ir, region_dimension_ir=rdim))

    assert "region_pruning_semantics @synthetic" in text
    assert 'region "Layer 0 Feed Forward" [FeedForwardRegion]' in text
    assert "same_indices" in text
    assert "attention_head_mapping_unproven" in text
