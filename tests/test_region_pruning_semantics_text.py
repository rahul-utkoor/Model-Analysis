from __future__ import annotations

from model_analysis.region_pruning_semantics import build_region_pruning_semantics, region_pruning_semantics_to_dict, region_pruning_semantics_to_markdown
from model_analysis.region_pruning_semantics_text import region_pruning_semantics_to_text
from test_region_pruning_semantics import synthetic_inputs


def test_text_dump_contains_readable_region_entries() -> None:
    tree, tensor_ir, rdim = synthetic_inputs()
    text = region_pruning_semantics_to_text(build_region_pruning_semantics(tree, tensor_ir, region_dimension_ir=rdim))

    assert "region_pruning_semantics @synthetic" in text
    assert 'region "Layer 0 Feed Forward" [FeedForwardRegion]' in text
    assert "same_indices" in text
    assert "attention_head_mapping_unproven" in text


def test_default_markdown_summarizes_auxiliary_details_without_spam() -> None:
    tree, tensor_ir, rdim = synthetic_inputs()
    semantics = region_pruning_semantics_to_dict(build_region_pruning_semantics(tree, tensor_ir, region_dimension_ir=rdim))
    for index in range(20):
        semantics["regions"].append(
            {
                "region_id": f"axis_{index}",
                "region_name": f"AxisTransformRegion_{index:03d}",
                "region_type": "AxisTransformRegion",
                "section": "Auxiliary Shape / Mask Flow",
                "op_range": "-",
                "primitive_leaf_count": 1,
                "pruning_role": "propagation_only",
                "dimensions": [{"dim_name": "symbolic_axis", "symbolic_role": "axis_mapping", "status": "propagated", "source": "test", "reason": "test"}],
                "propagation_rules": [],
                "repair_obligations": [],
                "blockers": [{"blocker_id": "b", "blocker_type": "unknown_axis_mapping", "severity": "warning", "explanation": "test"}],
                "evidence": {},
            }
        )

    default_md = region_pruning_semantics_to_markdown(semantics)
    debug_md = region_pruning_semantics_to_markdown(semantics, include_auxiliary_details=True)

    assert "Auxiliary Shape / Axis Propagation Summary" in default_md
    assert "AxisTransformRegion_019" not in default_md
    assert "AxisTransformRegion_019" in debug_md
