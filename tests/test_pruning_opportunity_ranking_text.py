from __future__ import annotations

from model_analysis.pruning_opportunity_ranking import build_pruning_opportunity_ranking, pruning_opportunity_ranking_to_markdown
from model_analysis.pruning_opportunity_ranking_text import pruning_opportunity_ranking_to_text


def ranking():
    return build_pruning_opportunity_ranking(
        {
            "model_name": "synthetic",
            "regions": [
                {
                    "region_id": "ffn",
                    "region_name": "Layer 0 Feed Forward",
                    "source_region_type": "FeedForwardRegion",
                    "semantic_category": "feed_forward_block",
                    "pruning_role": "directly_prunable",
                    "dimensions": [{"dim_name": "intermediate_dim", "status": "prunable"}],
                    "repair_obligations": [{"obligation_type": "same_indices_across_mlp"}, {"obligation_type": "prune_consumer_input"}],
                    "blockers": [],
                    "propagation_rules": [],
                    "evidence": {"source_ops": []},
                },
                {
                    "region_id": "shape",
                    "region_name": "ShapeRegion",
                    "source_region_type": "AxisTransformRegion",
                    "semantic_category": "shape_axis_transform",
                    "pruning_role": "propagation_only",
                    "dimensions": [],
                    "repair_obligations": [],
                    "blockers": [],
                    "propagation_rules": [],
                    "evidence": {"source_ops": []},
                },
            ],
        }
    )


def test_text_dump_contains_candidate_class_and_kind() -> None:
    text = pruning_opportunity_ranking_to_text(ranking())

    assert "pruning_opportunity_ranking @synthetic" in text
    assert "class = safe" in text
    assert "kind = feedforward_intermediate_pruning" in text


def test_markdown_suppresses_auxiliary_details_by_default() -> None:
    markdown = pruning_opportunity_ranking_to_markdown(ranking())

    assert "Auxiliary Metadata Flow" in markdown
    assert "### Auxiliary Details" not in markdown


def test_markdown_can_include_auxiliary_details() -> None:
    markdown = pruning_opportunity_ranking_to_markdown(ranking(), include_auxiliary_details=True)

    assert "### Auxiliary Details" in markdown
    assert "ShapeRegion" in markdown

