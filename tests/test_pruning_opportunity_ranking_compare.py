from __future__ import annotations

from model_analysis.pruning_opportunity_ranking import build_pruning_opportunity_ranking, pruning_opportunity_ranking_to_dict
from model_analysis.pruning_opportunity_ranking_compare import compare_pruning_opportunity_rankings, comparison_to_markdown


def report(model_name: str, category: str, role: str) -> dict:
    return pruning_opportunity_ranking_to_dict(
        build_pruning_opportunity_ranking(
            {
                "model_name": model_name,
                "regions": [
                    {
                        "region_id": "r0",
                        "region_name": "r0",
                        "source_region_type": "Region",
                        "semantic_category": category,
                        "pruning_role": role,
                        "dimensions": [{"dim_name": "intermediate_dim", "status": "prunable"}],
                        "repair_obligations": [{"obligation_type": "same_indices_across_mlp"}, {"obligation_type": "prune_consumer_input"}],
                        "blockers": [],
                        "propagation_rules": [],
                        "evidence": {"source_ops": []},
                    }
                ],
            }
        )
    )


def test_compare_aggregates_class_counts() -> None:
    comparison = compare_pruning_opportunity_rankings(
        [
            report("a", "feed_forward_block", "directly_prunable"),
            report("b", "attention_score_matmul", "constraint_carrier"),
        ]
    )

    assert comparison["pruning_class_matrix"]["safe"]["a"] == 1
    assert comparison["pruning_class_matrix"]["blocked"]["b"] == 1
    markdown = comparison_to_markdown(comparison)
    assert "Pruning Opportunity Ranking Comparison" in markdown
    assert "Candidate Kinds" in markdown

