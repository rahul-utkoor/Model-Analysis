from __future__ import annotations

from model_analysis.region_dimension_ir_compare import compare_region_dimension_irs


def make_ir(model_name: str, axis_role: str, region_type: str, relation: str, blocked: int, unresolved: int) -> dict:
    return {
        "model_name": model_name,
        "blocked_dimensions": [f"d{index}" for index in range(blocked)],
        "unresolved_constraints": [f"c{index}" for index in range(unresolved)],
        "summary": {
            "axis_role_counts": {axis_role: 1},
            "region_type_counts": {region_type: 1},
            "constraint_type_counts": {"mlp_intermediate_same_indices": 1},
            "relation_counts": {relation: 1},
            "num_dimension_variables": 1,
            "num_constraint_equations": 1,
            "num_equivalence_classes": 1,
            "num_blocked_dimensions": blocked,
            "num_unresolved_constraints": unresolved,
        },
    }


def test_compare_region_dimension_irs_builds_summary_matrices() -> None:
    comparison = compare_region_dimension_irs(
        [
            make_ir("bert", "intermediate", "FeedForwardRegion", "same_indices", 0, 0),
            make_ir("gpt2", "hidden", "ResidualMergeRegion", "join_equal", 2, 1),
        ]
    )

    assert comparison["num_models"] == 2
    assert comparison["axis_role_matrix"]["bert"]["intermediate"] == 1
    assert comparison["region_type_matrix"]["gpt2"]["ResidualMergeRegion"] == 1
    assert comparison["relation_matrix"]["gpt2"]["join_equal"] == 1
    assert comparison["blocked_dimension_matrix"]["gpt2"]["blocked_dimensions"] == 2
    assert comparison["summary"]["total_unresolved_constraints"] == 1
