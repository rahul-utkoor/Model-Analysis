from __future__ import annotations

from copy import deepcopy

from model_analysis.pruning_plan_validation_compare import compare_pruning_plan_validations, comparison_to_markdown
from test_pruning_plan_validation import fixtures, validated


def test_compare_summary_aggregates_valid_and_invalid_counts() -> None:
    valid = validated()
    plan_set, ranking, regions, ops = fixtures()
    invalid_plan_set = deepcopy(plan_set)
    invalid_plan_set["plans"][0]["actions"] = [a for a in invalid_plan_set["plans"][0]["actions"] if a["action_type"] != "prune_bias"]
    invalid = validated(invalid_plan_set, ranking, regions, ops)
    invalid["model_name"] = "other"

    comparison = compare_pruning_plan_validations([valid, invalid])

    assert comparison["summary"]["total_valid"] == 1
    assert comparison["summary"]["total_invalid"] == 1
    assert "valid" in comparison_to_markdown(comparison)
