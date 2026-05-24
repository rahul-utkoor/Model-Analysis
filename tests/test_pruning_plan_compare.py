from __future__ import annotations

from model_analysis.pruning_plan_compare import compare_pruning_plan_sets, comparison_to_markdown
from test_pruning_plan_synthesis import build_plan_set


def test_compare_aggregates_ready_symbolic_counts() -> None:
    a = build_plan_set()
    b = build_plan_set()
    b["model_name"] = "synthetic_b"
    comparison = compare_pruning_plan_sets([a, b])

    assert comparison["plan_status_matrix"]["ready_symbolic"]["synthetic"] == 1
    assert comparison["plan_status_matrix"]["ready_symbolic"]["synthetic_b"] == 1
    assert comparison["summary"]["total_ready_symbolic"] == 2
    markdown = comparison_to_markdown(comparison)
    assert "Pruning Plan Comparison" in markdown
