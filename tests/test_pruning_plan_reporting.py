from model_analysis.dependency_graph import DependencyGraph, PrunableUnit
from model_analysis.propagation_engine import simulate_pruning_action
from model_analysis.pruning_action import PruningAction
from model_analysis.pruning_plan_reporting import candidate_actions_to_markdown, pruning_plan_to_markdown


def make_action(target_unit_id: str) -> PruningAction:
    return PruningAction(
        action_id="test_action",
        model_name="tiny",
        target_unit_id=target_unit_id,
        target_unit_name=None,
        target_unit_type=None,
        prune_dim="out_features",
        indices=[0, 1],
        amount=2,
        fraction=None,
        strategy="manual_indices",
        reason="unit test",
    )


def make_linear_graph() -> DependencyGraph:
    return DependencyGraph(
        model_name="tiny",
        hf_id="local/tiny",
        task="unit-test",
        prunable_units=[
            PrunableUnit("torch:linear:linear", "linear", "torch", "linear", "linear", ["out_features"], 32, [8, 4], "medium", "test linear")
        ],
    )


def test_pruning_plan_markdown_contains_expected_sections():
    plan = simulate_pruning_action(make_linear_graph(), make_action("torch:linear:linear"))
    markdown = pruning_plan_to_markdown(plan)

    assert "# Pruning Plan:" in markdown
    assert "## Status" in markdown
    assert "## Requested Action" in markdown
    assert "## Propagation Trace" in markdown
    assert "## Manual Review Items" in markdown


def test_candidate_actions_markdown_contains_expected_columns():
    action = make_action("torch:linear:linear")
    markdown = candidate_actions_to_markdown([action])

    assert "# Candidate Pruning Actions" in markdown
    assert "target_unit_id" in markdown
    assert "prune_dim" in markdown
