from pathlib import Path

from model_analysis.pruning_action import (
    PruningAction,
    load_pruning_action_json,
    pruning_action_to_dict,
    pruning_plan_from_dict,
    pruning_plan_to_dict,
)
from model_analysis.propagation_engine import simulate_pruning_action
from model_analysis.dependency_graph import DependencyGraph, PrunableUnit


def make_linear_graph() -> DependencyGraph:
    return DependencyGraph(
        model_name="tiny",
        hf_id="local/tiny",
        task="unit-test",
        prunable_units=[
            PrunableUnit(
                unit_id="torch:linear:linear",
                name="linear",
                source="torch",
                unit_type="linear",
                module_or_node_name="linear",
                prunable_dims=["out_features"],
                parameter_count=32,
                shape=[8, 4],
                confidence="medium",
                reason="test linear",
            )
        ],
    )


def test_pruning_action_serialization_round_trip(tmp_path: Path):
    action = PruningAction(
        action_id="manual__linear",
        model_name="tiny",
        target_unit_id="torch:linear:linear",
        target_unit_name="linear",
        target_unit_type="linear",
        prune_dim="out_features",
        indices=[0, 1],
        amount=2,
        fraction=None,
        strategy="manual_indices",
        reason="unit test",
    )
    path = tmp_path / "action.json"
    path.write_text(__import__("json").dumps(pruning_action_to_dict(action)), encoding="utf-8")

    loaded = load_pruning_action_json(path)

    assert loaded == action


def test_pruning_plan_serialization_round_trip():
    graph = make_linear_graph()
    action = PruningAction(
        action_id="manual__linear",
        model_name="tiny",
        target_unit_id="torch:linear:linear",
        target_unit_name=None,
        target_unit_type=None,
        prune_dim="out_features",
        indices=[0, 1],
        amount=2,
        fraction=None,
        strategy="manual_indices",
        reason=None,
    )
    plan = simulate_pruning_action(graph, action)
    loaded = pruning_plan_from_dict(pruning_plan_to_dict(plan))

    assert loaded.status == "valid_local"
    assert loaded.action.target_unit_id == action.target_unit_id
