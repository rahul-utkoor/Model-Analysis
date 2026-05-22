from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from model_analysis.dependency_graph import DependencyGraph, PrunableUnit
from model_analysis.propagation_engine import simulate_pruning_action
from model_analysis.pruning_action import PruningAction
from model_analysis.pruning_plan_executor import execute_linear_pruning_plan, extract_linear_prune_specs_from_plan


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 6)
        self.fc2 = torch.nn.Linear(6, 3)

    def forward(self, inputs):
        return self.fc2(self.fc1(inputs))


def make_graph():
    return DependencyGraph(
        model_name="tiny",
        hf_id="local/tiny",
        task="unit-test",
        prunable_units=[
            PrunableUnit("torch:linear:fc1", "fc1", "torch", "mlp_expansion", "fc1", ["out_features", "intermediate_dim"], 30, [6, 4], "medium", "fc1"),
            PrunableUnit("torch:linear:fc2", "fc2", "torch", "mlp_projection", "fc2", ["in_features", "intermediate_dim"], 21, [3, 6], "medium", "fc2"),
        ],
        dependency_edges=[],
    )


def make_plan(graph=None):
    graph = graph or make_graph()
    action = PruningAction("a1", "tiny", "torch:linear:fc1", None, None, "out_features", [0, 1], 2, None, "manual_indices", None)
    return simulate_pruning_action(graph, action)


def test_extract_linear_prune_specs_from_plan_only_target():
    graph = make_graph()
    plan = make_plan(graph)

    specs = extract_linear_prune_specs_from_plan(plan, graph, only_target=True)

    assert len(specs) == 1
    assert specs[0].module_name == "fc1"
    assert specs[0].prune_dim == "out_features"


def test_execute_linear_pruning_plan_dry_run_does_not_modify_model(tmp_path: Path):
    model = TinyModel()
    graph = make_graph()
    plan = make_plan(graph)

    report = execute_linear_pruning_plan(
        model,
        "tiny",
        tmp_path / "source",
        tmp_path / "out",
        plan,
        graph,
        only_target=True,
        dry_run=True,
    )

    assert report.status == "success"
    assert report.skipped_records
    assert list(model.fc1.weight.shape) == [6, 4]


def test_execute_linear_pruning_plan_modifies_expected_shape(tmp_path: Path):
    model = TinyModel()
    graph = make_graph()
    plan = make_plan(graph)

    report = execute_linear_pruning_plan(
        model,
        "tiny",
        tmp_path / "source",
        tmp_path / "out",
        plan,
        graph,
        only_target=True,
        dry_run=False,
    )

    assert report.status == "success"
    assert report.applied_records[0].module_name == "fc1"
    assert list(model.fc1.weight.shape) == [4, 4]
    assert list(model.fc2.weight.shape) == [3, 6]


def test_only_target_prunes_only_target(tmp_path: Path):
    model = TinyModel()
    graph = make_graph()
    plan = make_plan(graph)
    plan.affected_units.append(
        {
            "unit_id": "torch:linear:fc2",
            "name": "fc2",
            "unit_type": "mlp_projection",
            "source": "torch",
            "affected_dim": "in_features",
            "indices": [0, 1],
            "reason": "propagated",
        }
    )

    execute_linear_pruning_plan(model, "tiny", tmp_path / "source", tmp_path / "out", plan, graph, only_target=True)

    assert list(model.fc1.weight.shape) == [4, 4]
    assert list(model.fc2.weight.shape) == [3, 6]
