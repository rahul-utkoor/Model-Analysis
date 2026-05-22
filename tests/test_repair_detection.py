from __future__ import annotations

from model_analysis.dependency_graph import DependencyEdge, DependencyGraph, PrunableUnit
from model_analysis.propagation_engine import simulate_pruning_action
from model_analysis.pruning_action import PruningAction
from model_analysis.repair_detection import detect_linear_repair_plan


def make_action(target_unit_id: str = "torch:linear:fc1") -> PruningAction:
    return PruningAction("a1", "tiny", target_unit_id, None, None, "out_features", [0, 1], 2, None, "manual_indices", "test")


def make_mlp_graph(edge_type: str = "mlp_hidden_coupling") -> DependencyGraph:
    return DependencyGraph(
        model_name="tiny",
        hf_id="local/tiny",
        task="unit-test",
        prunable_units=[
            PrunableUnit("torch:linear:fc1", "fc1", "torch", "mlp_expansion", "fc1", ["out_features", "intermediate_dim"], 40, [8, 4], "medium", "fc1"),
            PrunableUnit("torch:linear:fc2", "fc2", "torch", "mlp_projection", "fc2", ["in_features", "intermediate_dim"], 18, [2, 8], "medium", "fc2"),
        ],
        dependency_edges=[
            DependencyEdge("torch:linear:fc1", "torch:linear:fc2", edge_type, ["intermediate_dim"], "bidirectional", "medium", "test edge")
        ],
    )


def test_detects_mlp_pair_repair_from_plan():
    graph = make_mlp_graph()
    plan = simulate_pruning_action(graph, make_action())

    repair_plan = detect_linear_repair_plan(plan, graph)

    assert repair_plan.status == "executable"
    assert len(repair_plan.repair_specs) == 1
    spec = repair_plan.repair_specs[0]
    assert spec.source_module == "fc1"
    assert spec.source_prune_dim == "out_features"
    assert spec.target_module == "fc2"
    assert spec.target_prune_dim == "in_features"
    assert spec.indices == [0, 1]


def test_residual_coupling_is_manual_review_only():
    graph = make_mlp_graph("residual_coupling")
    plan = simulate_pruning_action(graph, make_action())

    repair_plan = detect_linear_repair_plan(plan, graph, allow_ambiguous=True)

    assert repair_plan.status == "rejected"
    assert not repair_plan.repair_specs
    assert repair_plan.manual_review_items


def test_attention_head_dimension_coupling_is_manual_review_only():
    graph = make_mlp_graph("head_dimension_coupling")
    plan = simulate_pruning_action(graph, make_action())

    repair_plan = detect_linear_repair_plan(plan, graph, allow_ambiguous=True)

    assert repair_plan.status == "rejected"
    assert not repair_plan.repair_specs
    assert repair_plan.manual_review_items
