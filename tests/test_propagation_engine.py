from __future__ import annotations

from model_analysis.dependency_graph import DependencyEdge, DependencyGraph, PrunableUnit
from model_analysis.propagation_engine import simulate_pruning_action
from model_analysis.pruning_action import PruningAction


def make_action(target_unit_id: str, prune_dim: str = "out_features", indices: list[int] | None = None) -> PruningAction:
    return PruningAction(
        action_id="test_action",
        model_name="tiny",
        target_unit_id=target_unit_id,
        target_unit_name=None,
        target_unit_type=None,
        prune_dim=prune_dim,
        indices=indices or [0, 1],
        amount=len(indices or [0, 1]),
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


def make_qkv_graph() -> DependencyGraph:
    units = [
        PrunableUnit(
            unit_id=f"torch:linear:{name}",
            name=name,
            source="torch",
            unit_type="linear",
            module_or_node_name=name,
            prunable_dims=["out_features"],
            parameter_count=72,
            shape=[8, 8],
            confidence="medium",
            reason="qkv member",
        )
        for name in ["q_proj", "k_proj", "v_proj"]
    ]
    edges = [
        DependencyEdge(
            src="torch:linear:q_proj",
            dst="torch:linear:k_proj",
            edge_type="qkv_coupling",
            affected_dims=["num_heads", "head_dim", "hidden_dim"],
            direction="bidirectional",
            confidence="medium",
            reason="qkv coupling",
        ),
        DependencyEdge(
            src="torch:linear:q_proj",
            dst="torch:linear:v_proj",
            edge_type="qkv_coupling",
            affected_dims=["num_heads", "head_dim", "hidden_dim"],
            direction="bidirectional",
            confidence="medium",
            reason="qkv coupling",
        ),
    ]
    return DependencyGraph("tiny", "local/tiny", "unit-test", prunable_units=units, dependency_edges=edges)


def make_mlp_graph() -> DependencyGraph:
    return DependencyGraph(
        model_name="tiny",
        hf_id="local/tiny",
        task="unit-test",
        prunable_units=[
            PrunableUnit("torch:linear:fc1", "fc1", "torch", "mlp_expansion", "fc1", ["out_features", "intermediate_dim"], 144, [16, 8], "medium", "fc1"),
            PrunableUnit("torch:linear:fc2", "fc2", "torch", "mlp_projection", "fc2", ["in_features", "intermediate_dim"], 136, [8, 16], "medium", "fc2"),
        ],
        dependency_edges=[
            DependencyEdge(
                src="torch:linear:fc1",
                dst="torch:linear:fc2",
                edge_type="mlp_hidden_coupling",
                affected_dims=["intermediate_dim"],
                direction="bidirectional",
                confidence="medium",
                reason="mlp hidden coupling",
            )
        ],
    )


def test_valid_local_linear_pruning():
    plan = simulate_pruning_action(make_linear_graph(), make_action("torch:linear:linear"))

    assert plan.status == "valid_local"
    assert plan.conflicts == []
    assert plan.affected_units[0]["unit_id"] == "torch:linear:linear"


def test_rejected_invalid_dimension():
    plan = simulate_pruning_action(make_linear_graph(), make_action("torch:linear:linear", prune_dim="in_features"))

    assert plan.status == "rejected"
    assert plan.conflicts[0]["type"] == "invalid_prune_dim"


def test_rejected_out_of_bounds_index():
    plan = simulate_pruning_action(make_linear_graph(), make_action("torch:linear:linear", indices=[100]))

    assert plan.status == "rejected"
    assert plan.conflicts[0]["type"] == "index_out_of_bounds"


def test_qkv_coupling_propagates_to_k_and_v():
    plan = simulate_pruning_action(make_qkv_graph(), make_action("torch:linear:q_proj"))

    affected_ids = {unit["unit_id"] for unit in plan.affected_units}
    assert "torch:linear:k_proj" in affected_ids
    assert "torch:linear:v_proj" in affected_ids
    assert any(step.edge_type == "qkv_coupling" and step.status == "propagated" for step in plan.propagation_steps)
    assert plan.status in {"valid_global", "ambiguous"}
    assert plan.status != "valid_local"


def test_mlp_hidden_coupling_adds_projection_constraint():
    plan = simulate_pruning_action(make_mlp_graph(), make_action("torch:linear:fc1"))

    affected_ids = {unit["unit_id"] for unit in plan.affected_units}
    assert "torch:linear:fc2" in affected_ids
    assert any(step.edge_type == "mlp_hidden_coupling" for step in plan.propagation_steps)
    assert any(constraint["edge_type"] == "mlp_hidden_coupling" for constraint in plan.constraints)


def test_residual_coupling_is_ambiguous():
    graph = make_linear_graph()
    graph.prunable_units.append(
        PrunableUnit("torch:linear:residual_peer", "residual_peer", "torch", "linear", "residual_peer", ["out_features"], 32, [8, 4], "medium", "peer")
    )
    graph.dependency_edges.append(
        DependencyEdge(
            src="torch:linear:linear",
            dst="torch:linear:residual_peer",
            edge_type="residual_coupling",
            affected_dims=["hidden_dim"],
            direction="bidirectional",
            confidence="medium",
            reason="residual add",
        )
    )

    plan = simulate_pruning_action(graph, make_action("torch:linear:linear"))

    assert plan.status == "ambiguous"
    assert plan.manual_review_items
    assert any(step.edge_type == "residual_coupling" for step in plan.propagation_steps)
