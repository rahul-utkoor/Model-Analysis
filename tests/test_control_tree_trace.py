from __future__ import annotations

from model_analysis.control_tree_trace import (
    build_control_tree_trace,
    build_initial_working_graph_from_tensor_ir,
    build_ordered_region_candidates_for_trace,
    collapse_nodes_into_region,
    initialize_control_tree_trace,
)


def tiny_tensor_graph() -> dict:
    ops = [
        {
            "op_id": "op::linear1",
            "name": "linear1",
            "op_type": "MatMul",
            "canonical_op_type": "linear",
            "inputs": ["x", "w1"],
            "outputs": ["h"],
            "predecessor_ops": [],
            "successor_ops": ["op::gelu", "op::skip"],
            "is_fork": True,
            "is_join": False,
            "source_frontend": "onnx",
        },
        {
            "op_id": "op::gelu",
            "name": "gelu",
            "op_type": "Gelu",
            "canonical_op_type": "activation",
            "inputs": ["h"],
            "outputs": ["a"],
            "predecessor_ops": ["op::linear1"],
            "successor_ops": ["op::linear2"],
            "is_fork": False,
            "is_join": False,
            "source_frontend": "onnx",
        },
        {
            "op_id": "op::linear2",
            "name": "linear2",
            "op_type": "MatMul",
            "canonical_op_type": "linear",
            "inputs": ["a", "w2"],
            "outputs": ["y"],
            "predecessor_ops": ["op::gelu"],
            "successor_ops": ["op::add"],
            "is_fork": False,
            "is_join": False,
            "source_frontend": "onnx",
        },
        {
            "op_id": "op::skip",
            "name": "skip",
            "op_type": "Identity",
            "canonical_op_type": "unknown",
            "inputs": ["h"],
            "outputs": ["s"],
            "predecessor_ops": ["op::linear1"],
            "successor_ops": ["op::add"],
            "is_fork": False,
            "is_join": False,
            "source_frontend": "onnx",
        },
        {
            "op_id": "op::add",
            "name": "add",
            "op_type": "Add",
            "canonical_op_type": "residual_add",
            "inputs": ["y", "s"],
            "outputs": ["z"],
            "predecessor_ops": ["op::linear2", "op::skip"],
            "successor_ops": [],
            "is_fork": False,
            "is_join": True,
            "source_frontend": "onnx",
        },
    ]
    values = [
        {"value_id": "x", "name": "x", "producer": None, "consumers": ["op::linear1"], "semantic_role": "activation"},
        {"value_id": "w1", "name": "w1", "producer": None, "consumers": ["op::linear1"], "semantic_role": "parameter"},
        {"value_id": "h", "name": "h", "producer": "op::linear1", "consumers": ["op::gelu", "op::skip"], "semantic_role": "activation"},
        {"value_id": "a", "name": "a", "producer": "op::gelu", "consumers": ["op::linear2"], "semantic_role": "activation"},
        {"value_id": "w2", "name": "w2", "producer": None, "consumers": ["op::linear2"], "semantic_role": "parameter"},
        {"value_id": "y", "name": "y", "producer": "op::linear2", "consumers": ["op::add"], "semantic_role": "activation"},
        {"value_id": "s", "name": "s", "producer": "op::skip", "consumers": ["op::add"], "semantic_role": "activation"},
        {"value_id": "z", "name": "z", "producer": "op::add", "consumers": [], "semantic_role": "activation"},
    ]
    return {
        "graph_id": "graph::tiny",
        "model_name": "tiny",
        "source_frontend": "onnx",
        "ops": ops,
        "values": values,
        "graph_inputs": ["x"],
        "graph_outputs": ["z"],
        "initializers": ["w1", "w2"],
        "summary": {},
        "metadata": {},
    }


def tiny_tree() -> dict:
    return {
        "model_name": "tiny",
        "source_frontend": "onnx",
        "root_region_id": "region::model",
        "regions": [
            {
                "region_id": "region::linear1",
                "region_type": "LinearProjectionRegion",
                "name": "linear1",
                "op_ids": ["op::linear1"],
                "confidence": "high",
                "reason": "linear op",
                "metadata": {},
            },
            {
                "region_id": "region::activation",
                "region_type": "ActivationRegion",
                "name": "gelu",
                "op_ids": ["op::gelu"],
                "confidence": "high",
                "reason": "activation op",
                "metadata": {},
            },
            {
                "region_id": "region::linear2",
                "region_type": "LinearProjectionRegion",
                "name": "linear2",
                "op_ids": ["op::linear2"],
                "confidence": "high",
                "reason": "linear op",
                "metadata": {},
            },
            {
                "region_id": "region::ffn",
                "region_type": "FeedForwardRegion",
                "name": "ffn",
                "op_ids": ["op::linear1", "op::gelu", "op::linear2"],
                "confidence": "high",
                "reason": "projection activation projection",
                "metadata": {},
            },
            {
                "region_id": "region::residual",
                "region_type": "ResidualMergeRegion",
                "name": "residual",
                "op_ids": ["op::add"],
                "confidence": "high",
                "reason": "residual join",
                "metadata": {},
            },
            {
                "region_id": "region::ffn_duplicate",
                "region_type": "FeedForwardRegion",
                "name": "ffn duplicate",
                "op_ids": ["op::linear1", "op::gelu", "op::linear2"],
                "confidence": "high",
                "reason": "duplicate candidate for skip behavior",
                "metadata": {},
            },
        ],
        "interfaces": [
            {"region_id": "region::linear1", "pruning_role": "directly_prunable"},
            {"region_id": "region::linear2", "pruning_role": "directly_prunable"},
            {"region_id": "region::ffn", "pruning_role": "directly_prunable"},
            {"region_id": "region::residual", "pruning_role": "blocked"},
        ],
        "summary": {},
        "metadata": {},
    }


def test_initial_working_graph_has_one_node_per_tensor_op_and_dataflow_edges() -> None:
    graph = build_initial_working_graph_from_tensor_ir(tiny_tensor_graph())

    assert len(graph.active_nodes) == 5
    assert len([edge for edge in graph.edges if edge.edge_kind == "dataflow"]) == 5
    assert graph.source_op_to_active_node["op::linear1"] in graph.active_nodes


def test_collapse_nodes_into_region_reduces_active_count_and_redirects_edges() -> None:
    graph = build_initial_working_graph_from_tensor_ir(tiny_tensor_graph())
    linear = graph.source_op_to_active_node["op::linear1"]
    gelu = graph.source_op_to_active_node["op::gelu"]

    graph, info = collapse_nodes_into_region(
        graph,
        [linear, gelu],
        "region::linear_gelu",
        "ActivationRegion",
        "linear gelu",
        "medium",
        "propagation_only",
        "test collapse",
    )

    assert "region::linear_gelu" in graph.active_nodes
    assert linear not in graph.active_nodes
    assert len(graph.active_nodes) == 4
    assert info["output_boundary_values"] == ["a", "h"]
    assert any(edge.src == "region::linear_gelu" and edge.dst == graph.source_op_to_active_node["op::linear2"] for edge in graph.edges)


def test_initialization_step_is_emitted() -> None:
    _, step = initialize_control_tree_trace(tiny_tensor_graph())

    assert step.pass_name == "initialize_primitives"
    assert step.action == "initialize"
    assert step.after_summary["num_active_nodes"] == 5


def test_ordered_candidates_prioritize_semantic_regions_above_primitives() -> None:
    candidates = build_ordered_region_candidates_for_trace(tiny_tensor_graph(), structural_region_tree=tiny_tree())
    types = [candidate["region_type"] for candidate in candidates]

    assert "FeedForwardRegion" in types
    assert "PrimitiveRegion" not in types
    assert types.index("FeedForwardRegion") > types.index("LinearProjectionRegion")


def test_trace_contains_collapse_and_skip_steps() -> None:
    trace = build_control_tree_trace(tiny_tensor_graph(), structural_region_tree=tiny_tree(), max_snapshot_nodes=50)

    actions = [step.action for step in trace.steps]
    assert "collapse" in actions
    assert "skip" in actions
    assert trace.summary["num_collapse_steps"] >= 1
    assert any(step.created_region_type == "FeedForwardRegion" for step in trace.steps if step.action == "collapse")
