from __future__ import annotations

from model_analysis.structural_region_detection import (
    build_op_adjacency_from_tensor_graph,
    build_primitive_regions,
    compute_region_boundary,
    detect_axis_transform_regions,
    detect_feedforward_regions,
    detect_fork_regions,
    detect_linear_projection_regions,
    detect_residual_merge_regions,
)


def _op(op_id, canonical, inputs, outputs, predecessors, successors, fork=False, join=False):
    return {
        "op_id": op_id,
        "name": op_id,
        "op_type": canonical,
        "canonical_op_type": canonical,
        "inputs": inputs,
        "outputs": outputs,
        "attributes": {},
        "predecessor_ops": predecessors,
        "successor_ops": successors,
        "is_fork": fork,
        "is_join": join,
        "region_hint": None,
        "source_frontend": "synthetic",
        "source_node_name": op_id,
        "source_location": {},
        "metadata": {},
    }


def _value(value_id, producer, consumers, graph_input=False, graph_output=False):
    return {
        "value_id": value_id,
        "name": value_id,
        "producer": producer,
        "consumers": consumers,
        "shape": [1, 4],
        "dtype": "FLOAT",
        "is_initializer": value_id in {"w1", "b1", "w2", "b2"},
        "is_graph_input": graph_input,
        "is_graph_output": graph_output,
        "semantic_role": "parameter" if value_id.startswith(("w", "b")) else "activation",
        "metadata": {},
    }


def synthetic_tensor_graph() -> dict:
    ops = [
        _op("linear1", "linear", ["x", "w1"], ["v1"], [], ["bias1"]),
        _op("bias1", "bias_add", ["v1", "b1"], ["v2"], ["linear1"], ["activation"]),
        _op("activation", "activation", ["v2"], ["v3"], ["bias1"], ["linear2"]),
        _op("linear2", "linear", ["v3", "w2"], ["v4"], ["activation"], ["bias2"]),
        _op("bias2", "bias_add", ["v4", "b2"], ["v5"], ["linear2"], ["residual"]),
        _op("residual", "elementwise_join", ["v5", "skip"], ["v6"], ["bias2"], ["norm"], join=True),
        _op("norm", "layer_norm", ["v6"], ["v7"], ["residual"], ["shape"]),
        _op("shape", "shape_op", ["v7"], ["v8"], ["norm"], ["transpose"]),
        _op("transpose", "shape_op", ["v8"], ["out"], ["shape"], [], False, False),
        _op("fan", "unknown", ["fan_in"], ["fan_out"], [], ["branch_a", "branch_b"], fork=True),
        _op("branch_a", "activation", ["fan_out"], ["a"], ["fan"], []),
        _op("branch_b", "activation", ["fan_out"], ["b"], ["fan"], []),
    ]
    consumer_map = {}
    producer_map = {}
    for op in ops:
        for input_value in op["inputs"]:
            consumer_map.setdefault(input_value, []).append(op["op_id"])
        for output in op["outputs"]:
            producer_map[output] = op["op_id"]
    ids = list(dict.fromkeys(["x", "skip", "fan_in", "w1", "b1", "w2", "b2", *producer_map]))
    values = [
        _value(
            value_id,
            producer_map.get(value_id),
            consumer_map.get(value_id, []),
            graph_input=value_id in {"x", "skip", "fan_in"},
            graph_output=value_id in {"out", "a", "b"},
        )
        for value_id in ids
    ]
    return {
        "graph_id": "tensor_graph::tiny",
        "model_name": "tiny",
        "source_frontend": "synthetic",
        "ops": ops,
        "values": values,
        "graph_inputs": ["x", "skip", "fan_in"],
        "graph_outputs": ["out", "a", "b"],
        "initializers": ["w1", "b1", "w2", "b2"],
        "summary": {},
        "metadata": {},
    }


def test_primitive_regions_created_for_all_tensor_ops() -> None:
    graph = synthetic_tensor_graph()

    assert len(build_primitive_regions(graph)) == len(graph["ops"])


def test_boundaries_and_adjacency_use_tensor_ir_connectivity() -> None:
    graph = synthetic_tensor_graph()
    boundary = compute_region_boundary(graph, ["linear1", "bias1"])
    adjacency = build_op_adjacency_from_tensor_graph(graph)

    assert "x" in boundary["boundary_input_values"]
    assert "v2" in boundary["boundary_output_values"]
    assert "v1" in boundary["internal_values"]
    assert adjacency["linear1"]["successors"] == ["bias1"]


def test_semantic_detectors_find_supported_region_patterns() -> None:
    graph = synthetic_tensor_graph()

    linear = detect_linear_projection_regions(graph)
    feedforward = detect_feedforward_regions(graph)
    residual = detect_residual_merge_regions(graph)
    axis = detect_axis_transform_regions(graph)
    forks = detect_fork_regions(graph)

    assert any(region.op_ids == ["linear1", "bias1"] for region in linear)
    assert any(region.op_ids[:4] == ["linear1", "bias1", "activation", "linear2"] for region in feedforward)
    assert any(region.op_ids == ["residual", "norm"] for region in residual)
    assert any("shape" in region.op_ids and "transpose" in region.op_ids for region in axis)
    assert any(region.op_ids[0] == "fan" for region in forks)
