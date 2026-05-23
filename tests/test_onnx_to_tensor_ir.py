from __future__ import annotations

from model_analysis.onnx_to_tensor_ir import build_tensor_graph_from_onnx_summary


def synthetic_summary() -> dict:
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "test",
        "onnx_path": "tiny.onnx",
        "inputs": [
            {"name": "x", "shape": [1, 4], "data_type": "FLOAT"},
            {"name": "skip", "shape": [1, 3], "data_type": "FLOAT"},
            {"name": "ids", "shape": [1, 2], "data_type": "INT64"},
        ],
        "outputs": [
            {"name": "transposed", "shape": [3, 1], "data_type": "FLOAT"},
            {"name": "active", "shape": [1, 3], "data_type": "FLOAT"},
            {"name": "lookup", "shape": [1, 2, 3], "data_type": "FLOAT"},
        ],
        "initializers": [
            {"name": "weight", "dims": [4, 3], "data_type": "FLOAT"},
            {"name": "bias", "dims": [3], "data_type": "FLOAT"},
            {"name": "embedding", "dims": [10, 3], "data_type": "FLOAT"},
        ],
        "tensor_shape_map": {
            "projected": [1, 3],
            "biased": [1, 3],
            "joined": [1, 3],
            "shape": [2],
            "reshaped": [1, 3],
        },
        "nodes": [
            {"name": "project", "op_type": "MatMul", "inputs": ["x", "weight"], "outputs": ["projected"]},
            {"name": "bias", "op_type": "Add", "inputs": ["projected", "bias"], "outputs": ["biased"]},
            {"name": "join", "op_type": "Add", "inputs": ["biased", "skip"], "outputs": ["joined"]},
            {"name": "shape", "op_type": "Shape", "inputs": ["joined"], "outputs": ["shape"]},
            {"name": "reshape", "op_type": "Reshape", "inputs": ["joined", "shape"], "outputs": ["reshaped"]},
            {"name": "transpose", "op_type": "Transpose", "inputs": ["reshaped"], "outputs": ["transposed"]},
            {"name": "activate", "op_type": "Relu", "inputs": ["joined"], "outputs": ["active"]},
            {"name": "lookup", "op_type": "Gather", "inputs": ["embedding", "ids"], "outputs": ["lookup"]},
        ],
    }


def build_graph():
    return build_tensor_graph_from_onnx_summary(
        synthetic_summary(),
        {"name": "tiny", "hf_id": "local/tiny", "task": "test"},
    )


def test_tensor_graph_builds_producers_consumers_and_roles() -> None:
    graph = build_graph()
    values = {value.name: value for value in graph.values}
    ops = {op.name: op for op in graph.ops}

    assert values["projected"].producer == ops["project"].op_id
    assert ops["bias"].op_id in values["projected"].consumers
    assert values["ids"].semantic_role == "index"
    assert values["embedding"].semantic_role == "parameter"
    assert values["shape"].semantic_role == "shape_tensor"
    assert ops["lookup"].canonical_op_type == "embedding_lookup"


def test_tensor_graph_detects_fork_and_join_structure() -> None:
    graph = build_graph()
    ops = {op.name: op for op in graph.ops}

    assert ops["join"].is_join is True
    assert ops["join"].is_fork is True
    assert ops["join"].region_hint == "ResidualJoin"
    assert ops["reshape"].op_id in ops["join"].successor_ops
    assert ops["activate"].op_id in ops["join"].successor_ops
    assert graph.summary["num_fork_ops"] >= 1
    assert graph.summary["num_join_ops"] == 1
