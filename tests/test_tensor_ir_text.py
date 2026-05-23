from __future__ import annotations

from model_analysis.onnx_to_tensor_ir import build_tensor_graph_from_onnx_summary
from model_analysis.tensor_ir_text import tensor_graph_to_text


def synthetic_summary() -> dict:
    return {
        "model_name": "tiny",
        "inputs": [{"name": "x"}, {"name": "skip"}],
        "outputs": [{"name": "joined"}],
        "initializers": [{"name": "weight", "dims": [4, 4]}],
        "nodes": [
            {"name": "project", "op_type": "MatMul", "inputs": ["x", "weight"], "outputs": ["projected"]},
            {"name": "join", "op_type": "Add", "inputs": ["projected", "skip"], "outputs": ["joined"]},
            {"name": "consumer_a", "op_type": "Relu", "inputs": ["joined"], "outputs": ["a"]},
            {"name": "consumer_b", "op_type": "Reshape", "inputs": ["joined"], "outputs": ["b"]},
        ],
    }


def test_tensor_ir_text_contains_graph_operations_and_join_annotations() -> None:
    graph = build_tensor_graph_from_onnx_summary(
        synthetic_summary(),
        {"name": "tiny", "hf_id": "local/tiny", "task": "test"},
    )

    text = tensor_graph_to_text(graph)

    assert "tensor.graph @tiny" in text
    assert 'tensor.op "linear"' in text
    assert 'tensor.op "elementwise_join"' in text
    assert "join(true)" in text
    assert "fork(true)" in text
