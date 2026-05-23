from __future__ import annotations

from model_analysis.tensor_ir import (
    TensorGraph,
    TensorOp,
    TensorValue,
    canonicalize_op_type,
    tensor_graph_to_dict,
    tensor_op_to_dict,
    tensor_value_to_dict,
)


def test_tensor_value_and_op_serialization() -> None:
    value = TensorValue("value::x", "x", None, ["op::matmul"], [1, 4], "FLOAT", False, True, False, "activation")
    op = TensorOp(
        "op::matmul",
        "matmul",
        "MatMul",
        "linear",
        ["value::x"],
        ["value::y"],
        {},
        [],
        [],
        False,
        False,
        "LinearProjection",
        "onnx",
        "matmul",
        {"node_index": 0},
    )
    graph = TensorGraph("graph::tiny", "tiny", "onnx", [op], [value], ["value::x"], [], [], {})

    assert tensor_value_to_dict(value)["semantic_role"] == "activation"
    assert tensor_op_to_dict(op)["canonical_op_type"] == "linear"
    assert tensor_graph_to_dict(graph)["source_frontend"] == "onnx"


def test_canonical_op_typing_is_conservative() -> None:
    initializers = {"weight", "bias", "embedding"}

    assert canonicalize_op_type("MatMul", ["x", "weight"], ["y"], initializers)[0] == "linear"
    assert canonicalize_op_type("MatMul", ["q", "k"], ["scores"], initializers)[0] == "matmul"
    assert canonicalize_op_type("Add", ["y", "bias"], ["biased"], initializers)[0] == "bias_add"
    assert canonicalize_op_type("Add", ["left", "right"], ["join"], initializers)[0] in {"residual_add", "elementwise_join"}
    assert canonicalize_op_type("Gather", ["embedding", "ids"], ["lookup"], initializers)[0] == "embedding_lookup"


def test_shape_operations_receive_shape_transform_hint() -> None:
    for op_type in ("Shape", "Reshape", "Transpose"):
        canonical, hint, _ = canonicalize_op_type(op_type, ["x"], ["y"], set())

        assert canonical in {"shape_op", "axis_transform"}
        assert hint == "ShapeTransform"
