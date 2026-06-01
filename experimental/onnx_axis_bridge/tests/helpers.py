"""Synthetic ONNX builders for bridge tests."""

from __future__ import annotations

from pathlib import Path


def _onnx():
    import onnx

    return onnx


def _value(name: str, shape: list[int]):
    onnx = _onnx()
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _weights(name: str, shape: list[int]):
    onnx = _onnx()
    size = 1
    for dim in shape:
        size *= dim
    return onnx.helper.make_tensor(name, onnx.TensorProto.FLOAT, shape, [0.1] * size)


def _save(path: Path, nodes, name: str, inputs, outputs, *, value_info=(), initializers=()) -> Path:
    onnx = _onnx()
    graph = onnx.helper.make_graph(list(nodes), name, list(inputs), list(outputs), list(initializers), value_info=list(value_info))
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    onnx.save(model, path)
    return path


def make_ffn(path: Path) -> Path:
    onnx = _onnx()
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["Intermediate"], name="first_projection"),
        onnx.helper.make_node("Relu", ["Intermediate"], ["Activated"], name="arbitrary_unary"),
        onnx.helper.make_node("MatMul", ["Activated", "W2"], ["Y"], name="second_projection"),
    ]
    return _save(
        path,
        nodes,
        "synthetic_ffn",
        [_value("X", [1, 4])],
        [_value("Y", [1, 4])],
        value_info=[_value("Intermediate", [1, 8]), _value("Activated", [1, 8])],
        initializers=[_weights("W1", [4, 8]), _weights("W2", [8, 4])],
    )


def make_qk(path: Path) -> Path:
    onnx = _onnx()
    nodes = [onnx.helper.make_node("MatMul", ["Q", "K"], ["Score"], name="arbitrary_rank4_product")]
    return _save(path, nodes, "synthetic_qk", [_value("Q", [1, 2, 5, 3]), _value("K", [1, 2, 3, 5])], [_value("Score", [1, 2, 5, 5])])


def make_context(path: Path) -> Path:
    onnx = _onnx()
    nodes = [onnx.helper.make_node("MatMul", ["P", "V"], ["Context"], name="arbitrary_rank4_product")]
    return _save(path, nodes, "synthetic_context", [_value("P", [1, 2, 5, 5]), _value("V", [1, 2, 5, 3])], [_value("Context", [1, 2, 5, 3])])


def make_attention_value_path(path: Path) -> Path:
    onnx = _onnx()
    nodes = [
        onnx.helper.make_node("MatMul", ["ValueInput", "W_value"], ["V"], name="arbitrary_value_projection"),
        onnx.helper.make_node("MatMul", ["P", "V"], ["Context"], name="arbitrary_context_product"),
        onnx.helper.make_node("MatMul", ["Context", "W_output"], ["Output"], name="arbitrary_output_projection"),
    ]
    return _save(
        path,
        nodes,
        "synthetic_attention_value_path",
        [_value("ValueInput", [1, 2, 5, 4]), _value("P", [1, 2, 5, 5])],
        [_value("Output", [1, 2, 5, 4])],
        value_info=[_value("V", [1, 2, 5, 3]), _value("Context", [1, 2, 5, 3])],
        initializers=[_weights("W_value", [4, 3]), _weights("W_output", [3, 4])],
    )


def make_attention_value_path_with_cache_layout(path: Path) -> Path:
    onnx = _onnx()
    nodes = [
        onnx.helper.make_node("MatMul", ["ValueInput", "W_value"], ["V"], name="arbitrary_value_projection"),
        onnx.helper.make_node("Concat", ["V"], ["CachedV"], axis=2, name="cache_sequence_concat"),
        onnx.helper.make_node("Cast", ["CachedV"], ["TypedV"], to=onnx.TensorProto.FLOAT, name="value_cast"),
        onnx.helper.make_node("MatMul", ["P", "TypedV"], ["Context"], name="arbitrary_context_product"),
        onnx.helper.make_node("MatMul", ["Context", "W_output"], ["Output"], name="arbitrary_output_projection"),
    ]
    return _save(
        path,
        nodes,
        "synthetic_attention_value_path_with_cache_layout",
        [_value("ValueInput", [1, 2, 5, 4]), _value("P", [1, 2, 5, 5])],
        [_value("Output", [1, 2, 5, 4])],
        value_info=[
            _value("V", [1, 2, 5, 3]),
            _value("CachedV", [1, 2, 5, 3]),
            _value("TypedV", [1, 2, 5, 3]),
            _value("Context", [1, 2, 5, 3]),
        ],
        initializers=[_weights("W_value", [4, 3]), _weights("W_output", [3, 4])],
    )


def make_residual(path: Path) -> Path:
    onnx = _onnx()
    nodes = [onnx.helper.make_node("Add", ["A", "B"], ["Y"], name="arbitrary_add")]
    return _save(path, nodes, "synthetic_residual", [_value("A", [1, 5, 4]), _value("B", [1, 5, 4])], [_value("Y", [1, 5, 4])])
