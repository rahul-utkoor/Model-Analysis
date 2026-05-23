from __future__ import annotations

from model_analysis.semantic_fusion import (
    build_semantic_fusion_report,
    detect_feedforward_fusions,
    detect_gelu_fusions,
)


def _op(op_id: str, raw_type: str, canonical: str, inputs: list[str], outputs: list[str]) -> dict:
    return {
        "op_id": op_id,
        "name": op_id,
        "op_type": raw_type,
        "canonical_op_type": canonical,
        "inputs": inputs,
        "outputs": outputs,
        "attributes": {},
        "predecessor_ops": [],
        "successor_ops": [],
        "is_fork": False,
        "is_join": canonical in {"elementwise_join", "residual_add"},
        "region_hint": None,
        "source_frontend": "synthetic",
        "source_node_name": op_id,
        "source_location": {},
        "metadata": {},
    }


def gelu_tensor_graph(transform_type: str = "Div", with_multiply_back: bool = True) -> dict:
    transform_id = transform_type.lower()
    ops = [
        _op("linear1", "MatMul", "linear", ["x", "w1"], ["projection"]),
        _op("bias1", "Add", "bias_add", ["projection", "b1"], ["intermediate"]),
        _op("const_scale", "Constant", "constant", [], ["scale"]),
        _op(transform_id, transform_type, "unknown", ["intermediate", "scale"], ["erf_input"]),
        _op("erf", "Erf", "activation", ["erf_input"], ["erf_output"]),
        _op("const_one", "Constant", "constant", [], ["one"]),
        _op("add", "Add", "elementwise_join", ["erf_output", "one"], ["add_output"]),
    ]
    if with_multiply_back:
        ops.extend(
            [
                _op("multiply_back", "Mul", "unknown", ["intermediate", "add_output"], ["multiply_output"]),
                _op("const_half", "Constant", "constant", [], ["half"]),
                _op("final_scale", "Mul", "unknown", ["multiply_output", "half"], ["gelu_output"]),
                _op("linear2", "MatMul", "linear", ["gelu_output", "w2"], ["output_projection"]),
                _op("bias2", "Add", "bias_add", ["output_projection", "b2"], ["out"]),
            ]
        )
    else:
        ops.append(_op("relu_tail", "Relu", "activation", ["add_output"], ["out"]))
    produced = {value: op["op_id"] for op in ops for value in op["outputs"]}
    consumers: dict[str, list[str]] = {}
    for op in ops:
        for value in op["inputs"]:
            consumers.setdefault(value, []).append(op["op_id"])
    for op in ops:
        op["predecessor_ops"] = sorted(
            {produced[value] for value in op["inputs"] if value in produced}
        )
        op["successor_ops"] = sorted(
            {consumer for value in op["outputs"] for consumer in consumers.get(value, [])}
        )
        op["is_fork"] = any(len(consumers.get(value, [])) > 1 for value in op["outputs"])
    all_values = list(
        dict.fromkeys(
            ["x", "w1", "b1", "w2", "b2", *[value for op in ops for value in op["outputs"]]]
        )
    )
    initializers = {"w1", "b1", "w2", "b2"}
    values = [
        {
            "value_id": value,
            "name": value,
            "producer": produced.get(value),
            "consumers": consumers.get(value, []),
            "shape": [1, 4],
            "dtype": "FLOAT",
            "is_initializer": value in initializers,
            "is_graph_input": value == "x",
            "is_graph_output": value == "out",
            "semantic_role": "parameter" if value in initializers else "activation",
            "metadata": {},
        }
        for value in all_values
    ]
    return {
        "graph_id": "tensor_graph::gelu",
        "model_name": "tiny-gelu",
        "source_frontend": "synthetic",
        "ops": ops,
        "values": values,
        "graph_inputs": ["x"],
        "graph_outputs": ["out"],
        "initializers": sorted(initializers),
        "summary": {},
        "metadata": {},
    }


def test_detects_full_div_gelu_pattern() -> None:
    fusions = detect_gelu_fusions(gelu_tensor_graph("Div"))

    assert len(fusions) == 1
    assert fusions[0].fusion_type == "GeluActivation"
    assert fusions[0].confidence == "high"
    assert fusions[0].op_ids == ["div", "erf", "add", "multiply_back", "final_scale"]


def test_detects_mul_transform_gelu_variant() -> None:
    fusion = detect_gelu_fusions(gelu_tensor_graph("Mul"))[0]

    assert fusion.confidence == "high"
    assert "mul" in fusion.op_ids


def test_erf_fragment_without_multiply_back_is_not_high_confidence() -> None:
    fusion = detect_gelu_fusions(gelu_tensor_graph("Div", with_multiply_back=False))[0]

    assert fusion.confidence == "low"


def test_detects_feedforward_fusion_around_decomposed_gelu() -> None:
    graph = gelu_tensor_graph()
    gelu = detect_gelu_fusions(graph)
    feedforward = detect_feedforward_fusions(graph, gelu)
    report = build_semantic_fusion_report(graph)

    assert len(feedforward) == 1
    assert feedforward[0].confidence == "high"
    assert feedforward[0].metadata["first_projection_ops"] == ["linear1", "bias1"]
    assert feedforward[0].metadata["second_projection_ops"] == ["linear2", "bias2"]
    assert report.summary["num_gelu_fusions"] == 1
    assert report.summary["num_feedforward_fusions"] == 1
