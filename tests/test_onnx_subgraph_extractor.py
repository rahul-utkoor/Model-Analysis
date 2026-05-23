from __future__ import annotations

from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from model_analysis.onnx_subgraph_extractor import (
    extract_onnx_subgraph_model,
    make_subgraph_export_report,
    make_fallback_value_info,
    netron_index_to_markdown,
    select_subgraphs_for_export,
    subgraph_export_report_to_markdown,
)


def tiny_dag_model():
    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4]),
    ]
    outputs = [helper.make_tensor_value_info("f_out", TensorProto.FLOAT, [1, 4])]
    value_info = [
        helper.make_tensor_value_info("c_out", TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info("d_out", TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info("e_out", TensorProto.FLOAT, [1, 4]),
    ]
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], [1.0])
    nodes = [
        helper.make_node("Add", ["A", "B"], ["c_out"], name="C"),
        helper.make_node("Relu", ["c_out"], ["d_out"], name="D"),
        helper.make_node("Mul", ["c_out", "scale"], ["e_out"], name="E"),
        helper.make_node("Add", ["d_out", "e_out"], ["f_out"], name="F"),
    ]
    graph = helper.make_graph(nodes, "tiny_dag", inputs, outputs, [scale], value_info=value_info)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def dag_record() -> dict:
    return {
        "subgraph_id": "join_fork_join_000001",
        "subgraph_kind": "dag_region",
        "node_names": ["C", "D", "E", "F"],
        "op_types": ["Add", "Relu", "Mul", "Add"],
        "pattern": "JoinForkJoin(Add <- [A, B]; Add -> [Relu, Mul] -> Add)",
        "boundary_input_tensors": ["A", "B"],
        "boundary_output_tensors": ["f_out"],
        "internal_tensors": ["c_out", "d_out", "e_out"],
        "initializer_tensors": ["scale"],
        "metadata": {},
        "source_onnx_path": "source/model.onnx",
    }


def test_extract_dag_subgraph_as_standalone_onnx(tmp_path: Path) -> None:
    output = tmp_path / "fragment.onnx"
    result = extract_onnx_subgraph_model(tiny_dag_model(), dag_record(), output, "tiny")

    assert result.status == "success"
    assert output.exists()
    extracted = onnx.load(output)
    assert [node.name for node in extracted.graph.node] == ["C", "D", "E", "F"]
    assert [value.name for value in extracted.graph.input] == ["A", "B"]
    assert [value.name for value in extracted.graph.output] == ["f_out"]
    assert [initializer.name for initializer in extracted.graph.initializer] == ["scale"]
    metadata = {entry.key: entry.value for entry in extracted.metadata_props}
    assert metadata["subgraph_id"] == "join_fork_join_000001"
    assert "JoinForkJoin" in metadata["pattern"]
    assert metadata["generated_by"] == "model_analysis.onnx_subgraph_extractor"
    assert "Netron visualization" in metadata["extraction_reason"]
    assert result.metadata["checker_status"] == "passed"
    onnx.checker.check_model(extracted)


def test_selection_by_id_and_pattern() -> None:
    records = [
        {"subgraph_id": "path_1", "subgraph_kind": "path", "pattern": "Gemm"},
        {"subgraph_id": "join_1", "subgraph_kind": "join", "pattern": "Join(Add) -> LayerNormalization"},
    ]

    assert select_subgraphs_for_export(records, subgraph_ids=["join_1"])[0]["subgraph_id"] == "join_1"
    assert select_subgraphs_for_export(records, pattern_contains="Gemm")[0]["subgraph_id"] == "path_1"


def test_fallback_value_info_has_unknown_tensor_boundary() -> None:
    value_info = make_fallback_value_info("unknown")

    assert value_info.name == "unknown"
    assert value_info.type.tensor_type.elem_type == TensorProto.FLOAT
    assert value_info.type.tensor_type.shape.dim[0].dim_param == "unknown_dim"


def test_missing_selected_node_returns_failed_export(tmp_path: Path) -> None:
    record = dag_record()
    record["node_names"] = ["C", "Missing"]

    result = extract_onnx_subgraph_model(tiny_dag_model(), record, tmp_path / "bad.onnx", "tiny")

    assert result.status == "failed"
    assert "not found" in result.reason
    assert not (tmp_path / "bad.onnx").exists()


def test_netron_reports_include_original_model_comparison_baseline(tmp_path: Path) -> None:
    source_path = tmp_path / "source" / "model.onnx"
    result = extract_onnx_subgraph_model(
        tiny_dag_model(),
        dag_record(),
        tmp_path / "fragment.onnx",
        "tiny",
    )
    report = make_subgraph_export_report("tiny", source_path, tmp_path, [result])

    index = netron_index_to_markdown(report)
    export_report = subgraph_export_report_to_markdown(report)

    assert "Original Full Model (Comparison Baseline)" in index
    assert f"netron {source_path}" in index
    assert "Original Full Model Baseline" in export_report
    assert f"netron {source_path}" in export_report
