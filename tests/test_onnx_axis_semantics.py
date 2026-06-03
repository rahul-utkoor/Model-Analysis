from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_analysis.onnx_axis_semantics_export import annotate_onnx_axis_semantics


onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402


def _relu_model(path: Path) -> Path:
    graph = helper.make_graph(
        [helper.make_node("Relu", ["X"], ["Y"], name="relu")],
        "relu_graph",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    onnx.save(model, path)
    return path


def test_allow_no_mlir_does_not_infer_semantics(tmp_path: Path) -> None:
    source = _relu_model(tmp_path / "relu.onnx")
    payload = annotate_onnx_axis_semantics(
        input_path=source,
        output_path=tmp_path / "relu.axis_annotated.onnx",
        sidecar_json=tmp_path / "relu.json",
        mlir_output_dir=tmp_path / "mlir",
        onnx_mlir_path=str(tmp_path / "missing-onnx-mlir"),
        allow_no_mlir=True,
        annotation_mode="doc_string",
        check_onnx=True,
    )

    assert payload["strict_mlir_semantics"] is True
    node = payload["nodes"][0]
    assert node["op_type"] == "Relu"
    assert node["semantic_class"] in {"UNKNOWN", "NO_ACCESS_EVIDENCE"}
    assert node["evidence_tier"] == "NONE"
    assert "MLIR_DERIVED_ELEMENTWISE_PRESERVE" not in payload["semantic_counts"]
    assert payload["blocker_counts"].get("mlir_toolchain_missing", 0) == 1


def test_doc_string_annotation_and_graph_io_unchanged(tmp_path: Path) -> None:
    source = _relu_model(tmp_path / "relu.onnx")
    original_bytes = source.read_bytes()
    output = tmp_path / "relu.axis_annotated.onnx"
    sidecar = tmp_path / "relu.json"
    payload = annotate_onnx_axis_semantics(
        input_path=source,
        output_path=output,
        sidecar_json=sidecar,
        mlir_output_dir=tmp_path / "mlir",
        onnx_mlir_path=str(tmp_path / "missing-onnx-mlir"),
        allow_no_mlir=True,
        annotation_mode="doc_string",
        check_onnx=True,
    )

    assert source.read_bytes() == original_bytes
    annotated = onnx.load(output)
    assert [item.name for item in annotated.graph.input] == ["X"]
    assert [item.name for item in annotated.graph.output] == ["Y"]
    assert annotated.graph.node[0].doc_string.startswith("AxisSemantics:")
    assert "mlir_files" not in annotated.graph.node[0].doc_string
    assert "relations_json" not in annotated.graph.node[0].doc_string
    assert len(annotated.graph.node[0].doc_string) < 1000
    assert payload["original_graph_unchanged"] is True
    assert json.loads(sidecar.read_text(encoding="utf-8"))["strict_mlir_semantics"] is True


def test_sidecar_contains_required_counts(tmp_path: Path) -> None:
    source = _relu_model(tmp_path / "relu.onnx")
    payload = annotate_onnx_axis_semantics(
        input_path=source,
        output_path=tmp_path / "relu.axis_annotated.onnx",
        sidecar_json=tmp_path / "relu.json",
        dot_path=tmp_path / "relu.axis_annotated.dot",
        mlir_output_dir=tmp_path / "mlir",
        onnx_mlir_path=str(tmp_path / "missing-onnx-mlir"),
        allow_no_mlir=True,
        annotation_mode="doc_string",
    )

    assert payload["semantic_counts"]
    assert payload["evidence_tier_counts"]
    assert payload["blocker_counts"]
    assert payload["leader_candidate_counts"]
    assert (tmp_path / "relu.axis_annotated.dot").is_file()


def test_verbose_doc_string_preserves_detail_only_when_requested(tmp_path: Path) -> None:
    source = _relu_model(tmp_path / "relu.onnx")
    output = tmp_path / "relu.axis_annotated.onnx"
    annotate_onnx_axis_semantics(
        input_path=source,
        output_path=output,
        sidecar_json=tmp_path / "relu.json",
        mlir_output_dir=tmp_path / "mlir",
        onnx_mlir_path=str(tmp_path / "missing-onnx-mlir"),
        allow_no_mlir=True,
        annotation_mode="doc_string",
        doc_string_format="verbose",
    )

    annotated = onnx.load(output)
    assert annotated.graph.node[0].doc_string.startswith("axis_semantics=")
    assert "axis_semantics.mlir_evidence_json" in annotated.graph.node[0].doc_string


def test_attributes_mode_uses_compact_attributes_by_default(tmp_path: Path) -> None:
    source = _relu_model(tmp_path / "relu.onnx")
    output = tmp_path / "relu.axis_annotated.onnx"
    annotate_onnx_axis_semantics(
        input_path=source,
        output_path=output,
        sidecar_json=tmp_path / "relu.json",
        mlir_output_dir=tmp_path / "mlir",
        onnx_mlir_path=str(tmp_path / "missing-onnx-mlir"),
        allow_no_mlir=True,
        annotation_mode="attributes",
    )

    annotated = onnx.load(output)
    attr_names = {attr.name for attr in annotated.graph.node[0].attribute}
    assert "axis_semantics.summary" in attr_names
    assert "axis_semantics.mlir_evidence_json" not in attr_names
    assert "axis_semantics.relations_json" not in attr_names
