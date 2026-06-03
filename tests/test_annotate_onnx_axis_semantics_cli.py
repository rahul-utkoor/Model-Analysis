from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402


def _make_relu(path: Path) -> Path:
    graph = helper.make_graph(
        [helper.make_node("Relu", ["X"], ["Y"], name="relu")],
        "relu_cli",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]), path)
    return path


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/annotate_onnx_axis_semantics.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--allow-no-mlir" in completed.stdout


def test_cli_doc_string_mode_without_mlir_is_unknown(tmp_path: Path) -> None:
    source = _make_relu(tmp_path / "relu.onnx")
    output = tmp_path / "relu.axis_annotated.onnx"
    sidecar = tmp_path / "relu.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/annotate_onnx_axis_semantics.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--sidecar-json",
            str(sidecar),
            "--mlir-output-dir",
            str(tmp_path / "mlir"),
            "--onnx-mlir",
            str(tmp_path / "missing-onnx-mlir"),
            "--allow-no-mlir",
            "--annotation-mode",
            "doc_string",
            "--check-onnx",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert output.is_file()
    assert payload["strict_mlir_semantics"] is True
    assert payload["nodes"][0]["semantic_class"] in {"UNKNOWN", "NO_ACCESS_EVIDENCE"}
    assert payload["nodes"][0]["evidence_tier"] == "NONE"
