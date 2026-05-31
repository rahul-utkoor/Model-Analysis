from __future__ import annotations

import os
from pathlib import Path

import pytest

from experimental.mlir_axis_bridge.native_dependence import load_native_dependence_report
from experimental.mlir_axis_bridge.native_pass_runner import run_native_dependence_tool


@pytest.mark.skipif(os.environ.get("RUN_NATIVE_MLIR_TESTS") != "1", reason="native MLIR tool integration is opt-in")
def test_native_tool_emits_compatible_json(tmp_path) -> None:
    tool = Path("experimental/mlir_axis_bridge/native/build/pruning-axis-dependence")
    if not tool.is_file():
        pytest.skip("native tool has not been built")
    result = run_native_dependence_tool(
        "experimental/mlir_axis_bridge/native/samples/attention_context_affine.mlir",
        tool,
        tmp_path / "native.json",
    )

    assert result.returncode == 0
    report = load_native_dependence_report(result.json_path)
    assert report.analysis_tool == "native_mlir_pass"
    assert report.relations
