from __future__ import annotations

from experimental.mlir_axis_bridge.bridge_runner import analyze_onnx_with_mlir_bridge
from experimental.mlir_axis_bridge.onnx_mlir_runner import MlirLoweringResult
from experimental.mlir_axis_bridge.toolchain import ToolchainStatus
from experimental.onnx_axis_bridge.onnx_loader import OnnxSubgraph
from experimental.onnx_axis_bridge.pattern_hints import OnnxPatternHint, OnnxPatternHintKind


def test_runner_uses_fallback_when_mlir_accesses_are_unavailable(monkeypatch, tmp_path) -> None:
    source = tmp_path / "local.onnx"
    source.write_bytes(b"local-test-placeholder")
    hint = OnnxPatternHint(OnnxPatternHintKind.FFN_LIKE, "medium", ("node_000",), ("synthetic topology",), "synthetic")
    monkeypatch.setattr(
        "experimental.mlir_axis_bridge.bridge_runner.check_toolchain",
        lambda *_: ToolchainStatus("/tool/onnx-mlir", "/tool/mlir-opt", True, True),
    )
    monkeypatch.setattr(
        "experimental.mlir_axis_bridge.bridge_runner.load_onnx_subgraph",
        lambda *_: OnnxSubgraph(str(source), "local", (), {}, (), (), ()),
    )
    monkeypatch.setattr("experimental.mlir_axis_bridge.bridge_runner.infer_pattern_hints", lambda *_: [hint])
    monkeypatch.setattr(
        "experimental.mlir_axis_bridge.bridge_runner.lower_onnx_subgraph_to_mlir",
        lambda *_: MlirLoweringResult(str(source), str(tmp_path)),
    )

    result = analyze_onnx_with_mlir_bridge(source, tmp_path / "out")

    assert result.evidence_source == ["onnx_hint_fallback"]
    assert result.summary["dfa_propagation_results"] == 1
