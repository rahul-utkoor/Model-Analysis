from __future__ import annotations

from experimental.mlir_axis_bridge.bridge_runner import analyze_onnx_with_mlir_bridge
from experimental.mlir_axis_bridge.onnx_mlir_runner import MlirLoweringResult
from experimental.mlir_axis_bridge.toolchain import ToolchainStatus
from experimental.mlir_axis_bridge.report import render_markdown
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


def test_bridge_prefers_native_when_requested(monkeypatch, tmp_path) -> None:
    source = tmp_path / "local.onnx"
    source.write_bytes(b"local-test-placeholder")
    native_json = tmp_path / "native.json"
    native_json.write_text(
        """
        {
          "mlir_file": "synthetic.mlir",
          "analysis_tool": "native_mlir_pass",
          "dialects_seen": ["affine.load", "affine.store"],
          "relations": [{
            "relation_id": "qk",
            "source_tensor": "Q",
            "source_indices": ["b", "h", "q", "d"],
            "target_tensor": "Score",
            "target_indices": ["b", "h", "q", "k"],
            "loop_ivs": ["b", "h", "q", "k", "d"],
            "relation_kind": "mixed",
            "dependence_kind": "reduction",
            "affine_evidence": ["QK contraction"],
            "proof": "qk_score_contraction_mixes_channels",
            "confidence": "high"
          }]
        }
        """,
        encoding="utf-8",
    )
    hint = OnnxPatternHint(OnnxPatternHintKind.QK_SCORE_LIKE, "high", ("node_000",), ("synthetic topology",), "synthetic")
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

    result = analyze_onnx_with_mlir_bridge(source, tmp_path / "out", native_dependence_json=native_json, prefer_native_dependence=True)

    assert result.evidence_source == ["native_mlir_dependence_evidence"]
    assert "native_mlir_dependence_evidence" in render_markdown(result)


def test_bridge_falls_back_when_native_pass_missing(monkeypatch, tmp_path) -> None:
    source = tmp_path / "local.onnx"
    source.write_bytes(b"local-test-placeholder")
    mlir = tmp_path / "lowered.mlir"
    mlir.write_text(
        """
        affine.for %j = 0 to 8 {
          %0 = affine.load %X[%b, %s, %j] : memref<1x2x8xf32>
          affine.store %0, %Y[%b, %s, %j] : memref<1x2x8xf32>
        }
        """,
        encoding="utf-8",
    )
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
        lambda *_: MlirLoweringResult(str(source), str(tmp_path), generated_files=[str(mlir)]),
    )

    result = analyze_onnx_with_mlir_bridge(source, tmp_path / "out", run_native_pass=True, native_pass_tool="/definitely/missing/tool")

    assert result.evidence_source == ["actual_loop_access_evidence"]
    assert any("native pass unavailable" in warning for warning in result.warnings)


def test_report_evidence_hierarchy_table(monkeypatch, tmp_path) -> None:
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

    markdown = render_markdown(analyze_onnx_with_mlir_bridge(source, tmp_path / "out"))

    assert "## Evidence Hierarchy" in markdown
    assert "| Evidence tier | Available | Used | Notes |" in markdown
