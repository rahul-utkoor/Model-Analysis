from __future__ import annotations

from experimental.axis_transfer_analysis.axis_relations import AxisRelationKind
from experimental.axis_transfer_analysis.pattern_recognition import PatternKind
from experimental.mlir_axis_bridge.access_extractor import extract_mlir_access_summary
from experimental.mlir_axis_bridge.axis_summary_builder import build_axis_transfer_from_mlir
from experimental.mlir_axis_bridge.mlir_artifacts import artifact_from_path
from experimental.onnx_axis_bridge.pattern_hints import OnnxPatternHint, OnnxPatternHintKind


def _summary(tmp_path, name: str, text: str):
    source = tmp_path / name
    source.write_text(text, encoding="utf-8")
    return extract_mlir_access_summary(artifact_from_path(source, "synthetic"))


def test_build_qk_from_access_pattern(tmp_path) -> None:
    summary = _summary(
        tmp_path,
        "qk.mlir",
        """
        %0 = affine.load %Q[%b, %head, %q, %d] : memref<1x2x4x8xf32>
        %1 = affine.load %K[%b, %head, %k, %d] : memref<1x2x4x8xf32>
        affine.store %2, %Score[%b, %head, %q, %k] : memref<1x2x4x4xf32>
        """,
    )

    result = build_axis_transfer_from_mlir(summary)

    assert result.evidence_source == "actual_loop_access_evidence"
    assert any(pattern.pattern_kind == PatternKind.QK_SCORE_BLOCKER for pattern in result.pattern_matches)


def test_build_attention_context_from_access_pattern(tmp_path) -> None:
    summary = _summary(
        tmp_path,
        "context.mlir",
        """
        %0 = affine.load %P[%b, %head, %q, %k] : memref<1x2x4x4xf32>
        %1 = affine.load %V[%b, %head, %k, %d] : memref<1x2x4x8xf32>
        affine.store %2, %Context[%b, %head, %q, %d] : memref<1x2x4x8xf32>
        """,
    )

    result = build_axis_transfer_from_mlir(summary)
    transfers = [transfer for op in result.axis_summary.op_summaries for transfer in op.transfers]

    assert result.evidence_source == "actual_loop_access_evidence"
    assert any(
        transfer.relation == AxisRelationKind.PRESERVED
        and transfer.source_axis == "value_dim"
        and transfer.target_axis == "value_context_dim"
        for transfer in transfers
    )


def test_high_level_fallback_with_onnx_hint(tmp_path) -> None:
    summary = _summary(tmp_path, "high.mlir", '%0 = "onnx.MatMul"(%arg0, %arg1) : () -> ()\n')
    hint = OnnxPatternHint(OnnxPatternHintKind.FFN_LIKE, "medium", ("node_000",), ("synthetic topology",), "synthetic")

    result = build_axis_transfer_from_mlir(summary, hint)

    assert result.evidence_source == "high_level_mlir_dialect_evidence"
    assert any(pattern.pattern_kind == PatternKind.FFN_INTERMEDIATE_CHAIN for pattern in result.pattern_matches)
