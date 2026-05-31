from __future__ import annotations

import json

from experimental.axis_transfer_analysis.axis_relations import AxisRelationKind
from experimental.axis_transfer_analysis.pattern_recognition import PatternKind
from experimental.mlir_axis_bridge.access_extractor import extract_mlir_access_summary
from experimental.mlir_axis_bridge.axis_summary_builder import build_axis_transfer_from_native_dependence
from experimental.mlir_axis_bridge.mlir_artifacts import artifact_from_path
from experimental.mlir_axis_bridge.native_dependence import load_native_dependence_report
from experimental.onnx_axis_bridge.pattern_hints import OnnxPatternHint, OnnxPatternHintKind


def _write_report(tmp_path, relations):
    path = tmp_path / "native.json"
    path.write_text(
        json.dumps(
            {
                "mlir_file": "synthetic.mlir",
                "analysis_tool": "native_mlir_pass",
                "dialects_seen": ["affine.for", "affine.load", "affine.store"],
                "relations": relations,
                "reductions": ["d"],
                "preserved_axes": [],
                "blocked_axes": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_native_dependence_json_loader(tmp_path) -> None:
    report = load_native_dependence_report(
        _write_report(
            tmp_path,
            [
                {
                    "relation_id": "r1",
                    "source_tensor": "Q",
                    "source_indices": ["b", "h", "q", "d"],
                    "target_tensor": "Score",
                    "target_indices": ["b", "h", "q", "k"],
                    "loop_ivs": ["b", "h", "q", "k", "d"],
                    "relation_kind": "reduced",
                    "dependence_kind": "reduction",
                    "affine_evidence": ["affine.load", "affine.store"],
                    "proof": "d is reduced",
                    "confidence": "high",
                }
            ],
        )
    )

    assert report.analysis_tool == "native_mlir_pass"
    assert report.relations[0].relation_kind == "reduced"
    assert report.relations[0].source_indices[-1] == "d"


def test_native_qk_dependence_to_axis_summary(tmp_path) -> None:
    report = load_native_dependence_report(
        _write_report(
            tmp_path,
            [
                {
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
                    "confidence": "high",
                }
            ],
        )
    )
    hint = OnnxPatternHint(OnnxPatternHintKind.QK_SCORE_LIKE, "high", ("score",), (), "")

    result = build_axis_transfer_from_native_dependence(report, hint)

    assert result.evidence_source == "native_mlir_dependence_evidence"
    assert any(pattern.pattern_kind == PatternKind.QK_SCORE_BLOCKER for pattern in result.pattern_matches)


def test_native_attention_context_to_axis_summary() -> None:
    report = load_native_dependence_report("experimental/mlir_axis_bridge/native/sample_expected_output.json")
    hint = OnnxPatternHint(OnnxPatternHintKind.ATTENTION_CONTEXT_LIKE, "high", ("context",), (), "")

    result = build_axis_transfer_from_native_dependence(report, hint)
    transfers = [transfer for op in result.axis_summary.op_summaries for transfer in op.transfers]

    assert result.evidence_source == "native_mlir_dependence_evidence"
    assert any(
        transfer.relation == AxisRelationKind.PRESERVED
        and transfer.source_axis == "value_dim"
        and transfer.target_axis == "value_context_dim"
        for transfer in transfers
    )


def test_python_affine_dependence_summary_preserved(tmp_path) -> None:
    source = tmp_path / "preserved.mlir"
    source.write_text(
        """
        affine.for %j = 0 to 8 {
          %0 = affine.load %X[%b, %s, %j] : memref<1x2x8xf32>
          affine.store %0, %Y[%b, %s, %j] : memref<1x2x8xf32>
        }
        """,
        encoding="utf-8",
    )
    summary = extract_mlir_access_summary(artifact_from_path(source, "synthetic"))

    assert "j" in summary.dependence_report.preserved_axes
    assert summary.access_records[0].loop_ivs == ("j",)


def test_python_affine_dependence_summary_reduced(tmp_path) -> None:
    source = tmp_path / "reduced.mlir"
    source.write_text(
        """
        affine.for %j = 0 to 8 {
          %0 = affine.load %X[%b, %s, %j] : memref<1x2x8xf32>
          %1 = affine.load %W[%j, %h] : memref<8x4xf32>
          affine.store %2, %Y[%b, %s, %h] : memref<1x2x4xf32>
        }
        """,
        encoding="utf-8",
    )
    summary = extract_mlir_access_summary(artifact_from_path(source, "synthetic"))

    assert "j" in summary.dependence_report.reductions
    assert any(relation.relation_kind == "reduced" and "j" in relation.source_indices for relation in summary.dependence_report.relations)
