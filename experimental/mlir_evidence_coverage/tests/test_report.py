from experimental.mlir_evidence_coverage.aggregate import aggregate_coverage
from experimental.mlir_evidence_coverage.coverage_case import (
    CoverageCase,
    CoverageEvidenceTier,
    CoveragePatternKind,
    CoverageResult,
    CoverageVerdict,
)
from experimental.mlir_evidence_coverage.report import render_index_markdown


def test_report_contains_evidence_tier_definitions_and_interpretation() -> None:
    case = CoverageCase("bert_layer0_score", "bert", 0, CoveragePatternKind.ATTENTION_QK_SCORE, "score", "score.onnx", "QK_SCORE_BLOCKER", "blocked", True)
    result = CoverageResult(case, found=True, evidence_tier=CoverageEvidenceTier.NATIVE_MLIR_DEPENDENCE, verdict=CoverageVerdict.BLOCKED_AS_EXPECTED)
    text = render_index_markdown([result], aggregate_coverage([result]))
    assert "# MLIR Evidence Coverage Study" in text
    assert "## Evidence Tier Definitions" in text
    assert "`native_mlir_dependence_evidence`" in text
    assert "Native MLIR dependence evidence currently covers selected important cases, not every propagation case." in text
    assert "Attention context value-axis preservation can be native-proven locally" in text
    assert "MLIR remains a local evidence generator" in text
