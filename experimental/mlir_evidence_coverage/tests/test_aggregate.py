from experimental.mlir_evidence_coverage.aggregate import aggregate_coverage
from experimental.mlir_evidence_coverage.coverage_case import (
    CoverageCase,
    CoverageEvidenceTier,
    CoveragePatternKind,
    CoverageResult,
    CoverageVerdict,
)


def _result(model: str, pattern: CoveragePatternKind, verdict: CoverageVerdict, tier: CoverageEvidenceTier, found: bool = True) -> CoverageResult:
    case = CoverageCase(f"{model}_{pattern.value}", model, 0, pattern, "node", "node/subgraph.onnx", "EXPECTED", "result", True)
    return CoverageResult(case, found=found, evidence_tier=tier, verdict=verdict)


def test_aggregate_counts_by_verdict() -> None:
    aggregate = aggregate_coverage(
        [
            _result("bert", CoveragePatternKind.ATTENTION_QK_SCORE, CoverageVerdict.BLOCKED_AS_EXPECTED, CoverageEvidenceTier.NATIVE_MLIR_DEPENDENCE),
            _result("opt", CoveragePatternKind.FFN_MLP_INTERMEDIATE, CoverageVerdict.FALLBACK_PROVEN, CoverageEvidenceTier.HIGH_LEVEL_MLIR_DIALECT),
            _result("gpt2", CoveragePatternKind.ATTENTION_VALUE_PATH, CoverageVerdict.MISSING, CoverageEvidenceTier.UNAVAILABLE, False),
        ]
    )
    assert aggregate.total_cases == 3
    assert aggregate.found_cases == 2
    assert aggregate.missing_cases == 1
    assert aggregate.blocked_as_expected == 1
    assert aggregate.fallback_proven == 1


def test_aggregate_counts_by_model_and_pattern() -> None:
    aggregate = aggregate_coverage(
        [
            _result("bert", CoveragePatternKind.FFN_MLP_INTERMEDIATE, CoverageVerdict.NATIVE_PROVEN, CoverageEvidenceTier.NATIVE_MLIR_DEPENDENCE),
            _result("bert", CoveragePatternKind.ATTENTION_VALUE_PATH, CoverageVerdict.MISSING, CoverageEvidenceTier.UNAVAILABLE, False),
        ]
    )
    assert aggregate.per_model["bert"].cases == 2
    assert aggregate.per_model["bert"].missing == 1
    assert aggregate.per_pattern["FFN_MLP_INTERMEDIATE"].native_proven == 1
