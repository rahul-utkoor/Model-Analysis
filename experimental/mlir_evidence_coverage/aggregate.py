"""Aggregate MLIR evidence coverage across models and pruning patterns."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from experimental.mlir_evidence_coverage.coverage_case import CoverageResult, CoverageVerdict


@dataclass(frozen=True)
class CoverageBreakdown:
    cases: int = 0
    native_proven: int = 0
    access_proven: int = 0
    fallback_proven: int = 0
    blocked_as_expected: int = 0
    partial: int = 0
    missing: int = 0
    unknown: int = 0
    failed: int = 0


@dataclass(frozen=True)
class CoverageAggregate:
    total_cases: int
    found_cases: int
    missing_cases: int
    native_proven: int
    access_proven: int
    fallback_proven: int
    blocked_as_expected: int
    partial: int
    unknown: int
    failed: int
    evidence_tier_counts: dict[str, int] = field(default_factory=dict)
    verdict_counts: dict[str, int] = field(default_factory=dict)
    pattern_counts: dict[str, int] = field(default_factory=dict)
    model_counts: dict[str, int] = field(default_factory=dict)
    per_model: dict[str, CoverageBreakdown] = field(default_factory=dict)
    per_pattern: dict[str, CoverageBreakdown] = field(default_factory=dict)


def _breakdown(results: list[CoverageResult]) -> CoverageBreakdown:
    verdicts = Counter(result.verdict.value for result in results)
    return CoverageBreakdown(
        cases=len(results),
        native_proven=verdicts[CoverageVerdict.NATIVE_PROVEN.value],
        access_proven=verdicts[CoverageVerdict.ACCESS_PROVEN.value],
        fallback_proven=verdicts[CoverageVerdict.FALLBACK_PROVEN.value],
        blocked_as_expected=verdicts[CoverageVerdict.BLOCKED_AS_EXPECTED.value],
        partial=verdicts[CoverageVerdict.PARTIAL.value],
        missing=verdicts[CoverageVerdict.MISSING.value],
        unknown=verdicts[CoverageVerdict.UNKNOWN.value],
        failed=verdicts[CoverageVerdict.FAILED.value],
    )


def aggregate_coverage(results: list[CoverageResult]) -> CoverageAggregate:
    verdicts = Counter(result.verdict.value for result in results)
    by_model: dict[str, list[CoverageResult]] = defaultdict(list)
    by_pattern: dict[str, list[CoverageResult]] = defaultdict(list)
    for result in results:
        by_model[result.case.model_name].append(result)
        by_pattern[result.case.pattern_kind.value].append(result)
    return CoverageAggregate(
        total_cases=len(results),
        found_cases=sum(result.found for result in results),
        missing_cases=sum(not result.found for result in results),
        native_proven=verdicts[CoverageVerdict.NATIVE_PROVEN.value],
        access_proven=verdicts[CoverageVerdict.ACCESS_PROVEN.value],
        fallback_proven=verdicts[CoverageVerdict.FALLBACK_PROVEN.value],
        blocked_as_expected=verdicts[CoverageVerdict.BLOCKED_AS_EXPECTED.value],
        partial=verdicts[CoverageVerdict.PARTIAL.value],
        unknown=verdicts[CoverageVerdict.UNKNOWN.value],
        failed=verdicts[CoverageVerdict.FAILED.value],
        evidence_tier_counts=dict(sorted(Counter(result.evidence_tier.value for result in results).items())),
        verdict_counts=dict(sorted(verdicts.items())),
        pattern_counts=dict(sorted(Counter(result.case.pattern_kind.value for result in results).items())),
        model_counts=dict(sorted(Counter(result.case.model_name for result in results).items())),
        per_model={name: _breakdown(items) for name, items in sorted(by_model.items())},
        per_pattern={name: _breakdown(items) for name, items in sorted(by_pattern.items())},
    )


def aggregate_to_dict(aggregate: CoverageAggregate) -> dict[str, object]:
    return asdict(aggregate)
