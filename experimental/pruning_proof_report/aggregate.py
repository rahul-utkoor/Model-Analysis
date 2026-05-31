"""Aggregate selected-subgraph pruning proof evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from experimental.pruning_proof_report.proof_case import ProofEvidence


@dataclass(frozen=True)
class ProofAggregate:
    cases_total: int
    cases_found: int
    cases_missing: int
    proven: int
    fallback_proven: int
    blocked: int
    partial: int
    unknown: int
    failed: int
    evidence_source_counts: dict[str, int]
    pattern_counts: dict[str, int]
    model_counts: dict[str, int]
    limitations: list[str]


def aggregate_evidence(evidence: list[ProofEvidence]) -> ProofAggregate:
    verdicts = Counter(item.verdict for item in evidence)
    sources = Counter(item.evidence_source for item in evidence)
    patterns = Counter(pattern for item in evidence for pattern in item.recognized_patterns)
    models = Counter(item.model_name for item in evidence)
    limitations = list(dict.fromkeys(limit for item in evidence for limit in item.limitations))
    return ProofAggregate(
        cases_total=len(evidence),
        cases_found=sum(item.found for item in evidence),
        cases_missing=sum(not item.found for item in evidence),
        proven=verdicts["proven"],
        fallback_proven=verdicts["fallback_proven"],
        blocked=verdicts["blocked"],
        partial=verdicts["partial"],
        unknown=verdicts["unknown"],
        failed=verdicts["failed"],
        evidence_source_counts=dict(sorted(sources.items())),
        pattern_counts=dict(sorted(patterns.items())),
        model_counts=dict(sorted(models.items())),
        limitations=limitations,
    )


def aggregate_to_dict(aggregate: ProofAggregate) -> dict[str, object]:
    return asdict(aggregate)
