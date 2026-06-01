"""Data records for the all-model propagation proof."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from experimental.all_model_plan_proof.config import PlanFamily


@dataclass
class PlanProofCell:
    model_name: str
    artifact_name: str
    layer_index: int
    family: PlanFamily
    expected: bool = True
    found_artifact: bool = False
    evidence_tier: str = "unavailable"
    recognized_pattern: str = ""
    dfa_ran: bool = False
    dfa_reached_fixed_point: bool = False
    verdict: str = "missing"
    report_path: str | None = None
    artifact_path: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelPlanSummary:
    ffn_expected: int = 0
    ffn_found: int = 0
    ffn_proven: int = 0
    attention_expected: int = 0
    attention_found: int = 0
    attention_proven: int = 0
    attention_partial: int = 0
    attention_missing: int = 0
    attention_unsupported: int = 0
    qk_blockers_expected: int = 0
    qk_blockers_proven: int = 0
    total_expected: int = 0
    total_proven: int = 0
    native_evidence_count: int = 0
    fallback_count: int = 0
    partial_count: int = 0
    missing_count: int = 0
    unsupported_count: int = 0
    failed_count: int = 0


@dataclass
class ModelPlanProof:
    model_name: str
    artifact_name: str
    layer_count: int
    ffn_cells: list[PlanProofCell]
    attention_value_cells: list[PlanProofCell]
    qk_blocker_cells: list[PlanProofCell]
    summary: ModelPlanSummary
    final_verdict: str
    notes: str = ""


@dataclass
class AllModelAggregate:
    models: int = 0
    total_expected: int = 0
    total_proven: int = 0
    native_evidence_count: int = 0
    access_evidence_count: int = 0
    fallback_count: int = 0
    partial_count: int = 0
    missing_count: int = 0
    unsupported_count: int = 0
    failed_count: int = 0
    evidence_tier_counts: dict[str, int] = field(default_factory=dict)
    verdict_counts: dict[str, int] = field(default_factory=dict)
    model_verdict_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class AllModelPlanProof:
    generated_at: str
    models: list[ModelPlanProof]
    aggregate: AllModelAggregate
    limitations: list[str]

    @classmethod
    def create(cls, models: list[ModelPlanProof], aggregate: AllModelAggregate, limitations: list[str]) -> "AllModelPlanProof":
        return cls(datetime.now(timezone.utc).isoformat(), models, aggregate, limitations)
