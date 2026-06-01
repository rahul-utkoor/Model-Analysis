"""Aggregate all-model pruning propagation proof results."""

from __future__ import annotations

from collections import Counter

from experimental.all_model_plan_proof.proof_model import AllModelAggregate, ModelPlanProof, PlanProofCell


def _plan_cells(model: ModelPlanProof) -> list[PlanProofCell]:
    return [*model.ffn_cells, *model.attention_value_cells]


def aggregate_model_proofs(models: list[ModelPlanProof]) -> AllModelAggregate:
    plan_cells = [cell for model in models for cell in _plan_cells(model) if cell.expected]
    tiers = Counter(cell.evidence_tier for cell in plan_cells)
    verdicts = Counter(cell.verdict for cell in plan_cells)
    model_verdicts = Counter(model.final_verdict for model in models)
    return AllModelAggregate(
        models=len(models),
        total_expected=sum(model.summary.total_expected for model in models),
        total_proven=sum(model.summary.total_proven for model in models),
        native_evidence_count=tiers["native_mlir_dependence_evidence"],
        access_evidence_count=tiers["actual_loop_access_evidence"],
        fallback_count=sum(cell.verdict == "fallback_proven" for cell in plan_cells),
        partial_count=verdicts["partial"],
        missing_count=verdicts["missing"],
        unsupported_count=verdicts["unsupported"],
        failed_count=verdicts["failed"],
        evidence_tier_counts=dict(sorted(tiers.items())),
        verdict_counts=dict(sorted(verdicts.items())),
        model_verdict_counts=dict(sorted(model_verdicts.items())),
    )
