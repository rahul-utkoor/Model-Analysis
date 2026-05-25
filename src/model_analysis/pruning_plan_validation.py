"""Validate symbolic pruning plans against ranking, region, and op semantics."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class ValidationCheck:
    check_id: str
    check_type: str
    status: str
    explanation: str


@dataclass
class ActionCoverage:
    required_action_type: str
    status: str
    matching_action_ids: list[str]
    explanation: str


@dataclass
class SemanticConsistencyCheck:
    target: str
    expected: str
    observed: str
    status: str
    explanation: str


@dataclass
class RepairConsistencyCheck:
    repair_type: str
    required_by_region_semantics: bool
    present_in_plan: bool
    status: str
    explanation: str


@dataclass
class PreservationConsistencyCheck:
    dimension: str
    location: str
    status: str
    explanation: str


@dataclass
class ForbiddenActionCheck:
    forbidden_dimension: str
    forbidden_location: str
    violated: bool
    status: str
    explanation: str


@dataclass
class PruningPlanValidation:
    validation_id: str
    plan_id: str
    candidate_id: str
    candidate_region_name: str
    plan_kind: str
    plan_status: str
    validation_status: str
    validation_score: int
    checks: list[ValidationCheck]
    required_action_coverage: list[ActionCoverage]
    semantic_consistency: list[SemanticConsistencyCheck]
    repair_consistency: list[RepairConsistencyCheck]
    preservation_consistency: list[PreservationConsistencyCheck]
    forbidden_action_consistency: list[ForbiddenActionCheck]
    evidence: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class PruningPlanValidationSet:
    model_name: str
    generated_at: str
    source_plan_path: str
    source_ranking_path: str
    source_region_pruning_semantics_path: str
    source_op_semantics_path: str
    validations: list[PruningPlanValidation]
    summary: dict[str, Any]


def pruning_plan_validation_to_dict(value: PruningPlanValidation) -> dict[str, Any]:
    return asdict(value)


def pruning_plan_validation_set_to_dict(value: PruningPlanValidationSet) -> dict[str, Any]:
    return asdict(value)


def write_pruning_plan_validation_json(value: PruningPlanValidationSet | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    data = pruning_plan_validation_set_to_dict(value) if isinstance(value, PruningPlanValidationSet) else value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_pruning_plan_validation_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_map(ranking: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item.get("candidate_id", ""): item for item in ranking.get("candidates", [])}


def _region_map(region_semantics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item.get("region_id", ""): item for item in region_semantics.get("regions", [])}


def _op_maps(op_semantics: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id = {item.get("op_id", ""): item for item in op_semantics.get("ops", [])}
    by_source = {item.get("source_name", ""): item for item in op_semantics.get("ops", [])}
    return by_id, by_source


def _check(index: int, check_type: str, status: str, explanation: str) -> ValidationCheck:
    return ValidationCheck(f"pvalid_check::{index:03d}", check_type, status, explanation)


def _actions_by_type(plan: dict[str, Any], action_type: str) -> list[dict[str, Any]]:
    return [item for item in plan.get("actions", []) if item.get("action_type") == action_type]


def _first_action(plan: dict[str, Any], action_type: str) -> dict[str, Any] | None:
    actions = _actions_by_type(plan, action_type)
    return actions[0] if actions else None


def _action_op(action: dict[str, Any] | None, by_id: dict[str, dict[str, Any]], by_source: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not action:
        return None
    return by_id.get(action.get("target_op", "")) or by_source.get(action.get("target_source_name", ""))


def _prop_op(step: dict[str, Any], by_id: dict[str, dict[str, Any]], by_source: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return by_id.get(step.get("through_op", "")) or by_source.get(step.get("through_source_name", ""))


def _has_plan_repair(plan: dict[str, Any], repair_type: str) -> bool:
    return any(item.get("repair_type") == repair_type and item.get("required", True) for item in plan.get("required_repairs", []))


def _has_region_repair(region: dict[str, Any], repair_type: str) -> bool:
    return any(item.get("obligation_type") == repair_type and item.get("required", True) for item in region.get("repair_obligations", []))


def _has_region_rule(region: dict[str, Any], rule_type: str) -> bool:
    return any(item.get("rule_type") == rule_type for item in region.get("propagation_rules", []))


def _has_hidden_prune_action(plan: dict[str, Any]) -> bool:
    return any(
        str(action.get("action_type", "")).startswith("prune")
        and action.get("dimension") == "hidden_dim"
        for action in plan.get("actions", [])
    )


def _output_bias_pruned(plan: dict[str, Any]) -> bool:
    return any(
        action.get("action_type") == "prune_bias"
        and "/output/dense/add" in str(action.get("target_source_name", "")).lower()
        for action in plan.get("actions", [])
    )


def _has_forbidden(plan: dict[str, Any], text: str, dimension: str = "hidden_dim") -> bool:
    needle = text.lower()
    return any(
        item.get("dimension") == dimension
        and needle in str(item.get("location", "")).lower()
        for item in plan.get("forbidden_actions", [])
    )


def _has_preserved(plan: dict[str, Any], text: str, dimension: str = "hidden_dim") -> bool:
    needle = text.lower()
    return any(
        item.get("dimension") == dimension
        and needle in str(item.get("location", "")).lower()
        for item in plan.get("preserved_dimensions", [])
    )


def _semantic_check(target: str, expected: str, observed: Any, ok: bool, explanation: str) -> SemanticConsistencyCheck:
    return SemanticConsistencyCheck(target, expected, str(observed), "pass" if ok else "fail", explanation)


def _validate_op_semantics(plan: dict[str, Any], by_id: dict[str, dict[str, Any]], by_source: dict[str, dict[str, Any]]) -> tuple[list[SemanticConsistencyCheck], list[str]]:
    semantic: list[SemanticConsistencyCheck] = []
    unknown_critical: list[str] = []
    producer_action = _first_action(plan, "prune_producer_output")
    bias_action = _first_action(plan, "prune_bias")
    consumer_action = _first_action(plan, "prune_consumer_input")
    producer = _action_op(producer_action, by_id, by_source)
    bias = _action_op(bias_action, by_id, by_source)
    consumer = _action_op(consumer_action, by_id, by_source)

    def mark_unknown(name: str, op: dict[str, Any] | None) -> None:
        if op is None or op.get("semantic_kind") == "unknown":
            unknown_critical.append(name)

    mark_unknown("producer", producer)
    mark_unknown("bias", bias)
    mark_unknown("consumer", consumer)

    semantic.extend(
        [
            _semantic_check("producer.semantic_kind", "parameterized_linear_matmul", producer.get("semantic_kind") if producer else None, bool(producer and producer.get("semantic_kind") == "parameterized_linear_matmul"), "Producer output action must target learned projection MatMul."),
            _semantic_check("producer.semantic_category", "parameterized_projection", producer.get("semantic_category") if producer else None, bool(producer and producer.get("semantic_category") == "parameterized_projection"), "Producer MatMul should be a parameterized projection."),
            _semantic_check("producer.parameterized", "true", producer.get("parameterized") if producer else None, bool(producer and producer.get("parameterized") is True), "Producer MatMul should be parameterized."),
            _semantic_check("producer.output_role", "intermediate_dim", producer.get("dimension_roles", {}).get("output") if producer else None, bool(producer and producer.get("dimension_roles", {}).get("output") == "intermediate_dim"), "Producer output axis should be intermediate_dim."),
            _semantic_check("producer.direct_pruning", "allowed", producer.get("pruning_effect", {}).get("direct_pruning") if producer else None, bool(producer and producer.get("pruning_effect", {}).get("direct_pruning") == "allowed"), "Producer output pruning should be allowed at op-semantics level."),
            _semantic_check("bias.semantic_kind", "linear_bias_add", bias.get("semantic_kind") if bias else None, bool(bias and bias.get("semantic_kind") == "linear_bias_add"), "Bias action must target the intermediate dense bias Add."),
            _semantic_check("bias.semantic_category", "parameterized_projection", bias.get("semantic_category") if bias else None, bool(bias and bias.get("semantic_category") == "parameterized_projection"), "Bias Add should belong to parameterized projection semantics."),
            _semantic_check("consumer.semantic_kind", "parameterized_linear_matmul", consumer.get("semantic_kind") if consumer else None, bool(consumer and consumer.get("semantic_kind") == "parameterized_linear_matmul"), "Consumer action must target FFN output projection MatMul."),
            _semantic_check("consumer.input_role", "intermediate_dim", consumer.get("dimension_roles", {}).get("input") if consumer else None, bool(consumer and consumer.get("dimension_roles", {}).get("input") == "intermediate_dim"), "Consumer input axis should be intermediate_dim."),
            _semantic_check("consumer.output_role", "hidden_dim", consumer.get("dimension_roles", {}).get("output") if consumer else None, bool(consumer and consumer.get("dimension_roles", {}).get("output") == "hidden_dim"), "Consumer output axis should remain hidden_dim."),
            _semantic_check("consumer.target_axis", "input_dim", consumer_action.get("target_axis") if consumer_action else None, bool(consumer_action and consumer_action.get("target_axis") == "input_dim"), "Consumer repair should target input_dim."),
        ]
    )
    for idx, step in enumerate(plan.get("propagation", [])):
        op = _prop_op(step, by_id, by_source)
        if op is None or op.get("semantic_kind") == "unknown":
            unknown_critical.append(f"propagation::{idx}")
    return semantic, unknown_critical


def _action_coverage(plan: dict[str, Any]) -> list[ActionCoverage]:
    out: list[ActionCoverage] = []
    for action_type in ["prune_producer_output", "prune_bias", "prune_consumer_input"]:
        matches = _actions_by_type(plan, action_type)
        if not matches:
            status = "missing"
        elif len(matches) > 1:
            status = "ambiguous"
        else:
            status = "covered"
        out.append(
            ActionCoverage(
                required_action_type=action_type,
                status=status,
                matching_action_ids=[item.get("action_id", "") for item in matches],
                explanation=f"Required action {action_type} is {status}.",
            )
        )
    return out


def _repair_consistency(plan: dict[str, Any], region: dict[str, Any]) -> list[RepairConsistencyCheck]:
    out = []
    for repair in ["same_indices_across_mlp", "prune_bias", "prune_consumer_input"]:
        required_by_region = _has_region_repair(region, repair) or _has_region_rule(region, repair)
        present = _has_plan_repair(plan, repair)
        ok = present or required_by_region
        out.append(
            RepairConsistencyCheck(
                repair_type=repair,
                required_by_region_semantics=required_by_region,
                present_in_plan=present,
                status="pass" if ok else "fail",
                explanation=f"Repair {repair} must be present in plan or region semantics.",
            )
        )
    return out


def _preservation_consistency(plan: dict[str, Any]) -> list[PreservationConsistencyCheck]:
    targets = [
        ("hidden_dim", "output/dense/matmul", "Output dense hidden_dim should be preserved."),
        ("hidden_dim", "output/dense/add", "Output dense bias belongs to hidden_dim and should be preserved."),
        ("hidden_dim", "output/add", "FFN residual hidden_dim should be preserved."),
        ("hidden_dim", "layernorm", "LayerNorm hidden_dim should be preserved."),
    ]
    out = []
    for dim, location, explanation in targets:
        ok = _has_preserved(plan, location, dim) or _has_forbidden(plan, location, dim)
        out.append(PreservationConsistencyCheck(dim, location, "pass" if ok else "fail", explanation))
    return out


def _forbidden_consistency(plan: dict[str, Any]) -> list[ForbiddenActionCheck]:
    targets = [
        ("hidden_dim", "output/add", _has_hidden_prune_action(plan), "Residual hidden_dim must not be pruned."),
        ("hidden_dim", "layernorm", _has_hidden_prune_action(plan), "LayerNorm hidden_dim must not be pruned."),
        ("hidden_dim", "output/dense/add", _output_bias_pruned(plan), "Output dense bias must not be pruned for intermediate_dim pruning."),
    ]
    out = []
    for dim, location, violated, explanation in targets:
        has_forbidden = _has_forbidden(plan, location, dim)
        status = "fail" if violated or not has_forbidden else "pass"
        out.append(ForbiddenActionCheck(dim, location, violated, status, explanation))
    return out


def _gelu_ok(plan: dict[str, Any], by_id: dict[str, dict[str, Any]], by_source: dict[str, dict[str, Any]]) -> bool:
    allowed = {"gelu_elementwise", "gelu_erf", "gelu_mul"}
    for step in plan.get("propagation", []):
        op = _prop_op(step, by_id, by_source)
        kind = op.get("semantic_kind") if op else step.get("semantic_kind")
        if kind in allowed and step.get("index_mapping") in {"same_indices", "no_index_change"}:
            return True
    return False


def _validate_ffn_plan(
    plan: dict[str, Any],
    candidate: dict[str, Any],
    region: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    by_source: dict[str, dict[str, Any]],
    index: int,
) -> PruningPlanValidation:
    coverage = _action_coverage(plan)
    semantic, unknown_critical = _validate_op_semantics(plan, by_id, by_source)
    repairs = _repair_consistency(plan, region)
    preservation = _preservation_consistency(plan)
    forbidden = _forbidden_consistency(plan)
    producer = _first_action(plan, "prune_producer_output")
    bias = _first_action(plan, "prune_bias")
    consumer = _first_action(plan, "prune_consumer_input")
    checks: list[ValidationCheck] = []
    checks.append(_check(0, "candidate_is_safe", "pass" if (
        candidate.get("pruning_class") == "safe"
        and candidate.get("candidate_kind") == "feedforward_intermediate_pruning"
        and candidate.get("semantic_category") == "feed_forward_block"
        and candidate.get("target_dimension") == "intermediate_dim"
        and (int(candidate.get("rank_score", 0)) >= 90 or candidate.get("confidence") == "high")
    ) else "fail", "Ranking candidate must be a high-confidence safe FFN intermediate opportunity."))
    checks.append(_check(1, "plan_status_ready", "pass" if plan.get("plan_status") == "ready_symbolic" else "fail", "Plan status must be ready_symbolic."))
    index_name = plan.get("symbolic_index_set", {}).get("name", "")
    checks.append(_check(2, "symbolic_index_set_present", "pass" if index_name and plan.get("target_dimension") == "intermediate_dim" else "fail", "Plan must carry a symbolic intermediate_dim index set."))
    checks.append(_check(3, "required_action_present", "pass" if all(item.status == "covered" for item in coverage) else "fail", "All required FFN actions must be present exactly once."))
    checks.append(_check(4, "producer_output_pruned", "pass" if producer and "/intermediate/dense/matmul" in producer.get("target_source_name", "").lower() and producer.get("target_axis") == "output_dim" and producer.get("dimension") == "intermediate_dim" else "fail", "Producer action must prune intermediate.dense MatMul output_dim."))
    checks.append(_check(5, "bias_pruned", "pass" if bias and "/intermediate/dense/add" in bias.get("target_source_name", "").lower() and bias.get("target_axis") == "bias_dim" and bias.get("dimension") == "intermediate_dim" else "fail", "Bias action must prune intermediate.dense Add bias_dim."))
    checks.append(_check(6, "consumer_input_pruned", "pass" if consumer and "/output/dense/matmul" in consumer.get("target_source_name", "").lower() and "/attention/output/dense/" not in consumer.get("target_source_name", "").lower() and consumer.get("target_axis") == "input_dim" and consumer.get("dimension") == "intermediate_dim" else "fail", "Consumer action must prune FFN output.dense MatMul input_dim."))
    checks.append(_check(7, "gelu_index_preserving", "pass" if _gelu_ok(plan, by_id, by_source) else "fail", "GELU propagation must preserve the same intermediate_dim indices."))
    checks.append(_check(8, "same_indices_across_mlp", "pass" if any(item.repair_type == "same_indices_across_mlp" and item.status == "pass" for item in repairs) else "fail", "same_indices_across_mlp repair/rule must be present."))
    checks.append(_check(9, "hidden_dim_preserved", "pass" if any(item.dimension == "hidden_dim" and item.status == "pass" for item in preservation) and not _has_hidden_prune_action(plan) else "fail", "Hidden dimensions must be preserved and not pruned."))
    checks.append(_check(10, "residual_hidden_not_pruned", "pass" if any(item.forbidden_location == "output/add" and item.status == "pass" for item in forbidden) else "fail", "Residual hidden_dim must be forbidden from pruning."))
    checks.append(_check(11, "layernorm_hidden_not_pruned", "pass" if any(item.forbidden_location == "layernorm" and item.status == "pass" for item in forbidden) else "fail", "LayerNorm hidden_dim must be forbidden from pruning."))
    checks.append(_check(12, "output_bias_not_pruned", "pass" if any(item.forbidden_location == "output/dense/add" and item.status == "pass" for item in forbidden) else "fail", "Output dense hidden bias must not be pruned by intermediate_dim plan."))
    checks.append(_check(13, "op_semantics_agree", "pass" if semantic and all(item.status == "pass" for item in semantic) else "fail", "Critical action op semantics must agree with FFN pruning plan."))
    checks.append(_check(14, "region_semantics_agree", "pass" if region.get("semantic_category") in {"feed_forward_block", ""} and region.get("pruning_role") in {"directly_prunable", None, ""} else "fail", "Region semantics should identify a directly prunable feed-forward block."))
    checks.append(_check(15, "no_blockers", "pass" if not candidate.get("blockers") and not region.get("blockers") else "fail", "Safe FFN symbolic plans should not carry blockers."))
    checks.append(_check(16, "no_unknown_critical_ops", "pass" if not unknown_critical else "fail", "Critical producer, bias, consumer, and GELU ops must not be unknown."))
    checks.append(_check(17, "prune_consumer_input_repair", "pass" if any(item.repair_type == "prune_consumer_input" and item.status == "pass" for item in repairs) else "fail", "prune_consumer_input repair must be present."))
    checks.append(_check(18, "prune_bias_repair", "pass" if any(item.repair_type == "prune_bias" and item.status == "pass" for item in repairs) else "fail", "prune_bias repair must be present."))

    errors = [check.check_type for check in checks if check.status == "fail"]
    warnings = [check.check_type for check in checks if check.status == "warning"]
    score = max(0, 100 - 10 * len(errors) - 3 * len(warnings))
    if errors:
        status = "invalid"
    elif warnings:
        status = "warning"
    else:
        status = "valid"
    return PruningPlanValidation(
        validation_id=f"pruning_plan_validation::{index:06d}::{plan.get('plan_id', 'plan')}",
        plan_id=plan.get("plan_id", ""),
        candidate_id=plan.get("candidate_id", ""),
        candidate_region_name=plan.get("candidate_region_name", ""),
        plan_kind=plan.get("plan_kind", ""),
        plan_status=plan.get("plan_status", ""),
        validation_status=status,
        validation_score=score,
        checks=checks,
        required_action_coverage=coverage,
        semantic_consistency=semantic,
        repair_consistency=repairs,
        preservation_consistency=preservation,
        forbidden_action_consistency=forbidden,
        evidence={
            "plan_summary": {
                "plan_id": plan.get("plan_id"),
                "symbolic_index_set": plan.get("symbolic_index_set", {}).get("name"),
                "target_dimension": plan.get("target_dimension"),
            },
            "candidate_summary": {
                "candidate_id": candidate.get("candidate_id"),
                "pruning_class": candidate.get("pruning_class"),
                "rank_score": candidate.get("rank_score"),
                "confidence": candidate.get("confidence"),
            },
            "region_semantics_summary": {
                "region_id": region.get("region_id"),
                "semantic_category": region.get("semantic_category"),
                "pruning_role": region.get("pruning_role"),
                "blockers": region.get("blockers", []),
            },
            "op_semantics_summary": {
                "producer": producer.get("target_source_name") if producer else "",
                "bias": bias.get("target_source_name") if bias else "",
                "consumer": consumer.get("target_source_name") if consumer else "",
                "gelu": [step.get("through_source_name", "") for step in plan.get("propagation", [])],
                "unknown_critical_ops": unknown_critical,
            },
        },
        warnings=warnings,
        errors=errors,
    )


def validate_pruning_plans(
    plan_set: dict[str, Any],
    ranking: dict[str, Any],
    region_pruning_semantics: dict[str, Any],
    op_semantics: dict[str, Any],
    *,
    source_plan_path: str = "",
    source_ranking_path: str = "",
    source_region_pruning_semantics_path: str = "",
    source_op_semantics_path: str = "",
) -> PruningPlanValidationSet:
    model_name = plan_set.get("model_name", ranking.get("model_name", region_pruning_semantics.get("model_name", "model")))
    candidates = _candidate_map(ranking)
    regions = _region_map(region_pruning_semantics)
    by_id, by_source = _op_maps(op_semantics)
    validations: list[PruningPlanValidation] = []
    for index, plan in enumerate(plan_set.get("plans", [])):
        candidate = candidates.get(plan.get("candidate_id", ""), {})
        region = regions.get(plan.get("candidate_region_id", ""), {})
        if plan.get("plan_kind") == "feedforward_intermediate_dim_plan":
            validations.append(_validate_ffn_plan(plan, candidate, region, by_id, by_source, index))
        else:
            validations.append(
                PruningPlanValidation(
                    validation_id=f"pruning_plan_validation::{index:06d}::{plan.get('plan_id', 'plan')}",
                    plan_id=plan.get("plan_id", ""),
                    candidate_id=plan.get("candidate_id", ""),
                    candidate_region_name=plan.get("candidate_region_name", ""),
                    plan_kind=plan.get("plan_kind", "unknown"),
                    plan_status=plan.get("plan_status", "unknown"),
                    validation_status="unknown",
                    validation_score=0,
                    checks=[_check(0, "region_semantics_agree", "unknown", "No validation policy implemented for this plan kind.")],
                    required_action_coverage=[],
                    semantic_consistency=[],
                    repair_consistency=[],
                    preservation_consistency=[],
                    forbidden_action_consistency=[],
                    evidence={"plan_summary": {"plan_id": plan.get("plan_id")}},
                    warnings=[],
                    errors=[],
                )
            )
    return PruningPlanValidationSet(
        model_name=model_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_plan_path=source_plan_path,
        source_ranking_path=source_ranking_path,
        source_region_pruning_semantics_path=source_region_pruning_semantics_path,
        source_op_semantics_path=source_op_semantics_path,
        validations=validations,
        summary=_summary(validations),
    )


def _summary(validations: list[PruningPlanValidation]) -> dict[str, Any]:
    status_counts = Counter(item.validation_status for item in validations)
    kind_counts = Counter(item.plan_kind for item in validations)
    failed = Counter(check.check_type for item in validations for check in item.checks if check.status == "fail")
    warnings = Counter(check.check_type for item in validations for check in item.checks if check.status == "warning")
    return {
        "total_plans": len(validations),
        "valid_plans": status_counts.get("valid", 0),
        "warning_plans": status_counts.get("warning", 0),
        "invalid_plans": status_counts.get("invalid", 0),
        "unknown_plans": status_counts.get("unknown", 0),
        "ready_symbolic_plans": sum(1 for item in validations if item.plan_status == "ready_symbolic"),
        "validated_ready_symbolic_plans": sum(1 for item in validations if item.plan_status == "ready_symbolic" and item.validation_status == "valid"),
        "failed_checks_by_type": dict(sorted(failed.items())),
        "warning_checks_by_type": dict(sorted(warnings.items())),
        "validation_status_counts": dict(sorted(status_counts.items())),
        "plan_kind_counts": dict(sorted(kind_counts.items())),
    }


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 100) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def _validation_row(item: dict[str, Any]) -> dict[str, Any]:
    failed = ", ".join(check.get("check_type", "") for check in item.get("checks", []) if check.get("status") == "fail")
    actions = ", ".join(row.get("required_action_type", "") for row in item.get("required_action_coverage", []) if row.get("status") == "covered")
    return {
        "plan": item.get("candidate_region_name", ""),
        "status": item.get("validation_status", ""),
        "score": item.get("validation_score", ""),
        "plan_status": item.get("plan_status", ""),
        "actions": actions,
        "failed": failed,
    }


def pruning_plan_validation_to_markdown(value: PruningPlanValidationSet | dict[str, Any]) -> str:
    data = pruning_plan_validation_set_to_dict(value) if isinstance(value, PruningPlanValidationSet) else value
    summary = data.get("summary", {})
    validations = data.get("validations", [])
    valid = [item for item in validations if item.get("validation_status") == "valid"]
    warning = [item for item in validations if item.get("validation_status") == "warning"]
    invalid = [item for item in validations if item.get("validation_status") == "invalid"]
    lines = [
        f"# Pruning Plan Validation: {data.get('model_name', '')}",
        "",
        "## Summary",
        "",
        f"- Total validations: `{summary.get('total_plans', 0)}`",
        f"- Valid: `{summary.get('valid_plans', 0)}`",
        f"- Warning: `{summary.get('warning_plans', 0)}`",
        f"- Invalid: `{summary.get('invalid_plans', 0)}`",
        f"- Unknown: `{summary.get('unknown_plans', 0)}`",
        f"- Validated ready symbolic: `{summary.get('validated_ready_symbolic_plans', 0)}`",
        f"- Failed checks: `{summary.get('failed_checks_by_type', {})}`",
        f"- Warning checks: `{summary.get('warning_checks_by_type', {})}`",
        "",
        "## Valid Plans",
        "",
        _table([_validation_row(item) for item in valid], ["plan", "status", "score", "plan_status", "actions"]),
        "",
        "## Warnings",
        "",
        _table([_validation_row(item) for item in warning], ["plan", "status", "score", "failed"]),
        "",
        "## Invalid Plans",
        "",
        _table([_validation_row(item) for item in invalid], ["plan", "status", "score", "failed"]),
        "",
        "## Validation Details",
        "",
    ]
    for item in validations:
        lines.extend(
            [
                f"### {item.get('candidate_region_name', item.get('plan_id', 'plan'))}",
                "",
                f"- Validation status: `{item.get('validation_status')}`",
                f"- Validation score: `{item.get('validation_score')}`",
                f"- Plan status: `{item.get('plan_status')}`",
                "",
                "#### Checks",
                "",
                _table(item.get("checks", []), ["check_type", "status", "explanation"]),
                "",
                "#### Action Coverage",
                "",
                _table(item.get("required_action_coverage", []), ["required_action_type", "status", "matching_action_ids", "explanation"]),
                "",
                "#### Semantic Consistency",
                "",
                _table(item.get("semantic_consistency", []), ["target", "expected", "observed", "status", "explanation"]),
                "",
                "#### Repair Consistency",
                "",
                _table(item.get("repair_consistency", []), ["repair_type", "required_by_region_semantics", "present_in_plan", "status", "explanation"]),
                "",
                "#### Preservation Consistency",
                "",
                _table(item.get("preservation_consistency", []), ["dimension", "location", "status", "explanation"]),
                "",
                "#### Forbidden Action Consistency",
                "",
                _table(item.get("forbidden_action_consistency", []), ["forbidden_dimension", "forbidden_location", "violated", "status", "explanation"]),
                "",
            ]
        )
    lines.extend([
        "## Interpretation",
        "",
        "This validation is a static consistency check over symbolic pruning plans. It does not choose concrete indices or modify models.",
        "",
    ])
    return "\n".join(lines)
