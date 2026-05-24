"""Synthesize symbolic pruning plans for top-ranked safe opportunities."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class SymbolicIndexSet:
    name: str
    description: str


@dataclass
class PlanAction:
    action_id: str
    action_type: str
    target_op: str
    target_op_type: str
    target_source_name: str
    target_axis: str
    dimension: str
    index_set: str
    required: bool
    explanation: str


@dataclass
class PlanPropagationStep:
    step_id: str
    through_op: str
    through_source_name: str
    semantic_kind: str
    from_dimension: str
    to_dimension: str
    index_mapping: str
    explanation: str


@dataclass
class PlanRepair:
    repair_type: str
    required: bool
    affected_actions: list[str]
    explanation: str


@dataclass
class PreservedDimension:
    dimension: str
    location: str
    reason: str


@dataclass
class ForbiddenAction:
    action_type: str
    dimension: str
    location: str
    reason: str


@dataclass
class PlanValidationCheck:
    check_id: str
    check_type: str
    status: str
    explanation: str


@dataclass
class PruningPlan:
    plan_id: str
    model_name: str
    candidate_id: str
    candidate_region_id: str
    candidate_region_name: str
    candidate_kind: str
    semantic_category: str
    pruning_class: str
    rank_score: int
    confidence: str
    plan_kind: str
    target_dimension: str
    symbolic_index_set: SymbolicIndexSet
    plan_status: str
    actions: list[PlanAction]
    propagation: list[PlanPropagationStep]
    required_repairs: list[PlanRepair]
    preserved_dimensions: list[PreservedDimension]
    forbidden_actions: list[ForbiddenAction]
    validation_checks: list[PlanValidationCheck]
    evidence: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@dataclass
class PruningPlanSet:
    model_name: str
    generated_at: str
    source_ranking_path: str
    source_region_pruning_semantics_path: str
    source_op_semantics_path: str
    plans: list[PruningPlan]
    summary: dict[str, Any]


def pruning_plan_to_dict(value: PruningPlan) -> dict[str, Any]:
    return asdict(value)


def pruning_plan_set_to_dict(value: PruningPlanSet) -> dict[str, Any]:
    return asdict(value)


def write_pruning_plan_set_json(value: PruningPlanSet | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    data = pruning_plan_set_to_dict(value) if isinstance(value, PruningPlanSet) else value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_pruning_plan_set_json(path: Path) -> PruningPlanSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PruningPlanSet(
        model_name=data.get("model_name", "model"),
        generated_at=data.get("generated_at", ""),
        source_ranking_path=data.get("source_ranking_path", ""),
        source_region_pruning_semantics_path=data.get("source_region_pruning_semantics_path", ""),
        source_op_semantics_path=data.get("source_op_semantics_path", ""),
        plans=[_plan_from_dict(item) for item in data.get("plans", [])],
        summary=data.get("summary", {}),
    )


def _plan_from_dict(item: dict[str, Any]) -> PruningPlan:
    return PruningPlan(
        plan_id=item["plan_id"],
        model_name=item.get("model_name", "model"),
        candidate_id=item.get("candidate_id", ""),
        candidate_region_id=item.get("candidate_region_id", ""),
        candidate_region_name=item.get("candidate_region_name", ""),
        candidate_kind=item.get("candidate_kind", ""),
        semantic_category=item.get("semantic_category", ""),
        pruning_class=item.get("pruning_class", ""),
        rank_score=int(item.get("rank_score", 0)),
        confidence=item.get("confidence", "unknown"),
        plan_kind=item.get("plan_kind", "unknown"),
        target_dimension=item.get("target_dimension", "unknown"),
        symbolic_index_set=SymbolicIndexSet(**item.get("symbolic_index_set", {"name": "I_unknown", "description": ""})),
        plan_status=item.get("plan_status", "unknown"),
        actions=[PlanAction(**action) for action in item.get("actions", [])],
        propagation=[PlanPropagationStep(**step) for step in item.get("propagation", [])],
        required_repairs=[PlanRepair(**repair) for repair in item.get("required_repairs", [])],
        preserved_dimensions=[PreservedDimension(**dim) for dim in item.get("preserved_dimensions", [])],
        forbidden_actions=[ForbiddenAction(**action) for action in item.get("forbidden_actions", [])],
        validation_checks=[PlanValidationCheck(**check) for check in item.get("validation_checks", [])],
        evidence=item.get("evidence", {}),
        warnings=item.get("warnings", []),
    )


def _layer_from_text(value: str) -> int | None:
    match = re.search(r"layer[ ._/:-]*(\d+)", value.lower())
    return int(match.group(1)) if match else None


def _layer_from_candidate(candidate: dict[str, Any]) -> int | None:
    for value in [candidate.get("candidate_region_name", ""), candidate.get("region_name", "")]:
        layer = _layer_from_text(str(value))
        if layer is not None:
            return layer
    for op in candidate.get("op_semantics_evidence", []):
        layer = _layer_from_text(str(op.get("source_name", "")))
        if layer is not None:
            return layer
    return None


def _index_set(layer: int | None) -> SymbolicIndexSet:
    suffix = f"layer_{layer}" if layer is not None else "unknown_layer"
    return SymbolicIndexSet(
        name=f"I_{suffix}_intermediate",
        description="Symbolic index set selecting FFN intermediate_dim entries to prune.",
    )


def _op_by_id(op_semantics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {op.get("op_id", ""): op for op in op_semantics.get("ops", [])}


def _ops_for_layer(op_semantics: dict[str, Any], layer: int | None) -> list[dict[str, Any]]:
    if layer is None:
        return op_semantics.get("ops", [])
    token = f"/encoder/layer.{layer}/"
    return [op for op in op_semantics.get("ops", []) if token in str(op.get("source_name", "")).lower()]


def _find_op(ops: list[dict[str, Any]], contains: str, semantic_kind: str | None = None) -> dict[str, Any] | None:
    needle = contains.lower()
    for op in ops:
        source = str(op.get("source_name", "")).lower()
        if needle in source and (semantic_kind is None or op.get("semantic_kind") == semantic_kind):
            return op
    return None


def _find_ffn_output_op(ops: list[dict[str, Any]], suffix: str, semantic_kind: str) -> dict[str, Any] | None:
    needle = f"/output/dense/{suffix}".lower()
    for op in ops:
        source = str(op.get("source_name", "")).lower()
        if "/attention/output/dense/" in source:
            continue
        if needle in source and op.get("semantic_kind") == semantic_kind:
            return op
    return None


def _find_ffn_output_support_op(ops: list[dict[str, Any]], suffix: str, semantic_kind: str) -> dict[str, Any] | None:
    needle = f"/output/{suffix}".lower()
    for op in ops:
        source = str(op.get("source_name", "")).lower()
        if "/attention/output/" in source:
            continue
        if needle in source and op.get("semantic_kind") == semantic_kind:
            return op
    return None


def _find_gelu_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        op for op in ops
        if str(op.get("semantic_category")) == "elementwise_index_preserving"
        and "/intermediate/intermediate_act_fn/" in str(op.get("source_name", "")).lower()
    ]


def _make_action(index: int, action_type: str, op: dict[str, Any] | None, axis: str, dimension: str, index_set: str, required: bool, explanation: str) -> PlanAction:
    return PlanAction(
        action_id=f"plan_action::{index:03d}",
        action_type=action_type,
        target_op=str(op.get("op_id", "")) if op else "",
        target_op_type=str(op.get("op_type", "")) if op else "",
        target_source_name=str(op.get("source_name", "")) if op else "",
        target_axis=axis,
        dimension=dimension,
        index_set=index_set,
        required=required,
        explanation=explanation,
    )


def _check(index: int, check_type: str, status: str, explanation: str) -> PlanValidationCheck:
    return PlanValidationCheck(f"plan_check::{index:03d}", check_type, status, explanation)


def _required_repairs(candidate: dict[str, Any], actions: list[PlanAction]) -> list[PlanRepair]:
    repair_types = {repair.get("obligation_type") for repair in candidate.get("required_repairs", [])}
    if "same_indices_across_mlp" not in repair_types:
        repair_types.add("same_indices_across_mlp")
    if "prune_consumer_input" not in repair_types:
        repair_types.add("prune_consumer_input")
    if any(action.action_type == "prune_bias" for action in actions):
        repair_types.add("prune_bias")
    affected = [action.action_id for action in actions if action.required]
    return [
        PlanRepair("same_indices_across_mlp", True, affected, "Use the same intermediate_dim index set across producer output, GELU, and consumer input."),
        PlanRepair("prune_bias", True, [a.action_id for a in actions if a.action_type == "prune_bias"], "Prune intermediate projection bias entries using the same index set."),
        PlanRepair("prune_consumer_input", True, [a.action_id for a in actions if a.action_type == "prune_consumer_input"], "Prune FFN output projection input columns using the same index set."),
        PlanRepair("preserve_hidden_output", True, [a.action_id for a in actions if a.action_type == "preserve_output"], "Keep output hidden_dim unchanged."),
        PlanRepair("validation_required", True, affected, "Validate symbolic structure before any optional executable backend."),
    ]


def _candidate_is_primary_ffn(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("pruning_class") == "safe"
        and candidate.get("candidate_kind") == "feedforward_intermediate_pruning"
        and candidate.get("semantic_category") == "feed_forward_block"
        and candidate.get("target_dimension") == "intermediate_dim"
    )


def _region_by_id(region_semantics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {region.get("region_id", ""): region for region in region_semantics.get("regions", [])}


def _synthesize_ffn_plan(
    model_name: str,
    candidate: dict[str, Any],
    region_semantics: dict[str, Any],
    op_semantics: dict[str, Any],
    index: int,
) -> PruningPlan:
    layer = _layer_from_candidate(candidate)
    index_set = _index_set(layer)
    layer_ops = _ops_for_layer(op_semantics, layer)
    intermediate_matmul = _find_op(layer_ops, "/intermediate/dense/matmul", "parameterized_linear_matmul")
    intermediate_bias = _find_op(layer_ops, "/intermediate/dense/add", "linear_bias_add")
    output_matmul = _find_ffn_output_op(layer_ops, "matmul", "parameterized_linear_matmul")
    output_bias = _find_ffn_output_op(layer_ops, "add", "linear_bias_add")
    residual_add = _find_ffn_output_support_op(layer_ops, "add", "residual_add")
    layernorm = _find_ffn_output_support_op(layer_ops, "layernorm/layernormalization", "layernorm")
    gelu_ops = _find_gelu_ops(layer_ops)
    actions: list[PlanAction] = []
    if intermediate_matmul:
        actions.append(_make_action(0, "prune_producer_output", intermediate_matmul, "output_dim", "intermediate_dim", index_set.name, True, "Prune FFN intermediate projection output features using the symbolic index set."))
    if intermediate_bias:
        actions.append(_make_action(1, "prune_bias", intermediate_bias, "bias_dim", "intermediate_dim", index_set.name, True, "Prune intermediate projection bias entries with the same indices."))
    if output_matmul:
        actions.append(_make_action(2, "prune_consumer_input", output_matmul, "input_dim", "intermediate_dim", index_set.name, True, "Prune FFN output projection input columns with the same indices."))
        actions.append(_make_action(3, "preserve_output", output_matmul, "output_dim", "hidden_dim", "", True, "Preserve FFN output hidden_dim."))

    propagation = [
        PlanPropagationStep(
            step_id=f"plan_prop::{idx:03d}",
            through_op=str(op.get("op_id", "")),
            through_source_name=str(op.get("source_name", "")),
            semantic_kind=str(op.get("semantic_kind", "")),
            from_dimension="intermediate_dim",
            to_dimension="intermediate_dim",
            index_mapping="no_index_change",
            explanation="GELU is elementwise and preserves the intermediate_dim index set.",
        )
        for idx, op in enumerate(gelu_ops)
    ]
    preserved = []
    if output_matmul:
        preserved.append(PreservedDimension("hidden_dim", str(output_matmul.get("source_name", "")), "FFN output hidden width is preserved."))
    if output_bias:
        preserved.append(PreservedDimension("hidden_dim", str(output_bias.get("source_name", "")), "Output dense bias belongs to hidden_dim and is not pruned by intermediate_dim pruning."))
    if residual_add:
        preserved.append(PreservedDimension("hidden_dim", str(residual_add.get("source_name", "")), "Residual hidden_dim remains unchanged."))
    if layernorm:
        preserved.append(PreservedDimension("hidden_dim", str(layernorm.get("source_name", "")), "LayerNorm hidden_dim remains unchanged."))
    forbidden = [
        ForbiddenAction("do_not_prune", "hidden_dim", str(residual_add.get("source_name", "residual hidden_dim")) if residual_add else "residual hidden_dim", "Residual hidden_dim pruning is outside this FFN intermediate plan."),
        ForbiddenAction("do_not_prune", "hidden_dim", str(layernorm.get("source_name", "LayerNorm hidden_dim")) if layernorm else "LayerNorm hidden_dim", "LayerNorm hidden_dim pruning is outside this FFN intermediate plan."),
        ForbiddenAction("do_not_prune", "hidden_dim", str(output_bias.get("source_name", "output.dense bias")) if output_bias else "output.dense bias", "Output dense bias follows hidden_dim, not intermediate_dim."),
    ]
    checks = [
        _check(0, "required_op_present", "pass" if intermediate_matmul else "fail", "Required intermediate dense MatMul evidence."),
        _check(1, "required_op_present", "pass" if intermediate_bias else "fail", "Required intermediate dense bias Add evidence."),
        _check(2, "required_op_present", "pass" if output_matmul else "fail", "Required output dense MatMul evidence."),
        _check(3, "required_op_present", "pass" if gelu_ops else "fail", "Required GELU/index-preserving activation evidence."),
        _check(4, "hidden_dim_preserved", "pass" if output_matmul else "fail", "Output dense hidden_dim is preserved."),
        _check(5, "residual_not_pruned", "pass" if residual_add else "warning", "Residual hidden_dim is not targeted by this plan."),
        _check(6, "layernorm_not_pruned", "pass" if layernorm else "warning", "LayerNorm hidden_dim is not targeted by this plan."),
        _check(7, "op_semantics_consistent", "pass", "Op semantics agree with FFN intermediate symbolic pruning."),
    ]
    warnings = []
    missing = [check.check_type + ":" + check.explanation for check in checks if check.status == "fail"]
    if missing:
        warnings.extend("missing_" + item for item in missing)
    status = "ready_symbolic" if not missing and not candidate.get("blockers") else "incomplete"
    if candidate.get("blockers"):
        status = "blocked"
        warnings.append("candidate_has_blockers")
    region_map = _region_by_id(region_semantics)
    region = region_map.get(candidate.get("candidate_region_id", ""), {})
    source_ops = [op.get("source_name", "") for op in [intermediate_matmul, intermediate_bias, *gelu_ops, output_matmul, output_bias, residual_add, layernorm] if op]
    return PruningPlan(
        plan_id=f"pruning_plan::{index:06d}::{candidate.get('region_id', candidate.get('candidate_region_id', 'ffn'))}",
        model_name=model_name,
        candidate_id=candidate.get("candidate_id", ""),
        candidate_region_id=candidate.get("region_id", candidate.get("candidate_region_id", "")),
        candidate_region_name=candidate.get("region_name", candidate.get("candidate_region_name", "")),
        candidate_kind=candidate.get("candidate_kind", ""),
        semantic_category=candidate.get("semantic_category", ""),
        pruning_class=candidate.get("pruning_class", ""),
        rank_score=int(candidate.get("rank_score", 0)),
        confidence=candidate.get("confidence", "unknown"),
        plan_kind="feedforward_intermediate_dim_plan",
        target_dimension="intermediate_dim",
        symbolic_index_set=index_set,
        plan_status=status,
        actions=actions,
        propagation=propagation,
        required_repairs=_required_repairs(candidate, actions),
        preserved_dimensions=preserved,
        forbidden_actions=forbidden,
        validation_checks=checks,
        evidence={
            "candidate_summary": {
                "candidate_id": candidate.get("candidate_id"),
                "rank_score": candidate.get("rank_score"),
                "confidence": candidate.get("confidence"),
                "reason": candidate.get("reason"),
            },
            "region_semantics_summary": {
                "region_id": region.get("region_id", candidate.get("region_id")),
                "semantic_category": region.get("semantic_category", candidate.get("semantic_category")),
                "pruning_role": region.get("pruning_role"),
            },
            "op_semantics_summary": {
                "intermediate_matmul": bool(intermediate_matmul),
                "intermediate_bias": bool(intermediate_bias),
                "gelu_ops": len(gelu_ops),
                "output_matmul": bool(output_matmul),
                "output_bias": bool(output_bias),
                "residual_add": bool(residual_add),
                "layernorm": bool(layernorm),
            },
            "source_ops": source_ops,
        },
        warnings=warnings,
    )


def synthesize_pruning_plans(
    ranking: dict[str, Any],
    region_pruning_semantics: dict[str, Any],
    op_semantics: dict[str, Any],
    *,
    source_ranking_path: str = "",
    source_region_pruning_semantics_path: str = "",
    source_op_semantics_path: str = "",
) -> PruningPlanSet:
    model_name = ranking.get("model_name", region_pruning_semantics.get("model_name", op_semantics.get("model_name", "model")))
    plans = [
        _synthesize_ffn_plan(model_name, candidate, region_pruning_semantics, op_semantics, index)
        for index, candidate in enumerate(ranking.get("candidates", []))
        if _candidate_is_primary_ffn(candidate)
    ]
    plans.sort(key=lambda plan: (plan.candidate_region_name, plan.plan_id))
    return PruningPlanSet(
        model_name=model_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_ranking_path=source_ranking_path,
        source_region_pruning_semantics_path=source_region_pruning_semantics_path,
        source_op_semantics_path=source_op_semantics_path,
        plans=plans,
        summary=_summary(plans),
    )


def _summary(plans: list[PruningPlan]) -> dict[str, Any]:
    status_counts = Counter(plan.plan_status for plan in plans)
    kind_counts = Counter(plan.plan_kind for plan in plans)
    missing_counts = Counter(warning for plan in plans for warning in plan.warnings if warning.startswith("missing_"))
    return {
        "total_plans": len(plans),
        "ready_symbolic": status_counts.get("ready_symbolic", 0),
        "incomplete": status_counts.get("incomplete", 0),
        "blocked": status_counts.get("blocked", 0),
        "unknown": status_counts.get("unknown", 0),
        "plan_kind_counts": dict(sorted(kind_counts.items())),
        "plan_status_counts": dict(sorted(status_counts.items())),
        "missing_evidence_counts": dict(sorted(missing_counts.items())),
    }


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 80) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    if len(rows) > limit:
        lines.append("| ... | " + f"{len(rows) - limit} more rows omitted" + " |" * (len(columns) - 2))
    return "\n".join(lines)


def _plan_row(plan: dict[str, Any]) -> dict[str, Any]:
    actions = ", ".join(action.get("action_type", "") for action in plan.get("actions", []) if action.get("required"))
    repairs = ", ".join(repair.get("repair_type", "") for repair in plan.get("required_repairs", []) if repair.get("required"))
    return {
        "plan": plan.get("candidate_region_name"),
        "status": plan.get("plan_status"),
        "score": plan.get("rank_score"),
        "confidence": plan.get("confidence"),
        "index_set": plan.get("symbolic_index_set", {}).get("name"),
        "actions": actions,
        "repairs": repairs,
        "warnings": ", ".join(plan.get("warnings", [])),
    }


def pruning_plan_set_to_markdown(value: PruningPlanSet | dict[str, Any]) -> str:
    data = pruning_plan_set_to_dict(value) if isinstance(value, PruningPlanSet) else value
    summary = data.get("summary", {})
    plans = data.get("plans", [])
    ready = [plan for plan in plans if plan.get("plan_status") == "ready_symbolic"]
    incomplete = [plan for plan in plans if plan.get("plan_status") == "incomplete"]
    lines = [
        f"# Pruning Plans: {data.get('model_name', '')}",
        "",
        "## Summary",
        "",
        f"- Total plans: `{summary.get('total_plans', 0)}`",
        f"- Ready symbolic: `{summary.get('ready_symbolic', 0)}`",
        f"- Incomplete: `{summary.get('incomplete', 0)}`",
        f"- Blocked: `{summary.get('blocked', 0)}`",
        f"- Unknown: `{summary.get('unknown', 0)}`",
        "",
        "## Ready Symbolic Plans",
        "",
        _table([_plan_row(plan) for plan in ready], ["plan", "status", "score", "confidence", "index_set", "actions", "repairs"], limit=80),
        "",
        "## Incomplete Plans",
        "",
        _table([_plan_row(plan) for plan in incomplete], ["plan", "status", "warnings"], limit=80),
        "",
        "## Plan Details",
        "",
    ]
    for plan in plans:
        lines.extend(
            [
                f"### {plan.get('candidate_region_name')}",
                "",
                f"- Status: `{plan.get('plan_status')}`",
                f"- Candidate score: `{plan.get('rank_score')}`",
                f"- Index set: `{plan.get('symbolic_index_set', {}).get('name')}`",
                "",
                "#### Actions",
                "",
                _table(plan.get("actions", []), ["action_type", "target_source_name", "target_axis", "dimension", "index_set", "required"], limit=40),
                "",
                "#### Propagation",
                "",
                _table(plan.get("propagation", []), ["through_source_name", "semantic_kind", "from_dimension", "to_dimension", "index_mapping"], limit=40),
                "",
                "#### Preserved Dimensions",
                "",
                _table(plan.get("preserved_dimensions", []), ["dimension", "location", "reason"], limit=40),
                "",
                "#### Forbidden Actions",
                "",
                _table(plan.get("forbidden_actions", []), ["action_type", "dimension", "location", "reason"], limit=40),
                "",
                "#### Validation Checks",
                "",
                _table(plan.get("validation_checks", []), ["check_type", "status", "explanation"], limit=40),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "These are symbolic pruning plans parameterized by index sets. They do not choose concrete indices, execute pruning, or modify models.",
            "",
        ]
    )
    return "\n".join(lines)
