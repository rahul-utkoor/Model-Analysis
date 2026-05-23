"""Static pruning-propagation and legality analysis over RegionDimensionIR."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir
from model_analysis.region_ir_graph import (
    extract_region_slice,
    get_region_equivalence_class_for_dimension,
)


@dataclass
class RegionPruningRequest:
    request_id: str
    model_name: str
    dimension_var_id: str
    requested_indices: list[int] | None
    requested_count: int | None
    requested_fraction: float | None
    symbolic: bool
    strategy: str
    reason: str | None


@dataclass
class RegionConstraintSatisfaction:
    constraint_id: str
    relation: str
    constraint_type: str
    lhs: str
    rhs: str
    status: str
    blocking: bool
    confidence: str
    reason: str


@dataclass
class RegionPropagationSlice:
    slice_id: str
    root_dimension: str
    direction: str
    dimensions: list[str]
    constraints: list[str]
    blocked_dimensions: list[str]
    blocking_constraints: list[str]
    unresolved_constraints: list[str]
    protected_dimensions: list[str]
    reason: str


@dataclass
class RegionRepairSetItem:
    repair_id: str
    repair_type: str
    source_dimension: str
    target_dimension: str | None
    constraint_id: str | None
    required_action: str
    executable_backend: str
    confidence: str
    reason: str


@dataclass
class RegionLegalityCheckResult:
    check_id: str
    model_name: str
    request: RegionPruningRequest
    status: str
    root_dimension: dict | None
    equivalent_dimensions: list[str]
    required_propagations: list[dict[str, Any]]
    constraint_satisfaction: list[RegionConstraintSatisfaction]
    forward_slice: RegionPropagationSlice
    backward_slice: RegionPropagationSlice
    minimal_repair_set: list[RegionRepairSetItem] = field(default_factory=list)
    blocking_reasons: list[dict[str, Any]] = field(default_factory=list)
    unresolved_items: list[dict[str, Any]] = field(default_factory=list)
    protected_dimensions: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def region_pruning_request_to_dict(request: RegionPruningRequest) -> dict[str, Any]:
    return asdict(request)


def region_legality_check_result_to_dict(result: RegionLegalityCheckResult) -> dict[str, Any]:
    return asdict(result)


def region_propagation_slice_to_dict(slice_result: RegionPropagationSlice) -> dict[str, Any]:
    return asdict(slice_result)


def write_region_legality_check_json(result: RegionLegalityCheckResult, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(region_legality_check_result_to_dict(result), indent=2), encoding="utf-8")


def _request(request: RegionPruningRequest | dict) -> RegionPruningRequest:
    return RegionPruningRequest(**request) if isinstance(request, dict) else request


def _safe(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace(" ", "_")


def make_region_pruning_request(
    model_name: str,
    dimension_var_id: str,
    indices: list[int] | None = None,
    count: int | None = None,
    fraction: float | None = None,
    strategy: str | None = None,
    reason: str | None = None,
) -> RegionPruningRequest:
    provided = sum(value is not None for value in (indices, count, fraction))
    if provided > 1:
        raise ValueError("Only one of indices, count, or fraction may be provided.")
    if indices is not None:
        resolved_strategy, symbolic = "explicit_indices", False
    elif count is not None:
        resolved_strategy, symbolic = strategy or "first_n", False
        if resolved_strategy not in {"first_n", "last_n"}:
            raise ValueError("Count requests support first_n or last_n strategies.")
    elif fraction is not None:
        resolved_strategy, symbolic = "fraction", False
    else:
        resolved_strategy, symbolic = strategy or "symbolic", True
        if resolved_strategy != "symbolic":
            raise ValueError("Non-symbolic strategies require indices, count, or fraction.")
    return RegionPruningRequest(
        request_id=f"region_request__{_safe(model_name)}__{_safe(dimension_var_id)}__{resolved_strategy}",
        model_name=model_name,
        dimension_var_id=dimension_var_id,
        requested_indices=indices,
        requested_count=count,
        requested_fraction=fraction,
        symbolic=symbolic,
        strategy=resolved_strategy,
        reason=reason,
    )


def _numeric_size(size: int | str | None) -> int | None:
    return size if isinstance(size, int) and not isinstance(size, bool) else None


def resolve_region_requested_indices(
    request: RegionPruningRequest | dict,
    dimension_size: int | str | None,
) -> tuple[list[int] | None, list[dict]]:
    req = _request(request)
    issues: list[dict[str, Any]] = []
    size = _numeric_size(dimension_size)
    if req.strategy == "symbolic":
        return None, issues
    if req.strategy == "explicit_indices":
        original = req.requested_indices or []
        indices = sorted(set(original))
        if any(index < 0 for index in indices):
            issues.append({"severity": "rejected", "reason": "Prune indices must be non-negative."})
        if len(indices) != len(original):
            issues.append({"severity": "warning", "reason": "Duplicate indices were normalized."})
    elif req.strategy in {"first_n", "last_n"}:
        if req.requested_count is None or req.requested_count <= 0:
            return None, [{"severity": "rejected", "reason": "Requested count must be positive."}]
        if size is None:
            return None, [{"severity": "unresolved", "reason": f"{req.strategy} requires a known numeric region dimension size."}]
        indices = list(range(req.requested_count)) if req.strategy == "first_n" else list(range(size - req.requested_count, size))
    elif req.strategy == "fraction":
        if req.requested_fraction is None or req.requested_fraction <= 0 or req.requested_fraction >= 1:
            return None, [{"severity": "rejected", "reason": "Requested fraction must be greater than 0 and less than 1."}]
        if size is None:
            return None, [{"severity": "unresolved", "reason": "Fraction requests require a known numeric region dimension size."}]
        indices = list(range(max(1, int(size * req.requested_fraction))))
    else:
        return None, [{"severity": "rejected", "reason": f"Unsupported strategy '{req.strategy}'."}]
    if size is None and req.strategy == "explicit_indices":
        issues.append({"severity": "unresolved", "reason": "Concrete indices cannot be range-checked because region dimension size is unknown."})
    if size is not None:
        if any(index >= size for index in indices):
            issues.append({"severity": "rejected", "reason": f"Prune index out of bounds for region dimension size {size}."})
        if len(indices) >= size:
            issues.append({"severity": "rejected", "reason": "Cannot prune all known region dimension entries."})
    return indices, issues


def _dimensions(ir: dict) -> dict[str, dict]:
    return {item.get("var_id"): item for item in ir.get("dimension_variables", [])}


def _constraints(ir: dict) -> dict[str, dict]:
    return {item.get("constraint_id"): item for item in ir.get("constraint_equations", [])}


def _other(constraint: dict, dimension: str) -> str | None:
    if constraint.get("lhs") == dimension:
        return constraint.get("rhs")
    if constraint.get("rhs") == dimension:
        return constraint.get("lhs")
    return constraint.get("rhs")


def _activated_constraints(ir: dict, dimension_ids: set[str]) -> list[dict]:
    return sorted(
        [
            item for item in ir.get("constraint_equations", [])
            if item.get("lhs") in dimension_ids or item.get("rhs") in dimension_ids
        ],
        key=lambda item: item.get("constraint_id", ""),
    )


def _evaluate_constraint(constraint: dict, root_id: str, indices: list[int] | None, dimensions: dict[str, dict]) -> RegionConstraintSatisfaction:
    relation = constraint.get("relation", "unknown")
    constraint_type = constraint.get("constraint_type", "unknown")
    lhs, rhs = constraint.get("lhs"), constraint.get("rhs")
    endpoints = [dimensions.get(lhs, {}), dimensions.get(rhs, {})]
    target = _other(constraint, root_id)
    status, reason = "not_applicable", "Constraint does not impose an action for this request."
    if relation == "same_indices":
        status = "requires_propagation" if target != root_id else "satisfied"
        reason = "The same pruning index set must be propagated to the paired region dimension."
        target_size = _numeric_size(dimensions.get(target, {}).get("size"))
        if indices is not None and target_size is not None and any(index >= target_size for index in indices):
            status, reason = "blocking", f"Requested indices exceed paired region dimension size {target_size}."
    elif constraint_type == "residual_hidden_equality":
        status, reason = "blocking", "Residual hidden equality blocks local hidden-width pruning without coordinated branch proof."
    elif constraint_type == "layernorm_hidden_equality":
        status, reason = "blocking", "LayerNorm hidden equality protects the normalized hidden width."
    elif relation in {"eq", "join_equal"}:
        protected = any(item.get("protected") or item.get("blocked") or item.get("axis_role") == "hidden" for item in endpoints)
        status = "blocking" if protected else "requires_propagation"
        reason = "A protected or hidden branch-compatibility equality blocks local dimension change." if protected else "Branch equality requires compatible propagated dimensions."
    elif relation == "preserve":
        status, reason = "satisfied", "Elementwise shape preservation does not introduce an additional pruning selection."
    elif relation == "reshape_map":
        status, reason = "unresolved", "Reshape/transpose axis mapping must be recovered before legality can be established."
    elif relation == "fanout":
        status, reason = "requires_propagation", "Producer pruning selections must propagate to every fanout consumer region."
    elif relation == "blocks":
        status, reason = "blocking", "This region equation explicitly blocks the requested transformation."
    elif constraint_type == "attention_head_axis_mapping" or relation == "unknown":
        status, reason = "unresolved", "Attention or unknown axis mapping has not been proven."
    if constraint.get("blocking") and status not in {"satisfied", "requires_propagation"} and status != "unresolved":
        status = "blocking"
    return RegionConstraintSatisfaction(
        constraint_id=constraint.get("constraint_id"),
        relation=relation,
        constraint_type=constraint_type,
        lhs=lhs,
        rhs=rhs,
        status=status,
        blocking=constraint.get("blocking", False),
        confidence=constraint.get("confidence", "low"),
        reason=reason,
    )


def _backend(constraint: dict, source: dict, target: dict) -> str:
    if constraint.get("constraint_type") == "mlp_intermediate_same_indices":
        name = f"{source.get('region_name', '')} {target.get('region_name', '')}".lower()
        return "experimental_bert_mlp" if "bert" in name else "analysis_only"
    if constraint.get("constraint_type") == "linear_bias_follow" and source.get("region_type") == "LinearProjectionRegion":
        return "experimental_linear"
    return "none" if constraint.get("blocking") else "analysis_only"


def compute_region_minimal_repair_set(
    ir: dict,
    request: RegionPruningRequest | dict,
    constraint_satisfaction: list[RegionConstraintSatisfaction | dict],
) -> list[RegionRepairSetItem]:
    req = _request(request)
    dimensions, constraints = _dimensions(ir), _constraints(ir)
    repairs: list[RegionRepairSetItem] = []
    for original in constraint_satisfaction:
        item = asdict(original) if hasattr(original, "__dataclass_fields__") else original
        if item.get("status") in {"satisfied", "not_applicable"}:
            continue
        constraint = constraints.get(item.get("constraint_id"), {})
        target_id = _other(constraint, req.dimension_var_id)
        source = dimensions.get(req.dimension_var_id, {})
        target = dimensions.get(target_id, {})
        ctype = item.get("constraint_type")
        if ctype in {"mlp_intermediate_same_indices", "linear_bias_follow"}:
            repair_type = "same_indices"
            action = f"Apply the same pruning index set to {target_id}."
        elif ctype == "fork_fanout_propagation":
            repair_type = "propagate_to_region"
            action = f"Propagate the pruning selection to fanout consumer dimension {target_id}."
        elif ctype == "residual_hidden_equality":
            repair_type = "block_protected_dimension"
            action = "Keep hidden_dim unchanged or prove coordinated branch repair."
        elif ctype == "layernorm_hidden_equality":
            repair_type = "preserve_dimension"
            action = "Preserve hidden width through normalization."
        elif ctype == "axis_transform_mapping":
            repair_type = "require_axis_mapping"
            action = "Recover reshape/transpose axis mapping before pruning analysis can proceed."
        elif ctype == "attention_head_axis_mapping":
            repair_type = "require_attention_axis_proof"
            action = "Prove head-axis and head-dimension mapping before attention pruning."
        elif ctype == "join_branch_compatibility":
            repair_type = "require_join_compatibility"
            action = "Prove compatible branch dimensions at the join."
        else:
            repair_type = "require_manual_analysis"
            action = "Provide explicit semantic dimension mapping for this region constraint."
        repairs.append(
            RegionRepairSetItem(
                repair_id=f"region_repair_{len(repairs) + 1:06d}",
                repair_type=repair_type,
                source_dimension=req.dimension_var_id,
                target_dimension=target_id,
                constraint_id=item.get("constraint_id"),
                required_action=action,
                executable_backend=_backend(constraint, source, target),
                confidence=item.get("confidence", "low"),
                reason=item.get("reason", ""),
            )
        )
    return repairs


def _empty_slice(root: str, direction: str) -> RegionPropagationSlice:
    return RegionPropagationSlice(
        slice_id=f"region_slice::{direction}::{root}",
        root_dimension=root,
        direction=direction,
        dimensions=[root],
        constraints=[],
        blocked_dimensions=[],
        blocking_constraints=[],
        unresolved_constraints=[],
        protected_dimensions=[],
        reason="No region slice available.",
    )


def _rejected(ir: dict, request: RegionPruningRequest, root: dict | None, code: str, reason: str) -> RegionLegalityCheckResult:
    return RegionLegalityCheckResult(
        check_id=f"region_check::{request.request_id}",
        model_name=ir.get("model_name", request.model_name),
        request=request,
        status="rejected",
        root_dimension=root,
        equivalent_dimensions=[],
        required_propagations=[],
        constraint_satisfaction=[],
        forward_slice=_empty_slice(request.dimension_var_id, "forward"),
        backward_slice=_empty_slice(request.dimension_var_id, "backward"),
        blocking_reasons=[{"type": code, "reason": reason}],
        unresolved_items=[],
        protected_dimensions=[request.dimension_var_id] if root and root.get("protected") else [],
        summary={"num_blocking_reasons": 1, "num_unresolved_items": 0, "num_repairs": 0},
        metadata={"analysis": "static_region_legality"},
    )


def check_region_pruning_legality(ir: dict, request: RegionPruningRequest | dict) -> RegionLegalityCheckResult:
    req = _request(request)
    dimensions = _dimensions(ir)
    root = dimensions.get(req.dimension_var_id)
    if root is None:
        return _rejected(ir, req, None, "root_dimension_missing", "Requested region dimension was not found.")
    if not root.get("prunable"):
        return _rejected(ir, req, root, "root_not_prunable", "Requested region dimension is not marked prunable.")
    if root.get("blocked") or req.dimension_var_id in set(ir.get("blocked_dimensions", [])):
        return _rejected(ir, req, root, "root_blocked", "Requested region dimension is statically blocked.")
    if root.get("protected") and not root.get("prunable"):
        return _rejected(ir, req, root, "root_protected", "Requested region dimension is protected.")
    indices, index_issues = resolve_region_requested_indices(req, root.get("size"))
    rejected_issues = [item for item in index_issues if item.get("severity") == "rejected"]
    if rejected_issues:
        return _rejected(ir, req, root, "invalid_indices", "; ".join(item["reason"] for item in rejected_issues))
    eq_class = get_region_equivalence_class_for_dimension(ir, req.dimension_var_id) or {}
    equivalent = sorted(eq_class.get("members", [req.dimension_var_id]))
    forward = extract_region_slice(ir, req.dimension_var_id, "forward")
    backward = extract_region_slice(ir, req.dimension_var_id, "backward")
    active_dimensions = set(equivalent) | set(forward.dimensions) | set(backward.dimensions)
    satisfaction = [
        _evaluate_constraint(item, req.dimension_var_id, indices, dimensions)
        for item in _activated_constraints(ir, active_dimensions)
    ]
    repairs = compute_region_minimal_repair_set(ir, req, satisfaction)
    blocking_reasons = [
        {"constraint_id": item.constraint_id, "constraint_type": item.constraint_type, "reason": item.reason}
        for item in satisfaction if item.status == "blocking"
    ]
    unresolved_items = [
        item for item in index_issues if item.get("severity") == "unresolved"
    ] + [
        {"constraint_id": item.constraint_id, "constraint_type": item.constraint_type, "reason": item.reason}
        for item in satisfaction if item.status == "unresolved"
    ]
    required = [
        {
            "constraint_id": item.constraint_id,
            "constraint_type": item.constraint_type,
            "target_dimension": _other(_constraints(ir).get(item.constraint_id, {}), req.dimension_var_id),
            "reason": item.reason,
        }
        for item in satisfaction if item.status == "requires_propagation"
    ]
    if blocking_reasons:
        status = "rejected"
    elif unresolved_items:
        status = "ambiguous"
    elif required or repairs:
        status = "legal_with_repairs"
    else:
        status = "legal"
    protected = sorted(
        item for item in active_dimensions if dimensions.get(item, {}).get("protected")
    )
    status_counts = Counter(item.status for item in satisfaction)
    return RegionLegalityCheckResult(
        check_id=f"region_check::{req.request_id}",
        model_name=ir.get("model_name", req.model_name),
        request=req,
        status=status,
        root_dimension=root,
        equivalent_dimensions=equivalent,
        required_propagations=required,
        constraint_satisfaction=satisfaction,
        forward_slice=forward,
        backward_slice=backward,
        minimal_repair_set=repairs,
        blocking_reasons=blocking_reasons,
        unresolved_items=unresolved_items,
        protected_dimensions=protected,
        summary={
            "num_equivalent_dimensions": len(equivalent),
            "num_required_propagations": len(required),
            "num_constraints": len(satisfaction),
            "num_repairs": len(repairs),
            "num_blocking_reasons": len(blocking_reasons),
            "num_unresolved_items": len(unresolved_items),
            "constraint_status_counts": dict(status_counts),
        },
        metadata={"analysis": "static_region_legality", "note": "No model or ONNX artifact is modified."},
    )


def explain_region_blocked_dimensions(ir: dict) -> list[dict]:
    dimensions = _dimensions(ir)
    constraints = ir.get("constraint_equations", [])
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for constraint in constraints:
        ctype = constraint.get("constraint_type")
        relevant = constraint.get("blocking") or constraint.get("relation") in {"reshape_map", "unknown"}
        if not relevant:
            continue
        for dimension_id in (constraint.get("lhs"), constraint.get("rhs")):
            dimension = dimensions.get(dimension_id)
            if not dimension or (dimension_id, constraint.get("constraint_id")) in seen:
                continue
            seen.add((dimension_id, constraint.get("constraint_id")))
            if ctype == "residual_hidden_equality":
                explanation = "Residual merge requires branch hidden dimensions to agree."
                mitigation = "Keep hidden_dim unchanged or prove coordinated branch repair."
                block_type, severity = "residual_hidden_equality", "blocked"
            elif ctype == "layernorm_hidden_equality":
                explanation = "Normalization parameters and incoming hidden width must agree."
                mitigation = "Preserve hidden_dim or prove synchronized normalization repair."
                block_type, severity = "layernorm_hidden_equality", "blocked"
            elif ctype == "axis_transform_mapping":
                explanation = "Shape-transform propagation lacks an explicit axis mapping."
                mitigation = "Recover axis mapping through reshape/transpose before legality analysis."
                block_type, severity = "axis_transform_mapping", "high"
            elif ctype == "attention_head_axis_mapping":
                explanation = "Attention head and hidden-axis mapping is not proven."
                mitigation = "Prove head-axis mapping before attention pruning."
                block_type, severity = "attention_head_axis_mapping", "high"
            else:
                explanation = constraint.get("reason", "Region constraint requires manual structural evidence.")
                mitigation = "Provide explicit region dimension mapping before executable use."
                block_type, severity = ctype or "unknown", "high"
            rows.append(
                {
                    "blocked_id": f"region_blocked_{len(rows) + 1:06d}",
                    "dimension_var_id": dimension_id,
                    "region_id": dimension.get("region_id"),
                    "region_type": dimension.get("region_type"),
                    "dim_name": dimension.get("dim_name"),
                    "axis_role": dimension.get("axis_role"),
                    "block_type": block_type,
                    "severity": severity,
                    "explanation": explanation,
                    "mitigation": mitigation,
                }
            )
    return sorted(rows, key=lambda item: (item["region_type"], item["dimension_var_id"], item["block_type"]))


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 300) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def region_propagation_slice_to_markdown(slice_result: RegionPropagationSlice | dict) -> str:
    data = asdict(slice_result) if hasattr(slice_result, "__dataclass_fields__") else slice_result
    return "\n".join(
        [
            f"# Region Propagation Slice: {data.get('slice_id')}",
            "",
            f"- Root dimension: `{data.get('root_dimension')}`",
            f"- Direction: `{data.get('direction')}`",
            f"- Dimensions: `{data.get('dimensions', [])}`",
            f"- Constraints: `{data.get('constraints', [])}`",
            f"- Blocking constraints: `{data.get('blocking_constraints', [])}`",
            f"- Unresolved constraints: `{data.get('unresolved_constraints', [])}`",
            f"- Protected dimensions: `{data.get('protected_dimensions', [])}`",
            "",
            "This is a static region-aware constraint slice; it does not modify models.",
            "",
        ]
    )


def region_repair_set_to_markdown(request_id: str, repairs: list[RegionRepairSetItem]) -> str:
    rows = [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in repairs]
    return "\n".join(
        [
            f"# Region Minimal Repair Set: {request_id}",
            "",
            "## Required Repairs",
            "",
            _table(rows, ["repair_id", "repair_type", "source_dimension", "target_dimension", "constraint_id", "required_action", "executable_backend", "confidence"]),
            "",
            "## Caveats",
            "",
            "These are static repair obligations, not executed transformations. Experimental backend labels identify prior prototype coverage only.",
            "",
        ]
    )


def region_legality_check_result_to_markdown(result: RegionLegalityCheckResult | dict) -> str:
    data = region_legality_check_result_to_dict(result) if isinstance(result, RegionLegalityCheckResult) else result
    request = data.get("request", {})
    return "\n".join(
        [
            f"# Region-Aware Pruning Legality Check: {request.get('request_id', '')}",
            "",
            "## Status",
            "",
            f"`{data.get('status')}`",
            "",
            "## Requested Region Dimension",
            "",
            f"- Dimension: `{request.get('dimension_var_id')}`",
            f"- Root evidence: `{data.get('root_dimension')}`",
            "",
            "## Requested Indices",
            "",
            f"- Strategy: `{request.get('strategy')}`",
            f"- Indices: `{request.get('requested_indices')}`",
            f"- Count: `{request.get('requested_count')}`",
            f"- Fraction: `{request.get('requested_fraction')}`",
            "",
            "## Equivalent Region Dimensions",
            "",
            "\n".join(f"- `{item}`" for item in data.get("equivalent_dimensions", [])) or "_None._",
            "",
            "## Constraint Satisfaction",
            "",
            _table(data.get("constraint_satisfaction", []), ["constraint_id", "relation", "constraint_type", "lhs", "rhs", "status", "blocking", "confidence", "reason"]),
            "",
            "## Forward Propagation Slice",
            "",
            f"- Dimensions: `{data.get('forward_slice', {}).get('dimensions', [])}`",
            f"- Constraints: `{data.get('forward_slice', {}).get('constraints', [])}`",
            "",
            "## Backward Constraint Slice",
            "",
            f"- Dimensions: `{data.get('backward_slice', {}).get('dimensions', [])}`",
            f"- Constraints: `{data.get('backward_slice', {}).get('constraints', [])}`",
            "",
            "## Minimal Repair Set",
            "",
            _table(data.get("minimal_repair_set", []), ["repair_id", "repair_type", "target_dimension", "required_action", "executable_backend", "confidence"]),
            "",
            "## Blocking Reasons",
            "",
            _table(data.get("blocking_reasons", []), ["constraint_id", "constraint_type", "type", "reason"]),
            "",
            "## Unresolved Items",
            "",
            _table(data.get("unresolved_items", []), ["constraint_id", "constraint_type", "severity", "reason"]),
            "",
            "## Protected Dimensions",
            "",
            "\n".join(f"- `{item}`" for item in data.get("protected_dimensions", [])) or "_None._",
            "",
            "## Interpretation",
            "",
            "This is static region-aware legality analysis over semantic region dimensions. It reports obligations and blockers conservatively and does not modify models.",
            "",
        ]
    )
