"""Static legality analysis over pruning Dimension IR."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.ir_graph import extract_slice, find_constraints_touching_dimension, get_equivalence_class_for_dimension
from model_analysis.paths import ensure_dir


@dataclass
class SymbolicPruningRequest:
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
class ConstraintSatisfaction:
    constraint_id: str
    relation: str
    constraint_type: str
    lhs: str
    rhs: str
    status: str
    confidence: str
    reason: str


@dataclass
class PropagationSlice:
    slice_id: str
    root_dimension: str
    direction: str
    dimensions: list[str]
    constraints: list[str]
    blocking_constraints: list[str]
    unresolved_constraints: list[str]
    reason: str


@dataclass
class RepairSetItem:
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
class LegalityCheckResult:
    check_id: str
    model_name: str
    request: SymbolicPruningRequest
    status: str
    root_dimension: dict | None
    equivalent_dimensions: list[str]
    required_propagations: list[dict[str, Any]]
    constraint_satisfaction: list[ConstraintSatisfaction]
    forward_slice: PropagationSlice
    backward_slice: PropagationSlice
    minimal_repair_set: list[RepairSetItem] = field(default_factory=list)
    blocking_reasons: list[dict[str, Any]] = field(default_factory=list)
    unresolved_items: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def symbolic_pruning_request_to_dict(request: SymbolicPruningRequest) -> dict[str, Any]:
    return asdict(request)


def legality_check_result_to_dict(result: LegalityCheckResult) -> dict[str, Any]:
    return asdict(result)


def write_legality_check_json(result: LegalityCheckResult, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(legality_check_result_to_dict(result), indent=2), encoding="utf-8")


def _request_obj(request: SymbolicPruningRequest | dict) -> SymbolicPruningRequest:
    return SymbolicPruningRequest(**request) if isinstance(request, dict) else request


def _safe(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace(" ", "_")


def make_symbolic_pruning_request(
    model_name: str,
    dimension_var_id: str,
    indices: list[int] | None = None,
    count: int | None = None,
    fraction: float | None = None,
    strategy: str | None = None,
    reason: str | None = None,
) -> SymbolicPruningRequest:
    provided = sum(value is not None for value in (indices, count, fraction))
    if provided > 1:
        raise ValueError("Only one of indices, count, or fraction may be provided.")
    if indices is not None:
        resolved_strategy = "explicit_indices"
        symbolic = False
    elif count is not None:
        resolved_strategy = strategy or "first_n"
        if resolved_strategy not in {"first_n", "last_n"}:
            raise ValueError("Count requests support first_n or last_n strategies.")
        symbolic = False
    elif fraction is not None:
        resolved_strategy = "fraction"
        symbolic = False
    else:
        resolved_strategy = strategy or "symbolic"
        if resolved_strategy != "symbolic":
            raise ValueError("Non-symbolic strategies require indices, count, or fraction.")
        symbolic = True
    request_id = f"request__{_safe(model_name)}__{_safe(dimension_var_id)}__{resolved_strategy}"
    return SymbolicPruningRequest(
        request_id=request_id,
        model_name=model_name,
        dimension_var_id=dimension_var_id,
        requested_indices=indices,
        requested_count=count,
        requested_fraction=fraction,
        symbolic=symbolic,
        strategy=resolved_strategy,
        reason=reason,
    )


def resolve_requested_indices(
    request: SymbolicPruningRequest | dict,
    dimension_size: int | None,
) -> tuple[list[int] | None, list[dict]]:
    req = _request_obj(request)
    issues: list[dict[str, Any]] = []
    indices: list[int] | None = None
    if req.strategy == "symbolic":
        return None, issues
    if req.strategy == "explicit_indices":
        indices = sorted(set(req.requested_indices or []))
    elif req.strategy in {"first_n", "last_n"}:
        if req.requested_count is None or req.requested_count <= 0:
            issues.append({"severity": "rejected", "reason": "Requested count must be positive."})
            return None, issues
        if dimension_size is None:
            issues.append({"severity": "unresolved", "reason": f"{req.strategy} requires known dimension size."})
            return None, issues
        if req.strategy == "first_n":
            indices = list(range(req.requested_count))
        else:
            indices = list(range(dimension_size - req.requested_count, dimension_size))
    elif req.strategy == "fraction":
        if req.requested_fraction is None or req.requested_fraction <= 0 or req.requested_fraction >= 1:
            issues.append({"severity": "rejected", "reason": "Requested fraction must be greater than 0 and less than 1."})
            return None, issues
        if dimension_size is None:
            issues.append({"severity": "unresolved", "reason": "Fraction requests require known dimension size."})
            return None, issues
        count = max(1, int(dimension_size * req.requested_fraction))
        indices = list(range(count))
    else:
        issues.append({"severity": "rejected", "reason": f"Unsupported strategy '{req.strategy}'."})
        return None, issues

    if any(index < 0 for index in indices):
        issues.append({"severity": "rejected", "reason": "Prune indices must be non-negative."})
    if len(indices) != len(req.requested_indices or indices) and req.strategy == "explicit_indices":
        issues.append({"severity": "warning", "reason": "Duplicate indices were normalized."})
    if dimension_size is not None:
        if any(index >= dimension_size for index in indices):
            issues.append({"severity": "rejected", "reason": f"Prune index out of bounds for dimension size {dimension_size}."})
        if len(indices) >= dimension_size:
            issues.append({"severity": "rejected", "reason": "Cannot prune all known dimension entries."})
    return indices, issues


def _dimensions_by_id(ir: dict) -> dict[str, dict]:
    return {item.get("var_id"): item for item in ir.get("dimension_variables", [])}


def _constraints_by_id(ir: dict) -> dict[str, dict]:
    return {item.get("constraint_id"): item for item in ir.get("constraint_equations", [])}


def _related_constraints(ir: dict, dimensions: set[str]) -> list[dict]:
    constraints = []
    for constraint in ir.get("constraint_equations", []):
        if constraint.get("lhs") in dimensions or constraint.get("rhs") in dimensions:
            constraints.append(constraint)
    return sorted(constraints, key=lambda item: item.get("constraint_id", ""))


def _other(constraint: dict, root: str) -> str | None:
    if constraint.get("lhs") == root:
        return constraint.get("rhs")
    if constraint.get("rhs") == root:
        return constraint.get("lhs")
    return constraint.get("rhs")


def _evaluate_constraint(constraint: dict, root_dimension: str, concrete_indices: list[int] | None, dimensions: dict[str, dict]) -> ConstraintSatisfaction:
    relation = constraint.get("relation")
    ctype = constraint.get("constraint_type")
    lhs = constraint.get("lhs")
    rhs = constraint.get("rhs")
    target = _other(constraint, root_dimension)
    status = "not_applicable"
    reason = "Constraint does not require action for this request."
    if relation == "same_indices":
        status = "requires_propagation" if target != root_dimension else "satisfied"
        reason = "Same-index constraint requires the pruning index set to propagate to the paired dimension."
        if concrete_indices is not None and target in dimensions and dimensions[target].get("size") is not None:
            size = dimensions[target]["size"]
            if any(index >= size for index in concrete_indices):
                status = "blocking"
                reason = f"Requested indices exceed target dimension size {size}."
    elif relation == "eq":
        if ctype == "residual_hidden_equality":
            status = "blocking"
            reason = "Residual hidden equality blocks hidden-size pruning unless coordinated residual repair is proven."
        elif ctype == "layernorm_hidden_equality":
            status = "blocking"
            reason = "LayerNorm hidden equality is high risk and must preserve equal hidden shape."
        else:
            status = "requires_propagation"
            reason = "Equality constraint requires shape-preserving propagation."
    elif relation == "tied":
        status = "blocking"
        reason = "Tied parameter update is not available in the static analysis backend."
    elif relation == "reshape":
        status = "unresolved"
        reason = "Reshape/index mapping must be recovered before legality can be proven."
    elif relation == "unknown":
        status = "unresolved"
        reason = "Unknown dimension mapping is unresolved."
    if constraint.get("blocking") and status not in {"requires_propagation", "satisfied"}:
        status = "blocking" if relation != "unknown" else "unresolved"
    return ConstraintSatisfaction(
        constraint_id=constraint.get("constraint_id"),
        relation=relation,
        constraint_type=ctype,
        lhs=lhs,
        rhs=rhs,
        status=status,
        confidence=constraint.get("confidence", "low"),
        reason=reason,
    )


def _backend_for_constraint(constraint: dict, dimensions: dict[str, dict]) -> str:
    if constraint.get("constraint_type") != "mlp_intermediate_consistency":
        return "none" if constraint.get("blocking") else "analysis_only"
    lhs = dimensions.get(constraint.get("lhs"), {})
    rhs = dimensions.get(constraint.get("rhs"), {})
    lhs_owner = lhs.get("owner_name", "")
    rhs_owner = rhs.get("owner_name", "")
    if "intermediate.dense" in lhs_owner and "output.dense" in rhs_owner:
        return "experimental_bert_mlp"
    if lhs.get("owner_type") in {"linear", "mlp_expansion"} and rhs.get("owner_type") in {"linear", "mlp_projection"}:
        return "experimental_linear"
    return "analysis_only"


def compute_minimal_repair_set(
    ir: dict,
    request: SymbolicPruningRequest | dict,
    constraint_satisfaction: list[ConstraintSatisfaction | dict],
) -> list[RepairSetItem]:
    req = _request_obj(request)
    dimensions = _dimensions_by_id(ir)
    constraints = _constraints_by_id(ir)
    repairs = []
    for item in constraint_satisfaction:
        satisfaction = asdict(item) if hasattr(item, "__dataclass_fields__") else item
        if satisfaction.get("status") in {"satisfied", "not_applicable"}:
            continue
        constraint = constraints.get(satisfaction.get("constraint_id"), {})
        target = _other(constraint, req.dimension_var_id)
        ctype = satisfaction.get("constraint_type")
        if satisfaction.get("relation") == "same_indices":
            repair_type = "same_indices"
            action = f"Apply the same pruning index set to {target}."
        elif ctype == "residual_hidden_equality":
            repair_type = "block_hidden_size_change"
            action = "Keep hidden size unchanged or prove coordinated residual repair."
        elif ctype == "layernorm_hidden_equality":
            repair_type = "preserve_equal_shape"
            action = "Preserve the equal hidden shape required by normalization."
        elif ctype == "embedding_output_tying":
            repair_type = "tied_parameter_update"
            action = "Detect tied weights and update both tied parameter owners."
        elif ctype == "unknown_mapping":
            repair_type = "require_manual_mapping"
            action = "Recover explicit dimension mapping before pruning execution."
        else:
            repair_type = "require_manual_mapping"
            action = "Manual mapping or propagation proof is required."
        repairs.append(
            RepairSetItem(
                repair_id=f"repair_{len(repairs) + 1:05d}",
                repair_type=repair_type,
                source_dimension=req.dimension_var_id,
                target_dimension=target,
                constraint_id=satisfaction.get("constraint_id"),
                required_action=action,
                executable_backend=_backend_for_constraint(constraint, dimensions),
                confidence=satisfaction.get("confidence", "low"),
                reason=satisfaction.get("reason", ""),
            )
        )
    return repairs


def check_pruning_legality(
    ir: dict,
    request: SymbolicPruningRequest | dict,
) -> LegalityCheckResult:
    req = _request_obj(request)
    dimensions = _dimensions_by_id(ir)
    root = dimensions.get(req.dimension_var_id)
    empty_slice = PropagationSlice("slice::empty", req.dimension_var_id, "none", [], [], [], [], "No slice available.")
    if not root:
        return _rejected_result(ir, req, None, "root_dimension_missing", "Root dimension variable was not found.")
    if not root.get("prunable"):
        return _rejected_result(ir, req, root, "root_not_prunable", "Root dimension is not marked prunable.")

    concrete_indices, index_issues = resolve_requested_indices(req, root.get("size"))
    if any(item.get("severity") == "rejected" for item in index_issues):
        return _rejected_result(ir, req, root, "invalid_indices", "; ".join(item["reason"] for item in index_issues if item.get("severity") == "rejected"))

    eq_class = get_equivalence_class_for_dimension(ir, req.dimension_var_id) or {}
    equivalent = sorted(eq_class.get("members", [req.dimension_var_id]))
    forward_slice = extract_slice(ir, req.dimension_var_id, "forward")
    backward_slice = extract_slice(ir, req.dimension_var_id, "backward")
    involved_dims = set(equivalent) | set(forward_slice.dimensions) | set(backward_slice.dimensions)
    satisfactions = [_evaluate_constraint(item, req.dimension_var_id, concrete_indices, dimensions) for item in _related_constraints(ir, involved_dims)]
    repairs = compute_minimal_repair_set(ir, req, satisfactions)
    blocking = [
        {"constraint_id": item.constraint_id, "reason": item.reason, "constraint_type": item.constraint_type}
        for item in satisfactions
        if item.status == "blocking"
    ]
    unresolved = list(index_issues) + [
        {"constraint_id": item.constraint_id, "reason": item.reason, "constraint_type": item.constraint_type}
        for item in satisfactions
        if item.status == "unresolved"
    ]
    required_propagations = [
        {"constraint_id": item.constraint_id, "lhs": item.lhs, "rhs": item.rhs, "reason": item.reason}
        for item in satisfactions
        if item.status == "requires_propagation"
    ]
    if blocking:
        status = "rejected"
    elif unresolved:
        status = "ambiguous"
    elif repairs:
        status = "legal_with_repairs"
    else:
        status = "legal"
    summary = {
        "num_equivalent_dimensions": len(equivalent),
        "num_required_propagations": len(required_propagations),
        "num_constraints_checked": len(satisfactions),
        "num_repairs": len(repairs),
        "num_blocking_reasons": len(blocking),
        "num_unresolved_items": len(unresolved),
        "constraint_status_counts": dict(Counter(item.status for item in satisfactions)),
    }
    return LegalityCheckResult(
        check_id=f"check__{req.request_id}",
        model_name=ir.get("model_name", req.model_name),
        request=req,
        status=status,
        root_dimension=root,
        equivalent_dimensions=equivalent,
        required_propagations=required_propagations,
        constraint_satisfaction=satisfactions,
        forward_slice=forward_slice,
        backward_slice=backward_slice,
        minimal_repair_set=repairs,
        blocking_reasons=blocking,
        unresolved_items=unresolved,
        summary=summary,
        metadata={"analysis_only": True},
    )


def _rejected_result(ir: dict, req: SymbolicPruningRequest, root: dict | None, reason_type: str, reason: str) -> LegalityCheckResult:
    empty = PropagationSlice("slice::empty", req.dimension_var_id, "none", [], [], [], [], "No propagation slice because request was rejected.")
    return LegalityCheckResult(
        check_id=f"check__{req.request_id}",
        model_name=ir.get("model_name", req.model_name),
        request=req,
        status="rejected",
        root_dimension=root,
        equivalent_dimensions=[],
        required_propagations=[],
        constraint_satisfaction=[],
        forward_slice=empty,
        backward_slice=empty,
        minimal_repair_set=[],
        blocking_reasons=[{"type": reason_type, "reason": reason}],
        unresolved_items=[],
        summary={"num_blocking_reasons": 1, "num_unresolved_items": 0, "num_repairs": 0},
        metadata={"analysis_only": True},
    )


def explain_blocked_regions(ir: dict) -> list[dict]:
    dimensions = _dimensions_by_id(ir)
    blocked = []
    seen = set()
    for constraint in ir.get("constraint_equations", []):
        if not constraint.get("blocking"):
            continue
        for dim_id in (constraint.get("lhs"), constraint.get("rhs")):
            if dim_id not in dimensions:
                continue
            key = (dim_id, constraint.get("constraint_id"))
            if key in seen:
                continue
            seen.add(key)
            blocked.append(
                {
                    "blocked_id": f"blocked_{len(blocked) + 1:05d}",
                    "dimension_var_id": dim_id,
                    "constraint_id": constraint.get("constraint_id"),
                    "block_type": constraint.get("constraint_type"),
                    "severity": "blocked" if constraint.get("relation") in {"eq", "tied"} else "high",
                    "explanation": _block_explanation(constraint),
                    "mitigation": _block_mitigation(constraint),
                }
            )
    for dim_id in ir.get("blocked_dimensions", []):
        if not any(item["dimension_var_id"] == dim_id for item in blocked):
            blocked.append(
                {
                    "blocked_id": f"blocked_{len(blocked) + 1:05d}",
                    "dimension_var_id": dim_id,
                    "constraint_id": None,
                    "block_type": "blocked_dimension",
                    "severity": "blocked",
                    "explanation": "Dimension is marked blocked in the Dimension IR.",
                    "mitigation": "Keep this dimension unchanged unless stronger structural evidence is added.",
                }
            )
    return blocked


def _block_explanation(constraint: dict) -> str:
    ctype = constraint.get("constraint_type")
    if ctype == "residual_hidden_equality":
        return "Hidden-size pruning crosses residual equality constraints."
    if ctype == "embedding_output_tying":
        return "Embedding/output tied parameters require coordinated updates that are not proven."
    if ctype == "unknown_mapping":
        return "Unknown mapping prevents proving how pruning indices propagate."
    if ctype == "reshape_preservation":
        return "Reshape or transpose path needs explicit index mapping."
    return constraint.get("reason", "Blocking constraint prevents legal pruning without additional proof.")


def _block_mitigation(constraint: dict) -> str:
    ctype = constraint.get("constraint_type")
    if ctype == "residual_hidden_equality":
        return "Keep hidden size unchanged or prove coordinated residual repair."
    if ctype == "embedding_output_tying":
        return "Detect tied weights and update both embedding and output projection."
    if ctype == "unknown_mapping":
        return "Recover reshape or graph index mapping from ONNX/PyTorch graph."
    return "Require explicit dimension mapping before executable pruning."


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def legality_check_result_to_markdown(result: LegalityCheckResult) -> str:
    data = legality_check_result_to_dict(result)
    request = data["request"]
    return "\n".join(
        [
            f"# Pruning Legality Check: {request['request_id']}",
            "",
            "## Status",
            "",
            f"- `{result.status}`",
            "",
            "## Requested Dimension",
            "",
            f"- `{request['dimension_var_id']}`",
            "",
            "## Requested Indices",
            "",
            f"- Strategy: `{request['strategy']}`",
            f"- Indices: `{request.get('requested_indices')}`",
            f"- Count: `{request.get('requested_count')}`",
            f"- Fraction: `{request.get('requested_fraction')}`",
            "",
            "## Equivalent Dimensions",
            "",
            "\n".join(f"- `{item}`" for item in result.equivalent_dimensions) or "_None._",
            "",
            "## Constraint Satisfaction",
            "",
            _table(data["constraint_satisfaction"], ["constraint_id", "relation", "constraint_type", "status", "confidence", "reason"]),
            "",
            "## Forward Slice",
            "",
            f"- Dimensions: `{result.forward_slice.dimensions}`",
            f"- Constraints: `{result.forward_slice.constraints}`",
            "",
            "## Backward Slice",
            "",
            f"- Dimensions: `{result.backward_slice.dimensions}`",
            f"- Constraints: `{result.backward_slice.constraints}`",
            "",
            "## Minimal Repair Set",
            "",
            _table(data["minimal_repair_set"], ["repair_id", "repair_type", "target_dimension", "constraint_id", "executable_backend", "required_action"]),
            "",
            "## Blocking Reasons",
            "",
            _table(data["blocking_reasons"], ["type", "constraint_id", "constraint_type", "reason"]),
            "",
            "## Unresolved Items",
            "",
            _table(data["unresolved_items"], ["severity", "constraint_id", "constraint_type", "reason"]),
            "",
            "## Interpretation",
            "",
            "This is a static Dimension-IR legality analysis. It does not modify weights, rewrite graphs, execute pruning, or evaluate accuracy.",
            "",
        ]
    )


def propagation_slice_to_dict(slice_obj: PropagationSlice) -> dict[str, Any]:
    return asdict(slice_obj)


def propagation_slice_to_markdown(slice_obj: PropagationSlice) -> str:
    data = asdict(slice_obj)
    return "\n".join(
        [
            f"# Propagation Slice: {slice_obj.slice_id}",
            "",
            "## Root Dimension",
            "",
            f"- `{slice_obj.root_dimension}`",
            "",
            "## Direction",
            "",
            f"- `{slice_obj.direction}`",
            "",
            "## Dimensions",
            "",
            "\n".join(f"- `{item}`" for item in slice_obj.dimensions) or "_None._",
            "",
            "## Constraints",
            "",
            "\n".join(f"- `{item}`" for item in slice_obj.constraints) or "_None._",
            "",
            "## Blocking Constraints",
            "",
            "\n".join(f"- `{item}`" for item in slice_obj.blocking_constraints) or "_None._",
            "",
            "## Unresolved Constraints",
            "",
            "\n".join(f"- `{item}`" for item in slice_obj.unresolved_constraints) or "_None._",
            "",
            "## Interpretation",
            "",
            data.get("reason", ""),
            "",
        ]
    )


def repair_set_to_markdown(request_id: str, repairs: list[RepairSetItem]) -> str:
    rows = [asdict(item) for item in repairs]
    return "\n".join(
        [
            f"# Minimal Repair Set: {request_id}",
            "",
            "## Summary",
            "",
            f"- Required repairs: `{len(repairs)}`",
            "",
            "## Required Repairs",
            "",
            _table(rows, ["repair_id", "repair_type", "source_dimension", "target_dimension", "constraint_id", "required_action"]),
            "",
            "## Experimental Backend Availability",
            "",
            _table(rows, ["repair_id", "executable_backend", "confidence", "reason"]),
            "",
            "## Manual Mapping Requirements",
            "",
            "\n".join(f"- `{row['repair_id']}` requires manual mapping." for row in rows if row["executable_backend"] in {"none", "analysis_only"}) or "_None._",
            "",
            "## Caveats",
            "",
            "- This report is analysis-only and does not execute any repair.",
            "- Experimental backend labels are descriptive, not execution requests.",
            "",
        ]
    )
