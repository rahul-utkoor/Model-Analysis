"""Symbolic dimension-variable IR for pruning analysis."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class DimensionVariable:
    var_id: str
    source_kind: str
    owner_id: str
    owner_name: str
    owner_type: str
    dim_name: str
    size: int | None
    axis: int | None
    semantic_role: str
    prunable: bool
    confidence: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class PruningIndexVariable:
    index_var_id: str
    dimension_var_id: str
    allowed_range: list[int] | None
    selected_indices: list[int] | None
    symbolic: bool
    reason: str


@dataclass
class ConstraintEquation:
    constraint_id: str
    lhs: str
    rhs: str
    relation: str
    direction: str
    constraint_type: str
    expression: str
    confidence: str
    blocking: bool
    evidence: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class DimensionEquivalenceClass:
    class_id: str
    members: list[str]
    representative: str
    size: int | None
    class_type: str
    confidence: str
    constraints: list[str]
    reason: str


@dataclass
class PruningIR:
    model_name: str
    hf_id: str
    task: str
    dimension_variables: list[DimensionVariable] = field(default_factory=list)
    index_variables: list[PruningIndexVariable] = field(default_factory=list)
    constraint_equations: list[ConstraintEquation] = field(default_factory=list)
    equivalence_classes: list[DimensionEquivalenceClass] = field(default_factory=list)
    blocked_dimensions: list[str] = field(default_factory=list)
    unresolved_constraints: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def dimension_variable_to_dict(value: DimensionVariable) -> dict[str, Any]:
    return asdict(value)


def index_variable_to_dict(value: PruningIndexVariable) -> dict[str, Any]:
    return asdict(value)


def constraint_equation_to_dict(value: ConstraintEquation) -> dict[str, Any]:
    return asdict(value)


def equivalence_class_to_dict(value: DimensionEquivalenceClass) -> dict[str, Any]:
    return asdict(value)


def pruning_ir_to_dict(ir: PruningIR) -> dict[str, Any]:
    return asdict(ir)


def write_pruning_ir_json(ir: PruningIR, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(pruning_ir_to_dict(ir), indent=2), encoding="utf-8")


def load_pruning_ir_json(path: Path) -> PruningIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _ir_from_dict(data)


def _ir_from_dict(data: dict[str, Any]) -> PruningIR:
    return PruningIR(
        model_name=data["model_name"],
        hf_id=data.get("hf_id", ""),
        task=data.get("task", ""),
        dimension_variables=[DimensionVariable(**item) for item in data.get("dimension_variables", [])],
        index_variables=[PruningIndexVariable(**item) for item in data.get("index_variables", [])],
        constraint_equations=[ConstraintEquation(**item) for item in data.get("constraint_equations", [])],
        equivalence_classes=[DimensionEquivalenceClass(**item) for item in data.get("equivalence_classes", [])],
        blocked_dimensions=data.get("blocked_dimensions", []),
        unresolved_constraints=data.get("unresolved_constraints", []),
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def _normalize(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def _source_kind(owner_id: str) -> str:
    if owner_id.startswith("torch:"):
        return "torch"
    if owner_id.startswith("onnx:"):
        return "onnx"
    if owner_id.startswith("synthetic:"):
        return "synthetic"
    return "unknown"


def _axis_for_dim(dim_name: str) -> int | None:
    if dim_name in {"out_features", "channel_out", "num_heads", "vocab_dim"}:
        return 0
    if dim_name in {"in_features", "channel_in", "embedding_dim", "hidden_dim", "intermediate_dim", "head_dim"}:
        return 1
    return None


def _dim_var_id(owner_id: str, dim_name: str) -> str:
    return f"dim::{_normalize(owner_id)}::{_normalize(dim_name)}"


def _index_var_id(var_id: str) -> str:
    return f"idx::{var_id}"


def _semantic_role(role: str, blocked: bool) -> str:
    if blocked:
        return "blocked"
    if role in {"producer", "consumer", "coupled", "boundary", "propagation_only", "blocked"}:
        return role
    return "unknown"


def _blocked_dim_ids(pruning_map: dict[str, Any]) -> set[str]:
    blocked = set()
    blocked_opportunities = set(pruning_map.get("blocked_opportunities", []))
    for opportunity in pruning_map.get("opportunities", []):
        if opportunity.get("opportunity_id") in blocked_opportunities or opportunity.get("risk_level") == "blocked" or opportunity.get("executability") == "blocked":
            blocked.update(opportunity.get("prunable_dimensions", []))
    return blocked


def build_dimension_variables_from_pruning_map(
    pruning_map: dict,
) -> list[DimensionVariable]:
    """Convert pruning-map dimensions into symbolic dimension variables."""
    blocked = _blocked_dim_ids(pruning_map)
    variables = []
    for item in pruning_map.get("pruning_dimensions", []):
        original_dim_id = item.get("dim_id")
        owner_id = item.get("unit_id", "unknown")
        var_id = _dim_var_id(owner_id, item.get("dim_name", "unknown"))
        role = _semantic_role(item.get("structural_role", "unknown"), original_dim_id in blocked)
        variables.append(
            DimensionVariable(
                var_id=var_id,
                source_kind=_source_kind(owner_id),
                owner_id=owner_id,
                owner_name=item.get("unit_name", owner_id),
                owner_type=item.get("unit_type", "unknown"),
                dim_name=item.get("dim_name", "unknown"),
                size=item.get("size"),
                axis=_axis_for_dim(item.get("dim_name", "")),
                semantic_role=role,
                prunable=role not in {"blocked", "propagation_only"},
                confidence=item.get("confidence", "low"),
                evidence=[{"source": "pruning_dimension", "data": item}],
                reason=item.get("reason", "Dimension imported from pruning map."),
            )
        )
    return sorted(variables, key=lambda value: value.var_id)


def build_index_variables(
    dimension_variables: list[DimensionVariable],
) -> list[PruningIndexVariable]:
    """Create symbolic pruning-index variables for prunable dimensions."""
    index_variables = []
    for dimension in dimension_variables:
        if not dimension.prunable:
            continue
        index_variables.append(
            PruningIndexVariable(
                index_var_id=_index_var_id(dimension.var_id),
                dimension_var_id=dimension.var_id,
                allowed_range=[0, dimension.size] if dimension.size is not None else None,
                selected_indices=None,
                symbolic=True,
                reason="Symbolic set of indices selected for pruning along this dimension.",
            )
        )
    return sorted(index_variables, key=lambda value: value.index_var_id)


def _var_by_original_dim(pruning_map: dict[str, Any], variables: list[DimensionVariable]) -> dict[str, DimensionVariable]:
    by_key = {(variable.owner_id, variable.dim_name): variable for variable in variables}
    result = {}
    for item in pruning_map.get("pruning_dimensions", []):
        variable = by_key.get((item.get("unit_id"), item.get("dim_name")))
        if variable:
            result[item.get("dim_id")] = variable
    return result


def _relation_and_type(constraint_type: str) -> tuple[str, str, bool]:
    if constraint_type == "mlp_same_intermediate_indices":
        return "same_indices", "mlp_intermediate_consistency", False
    if constraint_type == "qkv_same_heads":
        return "same_indices", "qkv_head_consistency", False
    if constraint_type == "residual_equal_shape":
        return "eq", "residual_hidden_equality", True
    if constraint_type in {"equality", "normalization_dependency"}:
        return "eq", "layernorm_hidden_equality", True
    if constraint_type == "tied_parameter":
        return "tied", "embedding_output_tying", True
    if constraint_type == "reshape_preserving":
        return "reshape", "reshape_preservation", False
    if constraint_type == "onnx_shape_dependency":
        return "unknown", "onnx_shape_dependency", True
    if constraint_type == "unknown_mapping":
        return "unknown", "unknown_mapping", True
    return "unknown", "unknown_mapping", True


def _expression(relation: str, lhs: str, rhs: str, constraint_type: str) -> str:
    if relation == "same_indices":
        return f"idx({lhs}) == idx({rhs})"
    if relation == "eq":
        return f"dim({lhs}) == dim({rhs})"
    if relation == "tied":
        return f"tied({lhs}, {rhs})"
    if relation == "reshape":
        return f"reshape_preserves({lhs}, {rhs})"
    return f"unknown_mapping({lhs}, {rhs}) /* {constraint_type} */"


def build_constraint_equations_from_pruning_map(
    pruning_map: dict,
    dimension_variables: list[DimensionVariable],
) -> list[ConstraintEquation]:
    """Convert pruning-map propagation constraints into symbolic equations."""
    var_map = _var_by_original_dim(pruning_map, dimension_variables)
    equations = []
    for item in pruning_map.get("propagation_constraints", []):
        lhs_var = var_map.get(item.get("src_dim_id"))
        rhs_var = var_map.get(item.get("dst_dim_id"))
        relation, symbolic_type, blocking = _relation_and_type(item.get("constraint_type", "unknown_mapping"))
        if not lhs_var or not rhs_var:
            lhs = item.get("src_dim_id", "missing_lhs")
            rhs = item.get("dst_dim_id", "missing_rhs")
            relation = "unknown"
            symbolic_type = "unknown_mapping"
            blocking = True
            reason = "Source or destination pruning dimension could not be mapped into the Dimension IR."
            evidence = [{"source": "propagation_constraint", "data": item}]
        else:
            lhs = lhs_var.var_id
            rhs = rhs_var.var_id
            reason = _constraint_reason(symbolic_type, item.get("reason", ""))
            evidence = list(item.get("evidence", [])) + [{"source": "propagation_constraint", "data": item}]
        equations.append(
            ConstraintEquation(
                constraint_id=item.get("constraint_id", f"constraint_{len(equations) + 1:05d}"),
                lhs=lhs,
                rhs=rhs,
                relation=relation,
                direction=item.get("direction", "none"),
                constraint_type=symbolic_type,
                expression=_expression(relation, lhs, rhs, symbolic_type),
                confidence=item.get("confidence", "low"),
                blocking=blocking,
                evidence=evidence,
                reason=reason,
            )
        )
    return sorted(equations, key=lambda value: value.constraint_id)


def _constraint_reason(symbolic_type: str, original_reason: str) -> str:
    if symbolic_type == "mlp_intermediate_consistency":
        return original_reason or "MLP expansion output channels feed projection input channels and require same pruning indices."
    if symbolic_type == "qkv_head_consistency":
        return original_reason or "Q/K/V head-related dimensions must use consistent pruning selections."
    if symbolic_type == "residual_hidden_equality":
        return original_reason or "Residual paths require hidden shape preservation; hidden-size pruning is blocking without repair proof."
    if symbolic_type == "layernorm_hidden_equality":
        return original_reason or "LayerNorm affine/normalized dimensions must remain aligned with hidden dimensions."
    if symbolic_type == "embedding_output_tying":
        return original_reason or "Tied embedding/output parameters require shared pruning decisions."
    if symbolic_type == "reshape_preservation":
        return original_reason or "Shape-changing operation requires dimension-preserving propagation evidence."
    return original_reason or "Dimension mapping is unknown and remains unresolved."


class _UnionFind:
    def __init__(self, members: list[str]):
        self.parent = {member: member for member in members}

    def find(self, item: str) -> str:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        representative = min(left_root, right_root)
        other = max(left_root, right_root)
        self.parent[other] = representative


def build_dimension_equivalence_classes(
    dimension_variables: list[DimensionVariable],
    constraint_equations: list[ConstraintEquation],
) -> list[DimensionEquivalenceClass]:
    """Build equality/index/tied equivalence classes with a conservative union-find."""
    variable_ids = [variable.var_id for variable in dimension_variables]
    by_id = {variable.var_id: variable for variable in dimension_variables}
    uf = _UnionFind(variable_ids)
    union_relations = {"eq", "same_indices", "tied"}
    for equation in constraint_equations:
        if equation.relation in union_relations and equation.lhs in by_id and equation.rhs in by_id:
            uf.union(equation.lhs, equation.rhs)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for var_id in variable_ids:
        members_by_root[uf.find(var_id)].append(var_id)

    constraint_ids_by_member: dict[str, list[ConstraintEquation]] = defaultdict(list)
    for equation in constraint_equations:
        if equation.lhs in by_id:
            constraint_ids_by_member[equation.lhs].append(equation)
        if equation.rhs in by_id:
            constraint_ids_by_member[equation.rhs].append(equation)

    classes = []
    for index, members in enumerate(sorted((sorted(values) for values in members_by_root.values()), key=lambda values: values[0]), start=1):
        equations = []
        for member in members:
            equations.extend(constraint_ids_by_member.get(member, []))
        unique_equations = {equation.constraint_id: equation for equation in equations}
        sizes = {by_id[member].size for member in members if by_id[member].size is not None}
        size = next(iter(sizes)) if len(sizes) == 1 else None
        confidence = _class_confidence(members, by_id, list(unique_equations.values()), len(sizes) > 1)
        classes.append(
            DimensionEquivalenceClass(
                class_id=f"eqclass_{index:05d}",
                members=members,
                representative=members[0],
                size=size,
                class_type=_class_type(list(unique_equations.values())),
                confidence=confidence,
                constraints=sorted(unique_equations),
                reason=_class_reason(list(unique_equations.values()), len(sizes) > 1),
            )
        )
    return classes


def _class_type(equations: list[ConstraintEquation]) -> str:
    types = {equation.constraint_type for equation in equations}
    if "embedding_output_tying" in types:
        return "tied_parameter"
    if "residual_hidden_equality" in types:
        return "residual_hidden"
    if "mlp_intermediate_consistency" in types:
        return "mlp_intermediate"
    if "qkv_head_consistency" in types:
        return "qkv_heads"
    if types:
        return "equality"
    return "unknown"


def _class_confidence(members: list[str], by_id: dict[str, DimensionVariable], equations: list[ConstraintEquation], size_conflict: bool) -> str:
    if size_conflict:
        return "low"
    confidences = {by_id[member].confidence for member in members}
    confidences.update(equation.confidence for equation in equations)
    if confidences == {"high"}:
        return "high"
    if "low" in confidences:
        return "low"
    return "medium"


def _class_reason(equations: list[ConstraintEquation], size_conflict: bool) -> str:
    if size_conflict:
        return "Known member sizes conflict; equivalence class remains low confidence."
    if equations:
        return "Members are connected by equality, same-index, or tied-parameter constraints."
    return "Singleton dimension class with no equality-producing constraints."


def build_pruning_ir(
    pruning_map: dict,
) -> PruningIR:
    """Build the full symbolic PruningIR from a model pruning map."""
    dimensions = build_dimension_variables_from_pruning_map(pruning_map)
    indices = build_index_variables(dimensions)
    equations = build_constraint_equations_from_pruning_map(pruning_map, dimensions)
    classes = build_dimension_equivalence_classes(dimensions, equations)
    dim_ids = {dimension.var_id for dimension in dimensions}
    blocked = sorted(
        {
            item
            for equation in equations
            if equation.blocking
            for item in (equation.lhs, equation.rhs)
            if item in dim_ids
        }
        | {dimension.var_id for dimension in dimensions if dimension.semantic_role == "blocked"}
    )
    unresolved = sorted(
        {
            equation.constraint_id
            for equation in equations
            if equation.relation == "unknown"
            or "missing" in equation.lhs
            or "missing" in equation.rhs
            or (equation.blocking and equation.confidence == "low")
        }
    )
    summary = {
        "num_dimension_variables": len(dimensions),
        "num_index_variables": len(indices),
        "num_constraint_equations": len(equations),
        "num_equivalence_classes": len(classes),
        "num_blocked_dimensions": len(blocked),
        "num_unresolved_constraints": len(unresolved),
        "relation_counts": dict(Counter(equation.relation for equation in equations)),
        "constraint_type_counts": dict(Counter(equation.constraint_type for equation in equations)),
        "semantic_role_counts": dict(Counter(dimension.semantic_role for dimension in dimensions)),
        "confidence_counts": dict(Counter(dimension.confidence for dimension in dimensions)),
        "prunable_dimension_count": sum(1 for dimension in dimensions if dimension.prunable),
    }
    return PruningIR(
        model_name=pruning_map.get("model_name"),
        hf_id=pruning_map.get("hf_id", ""),
        task=pruning_map.get("task", ""),
        dimension_variables=dimensions,
        index_variables=indices,
        constraint_equations=equations,
        equivalence_classes=classes,
        blocked_dimensions=blocked,
        unresolved_constraints=unresolved,
        summary=summary,
        metadata={
            "source": "dimension_variable_ir",
            "note": "Dimension IR and pruning maps are the primary research artifacts; executable pruning remains experimental backend support.",
        },
    )


def _ir_dict(ir: PruningIR | dict[str, Any]) -> dict[str, Any]:
    return pruning_ir_to_dict(ir) if isinstance(ir, PruningIR) else ir


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if not rows:
        return "_None._"
    selected = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if limit and len(rows) > limit:
        lines.append("| ... | " + f"{len(rows) - limit} more rows omitted" + " |" * (len(columns) - 1))
    return "\n".join(lines)


def pruning_ir_to_markdown(ir: PruningIR | dict) -> str:
    data = _ir_dict(ir)
    summary = data.get("summary", {})
    return "\n".join(
        [
            f"# Pruning Dimension IR: {data.get('model_name')}",
            "",
            "## Summary",
            "",
            f"- Dimension variables: `{summary.get('num_dimension_variables', 0)}`",
            f"- Index variables: `{summary.get('num_index_variables', 0)}`",
            f"- Constraint equations: `{summary.get('num_constraint_equations', 0)}`",
            f"- Equivalence classes: `{summary.get('num_equivalence_classes', 0)}`",
            f"- Blocked dimensions: `{summary.get('num_blocked_dimensions', 0)}`",
            f"- Unresolved constraints: `{summary.get('num_unresolved_constraints', 0)}`",
            f"- Prunable dimensions: `{summary.get('prunable_dimension_count', 0)}`",
            "",
            "## Dimension Variables",
            "",
            _markdown_table(
                data.get("dimension_variables", []),
                ["var_id", "owner_name", "owner_type", "dim_name", "size", "semantic_role", "prunable", "confidence"],
                limit=300,
            ),
            "",
            "## Index Variables",
            "",
            _markdown_table(
                data.get("index_variables", []),
                ["index_var_id", "dimension_var_id", "allowed_range", "symbolic"],
                limit=300,
            ),
            "",
            "## Constraint Equations",
            "",
            _markdown_table(
                data.get("constraint_equations", []),
                ["constraint_id", "lhs", "relation", "rhs", "constraint_type", "direction", "blocking", "confidence", "expression"],
                limit=300,
            ),
            "",
            "## Equivalence Classes",
            "",
            _markdown_table(
                data.get("equivalence_classes", []),
                ["class_id", "class_type", "representative", "members", "size", "confidence"],
                limit=300,
            ),
            "",
            "## Blocked Dimensions",
            "",
            "\n".join(f"- `{item}`" for item in data.get("blocked_dimensions", [])) or "_None._",
            "",
            "## Unresolved Constraints",
            "",
            "\n".join(f"- `{item}`" for item in data.get("unresolved_constraints", [])) or "_None._",
            "",
            "## Interpretation",
            "",
            "Dimension variables are compiler-style handles for prunable model dimensions. Index variables represent symbolic pruning selections. Constraint equations encode propagation rules, and equivalence classes group dimensions that must remain equal or share pruning indices.",
            "",
            "Blocking constraints and unresolved constraints identify regions where analysis can continue but executable pruning would require stronger mapping, shape evidence, or repair logic. Dimension IR and pruning maps are the primary research artifacts; executable pruning remains experimental backend support.",
            "",
        ]
    )
