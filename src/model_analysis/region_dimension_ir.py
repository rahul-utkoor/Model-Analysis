"""Region-scoped symbolic Dimension IR derived from Structural Region Trees."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class RegionDimensionVariable:
    var_id: str
    region_id: str
    region_type: str
    region_name: str
    dim_name: str
    axis_role: str
    size: int | str | None
    source: str
    prunable: bool
    protected: bool
    propagated: bool
    blocked: bool
    confidence: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class RegionConstraintEquation:
    constraint_id: str
    region_id: str
    region_type: str
    lhs: str
    rhs: str
    relation: str
    constraint_type: str
    blocking: bool
    confidence: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class RegionDimensionEquivalenceClass:
    class_id: str
    members: list[str]
    representative: str
    class_type: str
    size: int | str | None
    confidence: str
    constraints: list[str]
    reason: str


@dataclass
class RegionDimensionIR:
    model_name: str
    source_frontend: str
    root_region_id: str
    dimension_variables: list[RegionDimensionVariable] = field(default_factory=list)
    constraint_equations: list[RegionConstraintEquation] = field(default_factory=list)
    equivalence_classes: list[RegionDimensionEquivalenceClass] = field(default_factory=list)
    blocked_dimensions: list[str] = field(default_factory=list)
    unresolved_constraints: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def region_dimension_variable_to_dict(value: RegionDimensionVariable) -> dict[str, Any]:
    return asdict(value)


def region_constraint_equation_to_dict(value: RegionConstraintEquation) -> dict[str, Any]:
    return asdict(value)


def region_dimension_equivalence_class_to_dict(value: RegionDimensionEquivalenceClass) -> dict[str, Any]:
    return asdict(value)


def region_dimension_ir_to_dict(ir: RegionDimensionIR) -> dict[str, Any]:
    return asdict(ir)


def write_region_dimension_ir_json(ir: RegionDimensionIR, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(region_dimension_ir_to_dict(ir), indent=2), encoding="utf-8")


def load_region_dimension_ir_json(path: Path) -> RegionDimensionIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RegionDimensionIR(
        model_name=data["model_name"],
        source_frontend=data.get("source_frontend", "unknown"),
        root_region_id=data.get("root_region_id", ""),
        dimension_variables=[RegionDimensionVariable(**item) for item in data.get("dimension_variables", [])],
        constraint_equations=[RegionConstraintEquation(**item) for item in data.get("constraint_equations", [])],
        equivalence_classes=[RegionDimensionEquivalenceClass(**item) for item in data.get("equivalence_classes", [])],
        blocked_dimensions=data.get("blocked_dimensions", []),
        unresolved_constraints=data.get("unresolved_constraints", []),
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def _normalize(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def _var_id(region_id: str, dim_name: str, endpoint: str = "") -> str:
    suffix = f"::{_normalize(endpoint)}" if endpoint else ""
    return f"rdim::{_normalize(region_id)}::{_normalize(dim_name)}{suffix}"


def _regions_and_interfaces(tree: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    regions = {item["region_id"]: item for item in tree.get("regions", [])}
    interfaces = {item["region_id"]: item for item in tree.get("interfaces", [])}
    return regions, interfaces


def _has_constraint(interface: dict, constraint_type: str) -> bool:
    return any(item.get("type") == constraint_type for item in interface.get("constraints", []))


def _evidence(region: dict, interface: dict, role: str) -> list[dict[str, Any]]:
    return [
        {
            "source": "region_interface",
            "region_id": region.get("region_id"),
            "region_type": region.get("region_type"),
            "interface_role": role,
            "constraints": interface.get("constraints", []),
        }
    ]


def _add_dimension(
    dimensions: list[RegionDimensionVariable],
    region: dict,
    interface: dict,
    dim_name: str,
    axis_role: str,
    endpoint: str,
    *,
    prunable: bool = False,
    protected: bool = False,
    propagated: bool = False,
    blocked: bool = False,
    reason: str,
) -> None:
    dimensions.append(
        RegionDimensionVariable(
            var_id=_var_id(region["region_id"], dim_name, endpoint),
            region_id=region["region_id"],
            region_type=region["region_type"],
            region_name=region.get("name", region["region_id"]),
            dim_name=dim_name,
            axis_role=axis_role,
            size=None,
            source="region_interface",
            prunable=prunable,
            protected=protected,
            propagated=propagated,
            blocked=blocked,
            confidence=region.get("confidence", "low"),
            evidence=_evidence(region, interface, endpoint or dim_name),
            reason=reason,
        )
    )


def build_region_dimension_variables(structural_region_tree: dict) -> list[RegionDimensionVariable]:
    """Materialize symbolic dimensions exposed by structural region interfaces."""
    regions, interfaces = _regions_and_interfaces(structural_region_tree)
    dimensions: list[RegionDimensionVariable] = []
    for region_id in sorted(interfaces):
        interface = interfaces[region_id]
        region = regions.get(region_id, {"region_id": region_id, "region_type": interface.get("region_type", "UnknownRegion"), "confidence": "low"})
        region_type = region.get("region_type")
        if region_type == "LinearProjectionRegion":
            _add_dimension(dimensions, region, interface, "out_features", "hidden", "output", prunable=True, reason="Projection output exposes a locally prunable feature-like axis.")
            _add_dimension(dimensions, region, interface, "in_features", "hidden", "input", propagated=True, reason="Projection input receives propagated feature selections.")
            if _has_constraint(interface, "bias_follows_output"):
                _add_dimension(dimensions, region, interface, "bias_dim", "hidden", "bias", propagated=True, reason="Projection bias follows the output feature index selection.")
        elif region_type == "BiasAddRegion":
            _add_dimension(dimensions, region, interface, "out_features", "hidden", "producer", propagated=True, reason="Bias addition consumes producer output channels.")
            _add_dimension(dimensions, region, interface, "bias_dim", "hidden", "bias", propagated=True, reason="Bias channels must follow the producer output channels.")
        elif region_type == "FeedForwardRegion":
            _add_dimension(dimensions, region, interface, "intermediate_dim", "intermediate", "producer", prunable=True, reason="Feed-forward expansion output is a candidate intermediate pruning axis.")
            _add_dimension(dimensions, region, interface, "intermediate_dim", "intermediate", "consumer", propagated=True, reason="Feed-forward projection input must consume the selected intermediate indices.")
            _add_dimension(dimensions, region, interface, "hidden_dim", "hidden", "boundary", protected=True, reason="Feed-forward boundary hidden width remains protected.")
        elif region_type == "ResidualMergeRegion":
            _add_dimension(dimensions, region, interface, "hidden_dim", "hidden", "branch_a", protected=True, blocked=True, reason="Residual branch dimensions must agree at the merge.")
            _add_dimension(dimensions, region, interface, "hidden_dim", "hidden", "branch_b", protected=True, blocked=True, reason="Residual branch dimensions must agree at the merge.")
        elif region_type == "LayerNormRegion":
            _add_dimension(dimensions, region, interface, "hidden_dim", "hidden", "input", protected=True, propagated=True, reason="Incoming hidden width must match normalization width.")
            _add_dimension(dimensions, region, interface, "hidden_dim", "hidden", "normalization", protected=True, propagated=True, reason="LayerNorm affine/normalized width follows hidden width.")
        elif region_type == "AxisTransformRegion":
            _add_dimension(dimensions, region, interface, "symbolic_axis", "shape", "input", propagated=True, reason="Axis transform input needs symbolic mapping.")
            _add_dimension(dimensions, region, interface, "symbolic_axis", "shape", "output", propagated=True, reason="Axis transform output needs symbolic mapping.")
        elif region_type == "ActivationRegion":
            _add_dimension(dimensions, region, interface, "elementwise_dim", "unknown", "input", propagated=True, reason="Elementwise activation receives an unchanged shape.")
            _add_dimension(dimensions, region, interface, "elementwise_dim", "unknown", "output", propagated=True, reason="Elementwise activation preserves its input dimensions.")
        elif region_type == "AttentionSkeletonRegion":
            for name, role in (("num_heads", "head"), ("head_dim", "head_dim"), ("hidden_dim", "hidden"), ("sequence_dim", "sequence")):
                _add_dimension(dimensions, region, interface, name, role, "", protected=True, propagated=True, reason="Attention pruning is analysis-only until head/axis mapping is proven.")
        elif region_type == "ForkRegion":
            _add_dimension(dimensions, region, interface, "fanout_dim", "unknown", "producer", propagated=True, reason="Producer selections propagate along every fanout branch.")
            _add_dimension(dimensions, region, interface, "fanout_dim", "unknown", "consumer", propagated=True, reason="Fanout consumers receive producer dimension selections.")
        elif region_type == "JoinRegion":
            _add_dimension(dimensions, region, interface, "join_dim", "unknown", "branch_a", protected=True, reason="Join branches require compatible dimensions.")
            _add_dimension(dimensions, region, interface, "join_dim", "unknown", "branch_b", protected=True, reason="Join branches require compatible dimensions.")
        elif region_type in {"ProperAcyclicRegion", "UnknownRegion"} and interface.get("constraints"):
            _add_dimension(dimensions, region, interface, "symbolic_axis", "unknown", "", propagated=True, reason="Region has constraints but no precise dimension role.")
    return sorted(dimensions, key=lambda item: item.var_id)


def _variables_by_region(dimensions: list[RegionDimensionVariable]) -> dict[str, dict[tuple[str, str], RegionDimensionVariable]]:
    mapped: dict[str, dict[tuple[str, str], RegionDimensionVariable]] = defaultdict(dict)
    for dimension in dimensions:
        endpoint = dimension.var_id.rsplit("::", 1)[-1]
        mapped[dimension.region_id][(dimension.dim_name, endpoint)] = dimension
    return mapped


def _equation(
    equations: list[RegionConstraintEquation],
    region: dict,
    interface: dict,
    lhs: RegionDimensionVariable,
    rhs: RegionDimensionVariable,
    relation: str,
    constraint_type: str,
    blocking: bool,
    reason: str,
) -> None:
    equations.append(
        RegionConstraintEquation(
            constraint_id=f"region_constraint_{len(equations) + 1:06d}",
            region_id=region["region_id"],
            region_type=region["region_type"],
            lhs=lhs.var_id,
            rhs=rhs.var_id,
            relation=relation,
            constraint_type=constraint_type,
            blocking=blocking,
            confidence=region.get("confidence", "low"),
            evidence=[
                {"source": "structural_region", "region_id": region["region_id"], "region_type": region["region_type"]},
                {"source": "region_interface", "constraints": interface.get("constraints", []), "pruning_role": interface.get("pruning_role")},
            ],
            reason=reason,
        )
    )


def build_region_constraint_equations(
    structural_region_tree: dict,
    dimension_variables: list[RegionDimensionVariable],
) -> list[RegionConstraintEquation]:
    """Lower region-interface constraints into region-scoped symbolic equations."""
    regions, interfaces = _regions_and_interfaces(structural_region_tree)
    by_region = _variables_by_region(dimension_variables)
    equations: list[RegionConstraintEquation] = []
    for region_id in sorted(by_region):
        region = regions[region_id]
        interface = interfaces.get(region_id, {})
        variables = by_region[region_id]
        region_type = region["region_type"]
        if region_type == "LinearProjectionRegion" and ("bias_dim", "bias") in variables:
            _equation(equations, region, interface, variables[("out_features", "output")], variables[("bias_dim", "bias")], "same_indices", "linear_bias_follow", False, "Bias indices follow projection output feature selections.")
        elif region_type == "BiasAddRegion":
            _equation(equations, region, interface, variables[("out_features", "producer")], variables[("bias_dim", "bias")], "same_indices", "linear_bias_follow", False, "Bias indices follow producer output feature selections.")
        elif region_type == "FeedForwardRegion":
            _equation(equations, region, interface, variables[("intermediate_dim", "producer")], variables[("intermediate_dim", "consumer")], "same_indices", "mlp_intermediate_same_indices", False, "Feed-forward expansion output and projection input require identical intermediate pruning indices.")
        elif region_type == "ResidualMergeRegion":
            _equation(equations, region, interface, variables[("hidden_dim", "branch_a")], variables[("hidden_dim", "branch_b")], "join_equal", "residual_hidden_equality", True, "Residual merge requires branch hidden widths to remain equal; local hidden pruning is blocked.")
        elif region_type == "LayerNormRegion":
            _equation(equations, region, interface, variables[("hidden_dim", "input")], variables[("hidden_dim", "normalization")], "eq", "layernorm_hidden_equality", True, "LayerNorm width must equal incoming hidden width.")
        elif region_type == "AxisTransformRegion":
            _equation(equations, region, interface, variables[("symbolic_axis", "input")], variables[("symbolic_axis", "output")], "reshape_map", "axis_transform_mapping", True, "Reshape/transpose propagation is unresolved until input and output axes are mapped.")
        elif region_type == "ActivationRegion":
            _equation(equations, region, interface, variables[("elementwise_dim", "input")], variables[("elementwise_dim", "output")], "preserve", "activation_shape_preserve", False, "Elementwise activation preserves its input dimensions.")
        elif region_type == "AttentionSkeletonRegion":
            _equation(equations, region, interface, variables[("hidden_dim", "hidden_dim")], variables[("num_heads", "num_heads")], "unknown", "attention_head_axis_mapping", True, "Attention hidden/head/head-dimension mapping is not proven by the structural skeleton alone.")
        elif region_type == "ForkRegion":
            _equation(equations, region, interface, variables[("fanout_dim", "producer")], variables[("fanout_dim", "consumer")], "fanout", "fork_fanout_propagation", False, "Producer pruning selections propagate to all fork consumers.")
        elif region_type == "JoinRegion":
            _equation(equations, region, interface, variables[("join_dim", "branch_a")], variables[("join_dim", "branch_b")], "join_equal", "join_branch_compatibility", False, "Joined branches require compatible dimensions; exact semantic role remains conservative.")
        elif region_type in {"ProperAcyclicRegion", "UnknownRegion"} and interfaces.get(region_id, {}).get("constraints"):
            variable = next(iter(variables.values()))
            _equation(equations, region, interface, variable, variable, "unknown", "unknown", True, "Region constraint does not yet expose a precise dimension mapping.")
    return sorted(equations, key=lambda item: item.constraint_id)


class _UnionFind:
    def __init__(self, members: list[str]):
        self.parent = {member: member for member in members}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _class_type(constraints: list[RegionConstraintEquation]) -> str:
    types = {item.constraint_type for item in constraints}
    for constraint_type, class_type in (
        ("mlp_intermediate_same_indices", "mlp_intermediate"),
        ("residual_hidden_equality", "residual_hidden"),
        ("layernorm_hidden_equality", "layernorm_hidden"),
        ("attention_head_axis_mapping", "attention_head"),
        ("fork_fanout_propagation", "fanout"),
        ("linear_bias_follow", "linear_bias"),
    ):
        if constraint_type in types:
            return class_type
    if any(item.relation in {"eq", "same_indices", "join_equal", "preserve"} for item in constraints):
        return "generic_equality"
    return "unknown"


def build_region_dimension_equivalence_classes(
    dimensions: list[RegionDimensionVariable],
    constraints: list[RegionConstraintEquation],
) -> list[RegionDimensionEquivalenceClass]:
    """Group region dimensions joined by explicit equality-like equations."""
    by_id = {item.var_id: item for item in dimensions}
    uf = _UnionFind(list(by_id))
    for constraint in constraints:
        if constraint.relation in {"eq", "same_indices", "join_equal", "preserve"} and constraint.lhs in by_id and constraint.rhs in by_id:
            uf.union(constraint.lhs, constraint.rhs)
    groups: dict[str, list[str]] = defaultdict(list)
    for var_id in sorted(by_id):
        groups[uf.find(var_id)].append(var_id)
    classes = []
    for index, members in enumerate(sorted((sorted(items) for items in groups.values()), key=lambda items: items[0]), start=1):
        relevant = [
            item for item in constraints
            if item.lhs in members or item.rhs in members
        ]
        sizes = {by_id[item].size for item in members if by_id[item].size is not None}
        conflict = len(sizes) > 1
        confidences = {by_id[item].confidence for item in members} | {item.confidence for item in relevant}
        confidence = "low" if conflict or "low" in confidences else ("high" if confidences == {"high"} else "medium")
        classes.append(
            RegionDimensionEquivalenceClass(
                class_id=f"region_eqclass_{index:06d}",
                members=members,
                representative=members[0],
                class_type=_class_type(relevant),
                size=next(iter(sizes)) if len(sizes) == 1 else None,
                confidence=confidence,
                constraints=sorted({item.constraint_id for item in relevant}),
                reason="Members are joined by region-scoped equality or same-index constraints." if len(members) > 1 else "Singleton region dimension with no equality-producing peer.",
            )
        )
    return classes


def build_region_dimension_ir(structural_region_tree: dict) -> RegionDimensionIR:
    """Build the complete region-aware Dimension IR from a Structural Region Tree."""
    dimensions = build_region_dimension_variables(structural_region_tree)
    constraints = build_region_constraint_equations(structural_region_tree, dimensions)
    classes = build_region_dimension_equivalence_classes(dimensions, constraints)
    dimension_ids = {item.var_id for item in dimensions}
    blocked = sorted(
        {item.var_id for item in dimensions if item.blocked}
        | {
            endpoint
            for constraint in constraints
            if constraint.blocking
            for endpoint in (constraint.lhs, constraint.rhs)
            if endpoint in dimension_ids
        }
    )
    unresolved = sorted(
        item.constraint_id for item in constraints
        if item.relation in {"unknown", "reshape_map"}
    )
    summary = {
        "num_dimension_variables": len(dimensions),
        "num_constraint_equations": len(constraints),
        "num_equivalence_classes": len(classes),
        "num_blocked_dimensions": len(blocked),
        "num_unresolved_constraints": len(unresolved),
        "axis_role_counts": dict(Counter(item.axis_role for item in dimensions)),
        "region_type_counts": dict(Counter(item.region_type for item in dimensions)),
        "relation_counts": dict(Counter(item.relation for item in constraints)),
        "constraint_type_counts": dict(Counter(item.constraint_type for item in constraints)),
        "confidence_counts": dict(Counter(item.confidence for item in dimensions)),
        "prunable_dimension_count": sum(1 for item in dimensions if item.prunable),
        "protected_dimension_count": sum(1 for item in dimensions if item.protected),
        "propagated_dimension_count": sum(1 for item in dimensions if item.propagated),
    }
    return RegionDimensionIR(
        model_name=structural_region_tree.get("model_name", ""),
        source_frontend=structural_region_tree.get("source_frontend", "unknown"),
        root_region_id=structural_region_tree.get("root_region_id", ""),
        dimension_variables=dimensions,
        constraint_equations=constraints,
        equivalence_classes=classes,
        blocked_dimensions=blocked,
        unresolved_constraints=unresolved,
        summary=summary,
        metadata={
            "source": "structural_region_tree",
            "structural_region_summary": structural_region_tree.get("summary", {}),
            "note": "RegionDimensionIR is static analysis derived from semantic region interfaces; it does not modify models.",
        },
    )


def _data(ir: RegionDimensionIR | dict) -> dict:
    return region_dimension_ir_to_dict(ir) if isinstance(ir, RegionDimensionIR) else ir


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 300) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    if len(rows) > limit:
        lines.append("| ... | " + f"{len(rows) - limit} more rows omitted" + " |" * (len(columns) - 1))
    return "\n".join(lines)


def region_dimension_ir_to_markdown(ir: RegionDimensionIR | dict) -> str:
    data = _data(ir)
    summary = data.get("summary", {})
    return "\n".join(
        [
            f"# Region-Aware Dimension IR: {data.get('model_name', '')}",
            "",
            "## Summary",
            "",
            f"- Dimension variables: `{summary.get('num_dimension_variables', 0)}`",
            f"- Constraint equations: `{summary.get('num_constraint_equations', 0)}`",
            f"- Equivalence classes: `{summary.get('num_equivalence_classes', 0)}`",
            f"- Prunable dimensions: `{summary.get('prunable_dimension_count', 0)}`",
            f"- Protected dimensions: `{summary.get('protected_dimension_count', 0)}`",
            f"- Propagated dimensions: `{summary.get('propagated_dimension_count', 0)}`",
            f"- Blocked dimensions: `{summary.get('num_blocked_dimensions', 0)}`",
            f"- Unresolved constraints: `{summary.get('num_unresolved_constraints', 0)}`",
            "",
            "## Dimension Variables",
            "",
            _table(data.get("dimension_variables", []), ["var_id", "region_type", "dim_name", "axis_role", "size", "prunable", "protected", "propagated", "blocked", "confidence"]),
            "",
            "## Constraint Equations",
            "",
            _table(data.get("constraint_equations", []), ["constraint_id", "region_type", "lhs", "relation", "rhs", "constraint_type", "blocking", "confidence"]),
            "",
            "## Equivalence Classes",
            "",
            _table(data.get("equivalence_classes", []), ["class_id", "class_type", "representative", "members", "size", "confidence", "constraints"]),
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
            "RegionDimensionIR derives symbolic dimensions and equations from semantic Structural Region Tree interfaces. Projection and feed-forward regions expose candidate pruning axes; residual, normalization, attention, and transform regions preserve blocking or unresolved constraints conservatively.",
            "",
            "This report is static compiler-style analysis only. It does not execute pruning, modify models, or rewrite ONNX.",
            "",
        ]
    )
