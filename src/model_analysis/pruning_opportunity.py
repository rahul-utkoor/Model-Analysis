"""Compiler-style pruning opportunity map construction."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class PruningDimension:
    dim_id: str
    unit_id: str
    unit_name: str
    unit_type: str
    dim_name: str
    size: int | None
    structural_role: str
    confidence: str
    reason: str


@dataclass
class PropagationConstraint:
    constraint_id: str
    src_dim_id: str
    dst_dim_id: str
    constraint_type: str
    direction: str
    edge_type: str
    confidence: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class PruningOpportunity:
    opportunity_id: str
    opportunity_type: str
    root_unit_id: str
    root_unit_name: str
    prunable_dimensions: list[str]
    required_constraints: list[str]
    affected_units: list[str]
    forward_propagation: list[str]
    backward_propagation: list[str]
    blocked_by: list[dict[str, Any]]
    risk_level: str
    executability: str
    confidence: str
    reason: str


@dataclass
class ModelPruningMap:
    model_name: str
    hf_id: str
    task: str
    pruning_dimensions: list[PruningDimension] = field(default_factory=list)
    propagation_constraints: list[PropagationConstraint] = field(default_factory=list)
    opportunities: list[PruningOpportunity] = field(default_factory=list)
    independent_opportunities: list[str] = field(default_factory=list)
    coupled_opportunities: list[str] = field(default_factory=list)
    blocked_opportunities: list[str] = field(default_factory=list)
    structural_risks: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def pruning_dimension_to_dict(dimension: PruningDimension) -> dict[str, Any]:
    return asdict(dimension)


def propagation_constraint_to_dict(constraint: PropagationConstraint) -> dict[str, Any]:
    return asdict(constraint)


def pruning_opportunity_to_dict(opportunity: PruningOpportunity) -> dict[str, Any]:
    return asdict(opportunity)


def model_pruning_map_to_dict(model_map: ModelPruningMap) -> dict[str, Any]:
    return asdict(model_map)


def write_model_pruning_map_json(model_map: ModelPruningMap, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(model_pruning_map_to_dict(model_map), indent=2), encoding="utf-8")


def load_model_pruning_map_json(path: Path) -> ModelPruningMap:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelPruningMap(
        model_name=data["model_name"],
        hf_id=data.get("hf_id", ""),
        task=data.get("task", ""),
        pruning_dimensions=[PruningDimension(**item) for item in data.get("pruning_dimensions", [])],
        propagation_constraints=[PropagationConstraint(**item) for item in data.get("propagation_constraints", [])],
        opportunities=[PruningOpportunity(**item) for item in data.get("opportunities", [])],
        independent_opportunities=data.get("independent_opportunities", []),
        coupled_opportunities=data.get("coupled_opportunities", []),
        blocked_opportunities=data.get("blocked_opportunities", []),
        structural_risks=data.get("structural_risks", []),
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def _graph_dict(dependency_graph: dict[str, Any]) -> dict[str, Any]:
    return dependency_graph


def _units_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {unit.get("unit_id"): unit for unit in graph.get("prunable_units", [])}


def _edges_for_unit(graph: dict[str, Any], unit_id: str) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph.get("dependency_edges", [])
        if edge.get("src") == unit_id or edge.get("dst") == unit_id
    ]


def _dim_id(unit_id: str, dim_name: str) -> str:
    return f"dim::{unit_id}::{dim_name}"


def _safe_id(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace(" ", "_")


def _shape_size(unit: dict[str, Any], dim_name: str) -> int | None:
    shape = unit.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    if dim_name in {"out_features", "channel_out"} and len(shape) >= 1:
        return shape[0]
    if dim_name in {"in_features", "channel_in"} and len(shape) >= 2:
        return shape[1]
    if dim_name in {"embedding_dim"} and len(shape) >= 2:
        return shape[1]
    if dim_name in {"vocab_dim"} and len(shape) >= 1:
        return shape[0]
    if dim_name in {"intermediate_dim"}:
        if unit.get("unit_type") == "mlp_expansion" and len(shape) >= 1:
            return shape[0]
        if unit.get("unit_type") == "mlp_projection" and len(shape) >= 2:
            return shape[1]
    if dim_name in {"hidden_dim"}:
        if len(shape) >= 1 and unit.get("unit_type") in {"attention_qkv", "attention_output"}:
            return shape[0]
        if len(shape) >= 2:
            return shape[1]
    return None


def _validated_unit_ids(validation_report: dict[str, Any] | None) -> set[str]:
    if not validation_report:
        return set()
    return {item.get("unit_id") for item in validation_report.get("validated_units", [])}


def _edge_evidence(validation_report: dict[str, Any] | None, edge: dict[str, Any]) -> list[dict[str, Any]]:
    if not validation_report:
        return []
    evidence = []
    for key, source in (
        ("shape_supported_edges", "shape"),
        ("correspondence_supported_edges", "correspondence"),
        ("validated_edges", "validation"),
    ):
        for item in validation_report.get(key, []):
            if item.get("src") == edge.get("src") and item.get("dst") == edge.get("dst") and item.get("edge_type") == edge.get("edge_type"):
                evidence.append({"source": source, "confidence": item.get("confidence", "medium"), "reason": item.get("reason", "")})
    return evidence


def _role_for_dimension(unit: dict[str, Any], dim_name: str, coupled: bool) -> str:
    unit_type = unit.get("unit_type")
    if unit_type in {"embedding"}:
        return "boundary"
    if coupled:
        return "coupled"
    if dim_name in {"out_features", "channel_out", "intermediate_dim", "embedding_dim", "num_heads", "head_dim"}:
        return "producer"
    if dim_name in {"in_features", "channel_in"}:
        return "consumer"
    if unit.get("source") == "onnx":
        return "propagation_only"
    return "producer"


def _confidence_with_validation(unit: dict[str, Any], validation_report: dict[str, Any] | None) -> str:
    base = unit.get("confidence", "low")
    if unit.get("unit_id") in _validated_unit_ids(validation_report) and base == "low":
        return "medium"
    return base


def extract_pruning_dimensions(
    dependency_graph: dict,
    validation_report: dict | None = None,
) -> list[PruningDimension]:
    """Extract compiler-style dimension variables from prunable units."""
    graph = _graph_dict(dependency_graph)
    coupled_units = {
        edge.get("src")
        for edge in graph.get("dependency_edges", [])
        if edge.get("edge_type") not in {"feeds"}
    } | {
        edge.get("dst")
        for edge in graph.get("dependency_edges", [])
        if edge.get("edge_type") not in {"feeds"}
    }
    dimensions = []
    for unit in graph.get("prunable_units", []):
        unit_id = unit.get("unit_id")
        for dim_name in unit.get("prunable_dims", []):
            dimensions.append(
                PruningDimension(
                    dim_id=_dim_id(unit_id, dim_name),
                    unit_id=unit_id,
                    unit_name=unit.get("name") or unit.get("module_or_node_name") or unit_id,
                    unit_type=unit.get("unit_type", ""),
                    dim_name=dim_name,
                    size=_shape_size(unit, dim_name),
                    structural_role=_role_for_dimension(unit, dim_name, unit_id in coupled_units),
                    confidence=_confidence_with_validation(unit, validation_report),
                    reason=f"Dimension '{dim_name}' is declared prunable by dependency graph unit '{unit_id}'.",
                )
            )
    return dimensions


def _dimensions_by_unit(dimensions: list[PruningDimension]) -> dict[str, list[PruningDimension]]:
    result: dict[str, list[PruningDimension]] = defaultdict(list)
    for dimension in dimensions:
        result[dimension.unit_id].append(dimension)
    return result


def _select_dim_for_edge(unit_dims: list[PruningDimension], affected_dims: list[str], edge_type: str, side: str) -> PruningDimension | None:
    by_name = {dimension.dim_name: dimension for dimension in unit_dims}
    for dim_name in affected_dims:
        if dim_name in by_name:
            return by_name[dim_name]
    if edge_type == "mlp_hidden_coupling":
        return by_name.get("intermediate_dim") or by_name.get("out_features" if side == "src" else "in_features") or (unit_dims[0] if unit_dims else None)
    if edge_type == "qkv_coupling":
        return by_name.get("num_heads") or by_name.get("head_dim") or by_name.get("hidden_dim") or by_name.get("out_features") or (unit_dims[0] if unit_dims else None)
    if edge_type == "head_dimension_coupling":
        return by_name.get("hidden_dim") or by_name.get("in_features") or by_name.get("out_features") or (unit_dims[0] if unit_dims else None)
    if edge_type in {"residual_coupling", "normalization_dependency"}:
        return by_name.get("hidden_dim") or by_name.get("out_features") or by_name.get("in_features") or (unit_dims[0] if unit_dims else None)
    return unit_dims[0] if unit_dims else None


def _constraint_type_for_edge(edge_type: str, evidence: list[dict[str, Any]]) -> str:
    if edge_type == "qkv_coupling":
        return "qkv_same_heads"
    if edge_type == "head_dimension_coupling":
        return "same_indices" if evidence else "unknown_mapping"
    if edge_type == "mlp_hidden_coupling":
        return "mlp_same_intermediate_indices"
    if edge_type == "residual_coupling":
        return "residual_equal_shape"
    if edge_type == "normalization_dependency":
        return "equality"
    if edge_type == "embedding_tying":
        return "tied_parameter"
    if edge_type in {"propagation_only", "shape_dependency"}:
        return "reshape_preserving" if evidence else "unknown_mapping"
    return "unknown_mapping"


def extract_propagation_constraints(
    dependency_graph: dict,
    pruning_dimensions: list[PruningDimension],
    validation_report: dict | None = None,
) -> list[PropagationConstraint]:
    """Translate dependency edges into propagation constraints between dimensions."""
    graph = _graph_dict(dependency_graph)
    dims_by_unit = _dimensions_by_unit(pruning_dimensions)
    constraints = []
    for index, edge in enumerate(graph.get("dependency_edges", []), start=1):
        src_dim = _select_dim_for_edge(dims_by_unit.get(edge.get("src"), []), edge.get("affected_dims", []), edge.get("edge_type", ""), "src")
        dst_dim = _select_dim_for_edge(dims_by_unit.get(edge.get("dst"), []), edge.get("affected_dims", []), edge.get("edge_type", ""), "dst")
        if not src_dim or not dst_dim:
            continue
        evidence = _edge_evidence(validation_report, edge)
        confidence = "medium" if evidence and edge.get("confidence") == "low" else edge.get("confidence", "low")
        constraints.append(
            PropagationConstraint(
                constraint_id=f"constraint_{index:05d}",
                src_dim_id=src_dim.dim_id,
                dst_dim_id=dst_dim.dim_id,
                constraint_type=_constraint_type_for_edge(edge.get("edge_type", ""), evidence),
                direction=edge.get("direction", "forward"),
                edge_type=edge.get("edge_type", ""),
                confidence=confidence,
                evidence=evidence,
                reason=edge.get("reason", "Dependency graph edge induces a pruning propagation constraint."),
            )
        )
    return constraints


def _constraints_by_unit(constraints: list[PropagationConstraint]) -> dict[str, list[PropagationConstraint]]:
    result: dict[str, list[PropagationConstraint]] = defaultdict(list)
    for constraint in constraints:
        for dim_id in (constraint.src_dim_id, constraint.dst_dim_id):
            parts = dim_id.split("::")
            if len(parts) >= 3:
                result[parts[1]].append(constraint)
    return result


def _unit_type_opportunity(unit: dict[str, Any], constraints: list[PropagationConstraint]) -> tuple[str, str, str, str, list[dict[str, Any]]]:
    unit_type = unit.get("unit_type", "")
    unit_id = unit.get("unit_id", "")
    edge_types = {constraint.edge_type for constraint in constraints}
    blocked_by: list[dict[str, Any]] = []
    if "residual_coupling" in edge_types:
        return (
            "blocked_residual_hidden",
            "blocked",
            "blocked",
            "medium",
            [{"type": "residual_shape_coupling", "reason": "Hidden-size pruning across residual paths is blocked without explicit branch-shape proof."}],
        )
    if unit_type == "mlp_expansion" and "mlp_hidden_coupling" in edge_types:
        return ("mlp_intermediate", "medium", "pair_executable", "medium", blocked_by)
    if unit_type == "mlp_projection" and "mlp_hidden_coupling" in edge_types:
        return ("mlp_intermediate", "medium", "pair_executable", "medium", blocked_by)
    if unit_type == "attention_qkv" or "qkv_coupling" in edge_types:
        blocked_by.append({"type": "head_index_mapping", "reason": "Attention head index mapping is not executable yet."})
        return ("attention_qkv_heads", "high", "analysis_only", "medium", blocked_by)
    if unit_type == "attention_output" or "head_dimension_coupling" in edge_types:
        return ("attention_projection", "high", "requires_repair", "low", blocked_by)
    if unit_type == "embedding":
        blocked_by.append({"type": "embedding_output_tying", "reason": "Embedding/output tying may require shared parameter pruning."})
        return ("embedding_dimension", "high", "analysis_only", "low", blocked_by)
    if unit_type == "conv":
        return ("conv_output_channel", "medium", "analysis_only", unit.get("confidence", "low"), blocked_by)
    if unit_type in {"matmul", "gemm"} or unit_id.startswith("onnx:"):
        blocked_by.append({"type": "onnx_mapping", "reason": "ONNX-only pruning candidate is not mapped to a PyTorch transform."})
        return ("onnx_matmul_dimension", "high", "analysis_only", "low", blocked_by)
    if unit_type == "linear" and not edge_types.difference({"feeds", "propagation_only"}):
        return ("local_linear_output", "medium", "locally_executable", unit.get("confidence", "medium"), blocked_by)
    if unit_type == "linear":
        return ("local_linear_output", "medium", "requires_repair", unit.get("confidence", "medium"), blocked_by)
    return ("blocked_unknown_shape", "blocked", "blocked", "low", [{"type": "unknown_shape", "reason": "No known opportunity heuristic for this unit type."}])


def infer_pruning_opportunities(
    dependency_graph: dict,
    pruning_dimensions: list[PruningDimension],
    propagation_constraints: list[PropagationConstraint],
    validation_report: dict | None = None,
) -> list[PruningOpportunity]:
    """Infer model-level pruning opportunities from dimensions and constraints."""
    graph = _graph_dict(dependency_graph)
    dims_by_unit = _dimensions_by_unit(pruning_dimensions)
    constraints_by_unit = _constraints_by_unit(propagation_constraints)
    opportunities = []
    seen_mlp_pairs: set[frozenset[str]] = set()

    for unit in graph.get("prunable_units", []):
        unit_id = unit.get("unit_id")
        unit_dims = dims_by_unit.get(unit_id, [])
        if not unit_dims:
            continue
        unit_constraints = constraints_by_unit.get(unit_id, [])
        opportunity_type, risk_level, executability, confidence, blocked_by = _unit_type_opportunity(unit, unit_constraints)

        if opportunity_type == "mlp_intermediate":
            linked_units = {unit_id}
            for constraint in unit_constraints:
                if constraint.edge_type == "mlp_hidden_coupling":
                    for dim_id in (constraint.src_dim_id, constraint.dst_dim_id):
                        linked_units.add(dim_id.split("::")[1])
            key = frozenset(linked_units)
            if key in seen_mlp_pairs:
                continue
            seen_mlp_pairs.add(key)
            affected_units = sorted(linked_units)
            root_unit_id = sorted(linked_units)[0]
            root_name = (_units_by_id(graph).get(root_unit_id) or {}).get("name", root_unit_id)
        else:
            affected_units = sorted({unit_id, *[item.split("::")[1] for constraint in unit_constraints for item in (constraint.src_dim_id, constraint.dst_dim_id)]})
            root_unit_id = unit_id
            root_name = unit.get("name") or unit.get("module_or_node_name") or unit_id

        required_constraints = [constraint.constraint_id for constraint in unit_constraints if constraint.edge_type not in {"feeds"}]
        forward = [constraint.constraint_id for constraint in unit_constraints if constraint.direction in {"forward", "bidirectional"}]
        backward = [constraint.constraint_id for constraint in unit_constraints if constraint.direction in {"backward", "bidirectional"}]
        opportunities.append(
            PruningOpportunity(
                opportunity_id=f"opp::{_safe_id(opportunity_type)}::{_safe_id(root_unit_id)}",
                opportunity_type=opportunity_type,
                root_unit_id=root_unit_id,
                root_unit_name=root_name,
                prunable_dimensions=[dimension.dim_id for dimension in unit_dims],
                required_constraints=required_constraints,
                affected_units=affected_units,
                forward_propagation=forward,
                backward_propagation=backward,
                blocked_by=blocked_by,
                risk_level=risk_level,
                executability=executability,
                confidence=confidence,
                reason=_opportunity_reason(opportunity_type, unit, required_constraints),
            )
        )
    return opportunities


def _opportunity_reason(opportunity_type: str, unit: dict[str, Any], required_constraints: list[str]) -> str:
    if opportunity_type == "mlp_intermediate":
        return "MLP expansion/projection pair exposes an intermediate dimension with explicit same-index propagation constraints."
    if opportunity_type == "attention_qkv_heads":
        return "Attention Q/K/V structure is prunable in principle but requires proven head reshape and index mapping before execution."
    if opportunity_type == "blocked_residual_hidden":
        return "Hidden-size pruning crosses residual/LayerNorm-style constraints and is blocked for executable pruning."
    if opportunity_type == "local_linear_output":
        if required_constraints:
            return "Linear output dimension is prunable only with downstream repair or propagation constraints."
        return "Linear output dimension appears locally prunable from static structure."
    if opportunity_type == "embedding_dimension":
        return "Embedding dimensions are structurally prunable only with boundary and tied-parameter evidence."
    if opportunity_type == "onnx_matmul_dimension":
        return "ONNX MatMul/Gemm is a high-interest structural node but lacks executable PyTorch mapping."
    return unit.get("reason", "Static pruning opportunity inferred from dependency graph structure.")


def build_structural_risk_map(
    dependency_graph: dict,
    pruning_dimensions: list[PruningDimension],
    propagation_constraints: list[PropagationConstraint],
    opportunities: list[PruningOpportunity],
) -> list[dict]:
    """Build a model-level risk inventory for pruning analysis."""
    risks: list[dict[str, Any]] = []
    edge_to_risk = {
        "residual_coupling": ("residual_shape_coupling", "blocked", "Keep hidden size unchanged or prove branch equality."),
        "normalization_dependency": ("layernorm_hidden_dependency", "high", "Keep hidden size unchanged or propagate LayerNorm affine dimensions explicitly."),
        "head_dimension_coupling": ("attention_head_reshape", "high", "Require explicit head reshape mapping before executable pruning."),
        "qkv_coupling": ("qkv_consistency", "high", "Do not execute attention pruning until Q/K/V head mapping is proven."),
        "mlp_hidden_coupling": ("mlp_pair_consistency", "medium", "Propagate same intermediate indices from expansion to projection."),
        "embedding_tying": ("embedding_output_tying", "high", "Require tied-parameter detection before pruning embedding/output boundaries."),
        "shape_dependency": ("unknown_shape", "high", "Require explicit dimension mapping before executable pruning."),
        "propagation_only": ("unknown_shape", "medium", "Track propagation through shape-only nodes before execution."),
    }
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for edge in dependency_graph.get("dependency_edges", []):
        edge_type = edge.get("edge_type")
        if edge_type not in edge_to_risk:
            continue
        risk_type, severity, mitigation = edge_to_risk[edge_type]
        affected = tuple(sorted([edge.get("src"), edge.get("dst")]))
        key = (risk_type, edge_type, affected)
        if key in seen:
            continue
        seen.add(key)
        risks.append(
            {
                "risk_id": f"risk_{len(risks) + 1:05d}",
                "risk_type": risk_type,
                "affected_units": list(affected),
                "severity": severity,
                "reason": edge.get("reason", f"{edge_type} creates structural pruning risk."),
                "mitigation": mitigation,
            }
        )

    for opportunity in opportunities:
        if opportunity.opportunity_type == "onnx_matmul_dimension":
            risks.append(
                {
                    "risk_id": f"risk_{len(risks) + 1:05d}",
                    "risk_type": "onnx_unmapped_matmul",
                    "affected_units": opportunity.affected_units,
                    "severity": "high",
                    "reason": "ONNX-only MatMul/Gemm opportunity lacks PyTorch correspondence for executable pruning.",
                    "mitigation": "Require PyTorch-to-ONNX correspondence before executable pruning.",
                }
            )
        if opportunity.opportunity_type == "mlp_intermediate":
            risks.append(
                {
                    "risk_id": f"risk_{len(risks) + 1:05d}",
                    "risk_type": "nonuniform_layer_width",
                    "affected_units": opportunity.affected_units,
                    "severity": "medium",
                    "reason": "Per-layer MLP intermediate pruning may create nonuniform layer widths.",
                    "mitigation": "Record architecture metadata and verify reload behavior before relying on checkpoints.",
                }
            )
            risks.append(
                {
                    "risk_id": f"risk_{len(risks) + 1:05d}",
                    "risk_type": "artifact_reload_risk",
                    "affected_units": opportunity.affected_units,
                    "severity": "medium",
                    "reason": "Standard model configs may assume uniform feed-forward widths.",
                    "mitigation": "Treat executable pruning modules as experimental backends and keep pruning maps as the primary artifact.",
                }
            )
    return risks


def build_model_pruning_map(
    dependency_graph: dict,
    validation_report: dict | None = None,
) -> ModelPruningMap:
    """Build a compiler-style model pruning map from dependency evidence."""
    dimensions = extract_pruning_dimensions(dependency_graph, validation_report)
    constraints = extract_propagation_constraints(dependency_graph, dimensions, validation_report)
    opportunities = infer_pruning_opportunities(dependency_graph, dimensions, constraints, validation_report)
    risks = build_structural_risk_map(dependency_graph, dimensions, constraints, opportunities)
    independent = [
        opportunity.opportunity_id
        for opportunity in opportunities
        if not opportunity.required_constraints and opportunity.risk_level not in {"blocked", "high"}
    ]
    blocked = [
        opportunity.opportunity_id
        for opportunity in opportunities
        if opportunity.risk_level == "blocked" or opportunity.executability == "blocked"
    ]
    coupled = [
        opportunity.opportunity_id
        for opportunity in opportunities
        if opportunity.required_constraints and opportunity.opportunity_id not in blocked
    ]
    summary = {
        "num_pruning_dimensions": len(dimensions),
        "num_constraints": len(constraints),
        "num_opportunities": len(opportunities),
        "num_independent_opportunities": len(independent),
        "num_coupled_opportunities": len(coupled),
        "num_blocked_opportunities": len(blocked),
        "opportunity_type_counts": dict(Counter(item.opportunity_type for item in opportunities)),
        "risk_level_counts": dict(Counter(item.risk_level for item in opportunities)),
        "executability_counts": dict(Counter(item.executability for item in opportunities)),
        "confidence_counts": dict(Counter(item.confidence for item in opportunities)),
    }
    return ModelPruningMap(
        model_name=dependency_graph.get("model_name"),
        hf_id=dependency_graph.get("hf_id", ""),
        task=dependency_graph.get("task", ""),
        pruning_dimensions=dimensions,
        propagation_constraints=constraints,
        opportunities=opportunities,
        independent_opportunities=independent,
        coupled_opportunities=coupled,
        blocked_opportunities=blocked,
        structural_risks=risks,
        summary=summary,
        metadata={
            "source": "compiler_style_pruning_opportunity_analysis",
            "validation_used": validation_report is not None,
            "note": "Executable pruning modules are experimental backends; pruning maps are the primary research artifact.",
        },
    )


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


def _counts_lines(counts: dict[str, int]) -> list[str]:
    return [f"- {key}: `{value}`" for key, value in sorted(counts.items())] or ["- None"]


def model_pruning_map_to_markdown(model_map: ModelPruningMap) -> str:
    data = model_pruning_map_to_dict(model_map)
    summary = data.get("summary", {})
    lines = [
        f"# Model Pruning Map: {model_map.model_name}",
        "",
        "## Summary",
        "",
        f"- Pruning dimensions: `{summary.get('num_pruning_dimensions', 0)}`",
        f"- Propagation constraints: `{summary.get('num_constraints', 0)}`",
        f"- Opportunities: `{summary.get('num_opportunities', 0)}`",
        f"- Independent opportunities: `{summary.get('num_independent_opportunities', 0)}`",
        f"- Coupled opportunities: `{summary.get('num_coupled_opportunities', 0)}`",
        f"- Blocked opportunities: `{summary.get('num_blocked_opportunities', 0)}`",
        "",
        "### Opportunity Types",
        "",
        *_counts_lines(summary.get("opportunity_type_counts", {})),
        "",
        "## Pruning Dimensions",
        "",
        _markdown_table(
            data["pruning_dimensions"],
            ["dim_id", "unit_name", "unit_type", "dim_name", "size", "structural_role", "confidence", "reason"],
            limit=250,
        ),
        "",
        "## Propagation Constraints",
        "",
        _markdown_table(
            data["propagation_constraints"],
            ["constraint_id", "src_dim_id", "dst_dim_id", "constraint_type", "direction", "edge_type", "confidence", "reason"],
            limit=250,
        ),
        "",
        "## Pruning Opportunities",
        "",
        _markdown_table(
            data["opportunities"],
            ["opportunity_id", "opportunity_type", "root_unit_name", "prunable_dimensions", "risk_level", "executability", "confidence", "reason"],
            limit=250,
        ),
        "",
        "## Independent Opportunities",
        "",
        "\n".join(f"- `{item}`" for item in model_map.independent_opportunities) or "_None._",
        "",
        "## Coupled Opportunities",
        "",
        "\n".join(f"- `{item}`" for item in model_map.coupled_opportunities) or "_None._",
        "",
        "## Blocked Opportunities",
        "",
        "\n".join(f"- `{item}`" for item in model_map.blocked_opportunities) or "_None._",
        "",
        "## Structural Risk Map",
        "",
        _markdown_table(
            data["structural_risks"],
            ["risk_id", "risk_type", "affected_units", "severity", "reason", "mitigation"],
            limit=250,
        ),
        "",
        "## Interpretation",
        "",
        "This pruning map is a compiler-style static analysis artifact. It identifies candidate pruning dimensions, propagation constraints, coupled regions, blocked regions, and structural risks before any weight transformation.",
        "",
        "Structurally promising classes include locally prunable Linear outputs and MLP intermediate dimensions when same-index expansion/projection constraints are explicit. Attention and embedding opportunities remain analysis-only unless stronger dimension mapping evidence is available.",
        "",
        "Executable pruning modules from earlier milestones are experimental validation backends. The primary research artifact is the model pruning map and its dimension/constraint IR.",
        "",
    ]
    return "\n".join(lines)
