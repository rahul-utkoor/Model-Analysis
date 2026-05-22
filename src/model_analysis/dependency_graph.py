"""Conservative pruning dependency graph construction."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from model_analysis.paths import ensure_dir

Confidence = Literal["high", "medium", "low"]
Direction = Literal["forward", "backward", "bidirectional"]
Source = Literal["torch", "onnx"]


@dataclass
class PrunableUnit:
    unit_id: str
    name: str
    source: Source
    unit_type: str
    module_or_node_name: str
    prunable_dims: list[str]
    parameter_count: int | None
    shape: list[int] | None
    confidence: Confidence
    reason: str


@dataclass
class DependencyEdge:
    src: str
    dst: str
    edge_type: str
    affected_dims: list[str]
    direction: Direction
    confidence: Confidence
    reason: str


@dataclass
class DependencyGraph:
    model_name: str
    hf_id: str
    task: str
    prunable_units: list[PrunableUnit] = field(default_factory=list)
    dependency_edges: list[DependencyEdge] = field(default_factory=list)
    coupled_groups: list[dict[str, Any]] = field(default_factory=list)
    independent_units: list[str] = field(default_factory=list)
    ambiguous_units: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DependencyGraph":
        return cls(
            model_name=data["model_name"],
            hf_id=data.get("hf_id", ""),
            task=data.get("task", ""),
            prunable_units=[PrunableUnit(**unit) for unit in data.get("prunable_units", [])],
            dependency_edges=[DependencyEdge(**edge) for edge in data.get("dependency_edges", [])],
            coupled_groups=data.get("coupled_groups", []),
            independent_units=data.get("independent_units", []),
            ambiguous_units=data.get("ambiguous_units", []),
            metadata=data.get("metadata", {}),
        )


ATTENTION_OUTPUT_MARKERS = ("out_proj", "output.dense", "attention.output", ".attention.dense", ".attn.dense", "o_proj")
MLP_EXPANSION_MARKERS = ("fc1", "intermediate", "up_proj", "gate_proj")
MLP_PROJECTION_MARKERS = ("fc2", "output.dense", "down_proj")
PROPAGATION_OPS = {"Add", "Mul", "Reshape", "Transpose", "Gather", "Softmax", "LayerNormalization", "SkipLayerNormalization", "Attention"}


def _unit_id(source: str, unit_type: str, name: str) -> str:
    safe = name.replace("/", "__").replace(" ", "_")
    return f"{source}:{unit_type}:{safe}"


def _linear_unit_id(name: str) -> str:
    return _unit_id("torch", "linear", name)


def _parent_name(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else ""


def _leaf_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _contains_any(value: str, markers: tuple[str, ...] | set[str]) -> bool:
    lowered = value.lower()
    return any(marker.lower() in lowered for marker in markers)


def _is_qkv_name(name: str) -> bool:
    leaf = _leaf_name(name).lower()
    return leaf in {"q_proj", "k_proj", "v_proj", "query", "key", "value"} or any(
        marker in name.lower() for marker in (".q_proj", ".k_proj", ".v_proj", ".query", ".key", ".value")
    )


def _qkv_role(name: str) -> str | None:
    leaf = _leaf_name(name).lower()
    if leaf in {"q_proj", "query"}:
        return "q"
    if leaf in {"k_proj", "key"}:
        return "k"
    if leaf in {"v_proj", "value"}:
        return "v"
    return None


def _is_attention_output(name: str) -> bool:
    return _contains_any(name, ATTENTION_OUTPUT_MARKERS)


def _is_mlp_expansion(name: str) -> bool:
    return _contains_any(name, MLP_EXPANSION_MARKERS)


def _is_mlp_projection(name: str) -> bool:
    return _contains_any(name, MLP_PROJECTION_MARKERS)


def _add_edge_once(graph: DependencyGraph, edge: DependencyEdge) -> None:
    key = (edge.src, edge.dst, edge.edge_type, tuple(edge.affected_dims), edge.direction)
    existing = {
        (item.src, item.dst, item.edge_type, tuple(item.affected_dims), item.direction)
        for item in graph.dependency_edges
    }
    if key not in existing:
        graph.dependency_edges.append(edge)


def _find_unit(graph: DependencyGraph, unit_id: str) -> PrunableUnit | None:
    for unit in graph.prunable_units:
        if unit.unit_id == unit_id:
            return unit
    return None


def _recompute_independent_units(graph: DependencyGraph) -> None:
    coupled_edge_types = {
        "shape_dependency",
        "residual_coupling",
        "qkv_coupling",
        "head_dimension_coupling",
        "mlp_hidden_coupling",
        "embedding_tying",
        "normalization_dependency",
    }
    coupled = {
        edge.src
        for edge in graph.dependency_edges
        if edge.edge_type in coupled_edge_types
    } | {
        edge.dst
        for edge in graph.dependency_edges
        if edge.edge_type in coupled_edge_types
    }
    graph.independent_units = sorted(unit.unit_id for unit in graph.prunable_units if unit.unit_id not in coupled)


def _attention_group_key(members: list[str]) -> str:
    parents = {_parent_name(member) for member in members}
    if len(parents) == 1:
        return next(iter(parents)) or "root"
    return "mixed_attention_parent"


def _make_linear_unit(layer: dict[str, Any]) -> PrunableUnit:
    name = layer["name"]
    return PrunableUnit(
        unit_id=_linear_unit_id(name),
        name=name,
        source="torch",
        unit_type="linear",
        module_or_node_name=name,
        prunable_dims=["out_features"],
        parameter_count=layer.get("parameters"),
        shape=[layer.get("out_features"), layer.get("in_features")],
        confidence="medium",
        reason="Linear layer output features are a common pruning surface, subject to downstream dependency checks.",
    )


def _make_embedding_unit(layer: dict[str, Any]) -> PrunableUnit:
    name = layer["name"]
    return PrunableUnit(
        unit_id=_unit_id("torch", "embedding", name),
        name=name,
        source="torch",
        unit_type="embedding",
        module_or_node_name=name,
        prunable_dims=["embedding_dim"],
        parameter_count=layer.get("parameters"),
        shape=[layer.get("num_embeddings"), layer.get("embedding_dim")],
        confidence="low",
        reason="Embedding pruning may affect hidden dimensions and tied output heads; treat as conservative evidence.",
    )


def _mark_unit_type(graph: DependencyGraph, unit_id: str, unit_type: str, dims: list[str], confidence: Confidence, reason: str) -> None:
    unit = _find_unit(graph, unit_id)
    if not unit:
        return
    unit.unit_type = unit_type
    unit.prunable_dims = dims
    unit.confidence = confidence
    unit.reason = reason


def _add_torch_couplings(graph: DependencyGraph, torch_summary: dict[str, Any]) -> None:
    groups = torch_summary.get("pruning_relevant_groups", [])
    attention_group_units: list[PrunableUnit] = []

    for group in groups:
        group_type = group.get("group_type")
        members = list(group.get("members", []))
        if group_type == "attention_qkv" and any(_is_qkv_name(member) for member in members):
            parent = _attention_group_key(members)
            group_unit = PrunableUnit(
                unit_id=_unit_id("torch", "attention_qkv", parent),
                name=f"{parent}:attention_qkv",
                source="torch",
                unit_type="attention_qkv",
                module_or_node_name=parent,
                prunable_dims=["num_heads", "head_dim", "hidden_dim"],
                parameter_count=sum((_find_unit(graph, _linear_unit_id(member)).parameter_count or 0) for member in members if _find_unit(graph, _linear_unit_id(member))),
                shape=None,
                confidence="medium",
                reason="Q/K/V projections must usually be pruned consistently across attention heads.",
            )
            graph.prunable_units.append(group_unit)
            attention_group_units.append(group_unit)
            graph.coupled_groups.append(
                {
                    "group_id": group_unit.unit_id,
                    "group_type": "attention_qkv",
                    "members": [_linear_unit_id(member) for member in members],
                    "confidence": group.get("confidence", "medium"),
                    "reason": group.get("reason", group_unit.reason),
                }
            )

            for member in members:
                _mark_unit_type(
                    graph,
                    _linear_unit_id(member),
                    "linear",
                    ["out_features"],
                    "medium",
                    "Projection participates in an attention Q/K/V group.",
                )

            member_ids = [_linear_unit_id(member) for member in members if _find_unit(graph, _linear_unit_id(member))]
            for src in member_ids:
                for dst in member_ids:
                    if src != dst:
                        _add_edge_once(
                            graph,
                            DependencyEdge(
                                src=src,
                                dst=dst,
                                edge_type="qkv_coupling",
                                affected_dims=["num_heads", "head_dim", "hidden_dim"],
                                direction="bidirectional",
                                confidence="medium",
                                reason="Q/K/V projections must usually be pruned consistently across attention heads.",
                            ),
                        )
                _add_edge_once(
                    graph,
                    DependencyEdge(
                        src=group_unit.unit_id,
                        dst=src,
                        edge_type="qkv_coupling",
                        affected_dims=["num_heads", "head_dim", "hidden_dim"],
                        direction="bidirectional",
                        confidence="medium",
                        reason="Higher-level attention group constrains the member projection.",
                    ),
                )

        elif group_type == "attention_output_projection":
            for member in members:
                _mark_unit_type(
                    graph,
                    _linear_unit_id(member),
                    "attention_output",
                    ["in_features", "out_features", "hidden_dim"],
                    "medium",
                    "Layer name suggests attention output projection; pruning is coupled to attention head dimensions.",
                )

        elif group_type == "mlp_projection_pair" and len(members) >= 2:
            member_ids = [_linear_unit_id(member) for member in members if _find_unit(graph, _linear_unit_id(member))]
            expansion_ids = [unit_id for unit_id in member_ids if _is_mlp_expansion(_find_unit(graph, unit_id).name if _find_unit(graph, unit_id) else "")]
            projection_ids = [unit_id for unit_id in member_ids if _is_mlp_projection(_find_unit(graph, unit_id).name if _find_unit(graph, unit_id) else "")]

            for unit_id in expansion_ids:
                _mark_unit_type(
                    graph,
                    unit_id,
                    "mlp_expansion",
                    ["out_features", "intermediate_dim"],
                    "medium",
                    "MLP expansion layer output channels define the intermediate dimension.",
                )
            for unit_id in projection_ids:
                _mark_unit_type(
                    graph,
                    unit_id,
                    "mlp_projection",
                    ["in_features", "intermediate_dim"],
                    "medium",
                    "MLP projection layer input channels are coupled to the expansion layer output.",
                )

            graph.coupled_groups.append(
                {
                    "group_id": _unit_id("torch", "mlp", group.get("group_name", "mlp_projection_pair")),
                    "group_type": "mlp_projection_pair",
                    "members": member_ids,
                    "confidence": group.get("confidence", "medium"),
                    "reason": "Expansion/projection MLP layers share an intermediate hidden dimension.",
                }
            )
            for src in member_ids:
                for dst in member_ids:
                    if src != dst:
                        _add_edge_once(
                            graph,
                            DependencyEdge(
                                src=src,
                                dst=dst,
                                edge_type="mlp_hidden_coupling",
                                affected_dims=["intermediate_dim"],
                                direction="bidirectional",
                                confidence="medium",
                                reason="Pruning hidden channels in the expansion layer requires matching input channels in the projection layer.",
                            ),
                        )

    attention_outputs = [unit for unit in graph.prunable_units if unit.source == "torch" and unit.unit_type == "attention_output"]
    for group_unit in attention_group_units:
        group_parent = group_unit.module_or_node_name
        for output_unit in attention_outputs:
            output_parent = _parent_name(output_unit.name)
            if output_parent == group_parent or group_parent in output_unit.name or output_parent in group_parent:
                _add_edge_once(
                    graph,
                    DependencyEdge(
                        src=group_unit.unit_id,
                        dst=output_unit.unit_id,
                        edge_type="head_dimension_coupling",
                        affected_dims=["num_heads", "head_dim", "hidden_dim"],
                        direction="forward",
                        confidence="medium",
                        reason="Attention output projection consumes concatenated attention heads from Q/K/V projections.",
                    ),
                )


def _add_normalization_and_ambiguity(graph: DependencyGraph, torch_summary: dict[str, Any]) -> None:
    normalization_layers = torch_summary.get("normalization_layers", [])
    for norm in normalization_layers:
        norm_name = norm.get("name", "")
        parent = _parent_name(norm_name)
        nearby_units = [unit for unit in graph.prunable_units if unit.source == "torch" and parent and unit.name.startswith(parent)]
        if nearby_units:
            for unit in nearby_units:
                _add_edge_once(
                    graph,
                    DependencyEdge(
                        src=unit.unit_id,
                        dst=f"torch:norm:{norm_name}",
                        edge_type="normalization_dependency",
                        affected_dims=["hidden_dim"],
                        direction="forward",
                        confidence="low",
                        reason="LayerNorm-like parameters may need to follow the hidden dimension pruned upstream.",
                    ),
                )
        else:
            graph.ambiguous_units.append(
                {
                    "name": norm_name,
                    "source": "torch",
                    "reason": "Normalization layer detected, but no local prunable producer was confidently identified.",
                    "confidence": "low",
                }
            )

    qkv_roles_by_parent: dict[str, set[str]] = defaultdict(set)
    for unit in graph.prunable_units:
        if unit.source != "torch" or unit.unit_type not in {"linear", "attention_output"}:
            continue
        role = _qkv_role(unit.name)
        if role:
            qkv_roles_by_parent[_parent_name(unit.name)].add(role)

    for parent, roles in qkv_roles_by_parent.items():
        if roles and roles != {"q", "k", "v"}:
            graph.ambiguous_units.append(
                {
                    "name": parent or "root",
                    "source": "torch",
                    "reason": f"Incomplete attention projection group detected: {sorted(roles)}.",
                    "confidence": "medium",
                }
            )

    coupled_members = {member for group in graph.coupled_groups for member in group.get("members", [])}
    for unit in graph.prunable_units:
        if unit.source != "torch":
            continue
        if unit.unit_type == "embedding":
            graph.ambiguous_units.append(
                {
                    "name": unit.name,
                    "unit_id": unit.unit_id,
                    "source": "torch",
                    "reason": "Embedding may be tied to an output head or constrained by vocabulary semantics.",
                    "confidence": "low",
                }
            )
        elif unit.unit_type == "linear" and unit.unit_id not in coupled_members and "." in unit.name:
            graph.ambiguous_units.append(
                {
                    "name": unit.name,
                    "unit_id": unit.unit_id,
                    "source": "torch",
                    "reason": "Linear layer is inside a nested block but was not assigned to attention or MLP coupling.",
                    "confidence": "low",
                }
            )

    graph.metadata["torch_evidence"] = {
        "num_linear_layers": len(torch_summary.get("linear_layers", [])),
        "num_embedding_layers": len(torch_summary.get("embedding_layers", [])),
        "num_normalization_layers": len(torch_summary.get("normalization_layers", [])),
        "num_pruning_relevant_groups": len(torch_summary.get("pruning_relevant_groups", [])),
    }


def build_dependency_graph_from_torch_summary(torch_summary: dict) -> DependencyGraph:
    """Build a conservative pruning dependency graph from a PyTorch structural summary."""
    graph = DependencyGraph(
        model_name=torch_summary["model_name"],
        hf_id=torch_summary.get("hf_id", ""),
        task=torch_summary.get("task", ""),
        metadata={"sources": ["torch_structural_inventory"]},
    )

    graph.prunable_units.extend(_make_linear_unit(layer) for layer in torch_summary.get("linear_layers", []))
    graph.prunable_units.extend(_make_embedding_unit(layer) for layer in torch_summary.get("embedding_layers", []))

    _add_torch_couplings(graph, torch_summary)
    _add_normalization_and_ambiguity(graph, torch_summary)
    graph.ambiguous_units.append(
        {
            "name": "torch_to_onnx_mapping",
            "source": "torch",
            "reason": "PyTorch module names are not yet mapped to ONNX node names.",
            "confidence": "low",
        }
    )
    _recompute_independent_units(graph)
    return graph


def _onnx_shape_for_node(node: dict[str, Any], initializer_shapes: dict[str, list[int]]) -> list[int] | None:
    for input_name in node.get("inputs", []):
        if input_name in initializer_shapes:
            return initializer_shapes[input_name]
    return None


def augment_dependency_graph_with_onnx_summary(graph: DependencyGraph, onnx_summary: dict) -> DependencyGraph:
    """Augment a dependency graph with conservative ONNX graph evidence."""
    initializer_shapes = onnx_summary.get("initializer_shapes", {})
    previous_node_id: str | None = None
    propagation_relevant_count = 0

    for index, node in enumerate(onnx_summary.get("nodes", [])):
        op_type = node.get("op_type", "")
        node_name = node.get("name") or f"{op_type}_{index}"
        node_id = _unit_id("onnx", op_type.lower(), node_name)

        if op_type in {"Gemm", "MatMul", "Conv"}:
            unit_type = op_type.lower()
            dims = ["channel_out"] if op_type == "Conv" else ["out_features", "hidden_dim"]
            reason = "ONNX node is a high-interest parameterized operation, but mapping to PyTorch modules is not yet established."
            confidence: Confidence = "medium" if op_type == "Conv" else "low"
            if op_type == "Conv":
                reason = "ONNX Conv is a high-interest pruning target and may represent ViT patch projection in vision models."
            graph.prunable_units.append(
                PrunableUnit(
                    unit_id=node_id,
                    name=node_name,
                    source="onnx",
                    unit_type=unit_type,
                    module_or_node_name=node_name,
                    prunable_dims=dims,
                    parameter_count=None,
                    shape=_onnx_shape_for_node(node, initializer_shapes),
                    confidence=confidence,
                    reason=reason,
                )
            )
            graph.ambiguous_units.append(
                {
                    "name": node_name,
                    "unit_id": node_id,
                    "source": "onnx",
                    "reason": f"Unmapped ONNX {op_type} node needs PyTorch correspondence before concrete pruning.",
                    "confidence": "low",
                }
            )
            if previous_node_id:
                _add_edge_once(
                    graph,
                    DependencyEdge(
                        src=previous_node_id,
                        dst=node_id,
                        edge_type="feeds",
                        affected_dims=["hidden_dim"],
                        direction="forward",
                        confidence="low",
                        reason="Sequential ONNX node order suggests a possible feed relationship; tensor-level mapping is approximate.",
                    ),
                )
            previous_node_id = node_id

        elif op_type in PROPAGATION_OPS:
            propagation_relevant_count += 1
            propagation_id = _unit_id("onnx", "propagation", node_name)
            if previous_node_id:
                edge_type = "propagation_only"
                affected_dims = ["hidden_dim"]
                reason = f"ONNX {op_type} may propagate shape constraints."
                confidence: Confidence = "low"
                if op_type in {"Reshape", "Transpose", "Attention"}:
                    edge_type = "head_dimension_coupling"
                    affected_dims = ["num_heads", "head_dim", "hidden_dim"]
                    reason = f"ONNX {op_type} near attention can couple head and hidden dimensions."
                    confidence = "medium"
                elif op_type in {"LayerNormalization", "SkipLayerNormalization"}:
                    edge_type = "normalization_dependency"
                    reason = f"ONNX {op_type} may depend on the hidden dimension."
                    confidence = "medium"
                elif op_type == "Add":
                    edge_type = "residual_coupling"
                    reason = "ONNX Add is a residual-coupling candidate requiring branch shape consistency."
                    confidence = "medium"

                _add_edge_once(
                    graph,
                    DependencyEdge(
                        src=previous_node_id,
                        dst=propagation_id,
                        edge_type=edge_type,
                        affected_dims=affected_dims,
                        direction="forward" if edge_type == "propagation_only" else "bidirectional",
                        confidence=confidence,
                        reason=reason,
                    ),
                )

            graph.ambiguous_units.append(
                {
                    "name": node_name,
                    "unit_id": propagation_id,
                    "source": "onnx",
                    "reason": f"ONNX {op_type} is propagation-relevant but not directly prunable.",
                    "confidence": "medium" if op_type in {"Add", "Reshape", "Transpose", "LayerNormalization", "SkipLayerNormalization"} else "low",
                }
            )

    graph.metadata.setdefault("sources", ["torch_structural_inventory"])
    if "onnx_graph_summary" not in graph.metadata["sources"]:
        graph.metadata["sources"].append("onnx_graph_summary")
    graph.metadata["onnx_evidence"] = {
        "num_onnx_nodes": onnx_summary.get("graph_summary", {}).get("num_nodes", 0),
        "op_type_counts": onnx_summary.get("graph_summary", {}).get("op_type_counts", {}),
        "pruning_relevant_node_count": len(
            [
                node
                for node in onnx_summary.get("pruning_relevant_nodes", [])
                if node.get("op_type") in {"Gemm", "MatMul", "Conv"}
            ]
        ),
        "propagation_relevant_node_count": propagation_relevant_count,
    }
    _recompute_independent_units(graph)
    return graph


def write_dependency_graph_json(graph: DependencyGraph, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if not rows:
        return "_None detected._"
    selected = rows[:limit] if limit else rows
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if limit and len(rows) > limit:
        omitted = {columns[0]: "..."}
        if len(columns) > 1:
            omitted[columns[1]] = f"{len(rows) - limit} more rows omitted"
        lines.append("| " + " | ".join(str(omitted.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def dependency_graph_to_markdown(graph: DependencyGraph) -> str:
    data = graph.to_dict()
    units = data["prunable_units"]
    edges = data["dependency_edges"]
    lines = [
        f"# Dependency Graph: {graph.model_name}",
        "",
        "## Summary",
        "",
        f"- Prunable units: `{len(graph.prunable_units)}`",
        f"- Dependency edges: `{len(graph.dependency_edges)}`",
        f"- Coupled groups: `{len(graph.coupled_groups)}`",
        f"- Independent units: `{len(graph.independent_units)}`",
        f"- Ambiguous units: `{len(graph.ambiguous_units)}`",
        "",
        "## Prunable Units",
        "",
        _markdown_table(units, ["unit_id", "unit_type", "name", "prunable_dims", "confidence", "reason"], limit=300),
        "",
        "## Dependency Edges",
        "",
        _markdown_table(edges, ["src", "dst", "edge_type", "affected_dims", "direction", "confidence", "reason"], limit=400),
        "",
        "## Coupled Groups",
        "",
        _markdown_table(graph.coupled_groups, ["group_id", "group_type", "members", "confidence", "reason"], limit=200),
        "",
        "QKV groups, MLP groups, residual candidates, and embedding caveats are intentionally conservative. A group indicates likely shared pruning constraints, not a proof that pruning is valid.",
        "",
        "## Independent Units",
        "",
        "\n".join(f"- `{unit_id}`" for unit_id in graph.independent_units) if graph.independent_units else "_None identified conservatively._",
        "",
        "## Ambiguous / Manual Review",
        "",
        _markdown_table(graph.ambiguous_units, ["name", "unit_id", "source", "confidence", "reason"], limit=300),
        "",
        "## Interpretation for Pruning",
        "",
        "- Local pruning is most plausible for units without coupling edges, but still requires shape validation.",
        "- Forward propagation is required when a pruned output dimension feeds downstream projections, reshapes, normalization, residual paths, or ONNX propagation nodes.",
        "- Backward propagation is required when a unit's input dimension is constrained by an upstream projection, embedding, or tied parameter.",
        "- Structures marked ambiguous are unsafe without additional graph, shape, or framework-specific analysis.",
        "",
        "## Edge Type Counts",
        "",
        _markdown_table(
            [{"edge_type": key, "count": value} for key, value in sorted(Counter(edge.edge_type for edge in graph.dependency_edges).items())],
            ["edge_type", "count"],
        ),
        "",
    ]
    return "\n".join(lines)
