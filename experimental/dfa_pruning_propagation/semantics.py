"""Semantic annotations and structural pattern matching for the DFA prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experimental.dfa_pruning_propagation.ir import Axis, Graph, Node


class SemanticRole(str, Enum):
    UNKNOWN = "UNKNOWN"
    EXPANSION_PROJECTION = "EXPANSION_PROJECTION"
    INDEX_PRESERVING_ACTIVATION = "INDEX_PRESERVING_ACTIVATION"
    CONTRACTION_PROJECTION = "CONTRACTION_PROJECTION"
    RESIDUAL_MERGE = "RESIDUAL_MERGE"
    NORMALIZATION = "NORMALIZATION"
    VALUE_PROJECTION = "VALUE_PROJECTION"
    ATTENTION_CONTEXT = "ATTENTION_CONTEXT"
    ATTENTION_OUTPUT_PROJECTION = "ATTENTION_OUTPUT_PROJECTION"
    QUERY_PROJECTION = "QUERY_PROJECTION"
    KEY_PROJECTION = "KEY_PROJECTION"
    SCORE_CONTRACTION = "SCORE_CONTRACTION"


class SemanticAxisRole(str, Enum):
    UNKNOWN = "unknown"
    BATCH = "batch_dim"
    SEQUENCE = "sequence_dim"
    HIDDEN = "hidden_dim"
    INTERMEDIATE = "intermediate_dim"
    HEAD = "head_dim"
    VALUE = "value_dim"
    VALUE_CONTEXT = "value_context_dim"
    SCORE = "score_dim"
    MASK = "mask_dim"


class SemanticPattern(str, Enum):
    FFN_INTERMEDIATE_CHAIN = "FFN_INTERMEDIATE_CHAIN"
    ATTENTION_VALUE_CHAIN = "ATTENTION_VALUE_CHAIN"
    ATTENTION_QK_SCORE_CHAIN = "ATTENTION_QK_SCORE_CHAIN"
    RESIDUAL_PROTECTED_CHAIN = "RESIDUAL_PROTECTED_CHAIN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SemanticAnnotation:
    node_id: str
    semantic_role: SemanticRole
    confidence: str
    evidence: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AxisAnnotation:
    axis_key: str
    semantic_axis_role: SemanticAxisRole
    protected: bool
    evidence: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PatternMatch:
    pattern: SemanticPattern
    node_ids: tuple[str, ...]
    evidence: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class SemanticGraphAnnotations:
    nodes: dict[str, SemanticAnnotation] = field(default_factory=dict)
    axes: dict[str, AxisAnnotation] = field(default_factory=dict)
    patterns: list[PatternMatch] = field(default_factory=list)


def normalize_axis_role(role: SemanticAxisRole | str) -> SemanticAxisRole:
    if isinstance(role, SemanticAxisRole):
        return role
    normalized = role.strip().lower()
    aliases = {
        "batch": SemanticAxisRole.BATCH,
        "sequence": SemanticAxisRole.SEQUENCE,
        "hidden": SemanticAxisRole.HIDDEN,
        "intermediate": SemanticAxisRole.INTERMEDIATE,
        "head": SemanticAxisRole.HEAD,
        "value": SemanticAxisRole.VALUE,
        "value_context": SemanticAxisRole.VALUE_CONTEXT,
        "score": SemanticAxisRole.SCORE,
        "mask": SemanticAxisRole.MASK,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return SemanticAxisRole(normalized)
    except ValueError:
        return SemanticAxisRole.UNKNOWN


def _normalize_op_kind(op_kind: str) -> str:
    return "".join(character for character in op_kind.lower() if character.isalnum())


def _is_projection(node: Node) -> bool:
    return _normalize_op_kind(node.op_kind) in {"linear", "matmul", "gemm", "projection"}


def _is_activation(node: Node) -> bool:
    return _normalize_op_kind(node.op_kind) in {"activation", "gelu", "newgelu", "relu", "tanh", "silu"}


def _has_axis(axes: list[Axis], role: SemanticAxisRole) -> bool:
    return any(axis.role == role for axis in axes)


def _annotate_node(
    annotations: SemanticGraphAnnotations,
    node: Node,
    role: SemanticRole,
    confidence: str,
    *evidence: str,
) -> None:
    existing = annotations.nodes.get(node.node_id)
    if existing and existing.semantic_role != SemanticRole.UNKNOWN and node.semantic_role is not None:
        return
    annotation = SemanticAnnotation(node.node_id, role, confidence, tuple(evidence))
    annotations.nodes[node.node_id] = annotation
    node.semantic_role = role


def _record_pattern(
    annotations: SemanticGraphAnnotations,
    pattern: SemanticPattern,
    nodes: tuple[Node, ...],
    *evidence: str,
) -> None:
    match = PatternMatch(pattern, tuple(node.node_id for node in nodes), tuple(evidence))
    if match not in annotations.patterns:
        annotations.patterns.append(match)


def _seed_explicit_roles(graph: Graph, annotations: SemanticGraphAnnotations) -> None:
    for node in graph.nodes.values():
        if node.semantic_role is not None:
            annotations.nodes[node.node_id] = SemanticAnnotation(
                node.node_id,
                node.semantic_role,
                "high",
                ("explicit semantic_role metadata",),
            )


def _detect_ffn_chains(graph: Graph, annotations: SemanticGraphAnnotations) -> None:
    for expansion in graph.nodes.values():
        if not _is_projection(expansion) or not _has_axis(expansion.outputs, SemanticAxisRole.INTERMEDIATE):
            continue
        for activation in graph.successors(expansion.node_id):
            if (
                not _is_activation(activation)
                or not _has_axis(activation.inputs, SemanticAxisRole.INTERMEDIATE)
                or not _has_axis(activation.outputs, SemanticAxisRole.INTERMEDIATE)
            ):
                continue
            for contraction in graph.successors(activation.node_id):
                if (
                    not _is_projection(contraction)
                    or not _has_axis(contraction.inputs, SemanticAxisRole.INTERMEDIATE)
                    or not _has_axis(contraction.outputs, SemanticAxisRole.HIDDEN)
                ):
                    continue
                evidence = "projection INTERMEDIATE -> index-preserving activation INTERMEDIATE -> projection HIDDEN"
                _annotate_node(annotations, expansion, SemanticRole.EXPANSION_PROJECTION, "high", evidence)
                _annotate_node(annotations, activation, SemanticRole.INDEX_PRESERVING_ACTIVATION, "high", evidence)
                _annotate_node(annotations, contraction, SemanticRole.CONTRACTION_PROJECTION, "high", evidence)
                _record_pattern(annotations, SemanticPattern.FFN_INTERMEDIATE_CHAIN, (expansion, activation, contraction), evidence)


def _detect_attention_value_chains(graph: Graph, annotations: SemanticGraphAnnotations) -> None:
    for value_projection in graph.nodes.values():
        if not _is_projection(value_projection) or not _has_axis(value_projection.outputs, SemanticAxisRole.VALUE):
            continue
        for context in graph.successors(value_projection.node_id):
            if (
                not _has_axis(context.inputs, SemanticAxisRole.VALUE)
                or not _has_axis(context.outputs, SemanticAxisRole.VALUE_CONTEXT)
            ):
                continue
            for output_projection in graph.successors(context.node_id):
                if (
                    not _is_projection(output_projection)
                    or not _has_axis(output_projection.inputs, SemanticAxisRole.VALUE_CONTEXT)
                    or not _has_axis(output_projection.outputs, SemanticAxisRole.HIDDEN)
                ):
                    continue
                evidence = "projection VALUE -> context VALUE_CONTEXT -> projection HIDDEN"
                _annotate_node(annotations, value_projection, SemanticRole.VALUE_PROJECTION, "high", evidence)
                _annotate_node(annotations, context, SemanticRole.ATTENTION_CONTEXT, "high", evidence)
                _annotate_node(annotations, output_projection, SemanticRole.ATTENTION_OUTPUT_PROJECTION, "high", evidence)
                _record_pattern(annotations, SemanticPattern.ATTENTION_VALUE_CHAIN, (value_projection, context, output_projection), evidence)


def _detect_qk_score_chains(graph: Graph, annotations: SemanticGraphAnnotations) -> None:
    for score in graph.nodes.values():
        if (
            len([axis for axis in score.inputs if axis.role == SemanticAxisRole.HEAD]) < 2
            or not any(axis.role in {SemanticAxisRole.SCORE, SemanticAxisRole.SEQUENCE} for axis in score.outputs)
        ):
            continue
        projections = [
            predecessor
            for predecessor in graph.predecessors(score.node_id)
            if _is_projection(predecessor) and _has_axis(predecessor.outputs, SemanticAxisRole.HEAD)
        ]
        if len(projections) < 2:
            continue
        _annotate_node(
            annotations,
            score,
            SemanticRole.SCORE_CONTRACTION,
            "high",
            "two HEAD projections feed a SCORE/SEQUENCE contraction",
        )
        for projection in projections:
            projection_role = str(projection.attrs.get("attention_projection_role", "")).lower()
            if projection_role == "query":
                _annotate_node(annotations, projection, SemanticRole.QUERY_PROJECTION, "high", "explicit query projection metadata")
            elif projection_role == "key":
                _annotate_node(annotations, projection, SemanticRole.KEY_PROJECTION, "high", "explicit key projection metadata")
        _record_pattern(
            annotations,
            SemanticPattern.ATTENTION_QK_SCORE_CHAIN,
            (*projections, score),
            "HEAD projection pair feeds SCORE contraction",
        )


def _detect_protected_nodes(graph: Graph, annotations: SemanticGraphAnnotations) -> None:
    residuals: list[Node] = []
    norms: list[Node] = []
    for node in graph.nodes.values():
        normalized = _normalize_op_kind(node.op_kind)
        if normalized in {"add", "residualadd"} and (
            _has_axis(node.inputs, SemanticAxisRole.HIDDEN) or _has_axis(node.outputs, SemanticAxisRole.HIDDEN)
        ):
            _annotate_node(annotations, node, SemanticRole.RESIDUAL_MERGE, "high", "Add carries hidden_dim residual path")
            residuals.append(node)
        elif normalized in {"layernorm", "layernormalization", "normalization"} and (
            _has_axis(node.inputs, SemanticAxisRole.HIDDEN) or _has_axis(node.outputs, SemanticAxisRole.HIDDEN)
        ):
            _annotate_node(annotations, node, SemanticRole.NORMALIZATION, "high", "normalization carries protected hidden_dim")
            norms.append(node)
    for residual in residuals:
        for norm in graph.successors(residual.node_id):
            if norm in norms:
                _record_pattern(
                    annotations,
                    SemanticPattern.RESIDUAL_PROTECTED_CHAIN,
                    (residual, norm),
                    "residual hidden_dim feeds normalization hidden_dim",
                )


def annotate_graph(graph: Graph) -> SemanticGraphAnnotations:
    """Infer semantic node and axis roles without consulting syntactic names."""
    annotations = SemanticGraphAnnotations()
    _seed_explicit_roles(graph, annotations)
    _detect_ffn_chains(graph, annotations)
    _detect_attention_value_chains(graph, annotations)
    _detect_qk_score_chains(graph, annotations)
    _detect_protected_nodes(graph, annotations)
    for node in graph.nodes.values():
        if node.node_id not in annotations.nodes:
            role = node.semantic_role or SemanticRole.UNKNOWN
            annotations.nodes[node.node_id] = SemanticAnnotation(node.node_id, role, "low", ("no structural pattern matched",))
            node.semantic_role = role
    for axis in graph.all_axes():
        protected = axis.role == SemanticAxisRole.HIDDEN and any(
            node.semantic_role in {
                SemanticRole.CONTRACTION_PROJECTION,
                SemanticRole.ATTENTION_OUTPUT_PROJECTION,
                SemanticRole.RESIDUAL_MERGE,
                SemanticRole.NORMALIZATION,
            }
            for node in graph.touching_nodes(axis)
        )
        annotations.axes[axis.key] = AxisAnnotation(
            axis.key,
            axis.role,
            protected,
            ("hidden_dim boundary is structurally protected",) if protected else ("semantic axis role from IR",),
        )
    return annotations
