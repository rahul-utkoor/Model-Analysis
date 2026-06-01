"""Infer conservative local pruning-pattern hints from ONNX topology and shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from experimental.onnx_axis_bridge.onnx_graph_summary import (
    ELEMENTWISE_OPS,
    LAYOUT_OPS,
    OnnxGraphSummary,
    node_by_id,
    reachable_path,
    tensor_shape,
)
from experimental.onnx_axis_bridge.onnx_loader import OnnxNodeInfo, OnnxSubgraph


class OnnxPatternHintKind(str, Enum):
    UNKNOWN = "UNKNOWN"
    FFN_LIKE = "FFN_LIKE"
    ATTENTION_CONTEXT_LIKE = "ATTENTION_CONTEXT_LIKE"
    QK_SCORE_LIKE = "QK_SCORE_LIKE"
    ATTENTION_VALUE_PATH_LIKE = "ATTENTION_VALUE_PATH_LIKE"
    RESIDUAL_LIKE = "RESIDUAL_LIKE"
    LAYERNORM_LIKE = "LAYERNORM_LIKE"


@dataclass(frozen=True)
class OnnxPatternHint:
    kind: OnnxPatternHintKind
    confidence: str
    nodes: tuple[str, ...]
    evidence: tuple[str, ...] = field(default_factory=tuple)
    explanation: str = ""


def _known_equal(left: object, right: object) -> bool:
    return left is not None and right is not None and left == right


def _known_different(left: object, right: object) -> bool:
    return left is not None and right is not None and left != right


def _matmul_shapes(subgraph: OnnxSubgraph, node: OnnxNodeInfo) -> tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]] | None:
    if node.op_type != "MatMul" or len(node.inputs) < 2 or not node.outputs:
        return None
    left = tensor_shape(subgraph, node.inputs[0])
    right = tensor_shape(subgraph, node.inputs[1])
    output = tensor_shape(subgraph, node.outputs[0])
    if not (len(left) == len(right) == len(output) == 4):
        return None
    return left, right, output


def _infer_ffn_hints(subgraph: OnnxSubgraph, summary: OnnxGraphSummary) -> list[OnnxPatternHint]:
    nodes = node_by_id(summary)
    projections = [nodes[node_id] for node_id in summary.parameterized_ops]
    allowed = ELEMENTWISE_OPS | LAYOUT_OPS
    hints: list[OnnxPatternHint] = []
    for index, first in enumerate(projections):
        for second in projections[index + 1 :]:
            path = reachable_path(subgraph, summary, first, second, pass_through_ops=allowed)
            if not path:
                continue
            intermediaries = [nodes[node_id] for node_id in path[1:-1]]
            if not any(node.op_type in ELEMENTWISE_OPS for node in intermediaries):
                continue
            first_shape = tensor_shape(subgraph, first.outputs[0]) if first.outputs else ()
            second_shape = tensor_shape(subgraph, second.inputs[0]) if second.inputs else ()
            compatible = bool(first_shape and second_shape and _known_equal(first_shape[-1], second_shape[-1]))
            evidence = [
                f"{first.node_id} output reaches {second.node_id} input through local elementwise/layout flow",
                f"intermediary ops: {', '.join(node.op_type for node in intermediaries)}",
            ]
            if compatible:
                evidence.append(f"intermediate feature width is compatible: {first_shape[-1]}")
            hints.append(
                OnnxPatternHint(
                    OnnxPatternHintKind.FFN_LIKE,
                    "high" if compatible else "medium",
                    tuple(path),
                    tuple(evidence),
                    "Two parameterized projections are connected by an elementwise/index-preserving candidate chain.",
                )
            )
    return hints


def _infer_attention_contraction_hints(subgraph: OnnxSubgraph, summary: OnnxGraphSummary) -> list[OnnxPatternHint]:
    hints: list[OnnxPatternHint] = []
    for node in summary.topological_nodes:
        shapes = _matmul_shapes(subgraph, node)
        if shapes is None:
            continue
        left, right, output = shapes
        common = (
            _known_equal(left[-1], right[-2])
            and _known_equal(left[-2], output[-2])
            and _known_equal(right[-1], output[-1])
        )
        if not common:
            continue
        if _known_equal(output[-2], output[-1]) and _known_different(left[-1], output[-1]):
            hints.append(
                OnnxPatternHint(
                    OnnxPatternHintKind.QK_SCORE_LIKE,
                    "high",
                    (node.node_id,),
                    (
                        f"rank-4 MatMul: {left} x {right} -> {output}",
                        "projected feature axis is reduced while query/key position axes remain free",
                    ),
                    "Shape evidence matches QK score contraction; simple one-to-one projected-feature propagation is blocked.",
                )
            )
        elif _known_equal(left[-2], left[-1]) and _known_different(right[-2], right[-1]):
            hints.append(
                OnnxPatternHint(
                    OnnxPatternHintKind.ATTENTION_CONTEXT_LIKE,
                    "high",
                    (node.node_id,),
                    (
                        f"rank-4 MatMul: {left} x {right} -> {output}",
                        f"value feature axis {right[-1]} is preserved into context output",
                    ),
                    "Shape evidence matches attention context contraction; the value feature axis remains free.",
                )
            )
        else:
            hints.append(
                OnnxPatternHint(
                    OnnxPatternHintKind.UNKNOWN,
                    "low",
                    (node.node_id,),
                    (f"ambiguous rank-4 MatMul: {left} x {right} -> {output}",),
                    "The rank-4 contraction shape is compatible with attention, but score/context semantics are not proven.",
                )
            )
    return hints


def _infer_attention_value_path_hints(
    subgraph: OnnxSubgraph,
    summary: OnnxGraphSummary,
    contraction_hints: list[OnnxPatternHint],
) -> list[OnnxPatternHint]:
    nodes = node_by_id(summary)
    projections = [nodes[node_id] for node_id in summary.parameterized_ops]
    allowed = ELEMENTWISE_OPS | LAYOUT_OPS | {"Cast", "Concat", "Gather", "Shape", "Slice", "Split", "Squeeze", "Unsqueeze"}
    hints: list[OnnxPatternHint] = []
    for context_hint in contraction_hints:
        if context_hint.kind != OnnxPatternHintKind.ATTENTION_CONTEXT_LIKE:
            continue
        context = nodes[context_hint.nodes[0]]
        for producer in projections:
            producer_path = reachable_path(subgraph, summary, producer, context, pass_through_ops=allowed)
            if not producer_path:
                continue
            for consumer in projections:
                if consumer.node_id == producer.node_id:
                    continue
                consumer_path = reachable_path(subgraph, summary, context, consumer, pass_through_ops=allowed)
                if consumer_path:
                    hints.append(
                        OnnxPatternHint(
                            OnnxPatternHintKind.ATTENTION_VALUE_PATH_LIKE,
                            "medium",
                            tuple([*producer_path[:-1], *consumer_path]),
                            (*context_hint.evidence, "projection -> context -> output-projection topology is present"),
                            "Local topology and shape evidence support an attention value-path lowering.",
                        )
                    )
    return hints


def _infer_residual_hints(subgraph: OnnxSubgraph, summary: OnnxGraphSummary) -> list[OnnxPatternHint]:
    hints: list[OnnxPatternHint] = []
    initializer_names = set(subgraph.initializers)
    nodes = node_by_id(summary)

    def is_ancestor(candidate: str, node_id: str) -> bool:
        pending = [node_id]
        visited: set[str] = set()
        while pending:
            current_id = pending.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            for tensor in nodes[current_id].inputs:
                producer_id = summary.producer_by_tensor.get(tensor)
                if producer_id == candidate:
                    return True
                if producer_id:
                    pending.append(producer_id)
        return False

    for node in summary.topological_nodes:
        if node.op_type != "Add" or len(node.inputs) != 2 or not node.outputs:
            continue
        if any(tensor in initializer_names for tensor in node.inputs):
            continue
        left = tensor_shape(subgraph, node.inputs[0])
        right = tensor_shape(subgraph, node.inputs[1])
        output = tensor_shape(subgraph, node.outputs[0])
        producer_ids = [summary.producer_by_tensor.get(tensor) for tensor in node.inputs]
        branches_are_independent = not (
            producer_ids[0]
            and producer_ids[1]
            and (
                producer_ids[0] == producer_ids[1]
                or is_ancestor(producer_ids[0], producer_ids[1])
                or is_ancestor(producer_ids[1], producer_ids[0])
            )
        )
        if left and right and output and left == right == output and branches_are_independent:
            hints.append(
                OnnxPatternHint(
                    OnnxPatternHintKind.RESIDUAL_LIKE,
                    "high",
                    (node.node_id,),
                    (f"two independent non-initializer inputs and output share shape {output}",),
                    "Shape-aligned Add is a residual-like protected hidden-axis candidate.",
                )
            )
    return hints


def _infer_layernorm_hints(summary: OnnxGraphSummary) -> list[OnnxPatternHint]:
    return [
        OnnxPatternHint(
            OnnxPatternHintKind.LAYERNORM_LIKE,
            "high",
            (node.node_id,),
            ("LayerNormalization operator is explicit",),
            "Explicit LayerNormalization protects its normalized hidden axis.",
        )
        for node in summary.topological_nodes
        if node.op_type in {"LayerNormalization", "LayerNorm"}
    ]


def infer_pattern_hints(subgraph: OnnxSubgraph, summary: OnnxGraphSummary) -> list[OnnxPatternHint]:
    """Infer conservative local pattern hints from operator topology and shapes."""
    contraction_hints = _infer_attention_contraction_hints(subgraph, summary)
    hints = [
        *_infer_ffn_hints(subgraph, summary),
        *contraction_hints,
        *_infer_attention_value_path_hints(subgraph, summary, contraction_hints),
        *_infer_residual_hints(subgraph, summary),
        *_infer_layernorm_hints(summary),
    ]
    summary.candidate_pattern_hints = [hint.kind.value for hint in hints]
    if hints:
        return hints
    return [
        OnnxPatternHint(
            OnnxPatternHintKind.UNKNOWN,
            "low",
            tuple(node.node_id for node in summary.topological_nodes),
            ("no supported local ONNX pattern was proven",),
            "The subgraph remains available for inspection, but template lowering is skipped.",
        )
    ]
