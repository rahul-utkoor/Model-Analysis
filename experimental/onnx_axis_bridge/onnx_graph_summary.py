"""Summarize local ONNX graph structure for conservative pattern hints."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

from experimental.onnx_axis_bridge.onnx_loader import OnnxNodeInfo, OnnxSubgraph


PARAMETERIZED_OPS = {"MatMul", "Gemm", "Conv"}
ELEMENTWISE_OPS = {"Relu", "Sigmoid", "Tanh", "Erf", "Pow", "Mul", "Add", "Div", "Sub"}
LAYOUT_OPS = {"Reshape", "Transpose", "Flatten", "Unsqueeze", "Squeeze", "Gather", "Shape", "Expand", "Constant"}
CONTRACTION_OPS = {"MatMul", "Gemm", "Conv"}


@dataclass
class OnnxGraphSummary:
    num_nodes: int
    op_type_counts: dict[str, int]
    producer_by_tensor: dict[str, str]
    consumers_by_tensor: dict[str, list[str]]
    topological_nodes: list[OnnxNodeInfo]
    parameterized_ops: list[str]
    elementwise_ops: list[str]
    contraction_ops: list[str]
    reshape_transpose_ops: list[str]
    candidate_pattern_hints: list[str] = field(default_factory=list)
    explanation: str = ""


def summarize_subgraph(subgraph: OnnxSubgraph) -> OnnxGraphSummary:
    """Build a dependency summary without running ONNX shape inference."""
    producer_by_tensor = {
        tensor: node.node_id
        for node in subgraph.nodes
        for tensor in node.outputs
        if tensor
    }
    consumers: dict[str, list[str]] = defaultdict(list)
    for node in subgraph.nodes:
        for tensor in node.inputs:
            if tensor:
                consumers[tensor].append(node.node_id)
    return OnnxGraphSummary(
        num_nodes=len(subgraph.nodes),
        op_type_counts=dict(sorted(Counter(node.op_type for node in subgraph.nodes).items())),
        producer_by_tensor=producer_by_tensor,
        consumers_by_tensor=dict(consumers),
        topological_nodes=list(subgraph.nodes),
        parameterized_ops=[node.node_id for node in subgraph.nodes if node.op_type in PARAMETERIZED_OPS],
        elementwise_ops=[node.node_id for node in subgraph.nodes if node.op_type in ELEMENTWISE_OPS],
        contraction_ops=[node.node_id for node in subgraph.nodes if node.op_type in CONTRACTION_OPS],
        reshape_transpose_ops=[node.node_id for node in subgraph.nodes if node.op_type in LAYOUT_OPS],
        explanation="Local ONNX structure is summarized from topology, operator types, and available static shapes. Names are retained only as diagnostics.",
    )


def node_by_id(summary: OnnxGraphSummary) -> dict[str, OnnxNodeInfo]:
    return {node.node_id: node for node in summary.topological_nodes}


def reachable_path(
    subgraph: OnnxSubgraph,
    summary: OnnxGraphSummary,
    start: OnnxNodeInfo,
    goal: OnnxNodeInfo,
    *,
    pass_through_ops: set[str],
) -> list[str] | None:
    """Find one local tensor-flow path through allowed intermediary operations."""
    nodes = node_by_id(summary)
    queue = deque([(start.node_id, [start.node_id])])
    visited = {start.node_id}
    while queue:
        current_id, path = queue.popleft()
        current = nodes[current_id]
        for tensor in current.outputs:
            for consumer_id in summary.consumers_by_tensor.get(tensor, []):
                if consumer_id == goal.node_id:
                    return [*path, goal.node_id]
                consumer = nodes[consumer_id]
                if consumer_id not in visited and consumer.op_type in pass_through_ops:
                    visited.add(consumer_id)
                    queue.append((consumer_id, [*path, consumer_id]))
    return None


def tensor_shape(subgraph: OnnxSubgraph, tensor: str) -> tuple[int | str | None, ...]:
    info = subgraph.tensors.get(tensor)
    return info.shape if info else ()
