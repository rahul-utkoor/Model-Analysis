"""Local path and join-aware structural analysis over ONNX summary reports."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class OnnxGraphAdjacency:
    model_name: str
    node_names: list[str]
    node_by_name: dict[str, dict[str, Any]]
    successors: dict[str, list[str]]
    predecessors: dict[str, list[str]]
    producer_of_tensor: dict[str, str]
    consumers_of_tensor: dict[str, list[str]]
    node_inputs: dict[str, list[str]]
    node_outputs: dict[str, list[str]]
    graph_inputs: list[str]
    graph_outputs: list[str]
    initializers: list[str]


@dataclass
class NodePathSubgraph:
    subgraph_id: str
    model_name: str
    subgraph_kind: str
    size: int
    node_names: list[str]
    op_types: list[str]
    pattern: str
    input_tensors: list[str]
    output_tensors: list[str]
    internal_tensors: list[str]
    boundary_input_tensors: list[str]
    boundary_output_tensors: list[str]
    initializer_tensors: list[str]
    contains_initializers: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JoinSubgraph:
    subgraph_id: str
    model_name: str
    subgraph_kind: str
    join_node: str
    join_op_type: str
    branch_producer_nodes: list[str]
    branch_input_tensors: list[str]
    branch_depth: int
    post_join_nodes: list[str]
    post_join_depth: int
    node_names: list[str]
    op_types: list[str]
    pattern: str
    input_tensors: list[str]
    output_tensors: list[str]
    internal_tensors: list[str]
    boundary_input_tensors: list[str]
    boundary_output_tensors: list[str]
    initializer_tensors: list[str]
    contains_initializers: bool
    is_residual_like: bool
    residual_confidence: str
    residual_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubgraphPatternSummary:
    pattern: str
    size: int | None
    subgraph_kind: str
    count: int
    example_subgraph_ids: list[str]
    pruning_class: str
    risk_level: str
    reason: str


@dataclass
class SubgraphPruningEvidence:
    evidence_id: str
    subgraph_id: str
    subgraph_kind: str
    pattern: str
    evidence_type: str
    affected_nodes: list[str]
    affected_tensors: list[str]
    suggested_constraint_type: str | None
    confidence: str
    reason: str


@dataclass
class SubgraphAnalysisReport:
    model_name: str
    hf_id: str
    task: str
    max_nodes: int
    path_subgraphs: list[NodePathSubgraph] = field(default_factory=list)
    join_subgraphs: list[JoinSubgraph] = field(default_factory=list)
    pattern_summaries: list[SubgraphPatternSummary] = field(default_factory=list)
    pruning_evidence: list[SubgraphPruningEvidence] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def subgraph_analysis_report_to_dict(report: SubgraphAnalysisReport) -> dict[str, Any]:
    return asdict(report)


def write_subgraph_analysis_json(report: SubgraphAnalysisReport, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(subgraph_analysis_report_to_dict(report), indent=2), encoding="utf-8")


def load_subgraph_analysis_json(path: Path) -> SubgraphAnalysisReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SubgraphAnalysisReport(
        model_name=data["model_name"],
        hf_id=data.get("hf_id", ""),
        task=data.get("task", ""),
        max_nodes=data.get("max_nodes", 5),
        path_subgraphs=[NodePathSubgraph(**item) for item in data.get("path_subgraphs", [])],
        join_subgraphs=[JoinSubgraph(**item) for item in data.get("join_subgraphs", [])],
        pattern_summaries=[SubgraphPatternSummary(**item) for item in data.get("pattern_summaries", [])],
        pruning_evidence=[SubgraphPruningEvidence(**item) for item in data.get("pruning_evidence", [])],
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def _names(entries: list[Any]) -> list[str]:
    return [entry.get("name", "") if isinstance(entry, dict) else str(entry) for entry in entries]


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def build_onnx_adjacency(onnx_summary: dict) -> OnnxGraphAdjacency:
    """Construct tensor producer/consumer and node predecessor/successor maps."""
    nodes = onnx_summary.get("nodes", [])
    node_names = [node["name"] for node in nodes]
    order = {name: index for index, name in enumerate(node_names)}
    node_by_name = {node["name"]: node for node in nodes}
    node_inputs = {node["name"]: list(node.get("inputs", [])) for node in nodes}
    node_outputs = {node["name"]: list(node.get("outputs", [])) for node in nodes}
    producer_of_tensor = {
        tensor: node["name"]
        for node in nodes
        for tensor in node.get("outputs", [])
        if tensor
    }
    consumers_of_tensor: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for tensor in node.get("inputs", []):
            if tensor:
                consumers_of_tensor[tensor].append(node["name"])
    successors: dict[str, set[str]] = {name: set() for name in node_names}
    predecessors: dict[str, set[str]] = {name: set() for name in node_names}
    for tensor, producer in producer_of_tensor.items():
        for consumer in consumers_of_tensor.get(tensor, []):
            if consumer != producer:
                successors[producer].add(consumer)
                predecessors[consumer].add(producer)

    sort_key = lambda name: (order.get(name, len(order)), name)
    return OnnxGraphAdjacency(
        model_name=onnx_summary.get("model_name", ""),
        node_names=node_names,
        node_by_name=node_by_name,
        successors={name: sorted(values, key=sort_key) for name, values in successors.items()},
        predecessors={name: sorted(values, key=sort_key) for name, values in predecessors.items()},
        producer_of_tensor=producer_of_tensor,
        consumers_of_tensor={
            tensor: sorted(values, key=sort_key)
            for tensor, values in consumers_of_tensor.items()
        },
        node_inputs=node_inputs,
        node_outputs=node_outputs,
        graph_inputs=_names(onnx_summary.get("inputs", [])),
        graph_outputs=_names(onnx_summary.get("outputs", [])),
        initializers=_names(onnx_summary.get("initializers", [])),
    )


def compute_subgraph_tensor_sets(
    node_names: list[str],
    adjacency: OnnxGraphAdjacency,
    onnx_summary: dict,
) -> dict[str, Any]:
    """Identify subgraph-internal and boundary tensors for a node set."""
    del onnx_summary  # Adjacency already preserves the required graph-level tensor sets.
    selected = set(node_names)
    input_tensors = _ordered_unique(
        [tensor for name in node_names for tensor in adjacency.node_inputs.get(name, [])]
    )
    output_tensors = _ordered_unique(
        [tensor for name in node_names for tensor in adjacency.node_outputs.get(name, [])]
    )
    internal_tensors = [
        tensor
        for tensor in output_tensors
        if any(consumer in selected for consumer in adjacency.consumers_of_tensor.get(tensor, []))
    ]
    boundary_input_tensors = [
        tensor for tensor in input_tensors if adjacency.producer_of_tensor.get(tensor) not in selected
    ]
    graph_outputs = set(adjacency.graph_outputs)
    boundary_output_tensors = [
        tensor
        for tensor in output_tensors
        if tensor in graph_outputs
        or any(consumer not in selected for consumer in adjacency.consumers_of_tensor.get(tensor, []))
    ]
    initializer_names = set(adjacency.initializers)
    initializer_tensors = [tensor for tensor in input_tensors if tensor in initializer_names]
    return {
        "input_tensors": input_tensors,
        "output_tensors": output_tensors,
        "internal_tensors": internal_tensors,
        "boundary_input_tensors": boundary_input_tensors,
        "boundary_output_tensors": boundary_output_tensors,
        "initializer_tensors": initializer_tensors,
        "contains_initializers": bool(initializer_tensors),
    }


def classify_subgraph_pattern(
    op_types: list[str],
    subgraph_kind: str = "path",
    join_metadata: dict | None = None,
) -> tuple[str, str, str]:
    """Classify a local graph pattern without claiming execution safety."""
    metadata = join_metadata or {}
    ops = set(op_types)
    direct_ops = {"Gemm", "MatMul", "Conv"}
    activations = {"Gelu", "Relu", "QuickGelu", "Erf", "Tanh", "Sigmoid"}
    shape_ops = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Gather", "Slice", "Concat", "Split"}

    if subgraph_kind == "join":
        if metadata.get("is_residual_like") or metadata.get("add_kind") == "residual_add":
            return (
                "residual_like",
                "high",
                "Residual-style join candidates require equal hidden shapes across merged branches.",
            )
        return (
            "join_like",
            "high" if metadata.get("join_op_type") in {"Concat", "Sum"} else "medium",
            "A branch merge requires dimension reasoning across all joined values before pruning.",
        )
    if "MatMul" in ops and "Softmax" in ops:
        return (
            "attention_like",
            "high",
            "MatMul and Softmax indicate attention-like dataflow requiring head/sequence mapping.",
        )
    if direct_ops & ops and {"Reshape", "Transpose"} & ops:
        return (
            "attention_like",
            "high",
            "Projection adjacent to reshape/transpose requires explicit head or axis mapping.",
        )
    projected = [index for index, op in enumerate(op_types) if op in direct_ops]
    if len(projected) >= 2 and any(op in activations for op in op_types[projected[0] + 1 : projected[-1]]):
        return (
            "mlp_like",
            "medium",
            "Projection-activation-projection structure may couple intermediate producer and consumer dimensions.",
        )
    add_kinds = metadata.get("add_kinds", [])
    for index, op_type in enumerate(op_types[:-1]):
        if op_type in {"Add", "Sum"} and op_types[index + 1] == "LayerNormalization":
            if op_type == "Add" and "bias_add" in add_kinds:
                return (
                    "normalization_like",
                    "medium",
                    "Bias addition near normalization is not sufficient evidence for a residual branch merge.",
                )
            return (
                "residual_like",
                "high",
                "Join followed by normalization is a residual candidate; join-aware evidence is required.",
            )
    if "LayerNormalization" in ops:
        return (
            "normalization_like",
            "medium",
            "Normalization parameters and hidden dimension may require coordinated propagation.",
        )
    if shape_ops & ops:
        return (
            "shape_transform",
            "high",
            "Shape-changing operations require an explicit dimension-to-axis mapping for pruning propagation.",
        )
    if direct_ops & ops:
        return (
            "directly_prunable",
            "medium" if len(op_types) > 1 else "low",
            "Parameterized projection or convolution is a direct pruning surface, subject to its consumers.",
        )
    if "Add" in ops or "Sum" in ops or "Concat" in ops:
        return (
            "join_like",
            "medium",
            "This path passes through a possible merge; use the join-centered report for branch semantics.",
        )
    return ("unknown", "unknown", "No pruning-specific local pattern is established by these operations.")


def enumerate_node_path_subgraphs(
    adjacency: OnnxGraphAdjacency,
    onnx_summary: dict,
    model_name: str,
    max_nodes: int = 5,
    max_subgraphs_per_size: int | None = None,
) -> list[NodePathSubgraph]:
    """Enumerate directed simple node paths for lengths one through ``max_nodes``."""
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")
    paths_by_size: dict[int, list[list[str]]] = defaultdict(list)

    def record(path: list[str]) -> None:
        size = len(path)
        if max_subgraphs_per_size is None or len(paths_by_size[size]) < max_subgraphs_per_size:
            paths_by_size[size].append(list(path))

    def visit(path: list[str]) -> None:
        record(path)
        if len(path) >= max_nodes:
            return
        for successor in adjacency.successors.get(path[-1], []):
            if successor not in path:
                visit(path + [successor])

    for node_name in adjacency.node_names:
        visit([node_name])

    subgraphs = []
    for size in range(1, max_nodes + 1):
        for counter, node_names in enumerate(paths_by_size.get(size, []), start=1):
            op_types = [adjacency.node_by_name[name].get("op_type", "") for name in node_names]
            tensor_sets = compute_subgraph_tensor_sets(node_names, adjacency, onnx_summary)
            add_kinds = [
                classify_add_node_kind(adjacency.node_by_name[name], adjacency, onnx_summary)[0]
                for name in node_names
                if adjacency.node_by_name[name].get("op_type") == "Add"
            ]
            pruning_class, risk_level, reason = classify_subgraph_pattern(
                op_types,
                join_metadata={"add_kinds": add_kinds},
            )
            subgraphs.append(
                NodePathSubgraph(
                    subgraph_id=f"path_{size}_{counter:06d}",
                    model_name=model_name,
                    subgraph_kind="path",
                    size=size,
                    node_names=node_names,
                    op_types=op_types,
                    pattern=" -> ".join(op_types),
                    metadata={
                        "pruning_class": pruning_class,
                        "risk_level": risk_level,
                        "classification_reason": reason,
                        "add_kinds": add_kinds,
                    },
                    **tensor_sets,
                )
            )
    return subgraphs


def is_join_node(node: dict) -> bool:
    """Identify an op that may merge multiple dataflow branches."""
    return node.get("op_type") in {"Add", "Sum", "Concat", "Where"} and len(
        [value for value in node.get("inputs", []) if value]
    ) >= 2


def _shape_map(onnx_summary: dict) -> dict[str, list[Any]]:
    return onnx_summary.get("tensor_shape_map", {}) or onnx_summary.get("value_info_shapes", {})


def classify_add_node_kind(
    node: dict,
    adjacency: OnnxGraphAdjacency,
    onnx_summary: dict,
) -> tuple[str, str, str]:
    """Distinguish bias additions from residual-style Add candidates."""
    if node.get("op_type") != "Add":
        return ("unknown_add", "low", "Node is not an Add operation.")
    inputs = [tensor for tensor in node.get("inputs", []) if tensor]
    initializer_names = set(adjacency.initializers)
    initializer_inputs = [tensor for tensor in inputs if tensor in initializer_names]
    non_initializer_inputs = [tensor for tensor in inputs if tensor not in initializer_names]
    if initializer_inputs:
        return ("bias_add", "high", "At least one Add input is an initializer, consistent with bias addition.")
    if len(non_initializer_inputs) < 2:
        return ("unknown_add", "low", "Fewer than two non-initializer inputs prevent branch-join interpretation.")

    successor_ops = {
        adjacency.node_by_name[name].get("op_type", "")
        for name in adjacency.successors.get(node["name"], [])
    }
    produced_or_input = [
        tensor in adjacency.producer_of_tensor or tensor in set(adjacency.graph_inputs)
        for tensor in non_initializer_inputs
    ]
    shapes = _shape_map(onnx_summary)
    known_shapes = [shapes.get(tensor) for tensor in non_initializer_inputs if shapes.get(tensor)]
    same_shape = len(known_shapes) >= 2 and all(shape == known_shapes[0] for shape in known_shapes[1:])
    producers = [
        adjacency.producer_of_tensor.get(tensor)
        for tensor in non_initializer_inputs
        if adjacency.producer_of_tensor.get(tensor)
    ]
    order = {name: index for index, name in enumerate(adjacency.node_names)}
    distant_producers = (
        len(producers) >= 2
        and max(order.get(name, 0) for name in producers) - min(order.get(name, 0) for name in producers) > 2
    )

    if "LayerNormalization" in successor_ops:
        return (
            "residual_add",
            "high",
            "Two dataflow inputs merge before LayerNormalization, a strong residual-style join signal.",
        )
    if all(produced_or_input) and same_shape:
        return (
            "residual_add",
            "medium",
            "Two non-initializer inputs with matching known shapes form a residual-style Add candidate.",
        )
    if all(produced_or_input) and distant_producers:
        return (
            "residual_add",
            "medium",
            "Add inputs arrive from separated graph regions, suggesting a skip/transformed branch merge.",
        )
    if all(produced_or_input):
        return (
            "elementwise_add",
            "low",
            "Multiple dataflow values merge, but residual identity or equal-shape evidence is incomplete.",
        )
    return ("unknown_add", "low", "Input provenance is insufficient to classify the Add operation.")


def _walk_context(
    starts: list[str],
    edges: dict[str, list[str]],
    depth: int,
) -> list[str]:
    visited: set[str] = set()
    frontier = list(starts)
    for _ in range(max(depth, 0)):
        next_frontier: list[str] = []
        for name in frontier:
            if name not in visited:
                visited.add(name)
                next_frontier.extend(edges.get(name, []))
        frontier = next_frontier
    return list(visited)


def enumerate_join_subgraphs(
    adjacency: OnnxGraphAdjacency,
    onnx_summary: dict,
    model_name: str,
    branch_depth: int = 2,
    post_join_depth: int = 2,
    max_join_subgraphs: int | None = None,
) -> list[JoinSubgraph]:
    """Enumerate merge-centered regions separately from ordinary node paths."""
    initializer_names = set(adjacency.initializers)
    order = {name: index for index, name in enumerate(adjacency.node_names)}
    result = []
    for node in (adjacency.node_by_name[name] for name in adjacency.node_names):
        if not is_join_node(node):
            continue
        non_initializer_inputs = [
            tensor for tensor in node.get("inputs", []) if tensor and tensor not in initializer_names
        ]
        if len(non_initializer_inputs) < 2:
            continue
        join_name = node["name"]
        branch_producers = _ordered_unique(
            [
                adjacency.producer_of_tensor[tensor]
                for tensor in non_initializer_inputs
                if tensor in adjacency.producer_of_tensor
            ]
        )
        branch_context = _walk_context(branch_producers, adjacency.predecessors, branch_depth)
        post_nodes = _walk_context(adjacency.successors.get(join_name, []), adjacency.successors, post_join_depth)
        branch_context = sorted(branch_context, key=lambda name: (order.get(name, len(order)), name))
        post_nodes = sorted(post_nodes, key=lambda name: (order.get(name, len(order)), name))
        node_names = _ordered_unique([*branch_context, join_name, *post_nodes])
        tensor_sets = compute_subgraph_tensor_sets(node_names, adjacency, onnx_summary)
        add_kind = "not_add"
        add_confidence = "low"
        add_reason = "Join operation is not Add."
        if node.get("op_type") == "Add":
            add_kind, add_confidence, add_reason = classify_add_node_kind(node, adjacency, onnx_summary)
        post_ops = [adjacency.node_by_name[name].get("op_type", "") for name in post_nodes]
        residual_like = (
            node.get("op_type") in {"Add", "Sum"}
            and (add_kind == "residual_add" or "LayerNormalization" in post_ops)
        )
        residual_confidence = add_confidence if residual_like else "low"
        residual_reason = add_reason if node.get("op_type") == "Add" else (
            "Sum followed by LayerNormalization is a residual-style join candidate."
            if residual_like
            else "No residual-style evidence was established for this join."
        )
        branch_patterns = []
        for tensor in non_initializer_inputs:
            producer = adjacency.producer_of_tensor.get(tensor)
            branch_patterns.append(
                adjacency.node_by_name[producer].get("op_type", "") if producer else "GraphInput"
            )
        post_pattern = f" -> {' -> '.join(post_ops)}" if post_ops else ""
        pattern = f"Join({node.get('op_type', '')})[{', '.join(branch_patterns)}]{post_pattern}"
        classification = classify_subgraph_pattern(
            [adjacency.node_by_name[name].get("op_type", "") for name in node_names],
            "join",
            {
                "is_residual_like": residual_like,
                "add_kind": add_kind,
                "join_op_type": node.get("op_type", ""),
            },
        )
        result.append(
            JoinSubgraph(
                subgraph_id=f"join_{len(result) + 1:06d}",
                model_name=model_name,
                subgraph_kind="join",
                join_node=join_name,
                join_op_type=node.get("op_type", ""),
                branch_producer_nodes=branch_producers,
                branch_input_tensors=non_initializer_inputs,
                branch_depth=branch_depth,
                post_join_nodes=post_nodes,
                post_join_depth=post_join_depth,
                node_names=node_names,
                op_types=[adjacency.node_by_name[name].get("op_type", "") for name in node_names],
                pattern=pattern,
                is_residual_like=residual_like,
                residual_confidence=residual_confidence,
                residual_reason=residual_reason,
                metadata={
                    "add_kind": add_kind,
                    "add_confidence": add_confidence,
                    "add_reason": add_reason,
                    "is_residual_like": residual_like,
                    "pruning_class": classification[0],
                    "risk_level": classification[1],
                    "classification_reason": classification[2],
                },
                **tensor_sets,
            )
        )
        if max_join_subgraphs is not None and len(result) >= max_join_subgraphs:
            break
    return result


def generate_subgraph_pruning_evidence(
    path_subgraphs: list[NodePathSubgraph],
    join_subgraphs: list[JoinSubgraph],
    onnx_summary: dict,
) -> list[SubgraphPruningEvidence]:
    """Generate report-level evidence from local structural patterns."""
    del onnx_summary
    evidence: list[SubgraphPruningEvidence] = []

    def add(
        subgraph: NodePathSubgraph | JoinSubgraph,
        evidence_type: str,
        suggested_constraint_type: str | None,
        confidence: str,
        reason: str,
    ) -> None:
        evidence.append(
            SubgraphPruningEvidence(
                evidence_id=f"evidence_{len(evidence) + 1:06d}",
                subgraph_id=subgraph.subgraph_id,
                subgraph_kind=subgraph.subgraph_kind,
                pattern=subgraph.pattern,
                evidence_type=evidence_type,
                affected_nodes=subgraph.node_names,
                affected_tensors=subgraph.boundary_input_tensors + subgraph.boundary_output_tensors,
                suggested_constraint_type=suggested_constraint_type,
                confidence=confidence,
                reason=reason,
            )
        )

    for subgraph in path_subgraphs:
        ops = set(subgraph.op_types)
        pruning_class = subgraph.metadata.get("pruning_class")
        if {"Gemm", "MatMul", "Conv"} & ops:
            add(
                subgraph,
                "direct_prunable_op",
                None,
                "medium",
                "Parameterized ONNX operation is locally pruning-relevant, subject to consumers and joins.",
            )
        if pruning_class == "mlp_like":
            add(
                subgraph,
                "mlp_intermediate_coupling",
                "mlp_same_intermediate_indices",
                "medium",
                "Projection-activation-projection pattern suggests a shared MLP intermediate pruning dimension.",
            )
        if pruning_class == "attention_like":
            add(
                subgraph,
                "attention_head_mapping_required",
                "reshape_preservation" if {"Reshape", "Transpose"} & ops else "qkv_same_heads",
                "low",
                "Attention-like pattern requires explicit head and axis mapping before pruning propagation.",
            )
        if {"Reshape", "Transpose"} & ops:
            add(
                subgraph,
                "reshape_dimension_mapping",
                "reshape_preservation",
                "low",
                "Shape-transform path requires a mapping between tensor axes and symbolic pruning dimensions.",
            )
        if {"Squeeze", "Unsqueeze", "Flatten", "Slice", "Split", "Concat"} & ops:
            add(
                subgraph,
                "unknown_mapping",
                "unknown_mapping",
                "low",
                "Complex shape or partition operation requires explicit axis semantics before dimension propagation.",
            )
        if "Gather" in ops and subgraph.contains_initializers:
            add(
                subgraph,
                "embedding_lookup",
                "tied_parameter",
                "low",
                "Initializer-backed Gather is embedding-like; output tying and embedding dimensions remain uncertain.",
            )
        for index, op_type in enumerate(subgraph.op_types[:-1]):
            if op_type in {"Add", "Sum"} and subgraph.op_types[index + 1] == "LayerNormalization":
                add(
                    subgraph,
                    "layernorm_hidden_dependency",
                    "layernorm_hidden_equality",
                    "medium",
                    "Join-adjacent normalization preserves a hidden-size dependency requiring branch-aware review.",
                )

    for subgraph in join_subgraphs:
        if subgraph.is_residual_like:
            add(
                subgraph,
                "residual_hidden_equality",
                "residual_equal_shape",
                "high" if subgraph.residual_confidence == "high" else "medium",
                "Residual-style join requires equal hidden shape across skip and transformed branches.",
            )
            if "LayerNormalization" in subgraph.op_types:
                add(
                    subgraph,
                    "layernorm_hidden_dependency",
                    "layernorm_hidden_equality",
                    "medium",
                    "LayerNormalization after a residual-style join retains the merged hidden dimension.",
                )
        else:
            add(
                subgraph,
                "join_dimension_equality",
                "unknown_mapping",
                "low",
                "Join branches require a dimension compatibility analysis that is not established locally.",
            )
    return evidence


def summarize_subgraph_patterns(
    path_subgraphs: list[NodePathSubgraph],
    join_subgraphs: list[JoinSubgraph],
) -> list[SubgraphPatternSummary]:
    """Aggregate repeated local patterns while retaining path/join separation."""
    grouped: dict[tuple[str, int | None, str], list[Any]] = defaultdict(list)
    for subgraph in path_subgraphs:
        grouped[("path", subgraph.size, subgraph.pattern)].append(subgraph)
    for subgraph in join_subgraphs:
        grouped[("join", None, subgraph.pattern)].append(subgraph)
    summaries = []
    for (kind, size, pattern), subgraphs in sorted(grouped.items(), key=lambda item: item[0]):
        first = subgraphs[0]
        pruning_class, risk_level, reason = classify_subgraph_pattern(
            first.op_types,
            kind,
            first.metadata,
        )
        summaries.append(
            SubgraphPatternSummary(
                pattern=pattern,
                size=size,
                subgraph_kind=kind,
                count=len(subgraphs),
                example_subgraph_ids=[item.subgraph_id for item in subgraphs[:5]],
                pruning_class=pruning_class,
                risk_level=risk_level,
                reason=reason,
            )
        )
    return summaries


def build_subgraph_analysis_report(
    onnx_summary: dict,
    model_config: dict,
    max_nodes: int = 5,
    max_subgraphs_per_size: int | None = None,
    branch_depth: int = 2,
    post_join_depth: int = 2,
    max_join_subgraphs: int | None = None,
) -> SubgraphAnalysisReport:
    """Build the full local path and join-aware structural analysis report."""
    model_name = model_config.get("name") or onnx_summary.get("model_name", "")
    adjacency = build_onnx_adjacency(onnx_summary)
    paths = enumerate_node_path_subgraphs(
        adjacency,
        onnx_summary,
        model_name,
        max_nodes,
        max_subgraphs_per_size,
    )
    joins = enumerate_join_subgraphs(
        adjacency,
        onnx_summary,
        model_name,
        branch_depth,
        post_join_depth,
        max_join_subgraphs,
    )
    patterns = summarize_subgraph_patterns(paths, joins)
    evidence = generate_subgraph_pruning_evidence(paths, joins, onnx_summary)
    adds = [
        classify_add_node_kind(node, adjacency, onnx_summary)[0]
        for node in onnx_summary.get("nodes", [])
        if node.get("op_type") == "Add"
    ]
    class_counts = Counter(item.pruning_class for item in patterns for _ in range(item.count))
    risk_counts = Counter(item.risk_level for item in patterns for _ in range(item.count))
    evidence_counts = Counter(item.evidence_type for item in evidence)
    path_size_counts = Counter(item.size for item in paths)
    top_patterns = sorted(patterns, key=lambda item: (-item.count, item.subgraph_kind, item.pattern))[:10]
    summary = {
        "num_path_subgraphs": len(paths),
        "num_join_subgraphs": len(joins),
        "num_residual_like_join_subgraphs": sum(item.is_residual_like for item in joins),
        "num_patterns": len(patterns),
        "path_subgraph_count_by_size": {str(key): value for key, value in sorted(path_size_counts.items())},
        "top_patterns_by_count": [
            {
                "pattern": item.pattern,
                "subgraph_kind": item.subgraph_kind,
                "size": item.size,
                "count": item.count,
            }
            for item in top_patterns
        ],
        "pruning_class_counts": dict(class_counts),
        "risk_level_counts": dict(risk_counts),
        "evidence_type_counts": dict(evidence_counts),
        "directly_prunable_pattern_count": class_counts.get("directly_prunable", 0),
        "mlp_like_pattern_count": class_counts.get("mlp_like", 0),
        "attention_like_pattern_count": class_counts.get("attention_like", 0),
        "residual_like_pattern_count": class_counts.get("residual_like", 0),
        "join_like_pattern_count": class_counts.get("join_like", 0),
        "shape_transform_pattern_count": class_counts.get("shape_transform", 0),
        "bias_add_count": adds.count("bias_add"),
        "residual_add_count": adds.count("residual_add"),
        "elementwise_add_count": adds.count("elementwise_add"),
        "unknown_add_count": adds.count("unknown_add"),
    }
    return SubgraphAnalysisReport(
        model_name=model_name,
        hf_id=model_config.get("hf_id") or onnx_summary.get("hf_id", ""),
        task=model_config.get("task") or onnx_summary.get("task", ""),
        max_nodes=max_nodes,
        path_subgraphs=paths,
        join_subgraphs=joins,
        pattern_summaries=patterns,
        pruning_evidence=evidence,
        summary=summary,
        metadata={
            "branch_depth": branch_depth,
            "post_join_depth": post_join_depth,
            "max_subgraphs_per_size": max_subgraphs_per_size,
            "max_join_subgraphs": max_join_subgraphs,
            "source_onnx_summary": onnx_summary.get("onnx_path", ""),
        },
    )


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 200) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if len(rows) > limit:
        lines.append(f"| ... | {len(rows) - limit} more rows omitted |" + " |" * (len(columns) - 2))
    return "\n".join(lines)


def subgraph_analysis_report_to_markdown(report: SubgraphAnalysisReport | dict) -> str:
    data = subgraph_analysis_report_to_dict(report) if isinstance(report, SubgraphAnalysisReport) else report
    summary = data.get("summary", {})
    return "\n".join(
        [
            f"# Subgraph Structural Analysis: {data.get('model_name', '')}",
            "",
            "## Summary",
            "",
            f"- Directed path subgraphs: `{summary.get('num_path_subgraphs', 0)}`",
            f"- Join-centered subgraphs: `{summary.get('num_join_subgraphs', 0)}`",
            f"- Residual-like joins: `{summary.get('num_residual_like_join_subgraphs', 0)}`",
            f"- Distinct patterns: `{summary.get('num_patterns', 0)}`",
            f"- Bias Add nodes: `{summary.get('bias_add_count', 0)}`",
            f"- Residual Add candidates: `{summary.get('residual_add_count', 0)}`",
            "",
            "## Path Count By Size",
            "",
            _table(
                [{"size": key, "count": value} for key, value in summary.get("path_subgraph_count_by_size", {}).items()],
                ["size", "count"],
            ),
            "",
            "## Top Patterns",
            "",
            _table(summary.get("top_patterns_by_count", []), ["subgraph_kind", "size", "pattern", "count"]),
            "",
            "## Join-Centered Subgraphs",
            "",
            _table(
                data.get("join_subgraphs", []),
                ["subgraph_id", "join_node", "join_op_type", "pattern", "is_residual_like", "residual_confidence", "residual_reason"],
            ),
            "",
            "## Interpretation",
            "",
            "Directed paths capture local sequential structure. Join-centered subgraphs separately preserve branch-merge semantics, so residual Add candidates are not treated as ordinary linear paths or bias additions.",
            "",
            "This is static structural evidence only. It does not modify models or establish pruning legality by itself.",
            "",
        ]
    )


def pattern_summaries_to_markdown(report: SubgraphAnalysisReport | dict) -> str:
    data = subgraph_analysis_report_to_dict(report) if isinstance(report, SubgraphAnalysisReport) else report
    return "\n".join(
        [
            f"# Subgraph Pattern Summary: {data.get('model_name', '')}",
            "",
            _table(
                data.get("pattern_summaries", []),
                ["subgraph_kind", "size", "pattern", "count", "pruning_class", "risk_level", "reason"],
                limit=500,
            ),
            "",
            "Path and join patterns are kept separate because a merge operation carries branch constraints that a single directed path cannot represent.",
            "",
        ]
    )


def pruning_evidence_to_markdown(report: SubgraphAnalysisReport | dict) -> str:
    data = subgraph_analysis_report_to_dict(report) if isinstance(report, SubgraphAnalysisReport) else report
    return "\n".join(
        [
            f"# Subgraph Pruning Evidence: {data.get('model_name', '')}",
            "",
            _table(
                data.get("pruning_evidence", []),
                ["evidence_id", "subgraph_id", "subgraph_kind", "evidence_type", "suggested_constraint_type", "confidence", "reason"],
                limit=500,
            ),
            "",
            "## Interpretation",
            "",
            "Evidence is intended for later refinement of pruning maps and Dimension IR. It is not an instruction to execute pruning.",
            "",
        ]
    )


def join_subgraphs_to_markdown(
    report: SubgraphAnalysisReport | dict,
    residual_only: bool = False,
) -> str:
    data = subgraph_analysis_report_to_dict(report) if isinstance(report, SubgraphAnalysisReport) else report
    joins = data.get("join_subgraphs", [])
    if residual_only:
        joins = [item for item in joins if item.get("is_residual_like")]
    title = "Residual Subgraphs" if residual_only else "Join-Centered Subgraphs"
    return "\n".join(
        [
            f"# {title}: {data.get('model_name', '')}",
            "",
            _table(
                joins,
                ["subgraph_id", "join_node", "join_op_type", "pattern", "branch_producer_nodes", "post_join_nodes", "is_residual_like", "residual_confidence", "residual_reason"],
                limit=500,
            ),
            "",
            "Residual candidates preserve merge semantics explicitly; bias Add nodes are excluded from this report.",
            "",
        ]
    )
