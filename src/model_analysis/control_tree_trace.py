"""Stepwise construction trace for compiler-style dataflow control trees."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir
from model_analysis.structural_region_detection import detect_region_candidates


@dataclass
class ControlTreeGraphNode:
    node_id: str
    node_kind: str
    label: str
    region_type: str | None
    op_type: str | None
    canonical_op_type: str | None
    source_op_ids: list[str]
    source_region_ids: list[str]
    parent_region_id: str | None
    confidence: str
    pruning_role: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlTreeGraphEdge:
    src: str
    dst: str
    edge_kind: str
    tensor_or_value_id: str | None
    label: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlTreeStep:
    step_id: str
    step_index: int
    pass_name: str
    action: str
    created_region_id: str | None
    created_region_type: str | None
    collapsed_node_ids: list[str]
    collapsed_op_ids: list[str]
    collapsed_region_ids: list[str]
    input_boundary_values: list[str]
    output_boundary_values: list[str]
    reason: str
    confidence: str
    before_summary: dict[str, Any]
    after_summary: dict[str, Any]
    graph_snapshot: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlTreeTrace:
    model_name: str
    source_frontend: str
    root_region_id: str | None
    steps: list[ControlTreeStep]
    final_region_tree_path: str | None
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingControlGraph:
    nodes: dict[str, ControlTreeGraphNode]
    edges: list[ControlTreeGraphEdge]
    active_nodes: set[str]
    source_op_to_active_node: dict[str, str]
    created_regions: dict[str, dict[str, Any]]


def control_tree_graph_node_to_dict(node: ControlTreeGraphNode | dict) -> dict[str, Any]:
    return asdict(node) if isinstance(node, ControlTreeGraphNode) else dict(node)


def control_tree_graph_edge_to_dict(edge: ControlTreeGraphEdge | dict) -> dict[str, Any]:
    return asdict(edge) if isinstance(edge, ControlTreeGraphEdge) else dict(edge)


def control_tree_step_to_dict(step: ControlTreeStep | dict) -> dict[str, Any]:
    return asdict(step) if isinstance(step, ControlTreeStep) else dict(step)


def control_tree_trace_to_dict(trace: ControlTreeTrace | dict) -> dict[str, Any]:
    return asdict(trace) if isinstance(trace, ControlTreeTrace) else dict(trace)


def write_control_tree_trace_json(trace: ControlTreeTrace | dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(control_tree_trace_to_dict(trace), indent=2), encoding="utf-8")


def load_control_tree_trace_json(path: Path) -> ControlTreeTrace:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ControlTreeTrace(
        model_name=data.get("model_name", ""),
        source_frontend=data.get("source_frontend", "unknown"),
        root_region_id=data.get("root_region_id"),
        steps=[ControlTreeStep(**item) for item in data.get("steps", [])],
        final_region_tree_path=data.get("final_region_tree_path"),
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def _as_dict_list(items: list[Any]) -> list[dict[str, Any]]:
    return [item if isinstance(item, dict) else asdict(item) for item in items]


def _op_order(tensor_graph: dict) -> dict[str, int]:
    return {op.get("op_id", ""): index for index, op in enumerate(tensor_graph.get("ops", []))}


def _node_id_for_op(op_id: str) -> str:
    return f"node::op::{op_id}"


def _edge_key(edge: ControlTreeGraphEdge) -> tuple[str, str, str, str | None, str | None]:
    return (edge.src, edge.dst, edge.edge_kind, edge.tensor_or_value_id, edge.label)


def _dedupe_edges(edges: list[ControlTreeGraphEdge]) -> list[ControlTreeGraphEdge]:
    seen: set[tuple[str, str, str, str | None, str | None]] = set()
    out: list[ControlTreeGraphEdge] = []
    for edge in edges:
        key = _edge_key(edge)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return sorted(out, key=lambda edge: (edge.src, edge.dst, edge.edge_kind, edge.tensor_or_value_id or ""))


def build_initial_working_graph_from_tensor_ir(tensor_graph: dict) -> WorkingControlGraph:
    """Represent TensorOps as active control-tree graph nodes."""
    nodes: dict[str, ControlTreeGraphNode] = {}
    edges: list[ControlTreeGraphEdge] = []
    source_op_to_active_node: dict[str, str] = {}
    ops_by_id = {op.get("op_id"): op for op in tensor_graph.get("ops", [])}

    for op in tensor_graph.get("ops", []):
        op_id = op.get("op_id", "")
        node_id = _node_id_for_op(op_id)
        source_op_to_active_node[op_id] = node_id
        label = op.get("name") or op.get("op_type") or op_id
        nodes[node_id] = ControlTreeGraphNode(
            node_id=node_id,
            node_kind="tensor_op",
            label=str(label),
            region_type=None,
            op_type=op.get("op_type"),
            canonical_op_type=op.get("canonical_op_type"),
            source_op_ids=[op_id],
            source_region_ids=[],
            parent_region_id=None,
            confidence="high",
            pruning_role=None,
            metadata={
                "active": True,
                "source_frontend": op.get("source_frontend"),
                "source_node_name": op.get("source_node_name"),
                "is_fork": op.get("is_fork", False),
                "is_join": op.get("is_join", False),
                "region_hint": op.get("region_hint"),
            },
        )

    for value in tensor_graph.get("values", []):
        producer = value.get("producer")
        if not producer or producer not in ops_by_id:
            continue
        for consumer in value.get("consumers", []):
            if consumer not in ops_by_id:
                continue
            edges.append(
                ControlTreeGraphEdge(
                    src=_node_id_for_op(producer),
                    dst=_node_id_for_op(consumer),
                    edge_kind="dataflow",
                    tensor_or_value_id=value.get("value_id"),
                    label=value.get("name") or value.get("value_id"),
                    metadata={"semantic_role": value.get("semantic_role")},
                )
            )

    active_nodes = set(nodes)
    return WorkingControlGraph(
        nodes=nodes,
        edges=_dedupe_edges(edges),
        active_nodes=active_nodes,
        source_op_to_active_node=source_op_to_active_node,
        created_regions={},
    )


def summarize_working_graph(graph: WorkingControlGraph) -> dict[str, Any]:
    active = graph.active_nodes
    active_edges = [edge for edge in graph.edges if edge.src in active and edge.dst in active]
    active_node_kinds = Counter(graph.nodes[node_id].node_kind for node_id in active)
    active_region_types = Counter(
        graph.nodes[node_id].region_type or "tensor_op"
        for node_id in active
    )
    return {
        "num_nodes": len(graph.nodes),
        "num_active_nodes": len(active),
        "num_edges": len(graph.edges),
        "num_active_edges": len(active_edges),
        "active_node_kind_counts": dict(sorted(active_node_kinds.items())),
        "active_region_type_counts": dict(sorted(active_region_types.items())),
        "num_created_regions": len(graph.created_regions),
    }


def snapshot_working_graph(graph: WorkingControlGraph, max_nodes: int | None = None) -> dict[str, Any]:
    active_ids = sorted(graph.active_nodes)
    truncated = False
    if max_nodes is not None and len(active_ids) > max_nodes:
        active_ids = active_ids[:max_nodes]
        truncated = True
    selected = set(active_ids)
    nodes = [control_tree_graph_node_to_dict(graph.nodes[node_id]) for node_id in active_ids]
    edges = [
        control_tree_graph_edge_to_dict(edge)
        for edge in graph.edges
        if edge.src in selected and edge.dst in selected
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "truncated": truncated,
        "active_node_count": len(graph.active_nodes),
        "returned_node_count": len(nodes),
    }


def collapse_nodes_into_region(
    graph: WorkingControlGraph,
    node_ids: list[str],
    region_id: str,
    region_type: str,
    label: str,
    confidence: str,
    pruning_role: str | None,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[WorkingControlGraph, dict[str, Any]]:
    """Collapse active nodes into one abstract-region node and redirect boundary edges."""
    collapsed = [node_id for node_id in dict.fromkeys(node_ids) if node_id in graph.active_nodes]
    collapsed_set = set(collapsed)
    source_op_ids = sorted({
        op_id
        for node_id in collapsed
        for op_id in graph.nodes[node_id].source_op_ids
    })
    source_region_ids = sorted({
        region
        for node_id in collapsed
        for region in graph.nodes[node_id].source_region_ids
    } | ({region_id} if region_id else set()))
    input_boundary_values: list[str] = []
    output_boundary_values: list[str] = []
    new_edges: list[ControlTreeGraphEdge] = []

    for edge in graph.edges:
        src_inside = edge.src in collapsed_set
        dst_inside = edge.dst in collapsed_set
        if src_inside and dst_inside:
            continue
        if dst_inside and not src_inside:
            input_boundary_values.append(edge.tensor_or_value_id or "")
            new_edges.append(
                ControlTreeGraphEdge(
                    src=edge.src,
                    dst=region_id,
                    edge_kind=edge.edge_kind,
                    tensor_or_value_id=edge.tensor_or_value_id,
                    label=edge.label,
                    metadata={**edge.metadata, "redirected_by": region_id},
                )
            )
        elif src_inside and not dst_inside:
            output_boundary_values.append(edge.tensor_or_value_id or "")
            new_edges.append(
                ControlTreeGraphEdge(
                    src=region_id,
                    dst=edge.dst,
                    edge_kind=edge.edge_kind,
                    tensor_or_value_id=edge.tensor_or_value_id,
                    label=edge.label,
                    metadata={**edge.metadata, "redirected_by": region_id},
                )
            )
        else:
            new_edges.append(edge)

    for node_id in collapsed:
        graph.nodes[node_id].metadata["active"] = False
        graph.nodes[node_id].parent_region_id = region_id

    region_node = ControlTreeGraphNode(
        node_id=region_id,
        node_kind="abstract_region",
        label=label,
        region_type=region_type,
        op_type=None,
        canonical_op_type=None,
        source_op_ids=source_op_ids,
        source_region_ids=source_region_ids,
        parent_region_id=None,
        confidence=confidence,
        pruning_role=pruning_role,
        metadata={
            "active": True,
            "reason": reason,
            "collapsed_node_ids": collapsed,
            **(metadata or {}),
        },
    )
    graph.nodes[region_id] = region_node
    graph.active_nodes.difference_update(collapsed_set)
    graph.active_nodes.add(region_id)
    graph.edges = _dedupe_edges(new_edges)
    for op_id in source_op_ids:
        graph.source_op_to_active_node[op_id] = region_id
    graph.created_regions[region_id] = control_tree_graph_node_to_dict(region_node)

    collapse_summary = {
        "collapsed_node_ids": collapsed,
        "collapsed_op_ids": source_op_ids,
        "collapsed_region_ids": sorted({
            region_id
            for node_id in collapsed
            for region_id in graph.nodes[node_id].source_region_ids
        }),
        "input_boundary_values": sorted({item for item in input_boundary_values if item}),
        "output_boundary_values": sorted({item for item in output_boundary_values if item}),
    }
    return graph, collapse_summary


def initialize_control_tree_trace(tensor_graph: dict) -> tuple[WorkingControlGraph, ControlTreeStep]:
    graph = build_initial_working_graph_from_tensor_ir(tensor_graph)
    after = summarize_working_graph(graph)
    step = ControlTreeStep(
        step_id="step_000000",
        step_index=0,
        pass_name="initialize_primitives",
        action="initialize",
        created_region_id=None,
        created_region_type=None,
        collapsed_node_ids=[],
        collapsed_op_ids=[],
        collapsed_region_ids=[],
        input_boundary_values=[],
        output_boundary_values=[],
        reason="Initialize one active graph node for each primitive TensorOp.",
        confidence="high",
        before_summary={},
        after_summary=after,
        graph_snapshot=snapshot_working_graph(graph),
        metadata={},
    )
    return graph, step


_REGION_PRIORITY = {
    "LinearProjectionRegion": 2,
    "ActivationRegion": 3,
    "FeedForwardRegion": 4,
    "ResidualMergeRegion": 5,
    "LayerNormRegion": 6,
    "AxisTransformRegion": 7,
    "AttentionSkeletonRegion": 8,
    "ForkRegion": 9,
    "JoinRegion": 10,
    "BiasAddRegion": 11,
    "ProperAcyclicRegion": 12,
    "UnknownRegion": 13,
    "PrimitiveRegion": 99,
}


def _pass_name(region_type: str, source: str, metadata: dict[str, Any] | None = None) -> str:
    if region_type == "ActivationRegion" and source == "semantic_fusion":
        return "semantic_fusion_gelu"
    return {
        "LinearProjectionRegion": "collapse_linear_projection",
        "BiasAddRegion": "collapse_bias_add",
        "ActivationRegion": "collapse_activation",
        "FeedForwardRegion": "collapse_feedforward",
        "ResidualMergeRegion": "collapse_residual_merge",
        "LayerNormRegion": "collapse_layer_norm",
        "AxisTransformRegion": "collapse_axis_transform",
        "AttentionSkeletonRegion": "collapse_attention_skeleton",
        "ForkRegion": "collapse_fork_join",
        "JoinRegion": "collapse_fork_join",
        "ProperAcyclicRegion": "collapse_proper_acyclic",
    }.get(region_type, "collapse_unknown")


def _pruning_roles(structural_region_tree: dict | None) -> dict[str, str]:
    if not structural_region_tree:
        return {}
    return {
        item.get("region_id", ""): item.get("pruning_role", "unknown")
        for item in structural_region_tree.get("interfaces", [])
    }


def _candidate_from_region(region: dict[str, Any], pruning_roles: dict[str, str], source: str) -> dict[str, Any]:
    metadata = region.get("metadata", {}) or {}
    region_type = region.get("region_type", "UnknownRegion")
    return {
        "candidate_id": f"{source}::{region.get('region_id')}",
        "region_type": region_type,
        "op_ids": list(region.get("op_ids", [])),
        "region_id": region.get("region_id"),
        "confidence": region.get("confidence", "medium"),
        "reason": region.get("reason", ""),
        "pass_name": _pass_name(region_type, source, metadata),
        "pruning_role": pruning_roles.get(region.get("region_id", ""), metadata.get("pruning_role")),
        "priority": _REGION_PRIORITY.get(region_type, 50),
        "source": source,
        "metadata": metadata,
    }


def _semantic_fusion_candidates(report: dict | None) -> list[dict[str, Any]]:
    if not report:
        return []
    candidates: list[dict[str, Any]] = []
    for item in report.get("fusions", []):
        fusion_type = item.get("fusion_type")
        if fusion_type == "GeluActivation":
            region_type = "ActivationRegion"
            priority = 1
        elif fusion_type == "FeedForward":
            region_type = "FeedForwardRegion"
            priority = 4
        else:
            continue
        metadata = {"semantic_fusion": True, **(item.get("metadata", {}) or {})}
        candidates.append(
            {
                "candidate_id": f"semantic_fusion::{item.get('fusion_id')}",
                "region_type": region_type,
                "op_ids": list(item.get("op_ids", [])),
                "region_id": f"trace::{item.get('fusion_id')}",
                "confidence": item.get("confidence", "medium"),
                "reason": item.get("reason", ""),
                "pass_name": _pass_name(region_type, "semantic_fusion", metadata),
                "pruning_role": "propagation_only" if region_type == "ActivationRegion" else "directly_prunable",
                "priority": priority,
                "source": "semantic_fusion",
                "metadata": metadata,
            }
        )
    return candidates


def build_ordered_region_candidates_for_trace(
    tensor_graph: dict,
    structural_region_tree: dict | None = None,
    semantic_fusion_report: dict | None = None,
) -> list[dict[str, Any]]:
    op_order = _op_order(tensor_graph)
    candidates: list[dict[str, Any]] = []
    candidates.extend(_semantic_fusion_candidates(semantic_fusion_report))

    if structural_region_tree:
        pruning_roles = _pruning_roles(structural_region_tree)
        for region in structural_region_tree.get("regions", []):
            region_type = region.get("region_type")
            if region_type in {"ModelRegion", "PrimitiveRegion"}:
                continue
            candidates.append(_candidate_from_region(region, pruning_roles, "structural_region_tree"))
    else:
        for region in detect_region_candidates(tensor_graph):
            region_dict = asdict(region)
            if region_dict.get("region_type") in {"ModelRegion", "PrimitiveRegion"}:
                continue
            candidates.append(_candidate_from_region(region_dict, {}, "detector"))

    def key(candidate: dict[str, Any]) -> tuple[int, int, str]:
        op_ids = candidate.get("op_ids", [])
        min_order = min((op_order.get(op_id, 10**9) for op_id in op_ids), default=10**9)
        source_bias = 0 if candidate.get("source") == "semantic_fusion" else 1
        return (candidate.get("priority", 50), min_order, f"{source_bias}:{candidate.get('region_id') or candidate.get('candidate_id')}")

    return sorted(candidates, key=key)


def _make_step(
    index: int,
    candidate: dict[str, Any],
    action: str,
    before: dict[str, Any],
    after: dict[str, Any],
    snapshot: dict[str, Any],
    collapse_info: dict[str, Any] | None = None,
    reason_override: str | None = None,
) -> ControlTreeStep:
    info = collapse_info or {}
    return ControlTreeStep(
        step_id=f"step_{index:06d}",
        step_index=index,
        pass_name=candidate.get("pass_name", "unknown"),
        action=action,
        created_region_id=candidate.get("region_id") if action in {"collapse", "finalize"} else None,
        created_region_type=candidate.get("region_type") if action in {"collapse", "finalize"} else None,
        collapsed_node_ids=list(info.get("collapsed_node_ids", [])),
        collapsed_op_ids=list(info.get("collapsed_op_ids", [])),
        collapsed_region_ids=list(info.get("collapsed_region_ids", [])),
        input_boundary_values=list(info.get("input_boundary_values", [])),
        output_boundary_values=list(info.get("output_boundary_values", [])),
        reason=reason_override or candidate.get("reason", ""),
        confidence=candidate.get("confidence", "medium"),
        before_summary=before,
        after_summary=after,
        graph_snapshot=snapshot,
        metadata={
            "candidate_id": candidate.get("candidate_id"),
            "source": candidate.get("source"),
            **(candidate.get("metadata", {}) or {}),
        },
    )


def _active_nodes_for_candidate(graph: WorkingControlGraph, op_ids: list[str]) -> list[str]:
    active: list[str] = []
    for op_id in op_ids:
        node_id = graph.source_op_to_active_node.get(op_id)
        if node_id and node_id in graph.active_nodes and node_id not in active:
            active.append(node_id)
    return active


def _trace_summary(trace: ControlTreeTrace, final_graph: WorkingControlGraph) -> dict[str, Any]:
    steps = [control_tree_step_to_dict(step) for step in trace.steps]
    created_types = Counter(
        step.get("created_region_type")
        for step in steps
        if step.get("created_region_type")
    )
    pass_names = Counter(step.get("pass_name", "unknown") for step in steps)
    return {
        "num_steps": len(steps),
        "num_collapse_steps": sum(1 for step in steps if step.get("action") == "collapse"),
        "num_skip_steps": sum(1 for step in steps if step.get("action") == "skip"),
        "num_finalize_steps": sum(1 for step in steps if step.get("action") == "finalize"),
        "created_region_type_counts": dict(sorted(created_types.items())),
        "pass_name_counts": dict(sorted(pass_names.items())),
        "final_active_node_count": len(final_graph.active_nodes),
        "initial_tensor_op_count": steps[0].get("after_summary", {}).get("num_active_nodes", 0) if steps else 0,
    }


def build_control_tree_trace(
    tensor_graph: dict,
    structural_region_tree: dict | None = None,
    semantic_fusion_report: dict | None = None,
    max_snapshot_nodes: int | None = 500,
) -> ControlTreeTrace:
    graph, init_step = initialize_control_tree_trace(tensor_graph)
    init_step.graph_snapshot = snapshot_working_graph(graph, max_snapshot_nodes)
    steps = [init_step]
    candidates = build_ordered_region_candidates_for_trace(
        tensor_graph,
        structural_region_tree=structural_region_tree,
        semantic_fusion_report=semantic_fusion_report,
    )

    step_index = 1
    for candidate in candidates:
        before = summarize_working_graph(graph)
        active_nodes = _active_nodes_for_candidate(graph, candidate.get("op_ids", []))
        region_id = candidate.get("region_id") or f"trace_region::{step_index:06d}"
        candidate["region_id"] = region_id
        if not active_nodes:
            after = summarize_working_graph(graph)
            steps.append(
                _make_step(
                    step_index,
                    candidate,
                    "skip",
                    before,
                    after,
                    snapshot_working_graph(graph, max_snapshot_nodes),
                    reason_override="No active nodes remain for this candidate; it was already consumed by earlier collapses.",
                )
            )
            step_index += 1
            continue
        if len(active_nodes) == 1 and graph.nodes[active_nodes[0]].node_kind == "abstract_region":
            after = summarize_working_graph(graph)
            steps.append(
                _make_step(
                    step_index,
                    candidate,
                    "skip",
                    before,
                    after,
                    snapshot_working_graph(graph, max_snapshot_nodes),
                    reason_override="Candidate is already represented by one active abstract region.",
                )
            )
            step_index += 1
            continue

        label = candidate.get("region_type", "Region")
        graph, collapse_info = collapse_nodes_into_region(
            graph,
            active_nodes,
            region_id,
            candidate.get("region_type", "UnknownRegion"),
            label,
            candidate.get("confidence", "medium"),
            candidate.get("pruning_role"),
            candidate.get("reason", ""),
            metadata={
                "candidate_id": candidate.get("candidate_id"),
                "source": candidate.get("source"),
                **(candidate.get("metadata", {}) or {}),
            },
        )
        after = summarize_working_graph(graph)
        steps.append(
            _make_step(
                step_index,
                candidate,
                "collapse",
                before,
                after,
                snapshot_working_graph(graph, max_snapshot_nodes),
                collapse_info,
            )
        )
        step_index += 1

    before = summarize_working_graph(graph)
    root_id = (structural_region_tree or {}).get("root_region_id") or "trace::model_region"
    final_candidate = {
        "candidate_id": "finalize_model_region",
        "region_type": "ModelRegion",
        "op_ids": sorted(graph.source_op_to_active_node),
        "region_id": root_id,
        "confidence": "high",
        "reason": "Finalize the remaining active abstract graph into the model root region.",
        "pass_name": "finalize_model_region",
        "pruning_role": "analysis_only",
        "source": "trace",
        "metadata": {},
    }
    if len(graph.active_nodes) > 1:
        graph, collapse_info = collapse_nodes_into_region(
            graph,
            sorted(graph.active_nodes),
            root_id,
            "ModelRegion",
            "ModelRegion",
            "high",
            "analysis_only",
            final_candidate["reason"],
            metadata={"finalize": True},
        )
        graph.nodes[root_id].node_kind = "model_root"
        action = "finalize"
    else:
        only_node = next(iter(graph.active_nodes), None)
        collapse_info = {
            "collapsed_node_ids": [only_node] if only_node else [],
            "collapsed_op_ids": graph.nodes[only_node].source_op_ids if only_node else [],
            "collapsed_region_ids": graph.nodes[only_node].source_region_ids if only_node else [],
            "input_boundary_values": [],
            "output_boundary_values": [],
        }
        action = "skip"
        final_candidate["reason"] = "Final model root collapse skipped because the graph already has one active node."
    after = summarize_working_graph(graph)
    steps.append(
        _make_step(
            step_index,
            final_candidate,
            action,
            before,
            after,
            snapshot_working_graph(graph, max_snapshot_nodes),
            collapse_info,
            reason_override=final_candidate["reason"],
        )
    )

    trace = ControlTreeTrace(
        model_name=tensor_graph.get("model_name", ""),
        source_frontend=tensor_graph.get("source_frontend", "unknown"),
        root_region_id=root_id if action == "finalize" else None,
        steps=steps,
        final_region_tree_path=None,
        metadata={
            "trace_kind": "explanatory_structural_analysis_trace",
            "uses_existing_structural_region_tree": structural_region_tree is not None,
            "uses_semantic_fusion_report": semantic_fusion_report is not None,
        },
    )
    trace.summary = _trace_summary(trace, graph)
    return trace


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 500) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(_cell(row.get(column, "")) for column in columns) + " |")
    if len(rows) > limit:
        lines.append("| ... | " + f"{len(rows) - limit} more rows omitted |" + " |" * (len(columns) - 2))
    return "\n".join(lines)


def control_tree_trace_to_markdown(trace: ControlTreeTrace | dict) -> str:
    data = control_tree_trace_to_dict(trace)
    summary = data.get("summary", {})
    rows: list[dict[str, Any]] = []
    important: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    important_types = {
        "FeedForwardRegion",
        "ResidualMergeRegion",
        "AttentionSkeletonRegion",
        "AxisTransformRegion",
        "LinearProjectionRegion",
    }
    for step in data.get("steps", []):
        before_active = step.get("before_summary", {}).get("num_active_nodes", 0)
        after_active = step.get("after_summary", {}).get("num_active_nodes", 0)
        row = {
            "step": step.get("step_index"),
            "pass": step.get("pass_name"),
            "action": step.get("action"),
            "created_region": step.get("created_region_type") or "",
            "collapsed": len(step.get("collapsed_node_ids", [])),
            "ops": len(step.get("collapsed_op_ids", [])),
            "active": f"{before_active} -> {after_active}",
            "confidence": step.get("confidence"),
            "reason": step.get("reason"),
        }
        rows.append(row)
        if step.get("created_region_type") in important_types and step.get("action") in {"collapse", "finalize"}:
            important.append(row)
        if step.get("action") == "skip":
            skipped.append(row)

    return "\n".join(
        [
            f"# Dataflow Control-Tree Construction Trace: {data.get('model_name', '')}",
            "",
            "## Summary",
            "",
            f"- Source frontend: `{data.get('source_frontend', 'unknown')}`",
            f"- Initial TensorOps: `{summary.get('initial_tensor_op_count', 0)}`",
            f"- Total steps: `{summary.get('num_steps', 0)}`",
            f"- Collapse steps: `{summary.get('num_collapse_steps', 0)}`",
            f"- Skip steps: `{summary.get('num_skip_steps', 0)}`",
            f"- Final active nodes: `{summary.get('final_active_node_count', 0)}`",
            f"- Region type counts: `{summary.get('created_region_type_counts', {})}`",
            "",
            "## Step Table",
            "",
            _table(rows, ["step", "pass", "action", "created_region", "collapsed", "ops", "active", "confidence", "reason"]),
            "",
            "## Important Collapses",
            "",
            _table(important, ["step", "pass", "action", "created_region", "collapsed", "ops", "active", "confidence", "reason"]),
            "",
            "## Skipped / Ambiguous Collapses",
            "",
            _table(skipped, ["step", "pass", "action", "created_region", "collapsed", "ops", "active", "confidence", "reason"]),
            "",
            "## Interpretation",
            "",
            "This trace shows how primitive TensorOps are gradually collapsed into semantic regions. The final Structural Region Tree remains the authoritative final hierarchy. The trace is an explanatory construction artifact and does not modify models or pruning logic.",
            "",
        ]
    )
