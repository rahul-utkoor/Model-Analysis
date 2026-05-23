"""Bounded multi-branch DAG motif analysis over saved ONNX graph summaries."""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir
from model_analysis.subgraph_analysis import (
    OnnxGraphAdjacency,
    build_onnx_adjacency,
    classify_add_node_kind,
    compute_subgraph_tensor_sets,
    is_join_node,
)


@dataclass
class DagRegionSubgraph:
    region_id: str
    model_name: str
    region_kind: str
    entry_nodes: list[str]
    exit_nodes: list[str]
    join_nodes: list[str]
    fork_nodes: list[str]
    node_names: list[str]
    edges: list[dict[str, str | None]]
    op_types: list[str]
    branch_paths: list[list[str]]
    pattern: str
    input_tensors: list[str]
    output_tensors: list[str]
    internal_tensors: list[str]
    boundary_input_tensors: list[str]
    boundary_output_tensors: list[str]
    initializer_tensors: list[str]
    pruning_class: str
    risk_level: str
    suggested_constraints: list[dict[str, Any]]
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DagRegionAnalysisReport:
    model_name: str
    hf_id: str
    task: str
    max_branch_depth: int
    regions: list[DagRegionSubgraph] = field(default_factory=list)
    pattern_summaries: list[dict[str, Any]] = field(default_factory=list)
    pruning_evidence: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def dag_region_report_to_dict(report: DagRegionAnalysisReport) -> dict[str, Any]:
    return asdict(report)


def write_dag_region_report_json(report: DagRegionAnalysisReport, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(dag_region_report_to_dict(report), indent=2), encoding="utf-8")


def load_dag_region_report_json(path: Path) -> DagRegionAnalysisReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    return DagRegionAnalysisReport(
        model_name=data["model_name"],
        hf_id=data.get("hf_id", ""),
        task=data.get("task", ""),
        max_branch_depth=data.get("max_branch_depth", 4),
        regions=[DagRegionSubgraph(**item) for item in data.get("regions", [])],
        pattern_summaries=data.get("pattern_summaries", []),
        pruning_evidence=data.get("pruning_evidence", []),
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def _node_order(adjacency: OnnxGraphAdjacency) -> dict[str, int]:
    return {name: index for index, name in enumerate(adjacency.node_names)}


def _sort_nodes(nodes: set[str] | list[str], adjacency: OnnxGraphAdjacency) -> list[str]:
    order = _node_order(adjacency)
    return sorted(set(nodes), key=lambda name: (order.get(name, len(order)), name))


def _op_type(node_name: str, adjacency: OnnxGraphAdjacency) -> str:
    return adjacency.node_by_name.get(node_name, {}).get("op_type", "Unknown")


def _non_initializer_inputs(node_name: str, adjacency: OnnxGraphAdjacency) -> list[str]:
    initializers = set(adjacency.initializers)
    return [
        tensor
        for tensor in adjacency.node_inputs.get(node_name, [])
        if tensor and tensor not in initializers
    ]


def _is_dataflow_join(node_name: str, adjacency: OnnxGraphAdjacency) -> bool:
    node = adjacency.node_by_name[node_name]
    return is_join_node(node) and len(_non_initializer_inputs(node_name, adjacency)) >= 2


def is_fork_node(node_name: str, adjacency: OnnxGraphAdjacency) -> bool:
    """Return true when a node's produced values feed two or more consumers."""
    consumers = {
        consumer
        for tensor in adjacency.node_outputs.get(node_name, [])
        for consumer in adjacency.consumers_of_tensor.get(tensor, [])
        if consumer != node_name
    }
    return len(consumers) >= 2


def _region_edges(node_names: list[str], adjacency: OnnxGraphAdjacency) -> list[dict[str, str | None]]:
    selected = set(node_names)
    edges: list[dict[str, str | None]] = []
    for source in node_names:
        for tensor in adjacency.node_outputs.get(source, []):
            for destination in adjacency.consumers_of_tensor.get(tensor, []):
                if destination in selected:
                    edges.append({"src": source, "dst": destination, "tensor": tensor})
    return edges


def _reachable_paths(
    adjacency: OnnxGraphAdjacency,
    start: str,
    max_depth: int,
) -> dict[str, list[str]]:
    paths = {start: [start]}
    queue: deque[tuple[str, list[str]]] = deque([(start, [start])])
    while queue:
        current, path = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        for successor in adjacency.successors.get(current, []):
            if successor in path:
                continue
            candidate = path + [successor]
            if successor not in paths or len(candidate) < len(paths[successor]):
                paths[successor] = candidate
                queue.append((successor, candidate))
    return paths


def find_reconvergence(
    adjacency: OnnxGraphAdjacency,
    branch_start_a: str,
    branch_start_b: str,
    max_depth: int,
) -> tuple[str | None, list[str], list[str]]:
    """Find the nearest deterministic common descendant of two branches."""
    paths_a = _reachable_paths(adjacency, branch_start_a, max_depth)
    paths_b = _reachable_paths(adjacency, branch_start_b, max_depth)
    candidates = (set(paths_a) & set(paths_b)) - {branch_start_a, branch_start_b}
    if not candidates:
        return (None, [], [])
    order = _node_order(adjacency)
    selected = min(
        candidates,
        key=lambda name: (
            max(len(paths_a[name]), len(paths_b[name])),
            len(paths_a[name]) + len(paths_b[name]),
            order.get(name, len(order)),
            name,
        ),
    )
    return (selected, paths_a[selected], paths_b[selected])


def _branch_expression(path: list[str], adjacency: OnnxGraphAdjacency, trim_exit: bool = False) -> str:
    selected = path[1:-1] if trim_exit else path[1:]
    if not selected:
        return "Direct"
    return " -> ".join(_op_type(name, adjacency) for name in selected)


def canonicalize_dag_region_pattern(region: DagRegionSubgraph) -> str:
    """Render a deterministic op-type motif representation."""
    node_ops = region.metadata.get("node_op_types", {})
    op = lambda name: node_ops.get(name, "Unknown")
    if region.region_kind == "fork":
        root = region.fork_nodes[0]
        branches = [_path_ops(path, node_ops, omit_first=True) for path in region.branch_paths]
        return f"Fork({op(root)} -> [{', '.join(branches)}])"
    if region.region_kind in {"diamond", "fork_join"}:
        root = region.fork_nodes[0]
        exit_node = region.join_nodes[-1]
        branches = [_path_ops(path, node_ops, omit_first=True, omit_last=True) for path in region.branch_paths]
        return f"Diamond({op(root)} -> [{', '.join(branches)}] -> {op(exit_node)})"
    if region.region_kind == "join_fork_join":
        center = region.fork_nodes[0]
        exit_node = region.join_nodes[-1]
        entries = [op(name) for name in region.entry_nodes] or ["BoundaryInput"]
        branches = [_path_ops(path, node_ops, omit_first=True, omit_last=True) for path in region.branch_paths]
        return (
            f"JoinForkJoin({op(center)} <- [{', '.join(entries)}]; "
            f"{op(center)} -> [{', '.join(branches)}] -> {op(exit_node)})"
        )
    return f"{region.region_kind}({', '.join(region.op_types)})"


def _path_ops(
    path: list[str],
    node_ops: dict[str, str],
    omit_first: bool = False,
    omit_last: bool = False,
) -> str:
    start = 1 if omit_first else 0
    end = -1 if omit_last else None
    names = path[start:end]
    return " -> ".join(node_ops.get(name, "Unknown") for name in names) or "Direct"


def _residual_add_present(
    region: DagRegionSubgraph,
    adjacency: OnnxGraphAdjacency,
    onnx_summary: dict,
) -> bool:
    for node_name in region.join_nodes:
        node = adjacency.node_by_name.get(node_name, {})
        if node.get("op_type") == "Add":
            if classify_add_node_kind(node, adjacency, onnx_summary)[0] == "residual_add":
                return True
    return False


def classify_dag_region(
    region: DagRegionSubgraph,
) -> tuple[str, str, str, list[dict[str, Any]]]:
    """Classify multi-branch structural evidence conservatively."""
    ops = set(region.op_types)
    constraints: list[dict[str, Any]] = []
    metadata = region.metadata
    direct_ops = {"Gemm", "MatMul", "Conv"}
    shape_ops = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Slice", "Split"}
    branch_parameterized = [
        any(metadata["node_op_types"].get(name) in direct_ops for name in path[1:-1])
        for path in region.branch_paths
    ]

    if metadata.get("residual_like"):
        constraints.append(
            {
                "constraint_type": "residual_equal_shape",
                "confidence": "medium",
                "reason": "Residual-style joins require equal hidden dimensions across merged branches.",
            }
        )
        return (
            "residual_like",
            "high",
            "A residual-style join occurs inside a multi-branch region; hidden-size pruning requires branch equality.",
            constraints,
        )
    if shape_ops & ops:
        constraints.append(
            {
                "constraint_type": "reshape_preservation",
                "confidence": "low",
                "reason": "Shape-transform branches require explicit axis semantics before propagation.",
            }
        )
        if {"MatMul", "Gemm"} & ops:
            return (
                "attention_like",
                "high",
                "Parameterized branches pass through reshape/transpose operations; head or axis mapping is required.",
                constraints,
            )
        return (
            "shape_transform",
            "high",
            "Multi-branch shape-transform dataflow requires explicit dimension-to-axis mapping.",
            constraints,
        )
    if region.region_kind in {"diamond", "fork_join", "join_fork_join"} and sum(branch_parameterized) >= 2:
        constraints.append(
            {
                "constraint_type": "branch_output_compatibility",
                "confidence": "medium",
                "reason": "Parallel parameterized branches reconverge and must present compatible output dimensions.",
            }
        )
        return (
            "multi_branch_constraint",
            "high",
            "Parallel parameterized branches reconverge at a join, requiring coordinated dimension reasoning.",
            constraints,
        )
    if region.region_kind == "fork":
        constraints.append(
            {
                "constraint_type": "fanout_same_indices",
                "confidence": "medium",
                "reason": "Pruning a producer output must be propagated consistently to each consuming branch.",
            }
        )
        return (
            "propagation_relevant",
            "medium" if direct_ops & ops else "high",
            "A produced value has multiple consumers; local pruning must account for every branch.",
            constraints,
        )
    if region.join_nodes:
        constraints.append(
            {
                "constraint_type": "join_dimension_equality",
                "confidence": "low",
                "reason": "Reconvergent branch dimensions must be compatible at the join.",
            }
        )
        return (
            "multi_branch_constraint",
            "high",
            "A reconvergent branch region requires join compatibility analysis.",
            constraints,
        )
    return ("unknown", "unknown", "No supported DAG motif classification was established.", constraints)


def _make_region(
    *,
    region_id: str,
    model_name: str,
    region_kind: str,
    entry_nodes: list[str],
    exit_nodes: list[str],
    join_nodes: list[str],
    fork_nodes: list[str],
    branch_paths: list[list[str]],
    adjacency: OnnxGraphAdjacency,
    onnx_summary: dict,
) -> DagRegionSubgraph:
    nodes = _sort_nodes(
        [*entry_nodes, *exit_nodes, *join_nodes, *fork_nodes, *(name for path in branch_paths for name in path)],
        adjacency,
    )
    tensors = compute_subgraph_tensor_sets(nodes, adjacency, onnx_summary)
    node_op_types = {name: _op_type(name, adjacency) for name in nodes}
    region = DagRegionSubgraph(
        region_id=region_id,
        model_name=model_name,
        region_kind=region_kind,
        entry_nodes=_sort_nodes(entry_nodes, adjacency),
        exit_nodes=_sort_nodes(exit_nodes, adjacency),
        join_nodes=_sort_nodes(join_nodes, adjacency),
        fork_nodes=_sort_nodes(fork_nodes, adjacency),
        node_names=nodes,
        edges=_region_edges(nodes, adjacency),
        op_types=[node_op_types[name] for name in nodes],
        branch_paths=branch_paths,
        pattern="",
        pruning_class="unknown",
        risk_level="unknown",
        suggested_constraints=[],
        reason="",
        metadata={"node_op_types": node_op_types},
        input_tensors=tensors["input_tensors"],
        output_tensors=tensors["output_tensors"],
        internal_tensors=tensors["internal_tensors"],
        boundary_input_tensors=tensors["boundary_input_tensors"],
        boundary_output_tensors=tensors["boundary_output_tensors"],
        initializer_tensors=tensors["initializer_tensors"],
    )
    region.metadata["residual_like"] = _residual_add_present(region, adjacency, onnx_summary)
    region.pattern = canonicalize_dag_region_pattern(region)
    (
        region.pruning_class,
        region.risk_level,
        region.reason,
        region.suggested_constraints,
    ) = classify_dag_region(region)
    return region


def enumerate_fork_regions(
    adjacency: OnnxGraphAdjacency,
    onnx_summary: dict,
    model_name: str,
    max_regions: int | None = None,
) -> list[DagRegionSubgraph]:
    """Emit a local producer-to-consumers region for each fork."""
    regions = []
    for node_name in adjacency.node_names:
        if not is_fork_node(node_name, adjacency):
            continue
        successors = adjacency.successors.get(node_name, [])
        paths = [[node_name, successor] for successor in successors]
        regions.append(
            _make_region(
                region_id=f"fork_{len(regions) + 1:06d}",
                model_name=model_name,
                region_kind="fork",
                entry_nodes=[node_name],
                exit_nodes=successors,
                join_nodes=[],
                fork_nodes=[node_name],
                branch_paths=paths,
                adjacency=adjacency,
                onnx_summary=onnx_summary,
            )
        )
        if max_regions is not None and len(regions) >= max_regions:
            break
    return regions


def enumerate_diamond_regions(
    adjacency: OnnxGraphAdjacency,
    onnx_summary: dict,
    model_name: str,
    max_branch_depth: int = 4,
    max_regions: int | None = None,
) -> list[DagRegionSubgraph]:
    """Emit bounded fork/reconvergence motifs."""
    regions = []
    for fork_node in adjacency.node_names:
        if not is_fork_node(fork_node, adjacency):
            continue
        for first, second in combinations(adjacency.successors.get(fork_node, []), 2):
            convergence, first_path, second_path = find_reconvergence(
                adjacency, first, second, max_branch_depth
            )
            if not convergence:
                continue
            branch_paths = [[fork_node, *first_path], [fork_node, *second_path]]
            regions.append(
                _make_region(
                    region_id=f"diamond_{len(regions) + 1:06d}",
                    model_name=model_name,
                    region_kind="diamond",
                    entry_nodes=[fork_node],
                    exit_nodes=[convergence],
                    join_nodes=[convergence],
                    fork_nodes=[fork_node],
                    branch_paths=branch_paths,
                    adjacency=adjacency,
                    onnx_summary=onnx_summary,
                )
            )
            if max_regions is not None and len(regions) >= max_regions:
                return regions
    return regions


def enumerate_join_fork_join_regions(
    adjacency: OnnxGraphAdjacency,
    onnx_summary: dict,
    model_name: str,
    max_branch_depth: int = 4,
    max_regions: int | None = None,
) -> list[DagRegionSubgraph]:
    """Detect bounded join-fork-join motifs such as A,B -> C -> D,E -> F."""
    regions = []
    for center in adjacency.node_names:
        if not _is_dataflow_join(center, adjacency) or not is_fork_node(center, adjacency):
            continue
        entry_nodes = [
            adjacency.producer_of_tensor[tensor]
            for tensor in _non_initializer_inputs(center, adjacency)
            if tensor in adjacency.producer_of_tensor
        ]
        for first, second in combinations(adjacency.successors.get(center, []), 2):
            convergence, first_path, second_path = find_reconvergence(
                adjacency, first, second, max_branch_depth
            )
            if not convergence:
                continue
            branch_paths = [[center, *first_path], [center, *second_path]]
            regions.append(
                _make_region(
                    region_id=f"join_fork_join_{len(regions) + 1:06d}",
                    model_name=model_name,
                    region_kind="join_fork_join",
                    entry_nodes=entry_nodes,
                    exit_nodes=[convergence],
                    join_nodes=[center, convergence],
                    fork_nodes=[center],
                    branch_paths=branch_paths,
                    adjacency=adjacency,
                    onnx_summary=onnx_summary,
                )
            )
            if max_regions is not None and len(regions) >= max_regions:
                return regions
    return regions


def _region_key(region: DagRegionSubgraph) -> tuple[Any, ...]:
    return (
        region.region_kind,
        tuple(sorted(region.node_names)),
        tuple(tuple(path) for path in region.branch_paths),
    )


def _generate_pruning_evidence(regions: list[DagRegionSubgraph]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for region in regions:
        if {"Gemm", "MatMul", "Conv"} & set(region.op_types):
            evidence.append(
                {
                    "evidence_id": f"dag_evidence_{len(evidence) + 1:06d}",
                    "region_id": region.region_id,
                    "evidence_type": "direct_prunable_op",
                    "suggested_constraint_type": None,
                    "confidence": "medium",
                    "reason": "The region contains a parameterized operation, but branch constraints remain active.",
                }
            )
        for constraint in region.suggested_constraints:
            evidence.append(
                {
                    "evidence_id": f"dag_evidence_{len(evidence) + 1:06d}",
                    "region_id": region.region_id,
                    "evidence_type": constraint["constraint_type"],
                    "suggested_constraint_type": constraint["constraint_type"],
                    "confidence": constraint.get("confidence", "low"),
                    "reason": constraint.get("reason", region.reason),
                }
            )
    return evidence


def _summarize_patterns(regions: list[DagRegionSubgraph]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[DagRegionSubgraph]] = defaultdict(list)
    for region in regions:
        grouped[(region.region_kind, region.pattern)].append(region)
    return [
        {
            "region_kind": kind,
            "pattern": pattern,
            "count": len(items),
            "example_region_ids": [item.region_id for item in items[:5]],
            "pruning_class": items[0].pruning_class,
            "risk_level": items[0].risk_level,
            "reason": items[0].reason,
        }
        for (kind, pattern), items in sorted(grouped.items())
    ]


def build_dag_region_analysis_report(
    onnx_summary: dict,
    model_config: dict,
    max_branch_depth: int = 4,
    max_regions: int | None = None,
) -> DagRegionAnalysisReport:
    """Build bounded fork, diamond, and join-fork-join region evidence."""
    model_name = model_config.get("name") or onnx_summary.get("model_name", "")
    adjacency = build_onnx_adjacency(onnx_summary)
    # Retain the most structurally informative motifs first when report size is bounded.
    candidates = [
        *enumerate_join_fork_join_regions(adjacency, onnx_summary, model_name, max_branch_depth, max_regions),
        *enumerate_diamond_regions(adjacency, onnx_summary, model_name, max_branch_depth, max_regions),
        *enumerate_fork_regions(adjacency, onnx_summary, model_name, max_regions),
    ]
    unique: list[DagRegionSubgraph] = []
    seen: set[tuple[Any, ...]] = set()
    for region in candidates:
        key = _region_key(region)
        if key not in seen:
            seen.add(key)
            unique.append(region)
            if max_regions is not None and len(unique) >= max_regions:
                break
    regions = unique
    patterns = _summarize_patterns(regions)
    evidence = _generate_pruning_evidence(regions)
    region_counts = Counter(region.region_kind for region in regions)
    class_counts = Counter(region.pruning_class for region in regions)
    risk_counts = Counter(region.risk_level for region in regions)
    constraint_counts = Counter(
        constraint.get("constraint_type")
        for region in regions
        for constraint in region.suggested_constraints
        if constraint.get("constraint_type")
    )
    summary = {
        "num_regions": len(regions),
        "region_kind_counts": dict(region_counts),
        "pruning_class_counts": dict(class_counts),
        "risk_level_counts": dict(risk_counts),
        "suggested_constraint_counts": dict(constraint_counts),
        "num_fork_regions": region_counts.get("fork", 0),
        "num_diamond_regions": region_counts.get("diamond", 0),
        "num_join_fork_join_regions": region_counts.get("join_fork_join", 0),
        "num_residual_like_regions": class_counts.get("residual_like", 0),
        "num_attention_like_regions": class_counts.get("attention_like", 0),
        "num_shape_transform_regions": class_counts.get("shape_transform", 0),
    }
    return DagRegionAnalysisReport(
        model_name=model_name,
        hf_id=model_config.get("hf_id") or onnx_summary.get("hf_id", ""),
        task=model_config.get("task") or onnx_summary.get("task", ""),
        max_branch_depth=max_branch_depth,
        regions=regions,
        pattern_summaries=patterns,
        pruning_evidence=evidence,
        summary=summary,
        metadata={
            "max_regions": max_regions,
            "source_onnx_summary": onnx_summary.get("onnx_path", ""),
        },
    )


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 250) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if len(rows) > limit:
        lines.append(f"| ... | {len(rows) - limit} more rows omitted |" + " |" * (len(columns) - 2))
    return "\n".join(lines)


def dag_region_report_to_markdown(report: DagRegionAnalysisReport | dict) -> str:
    data = dag_region_report_to_dict(report) if isinstance(report, DagRegionAnalysisReport) else report
    summary = data.get("summary", {})
    return "\n".join(
        [
            f"# DAG Region Analysis: {data.get('model_name', '')}",
            "",
            "## Summary",
            "",
            f"- Regions: `{summary.get('num_regions', 0)}`",
            f"- Fork regions: `{summary.get('num_fork_regions', 0)}`",
            f"- Diamond regions: `{summary.get('num_diamond_regions', 0)}`",
            f"- Join-fork-join regions: `{summary.get('num_join_fork_join_regions', 0)}`",
            f"- Residual-like regions: `{summary.get('num_residual_like_regions', 0)}`",
            "",
            "## Regions",
            "",
            _table(
                data.get("regions", []),
                ["region_id", "region_kind", "pattern", "fork_nodes", "join_nodes", "pruning_class", "risk_level", "reason"],
            ),
            "",
            "## Interpretation",
            "",
            "Path subgraphs describe sequential neighborhoods and join subgraphs describe one merge. DAG regions preserve fanout and reconvergence together, making multi-branch propagation constraints visible.",
            "",
            "This is structural analysis only. It does not modify models or execute pruning.",
            "",
        ]
    )


def dag_region_patterns_to_markdown(report: DagRegionAnalysisReport | dict) -> str:
    data = dag_region_report_to_dict(report) if isinstance(report, DagRegionAnalysisReport) else report
    return "\n".join(
        [
            f"# DAG Region Patterns: {data.get('model_name', '')}",
            "",
            _table(
                data.get("pattern_summaries", []),
                ["region_kind", "pattern", "count", "pruning_class", "risk_level", "reason"],
            ),
            "",
        ]
    )


def dag_region_evidence_to_markdown(report: DagRegionAnalysisReport | dict) -> str:
    data = dag_region_report_to_dict(report) if isinstance(report, DagRegionAnalysisReport) else report
    return "\n".join(
        [
            f"# DAG Region Pruning Evidence: {data.get('model_name', '')}",
            "",
            _table(
                data.get("pruning_evidence", []),
                ["evidence_id", "region_id", "evidence_type", "suggested_constraint_type", "confidence", "reason"],
            ),
            "",
            "## Interpretation",
            "",
            "Evidence is a report-level input for future pruning-map and Dimension-IR refinement. It is not an executable transform.",
            "",
        ]
    )
