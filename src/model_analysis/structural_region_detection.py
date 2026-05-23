"""Detection and hierarchy construction for semantic regions over Tensor IR."""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from model_analysis.structural_region_tree import (
    StructuralRegion,
    StructuralRegionInterface,
    StructuralRegionTree,
)


PRIORITY = {
    "ModelRegion": 0,
    "FeedForwardRegion": 1,
    "AttentionSkeletonRegion": 2,
    "ResidualMergeRegion": 3,
    "LinearProjectionRegion": 4,
    "LayerNormRegion": 5,
    "ActivationRegion": 6,
    "AxisTransformRegion": 7,
    "ForkRegion": 8,
    "JoinRegion": 9,
    "BiasAddRegion": 10,
    "ProperAcyclicRegion": 11,
    "PrimitiveRegion": 12,
}


def load_tensor_graph_dict(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_ops_by_id(tensor_graph: dict) -> dict[str, dict]:
    return {item["op_id"]: item for item in tensor_graph.get("ops", [])}


def get_values_by_id(tensor_graph: dict) -> dict[str, dict]:
    return {item["value_id"]: item for item in tensor_graph.get("values", [])}


def build_op_adjacency_from_tensor_graph(tensor_graph: dict) -> dict[str, dict[str, list[str]]]:
    ops = get_ops_by_id(tensor_graph)
    return {
        op_id: {
            "successors": sorted(item for item in op.get("successor_ops", []) if item in ops),
            "predecessors": sorted(item for item in op.get("predecessor_ops", []) if item in ops),
        }
        for op_id, op in ops.items()
    }


def compute_region_boundary(tensor_graph: dict, op_ids: list[str]) -> dict[str, Any]:
    ops = get_ops_by_id(tensor_graph)
    values = get_values_by_id(tensor_graph)
    selected = set(op_ids)
    input_values = list(dict.fromkeys(value for op_id in op_ids for value in ops[op_id].get("inputs", [])))
    output_values = list(dict.fromkeys(value for op_id in op_ids for value in ops[op_id].get("outputs", [])))
    internal_values = [
        value_id
        for value_id in output_values
        if any(consumer in selected for consumer in values.get(value_id, {}).get("consumers", []))
    ]
    boundary_inputs = [
        value_id for value_id in input_values
        if values.get(value_id, {}).get("producer") not in selected
    ]
    graph_outputs = set(tensor_graph.get("graph_outputs", []))
    boundary_outputs = [
        value_id for value_id in output_values
        if value_id in graph_outputs
        or any(consumer not in selected for consumer in values.get(value_id, {}).get("consumers", []))
        or not values.get(value_id, {}).get("consumers", [])
    ]
    entries = [
        op_id for op_id in op_ids
        if any(value_id in boundary_inputs for value_id in ops[op_id].get("inputs", []))
        or not any(predecessor in selected for predecessor in ops[op_id].get("predecessor_ops", []))
    ]
    exits = [
        op_id for op_id in op_ids
        if any(value_id in boundary_outputs for value_id in ops[op_id].get("outputs", []))
        or not any(successor in selected for successor in ops[op_id].get("successor_ops", []))
    ]
    return {
        "input_values": input_values,
        "output_values": output_values,
        "internal_values": internal_values,
        "boundary_input_values": boundary_inputs,
        "boundary_output_values": boundary_outputs,
        "entry_ops": list(dict.fromkeys(entries)),
        "exit_ops": list(dict.fromkeys(exits)),
        "contains_fork": any(ops[op_id].get("is_fork", False) for op_id in op_ids),
        "contains_join": any(ops[op_id].get("is_join", False) for op_id in op_ids),
        "single_entry": len(set(entries)) == 1,
        "single_exit": len(set(exits)) == 1,
    }


def _region(
    tensor_graph: dict,
    region_type: str,
    op_ids: list[str],
    counter: int,
    confidence: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> StructuralRegion:
    ordered = [item["op_id"] for item in tensor_graph.get("ops", []) if item["op_id"] in set(op_ids)]
    boundary = compute_region_boundary(tensor_graph, ordered)
    return StructuralRegion(
        region_id=f"region::{region_type.lower()}::{counter:06d}",
        region_type=region_type,
        name=f"{region_type}_{counter:06d}",
        op_ids=ordered,
        value_ids=list(dict.fromkeys([*boundary["input_values"], *boundary["output_values"]])),
        children=[],
        parent=None,
        entry_ops=boundary["entry_ops"],
        exit_ops=boundary["exit_ops"],
        input_values=boundary["input_values"],
        output_values=boundary["output_values"],
        internal_values=boundary["internal_values"],
        boundary_input_values=boundary["boundary_input_values"],
        boundary_output_values=boundary["boundary_output_values"],
        contains_fork=boundary["contains_fork"],
        contains_join=boundary["contains_join"],
        single_entry=boundary["single_entry"],
        single_exit=boundary["single_exit"],
        region_depth=0,
        confidence=confidence,
        reason=reason,
        metadata={"candidate_priority": PRIORITY[region_type], "matched_ops": ordered, **(metadata or {})},
    )


def _successor(adjacency: dict, op_id: str) -> str | None:
    successors = adjacency.get(op_id, {}).get("successors", [])
    return successors[0] if len(successors) == 1 else None


def build_primitive_regions(tensor_graph: dict) -> list[StructuralRegion]:
    regions = []
    for index, op in enumerate(tensor_graph.get("ops", []), start=1):
        regions.append(
            _region(
                tensor_graph,
                "PrimitiveRegion",
                [op["op_id"]],
                index,
                "high",
                "Leaf region representing one TensorOp.",
                {
                    "canonical_op_type": op.get("canonical_op_type"),
                    "region_hint": op.get("region_hint"),
                    "source_frontend": op.get("source_frontend"),
                    "source_node_name": op.get("source_node_name"),
                },
            )
        )
    return regions


def detect_linear_projection_regions(tensor_graph: dict, start_counter: int = 1) -> list[StructuralRegion]:
    ops = get_ops_by_id(tensor_graph)
    adjacency = build_op_adjacency_from_tensor_graph(tensor_graph)
    regions = []
    counter = start_counter
    for op in tensor_graph.get("ops", []):
        if op.get("canonical_op_type") not in {"linear", "matmul"}:
            continue
        matched = [op["op_id"]]
        successor = _successor(adjacency, op["op_id"])
        if successor and ops[successor].get("canonical_op_type") == "bias_add" and len(adjacency[successor]["predecessors"]) == 1:
            matched.append(successor)
        regions.append(
            _region(
                tensor_graph,
                "LinearProjectionRegion",
                matched,
                counter,
                "high" if op.get("canonical_op_type") == "linear" else "medium",
                "Projection operation with optional adjacent bias addition.",
            )
        )
        counter += 1
    return regions


def detect_single_op_regions(tensor_graph: dict, canonical_type: str, region_type: str, reason: str) -> list[StructuralRegion]:
    regions = []
    for index, op in enumerate(tensor_graph.get("ops", []), start=1):
        if op.get("canonical_op_type") == canonical_type:
            regions.append(_region(tensor_graph, region_type, [op["op_id"]], index, "high", reason))
    return regions


def detect_axis_transform_regions(tensor_graph: dict) -> list[StructuralRegion]:
    ops = get_ops_by_id(tensor_graph)
    adjacency = build_op_adjacency_from_tensor_graph(tensor_graph)
    kinds = {"shape_op", "axis_transform", "constant", "mask_or_select"}
    visited: set[str] = set()
    regions = []
    for op in tensor_graph.get("ops", []):
        op_id = op["op_id"]
        if op_id in visited or op.get("canonical_op_type") not in kinds:
            continue
        predecessors = [
            item for item in adjacency[op_id]["predecessors"]
            if ops[item].get("canonical_op_type") in kinds
        ]
        if len(predecessors) == 1 and len(adjacency[predecessors[0]]["successors"]) == 1:
            continue
        chain = [op_id]
        current = op_id
        while True:
            successor = _successor(adjacency, current)
            if not successor or successor in visited or ops[successor].get("canonical_op_type") not in kinds:
                break
            if len(adjacency[successor]["predecessors"]) != 1:
                break
            chain.append(successor)
            current = successor
        visited.update(chain)
        regions.append(
            _region(
                tensor_graph,
                "AxisTransformRegion",
                chain,
                len(regions) + 1,
                "medium",
                "Tensor-axis/shape operations form a propagation-only structural region.",
            )
        )
    return regions


def detect_residual_merge_regions(tensor_graph: dict) -> list[StructuralRegion]:
    ops = get_ops_by_id(tensor_graph)
    adjacency = build_op_adjacency_from_tensor_graph(tensor_graph)
    regions = []
    for op in tensor_graph.get("ops", []):
        if op.get("canonical_op_type") not in {"residual_add", "elementwise_join"}:
            continue
        matched = [op["op_id"]]
        successor = _successor(adjacency, op["op_id"])
        if successor and ops[successor].get("canonical_op_type") == "layer_norm":
            matched.append(successor)
        regions.append(
            _region(
                tensor_graph,
                "ResidualMergeRegion",
                matched,
                len(regions) + 1,
                "medium",
                "Activation branches merge; hidden dimensions must remain compatible across the join.",
            )
        )
    return regions


def detect_feedforward_regions(tensor_graph: dict) -> list[StructuralRegion]:
    ops = get_ops_by_id(tensor_graph)
    adjacency = build_op_adjacency_from_tensor_graph(tensor_graph)
    regions = []
    for op in tensor_graph.get("ops", []):
        if op.get("canonical_op_type") not in {"linear", "matmul"}:
            continue
        matched = [op["op_id"]]
        current = _successor(adjacency, op["op_id"])
        if current and ops[current].get("canonical_op_type") == "bias_add":
            matched.append(current)
            current = _successor(adjacency, current)
        if not current or ops[current].get("canonical_op_type") != "activation":
            continue
        matched.append(current)
        current = _successor(adjacency, current)
        if not current or ops[current].get("canonical_op_type") not in {"linear", "matmul"}:
            continue
        matched.append(current)
        final = _successor(adjacency, current)
        if final and ops[final].get("canonical_op_type") == "bias_add":
            matched.append(final)
        regions.append(
            _region(
                tensor_graph,
                "FeedForwardRegion",
                matched,
                len(regions) + 1,
                "high",
                "Projection, activation, and projection expose a coupled intermediate dimension.",
            )
        )
    return regions


def _bounded_find(
    adjacency: dict,
    ops: dict[str, dict],
    start: str,
    direction: str,
    target: set[str],
    max_depth: int = 4,
) -> list[str] | None:
    key = "predecessors" if direction == "backward" else "successors"
    queue = deque([(start, [start], 0)])
    seen = {start}
    while queue:
        current, path, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in adjacency[current][key]:
            if neighbor in seen:
                continue
            candidate = [neighbor, *path] if direction == "backward" else [*path, neighbor]
            if ops[neighbor].get("canonical_op_type") in target:
                return candidate
            seen.add(neighbor)
            queue.append((neighbor, candidate, depth + 1))
    return None


def detect_attention_skeleton_regions(tensor_graph: dict) -> list[StructuralRegion]:
    ops = get_ops_by_id(tensor_graph)
    adjacency = build_op_adjacency_from_tensor_graph(tensor_graph)
    regions = []
    for op in tensor_graph.get("ops", []):
        if op.get("canonical_op_type") != "softmax":
            continue
        before = _bounded_find(adjacency, ops, op["op_id"], "backward", {"matmul"})
        after = _bounded_find(adjacency, ops, op["op_id"], "forward", {"matmul"})
        if not before or not after:
            continue
        matched = list(dict.fromkeys([*before, *after[1:]]))
        regions.append(
            _region(
                tensor_graph,
                "AttentionSkeletonRegion",
                matched,
                len(regions) + 1,
                "medium",
                "MatMul/Softmax/MatMul skeleton requires head and axis mapping before pruning.",
            )
        )
    return regions


def detect_fork_regions(tensor_graph: dict) -> list[StructuralRegion]:
    adjacency = build_op_adjacency_from_tensor_graph(tensor_graph)
    regions = []
    for op in tensor_graph.get("ops", []):
        if op.get("is_fork"):
            matched = [op["op_id"], *adjacency[op["op_id"]]["successors"]]
            regions.append(
                _region(
                    tensor_graph,
                    "ForkRegion",
                    matched,
                    len(regions) + 1,
                    "medium",
                    "A producer fans out to multiple immediate consumers.",
                )
            )
    return regions


def detect_join_regions(tensor_graph: dict) -> list[StructuralRegion]:
    return [
        _region(
            tensor_graph,
            "JoinRegion",
            [op["op_id"]],
            index,
            "medium",
            "A generic tensor-dataflow merge carries branch compatibility constraints.",
        )
        for index, op in enumerate(
            (
                item for item in tensor_graph.get("ops", [])
                if item.get("is_join") and item.get("canonical_op_type") not in {"residual_add", "elementwise_join"}
            ),
            start=1,
        )
    ]


def detect_region_candidates(tensor_graph: dict) -> list[StructuralRegion]:
    return [
        *detect_feedforward_regions(tensor_graph),
        *detect_attention_skeleton_regions(tensor_graph),
        *detect_residual_merge_regions(tensor_graph),
        *detect_linear_projection_regions(tensor_graph),
        *detect_single_op_regions(tensor_graph, "layer_norm", "LayerNormRegion", "Normalization carries hidden-dimension constraints."),
        *detect_single_op_regions(tensor_graph, "activation", "ActivationRegion", "Elementwise activation preserves tensor shape."),
        *detect_axis_transform_regions(tensor_graph),
        *detect_fork_regions(tensor_graph),
        *detect_join_regions(tensor_graph),
        *detect_single_op_regions(tensor_graph, "bias_add", "BiasAddRegion", "Bias follows the producer output dimension."),
    ]


def resolve_region_overlaps(
    candidates: list[StructuralRegion],
    primitive_regions: list[StructuralRegion],
) -> list[StructuralRegion]:
    """Select compatible semantic regions, retaining nested contained regions."""
    selected: list[StructuralRegion] = []
    shadowed: list[StructuralRegion] = []
    ordered = sorted(candidates, key=lambda item: (PRIORITY.get(item.region_type, 99), -len(item.op_ids), item.region_id))
    for candidate in ordered:
        candidate_ops = set(candidate.op_ids)
        conflict = False
        duplicate = False
        for existing in selected:
            existing_ops = set(existing.op_ids)
            overlap = candidate_ops & existing_ops
            if not overlap:
                continue
            if candidate_ops == existing_ops:
                duplicate = True
                break
            if candidate_ops < existing_ops or existing_ops < candidate_ops:
                continue
            conflict = True
            break
        if conflict or duplicate:
            candidate.metadata["shadowed"] = True
            shadowed.append(candidate)
        else:
            selected.append(candidate)
    for candidate in shadowed:
        candidate.metadata["selected"] = False
    for candidate in selected:
        candidate.metadata["selected"] = True
    return [*selected, *primitive_regions]


def _symbolic_dimension(name: str, role: str) -> dict[str, str]:
    return {"dim_name": name, "symbolic_role": role}


def infer_region_interface(region: StructuralRegion | dict, tensor_graph: dict) -> StructuralRegionInterface:
    item = region if isinstance(region, dict) else region.__dict__
    region_type = item["region_type"]
    prunable: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    propagated: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []
    role = "analysis_only"
    reason = "Region is retained for structural analysis."
    if region_type == "LinearProjectionRegion":
        role = "directly_prunable"
        prunable = [_symbolic_dimension("out_features", "projection_output")]
        propagated = [_symbolic_dimension("in_features", "projection_input")]
        if len(item.get("op_ids", [])) > 1:
            constraints.append({"type": "bias_follows_output", "reason": "Bias channels follow output projection indices."})
        reason = "Projection output is a local pruning surface subject to consumer propagation."
    elif region_type == "BiasAddRegion":
        role = "propagation_only"
        propagated = [_symbolic_dimension("out_features", "bias_axis")]
        constraints.append({"type": "bias_follows_output"})
        reason = "Bias is not independent from its projection output dimension."
    elif region_type == "LayerNormRegion":
        role = "constraint_carrier"
        protected = [_symbolic_dimension("hidden_dim", "normalization_hidden")]
        constraints.append({"type": "hidden_equals_normalization_parameter"})
        reason = "Normalization affine parameters constrain hidden width."
    elif region_type == "ResidualMergeRegion":
        role = "blocked"
        blocked = [_symbolic_dimension("hidden_dim", "residual_branch_hidden")]
        constraints.append({"type": "branch_hidden_equality"})
        reason = "Residual-like branch merge requires equal hidden dimensions at the join."
    elif region_type == "AxisTransformRegion":
        role = "propagation_only"
        propagated = [_symbolic_dimension("symbolic_axis", "axis_mapping")]
        constraints.append({"type": "reshape_transpose_axis_mapping_required"})
        reason = "Axis transformations require explicit mapping for pruning propagation."
    elif region_type == "ActivationRegion":
        role = "propagation_only"
        propagated = [_symbolic_dimension("elementwise_dim", "shape_preserving")]
        constraints.append({"type": "activation_preserves_shape"})
        reason = "Elementwise activation propagates its input dimensions."
    elif region_type == "FeedForwardRegion":
        role = "directly_prunable"
        prunable = [_symbolic_dimension("intermediate_dim", "mlp_hidden")]
        protected = [_symbolic_dimension("hidden_dim", "residual_boundary")]
        constraints.append({"type": "same_indices", "expression": "first_projection.out_features == second_projection.in_features"})
        reason = "Feed-forward intermediate channels require paired projection consistency."
    elif region_type == "AttentionSkeletonRegion":
        role = "analysis_only"
        protected = [_symbolic_dimension(name, "attention_axis") for name in ("num_heads", "head_dim", "sequence_dim", "hidden_dim")]
        constraints.append({"type": "head_axis_mapping_required"})
        reason = "Attention structure needs proven head/reshape axis mapping before transformation."
    elif region_type == "ForkRegion":
        role = "propagation_only"
        propagated = [_symbolic_dimension("producer_output", "fanout")]
        constraints.append({"type": "fanout_same_indices"})
        reason = "Producer pruning must propagate to all fork consumers."
    elif region_type == "JoinRegion":
        role = "constraint_carrier"
        protected = [_symbolic_dimension("branch_dim", "merge_compatibility")]
        constraints.append({"type": "branch_compatibility"})
        reason = "Joined branch dimensions require compatibility evidence."
    elif region_type == "PrimitiveRegion":
        role = "analysis_only"
        reason = "Primitive leaf contributes source-operation evidence to its enclosing region."
    elif region_type == "ModelRegion":
        role = "analysis_only"
        reason = "Root region owns the complete tensor-dataflow analysis hierarchy."
    return StructuralRegionInterface(
        region_id=item["region_id"],
        region_type=region_type,
        prunable_dimensions=prunable,
        protected_dimensions=protected,
        propagated_dimensions=propagated,
        blocked_dimensions=blocked,
        constraints=constraints,
        pruning_role=role,
        reason=reason,
    )


def _assign_hierarchy(regions: list[StructuralRegion], primitives: list[StructuralRegion], root_id: str) -> None:
    semantic = [item for item in regions if item.region_type not in {"PrimitiveRegion", "ModelRegion"}]
    for region in semantic:
        containing = [
            item for item in semantic
            if item.region_id != region.region_id and set(region.op_ids) < set(item.op_ids)
        ]
        if containing:
            region.parent = min(containing, key=lambda item: (len(item.op_ids), PRIORITY.get(item.region_type, 99))).region_id
        else:
            region.parent = root_id
    for primitive in primitives:
        containing = [item for item in semantic if set(primitive.op_ids) <= set(item.op_ids)]
        primitive.parent = (
            min(containing, key=lambda item: (len(item.op_ids), -PRIORITY.get(item.region_type, 99))).region_id
            if containing else root_id
        )
    by_id = {item.region_id: item for item in regions}
    for region in regions:
        region.children = []
    for region in [*semantic, *primitives]:
        by_id[region.parent or root_id].children.append(region.region_id)
    for region in regions:
        region.children.sort()


def _set_depths(regions: list[StructuralRegion], root_id: str) -> None:
    by_id = {item.region_id: item for item in regions}
    queue = deque([(root_id, 0)])
    while queue:
        region_id, depth = queue.popleft()
        by_id[region_id].region_depth = depth
        queue.extend((child, depth + 1) for child in by_id[region_id].children)


def build_structural_region_tree(tensor_graph: dict) -> StructuralRegionTree:
    primitive_regions = build_primitive_regions(tensor_graph)
    candidates = detect_region_candidates(tensor_graph)
    selected_with_primitives = resolve_region_overlaps(candidates, primitive_regions)
    selected = [item for item in selected_with_primitives if item.region_type != "PrimitiveRegion"]
    all_ops = [item["op_id"] for item in tensor_graph.get("ops", [])]
    root = _region(
        tensor_graph,
        "ModelRegion",
        all_ops,
        1,
        "high",
        "Root structural region containing all Tensor IR operations.",
    )
    regions = [root, *selected, *primitive_regions]
    _assign_hierarchy(regions, primitive_regions, root.region_id)
    _set_depths(regions, root.region_id)
    interfaces = [infer_region_interface(item, tensor_graph) for item in regions]
    type_counts = Counter(item.region_type for item in regions)
    role_counts = Counter(item.pruning_role for item in interfaces)
    confidence_counts = Counter(item.confidence for item in regions)
    return StructuralRegionTree(
        model_name=tensor_graph.get("model_name", ""),
        source_frontend=tensor_graph.get("source_frontend", "unknown"),
        root_region_id=root.region_id,
        regions=regions,
        interfaces=interfaces,
        summary={
            "num_regions": len(regions),
            "num_primitive_regions": type_counts.get("PrimitiveRegion", 0),
            "region_type_counts": dict(type_counts),
            "pruning_role_counts": dict(role_counts),
            "confidence_counts": dict(confidence_counts),
            "num_fork_regions": type_counts.get("ForkRegion", 0),
            "num_join_regions": type_counts.get("JoinRegion", 0),
            "num_residual_merge_regions": type_counts.get("ResidualMergeRegion", 0),
            "num_feedforward_regions": type_counts.get("FeedForwardRegion", 0),
            "num_attention_skeleton_regions": type_counts.get("AttentionSkeletonRegion", 0),
            "num_axis_transform_regions": type_counts.get("AxisTransformRegion", 0),
            "num_directly_prunable_regions": role_counts.get("directly_prunable", 0),
            "num_blocked_regions": role_counts.get("blocked", 0),
            "num_analysis_only_regions": role_counts.get("analysis_only", 0),
        },
        metadata={
            "tensor_graph_id": tensor_graph.get("graph_id"),
            "detection_note": "Conservative first-pass structural regions over frontend-independent Tensor IR.",
        },
    )
