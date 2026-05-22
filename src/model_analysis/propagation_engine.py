"""Dry-run pruning propagation over dependency graphs."""

from __future__ import annotations

from collections import deque
from typing import Any

from model_analysis.dependency_graph import DependencyEdge, DependencyGraph, PrunableUnit
from model_analysis.pruning_action import PruningAction, PropagationStep, PruningPlan


CRITICAL_EDGE_TYPES = {
    "qkv_coupling",
    "head_dimension_coupling",
    "mlp_hidden_coupling",
    "residual_coupling",
    "normalization_dependency",
    "embedding_tying",
    "shape_dependency",
}


def _find_unit(graph: DependencyGraph, unit_id: str) -> PrunableUnit | None:
    for unit in graph.prunable_units:
        if unit.unit_id == unit_id:
            return unit
    return None


def _shape_bound(unit: PrunableUnit, dim: str) -> int | None:
    if not unit.shape:
        return None
    if dim in {"out_features", "channel_out", "intermediate_dim"}:
        return unit.shape[0] if len(unit.shape) >= 1 and unit.shape[0] is not None else None
    if dim in {"in_features", "embedding_dim", "head_dim"}:
        return unit.shape[-1] if unit.shape and unit.shape[-1] is not None else None
    if dim == "vocab_dim":
        return unit.shape[0] if len(unit.shape) >= 1 and unit.shape[0] is not None else None
    if dim == "hidden_dim":
        return unit.shape[0] if len(unit.shape) >= 1 and unit.unit_type in {"linear", "mlp_expansion"} else unit.shape[-1]
    return None


def _normalize_indices(indices: list[int]) -> list[int]:
    return sorted(set(indices))


def _empty_plan(graph: DependencyGraph, action: PruningAction) -> PruningPlan:
    return PruningPlan(
        plan_id=f"{graph.model_name}__{action.action_id}",
        model_name=graph.model_name,
        action=action,
        metadata={"analysis": "dry_run_static_propagation", "graph_sources": graph.metadata.get("sources", [])},
    )


def _add_affected(plan: PruningPlan, unit: PrunableUnit, dim: str, indices: list[int], reason: str) -> None:
    existing = {item["unit_id"] for item in plan.affected_units}
    if unit.unit_id not in existing:
        plan.affected_units.append(
            {
                "unit_id": unit.unit_id,
                "name": unit.name,
                "unit_type": unit.unit_type,
                "source": unit.source,
                "affected_dim": dim,
                "indices": indices,
                "reason": reason,
            }
        )


def _traversable_edges(graph: DependencyGraph, current_unit_id: str) -> list[tuple[DependencyEdge, str, str]]:
    traversable = []
    for edge in graph.dependency_edges:
        if edge.src == current_unit_id and edge.direction in {"forward", "bidirectional"}:
            traversable.append((edge, edge.src, edge.dst))
        if edge.dst == current_unit_id and edge.direction in {"backward", "bidirectional"}:
            traversable.append((edge, edge.dst, edge.src))
        if edge.direction == "bidirectional" and edge.dst == current_unit_id:
            traversable.append((edge, edge.dst, edge.src))
    return traversable


def _dims_overlap(prune_dim: str, affected_dims: list[str]) -> bool:
    if not affected_dims:
        return False
    aliases = {
        "out_features": {"out_features", "hidden_dim", "intermediate_dim", "channel_out"},
        "in_features": {"in_features", "hidden_dim", "intermediate_dim"},
        "hidden_dim": {"hidden_dim", "out_features", "in_features", "embedding_dim"},
        "intermediate_dim": {"intermediate_dim", "out_features", "in_features"},
        "embedding_dim": {"embedding_dim", "hidden_dim"},
        "num_heads": {"num_heads", "head_dim", "hidden_dim"},
        "head_dim": {"num_heads", "head_dim", "hidden_dim"},
    }
    return bool(aliases.get(prune_dim, {prune_dim}) & set(affected_dims))


def _edge_semantics(edge: DependencyEdge, action_dim: str, src_unit: PrunableUnit | None, dst_unit: PrunableUnit | None) -> tuple[str, list[str], str, list[dict[str, Any]], list[dict[str, Any]]]:
    constraints: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []
    affected_dims = list(edge.affected_dims)

    if edge.edge_type == "qkv_coupling":
        if action_dim in {"num_heads", "head_dim", "hidden_dim", "out_features"}:
            constraints.append(
                {
                    "edge_type": edge.edge_type,
                    "src": edge.src,
                    "dst": edge.dst,
                    "affected_dims": affected_dims,
                    "reason": "Q/K/V projections must prune matching attention structure.",
                }
            )
            return "propagated", affected_dims, "Q/K/V coupling propagates matching indices.", constraints, manual_review
        return "ignored", affected_dims, "Prune dimension does not affect Q/K/V coupling.", constraints, manual_review

    if edge.edge_type == "head_dimension_coupling":
        propagated_dims = ["in_features", "hidden_dim"]
        constraints.append(
            {
                "edge_type": edge.edge_type,
                "src": edge.src,
                "dst": edge.dst,
                "affected_dims": propagated_dims,
                "reason": "Attention output projection consumes concatenated head dimensions.",
            }
        )
        manual_review.append(
            {
                "item": f"{edge.src} -> {edge.dst}",
                "reason": "Exact head-to-hidden index mapping is not represented yet.",
                "confidence": "medium",
            }
        )
        return "ambiguous", propagated_dims, "Attention head dimension mapping is not yet explicit.", constraints, manual_review

    if edge.edge_type == "mlp_hidden_coupling":
        constraints.append(
            {
                "edge_type": edge.edge_type,
                "src": edge.src,
                "dst": edge.dst,
                "affected_dims": ["intermediate_dim"],
                "reason": "MLP expansion output channels and projection input channels share the intermediate dimension.",
            }
        )
        if action_dim in {"out_features", "intermediate_dim", "in_features"}:
            return "propagated", ["intermediate_dim"], "MLP hidden coupling propagates intermediate indices.", constraints, manual_review
        return "ignored", ["intermediate_dim"], "Prune dimension does not affect MLP intermediate coupling.", constraints, manual_review

    if edge.edge_type == "residual_coupling":
        constraints.append(
            {
                "edge_type": edge.edge_type,
                "src": edge.src,
                "dst": edge.dst,
                "affected_dims": ["hidden_dim"],
                "reason": "Residual branches require compatible hidden dimensions.",
            }
        )
        manual_review.append(
            {
                "item": f"{edge.src} -> {edge.dst}",
                "reason": "Residual coupling touched; branch shape evidence is insufficient for execution.",
                "confidence": "medium",
            }
        )
        return "ambiguous", ["hidden_dim"], "Residual Add requires manual branch compatibility review.", constraints, manual_review

    if edge.edge_type == "normalization_dependency":
        constraints.append(
            {
                "edge_type": edge.edge_type,
                "src": edge.src,
                "dst": edge.dst,
                "affected_dims": ["hidden_dim"],
                "reason": "Normalization affine parameters may need matching hidden-dimension pruning.",
            }
        )
        if src_unit and src_unit.shape:
            return "propagated", ["hidden_dim"], "Normalization dependency follows a known upstream shape.", constraints, manual_review
        manual_review.append(
            {
                "item": f"{edge.src} -> {edge.dst}",
                "reason": "Normalization dependency has insufficient shape evidence.",
                "confidence": "low",
            }
        )
        return "ambiguous", ["hidden_dim"], "Normalization dependency shape mapping is incomplete.", constraints, manual_review

    if edge.edge_type == "embedding_tying":
        manual_review.append(
            {
                "item": f"{edge.src} -> {edge.dst}",
                "reason": "Embedding/output tying is not reliably detected.",
                "confidence": "low",
            }
        )
        return "ambiguous", affected_dims, "Embedding tying uncertainty requires manual review.", constraints, manual_review

    if edge.edge_type == "propagation_only":
        if _dims_overlap(action_dim, affected_dims):
            manual_review.append(
                {
                    "item": f"{edge.src} -> {edge.dst}",
                    "reason": "Shape-changing ONNX operation may need index remapping.",
                    "confidence": "low",
                }
            )
            return "ambiguous", affected_dims, "Propagation-only edge has explicit affected dimensions but no exact index mapping.", constraints, manual_review
        return "ignored", affected_dims, "Propagation-only edge does not overlap requested prune dimension.", constraints, manual_review

    if edge.edge_type == "feeds":
        if _dims_overlap(action_dim, affected_dims):
            return "propagated", affected_dims, "Feed edge overlaps affected dimensions and is traced forward.", constraints, manual_review
        return "ignored", affected_dims, "Feed edge traced without imposing pruning constraints.", constraints, manual_review

    if edge.edge_type == "shape_dependency":
        constraints.append(
            {
                "edge_type": edge.edge_type,
                "src": edge.src,
                "dst": edge.dst,
                "affected_dims": affected_dims,
                "reason": "Shape dependency requires explicit dimension mapping before execution.",
            }
        )
        return "ambiguous", affected_dims, "Shape dependency lacks exact mapping.", constraints, manual_review

    if edge.edge_type in CRITICAL_EDGE_TYPES:
        return "ambiguous", affected_dims, f"{edge.edge_type} is critical and lacks explicit propagation semantics.", constraints, manual_review
    return "ignored", affected_dims, f"{edge.edge_type} does not impose pruning propagation in the current engine.", constraints, manual_review


def _finalize_status(plan: PruningPlan, locally_valid: bool, traversed_required_edge: bool) -> None:
    if plan.conflicts:
        plan.status = "rejected"
    elif any(step.status == "ambiguous" for step in plan.propagation_steps) or plan.manual_review_items:
        plan.status = "ambiguous"
    elif traversed_required_edge and any(step.status == "propagated" for step in plan.propagation_steps):
        plan.status = "valid_global"
    elif locally_valid:
        plan.status = "valid_local"
    else:
        plan.status = "ambiguous"

    step_counts: dict[str, int] = {}
    for step in plan.propagation_steps:
        step_counts[step.status] = step_counts.get(step.status, 0) + 1
    plan.summary = {
        "status": plan.status,
        "num_affected_units": len(plan.affected_units),
        "num_propagation_steps": len(plan.propagation_steps),
        "num_constraints": len(plan.constraints),
        "num_conflicts": len(plan.conflicts),
        "num_manual_review_items": len(plan.manual_review_items),
        "propagation_step_status_counts": step_counts,
    }


def simulate_pruning_action(graph: DependencyGraph, action: PruningAction) -> PruningPlan:
    """Simulate a pruning action through a dependency graph without mutating model artifacts."""
    plan = _empty_plan(graph, action)
    target = _find_unit(graph, action.target_unit_id)
    normalized_indices = _normalize_indices(action.indices)
    action.indices = normalized_indices

    if not target:
        plan.conflicts.append(
            {
                "type": "missing_target_unit",
                "reason": f"Target unit '{action.target_unit_id}' does not exist in dependency graph.",
            }
        )
        _finalize_status(plan, locally_valid=False, traversed_required_edge=False)
        return plan

    action.target_unit_name = action.target_unit_name or target.name
    action.target_unit_type = action.target_unit_type or target.unit_type
    _add_affected(plan, target, action.prune_dim, normalized_indices, "Requested pruning target.")

    if action.prune_dim not in target.prunable_dims:
        plan.conflicts.append(
            {
                "type": "invalid_prune_dim",
                "target_unit_id": target.unit_id,
                "prune_dim": action.prune_dim,
                "valid_dims": target.prunable_dims,
                "reason": "Requested dimension is not listed as prunable for the target unit.",
            }
        )
        _finalize_status(plan, locally_valid=False, traversed_required_edge=False)
        return plan

    if any(index < 0 for index in normalized_indices):
        plan.conflicts.append(
            {
                "type": "invalid_indices",
                "indices": normalized_indices,
                "reason": "Pruning indices must be non-negative.",
            }
        )
        _finalize_status(plan, locally_valid=False, traversed_required_edge=False)
        return plan

    if not normalized_indices:
        plan.conflicts.append(
            {
                "type": "empty_indices",
                "reason": "At least one pruning index is required.",
            }
        )
        _finalize_status(plan, locally_valid=False, traversed_required_edge=False)
        return plan

    bound = _shape_bound(target, action.prune_dim)
    if bound is not None and max(normalized_indices) >= bound:
        plan.conflicts.append(
            {
                "type": "index_out_of_bounds",
                "target_unit_id": target.unit_id,
                "prune_dim": action.prune_dim,
                "shape": target.shape,
                "max_index": max(normalized_indices),
                "bound": bound,
                "reason": "Requested index exceeds known shape bound.",
            }
        )
        _finalize_status(plan, locally_valid=False, traversed_required_edge=False)
        return plan
    if bound is None:
        plan.manual_review_items.append(
            {
                "item": target.unit_id,
                "reason": "Target shape bound is unknown for requested prune dimension.",
                "confidence": "low",
            }
        )

    queue = deque([(target.unit_id, action.prune_dim)])
    visited_edges: set[tuple[str, str, str, str]] = set()
    traversed_required_edge = False
    step_index = 0

    while queue:
        current_id, current_dim = queue.popleft()
        current_unit = _find_unit(graph, current_id)
        for edge, traversal_src, traversal_dst in _traversable_edges(graph, current_id):
            edge_key = (traversal_src, traversal_dst, edge.edge_type, current_dim)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            dst_unit = _find_unit(graph, traversal_dst)
            status, affected_dims, reason, constraints, manual_review = _edge_semantics(edge, current_dim, current_unit, dst_unit)
            traversed_required_edge = traversed_required_edge or edge.edge_type in CRITICAL_EDGE_TYPES
            step = PropagationStep(
                step_id=f"step_{step_index:04d}",
                src_unit_id=traversal_src,
                dst_unit_id=traversal_dst,
                edge_type=edge.edge_type,
                direction=edge.direction,
                affected_dims=affected_dims,
                propagated_indices=normalized_indices if status in {"propagated", "ambiguous"} else [],
                status=status,
                reason=reason,
            )
            step_index += 1
            plan.propagation_steps.append(step)
            plan.constraints.extend(constraints)
            plan.manual_review_items.extend(manual_review)

            if dst_unit and status in {"propagated", "ambiguous"}:
                propagated_dim = affected_dims[0] if affected_dims else current_dim
                _add_affected(plan, dst_unit, propagated_dim, normalized_indices, reason)
                if status == "propagated":
                    queue.append((dst_unit.unit_id, propagated_dim))

    if graph.metadata.get("onnx_evidence") and not any(unit.get("source") == "onnx" for unit in plan.affected_units):
        plan.manual_review_items.append(
            {
                "item": "onnx_mapping",
                "reason": "ONNX evidence exists, but the requested action did not map to concrete ONNX nodes.",
                "confidence": "low",
            }
        )

    _finalize_status(plan, locally_valid=True, traversed_required_edge=traversed_required_edge)
    return plan
