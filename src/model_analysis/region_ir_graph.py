"""Graph traversal helpers for RegionDimensionIR constraint analysis."""

from __future__ import annotations

from collections import deque
from typing import Any


def _direction(constraint: dict[str, Any]) -> str:
    explicit = constraint.get("direction")
    if explicit in {"forward", "backward", "bidirectional", "none"}:
        return explicit
    relation = constraint.get("relation")
    if relation == "fanout":
        return "forward"
    if relation == "blocks":
        return "none"
    if relation in {"same_indices", "eq", "join_equal", "preserve", "reshape_map", "unknown"}:
        return "bidirectional"
    return "bidirectional"


def build_region_constraint_adjacency(ir: dict) -> dict:
    """Build deterministic directed/bidirectional adjacency over region dimensions."""
    adjacency: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for original in ir.get("constraint_equations", []):
        constraint = dict(original)
        constraint["inferred_direction"] = _direction(constraint)
        lhs, rhs = constraint.get("lhs"), constraint.get("rhs")
        for dimension in (lhs, rhs):
            adjacency.setdefault(dimension, {"outgoing": [], "incoming": [], "bidirectional": [], "none": []})
        direction = constraint["inferred_direction"]
        if direction == "bidirectional":
            adjacency[lhs]["bidirectional"].append(constraint)
            adjacency[rhs]["bidirectional"].append(constraint)
        elif direction == "forward":
            adjacency[lhs]["outgoing"].append(constraint)
            adjacency[rhs]["incoming"].append(constraint)
        elif direction == "backward":
            adjacency[rhs]["outgoing"].append(constraint)
            adjacency[lhs]["incoming"].append(constraint)
        else:
            adjacency[lhs]["none"].append(constraint)
            if rhs != lhs:
                adjacency[rhs]["none"].append(constraint)
    for entry in adjacency.values():
        for key, constraints in entry.items():
            entry[key] = sorted(constraints, key=lambda item: item.get("constraint_id", ""))
    return adjacency


def get_region_equivalence_class_for_dimension(ir: dict, dimension_var_id: str) -> dict | None:
    for equivalent in ir.get("equivalence_classes", []):
        if dimension_var_id in equivalent.get("members", []):
            return equivalent
    return None


def find_region_constraints_touching_dimension(ir: dict, dimension_var_id: str) -> list[dict]:
    return sorted(
        [
            item for item in ir.get("constraint_equations", [])
            if item.get("lhs") == dimension_var_id or item.get("rhs") == dimension_var_id
        ],
        key=lambda item: item.get("constraint_id", ""),
    )


def _neighbors(adjacency: dict, dimension: str, direction: str) -> list[dict]:
    item = adjacency.get(dimension, {"outgoing": [], "incoming": [], "bidirectional": [], "none": []})
    if direction == "forward":
        return item["outgoing"] + item["bidirectional"] + item["none"]
    if direction == "backward":
        return item["incoming"] + item["bidirectional"] + item["none"]
    return item["outgoing"] + item["incoming"] + item["bidirectional"] + item["none"]


def _other_dimension(constraint: dict, current: str) -> str:
    return constraint.get("rhs") if constraint.get("lhs") == current else constraint.get("lhs")


def extract_region_slice(
    ir: dict,
    root_dimension: str,
    direction: str,
    include_blocking: bool = True,
    max_depth: int | None = None,
):
    """Extract a deterministic region-aware propagation/constraint slice."""
    from model_analysis.region_ir_analysis import RegionPropagationSlice

    dimensions_by_id = {item.get("var_id"): item for item in ir.get("dimension_variables", [])}
    adjacency = build_region_constraint_adjacency(ir)
    visited_dimensions = {root_dimension}
    visited_constraints: dict[str, dict] = {}
    queue = deque([(root_dimension, 0)])
    while queue:
        current, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for constraint in _neighbors(adjacency, current, direction):
            if constraint.get("blocking") and not include_blocking:
                continue
            constraint_id = constraint.get("constraint_id")
            visited_constraints[constraint_id] = constraint
            if constraint.get("inferred_direction") == "none":
                continue
            neighbor = _other_dimension(constraint, current)
            if neighbor not in visited_dimensions:
                visited_dimensions.add(neighbor)
                queue.append((neighbor, depth + 1))
    constraints = [visited_constraints[key] for key in sorted(visited_constraints)]
    dimensions = sorted(visited_dimensions)
    return RegionPropagationSlice(
        slice_id=f"region_slice::{direction}::{root_dimension}",
        root_dimension=root_dimension,
        direction=direction,
        dimensions=dimensions,
        constraints=[item.get("constraint_id") for item in constraints],
        blocked_dimensions=sorted(item for item in dimensions if dimensions_by_id.get(item, {}).get("blocked")),
        blocking_constraints=[item.get("constraint_id") for item in constraints if item.get("blocking")],
        unresolved_constraints=[
            item.get("constraint_id") for item in constraints
            if item.get("relation") in {"unknown", "reshape_map"}
        ],
        protected_dimensions=sorted(item for item in dimensions if dimensions_by_id.get(item, {}).get("protected")),
        reason=f"{direction} region-dimension slice from {root_dimension}.",
    )
