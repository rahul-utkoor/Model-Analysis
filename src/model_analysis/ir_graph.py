"""Graph traversal helpers for pruning Dimension IR analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any


def _constraint_dict(constraint: Any) -> dict[str, Any]:
    return asdict(constraint) if hasattr(constraint, "__dataclass_fields__") else dict(constraint)


def build_constraint_adjacency(ir: dict) -> dict:
    """Build incoming/outgoing/bidirectional constraint adjacency keyed by dimension id."""
    adjacency: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for constraint in ir.get("constraint_equations", []):
        item = _constraint_dict(constraint)
        lhs = item.get("lhs")
        rhs = item.get("rhs")
        if lhs not in adjacency:
            adjacency[lhs] = {"outgoing": [], "incoming": [], "bidirectional": []}
        if rhs not in adjacency:
            adjacency[rhs] = {"outgoing": [], "incoming": [], "bidirectional": []}
        direction = item.get("direction")
        if direction == "bidirectional":
            adjacency[lhs]["bidirectional"].append(item)
            adjacency[rhs]["bidirectional"].append(item)
        elif direction == "backward":
            adjacency[lhs]["outgoing"].append(item)
            adjacency[rhs]["incoming"].append(item)
        else:
            adjacency[lhs]["outgoing"].append(item)
            adjacency[rhs]["incoming"].append(item)
    for entry in adjacency.values():
        for key in entry:
            entry[key] = sorted(entry[key], key=lambda value: value.get("constraint_id", ""))
    return adjacency


def get_equivalence_class_for_dimension(ir: dict, dimension_var_id: str) -> dict | None:
    for eq_class in ir.get("equivalence_classes", []):
        if dimension_var_id in eq_class.get("members", []):
            return eq_class
    return None


def find_constraints_touching_dimension(ir: dict, dimension_var_id: str) -> list[dict]:
    constraints = [
        constraint
        for constraint in ir.get("constraint_equations", [])
        if constraint.get("lhs") == dimension_var_id or constraint.get("rhs") == dimension_var_id
    ]
    return sorted(constraints, key=lambda value: value.get("constraint_id", ""))


def _neighbors_for_direction(adjacency: dict, dimension: str, direction: str) -> list[dict]:
    entry = adjacency.get(dimension, {"outgoing": [], "incoming": [], "bidirectional": []})
    if direction == "forward":
        return entry["outgoing"] + entry["bidirectional"]
    if direction == "backward":
        return entry["incoming"] + entry["bidirectional"]
    return entry["outgoing"] + entry["incoming"] + entry["bidirectional"]


def _other_dimension(constraint: dict, current: str) -> str:
    lhs = constraint.get("lhs")
    rhs = constraint.get("rhs")
    return rhs if lhs == current else lhs


def extract_slice(
    ir: dict,
    root_dimension: str,
    direction: str,
    include_blocking: bool = True,
    max_depth: int | None = None,
):
    """Extract a deterministic propagation slice from a root dimension."""
    from model_analysis.ir_analysis import PropagationSlice

    adjacency = build_constraint_adjacency(ir)
    visited_dims = {root_dimension}
    visited_constraints: dict[str, dict] = {}
    queue = deque([(root_dimension, 0)])
    while queue:
        current, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for constraint in _neighbors_for_direction(adjacency, current, direction):
            if constraint.get("blocking") and not include_blocking:
                continue
            constraint_id = constraint.get("constraint_id")
            visited_constraints[constraint_id] = constraint
            other = _other_dimension(constraint, current)
            if other not in visited_dims:
                visited_dims.add(other)
                queue.append((other, depth + 1))

    constraints = [visited_constraints[key] for key in sorted(visited_constraints)]
    return PropagationSlice(
        slice_id=f"slice::{direction}::{root_dimension}",
        root_dimension=root_dimension,
        direction=direction,
        dimensions=sorted(visited_dims),
        constraints=[item.get("constraint_id") for item in constraints],
        blocking_constraints=[item.get("constraint_id") for item in constraints if item.get("blocking")],
        unresolved_constraints=[item.get("constraint_id") for item in constraints if item.get("relation") == "unknown"],
        reason=f"{direction} propagation slice from {root_dimension}.",
    )
