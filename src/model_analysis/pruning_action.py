"""Schemas and serialization helpers for dry-run pruning actions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class PruningAction:
    action_id: str
    model_name: str
    target_unit_id: str
    target_unit_name: str | None
    target_unit_type: str | None
    prune_dim: str
    indices: list[int]
    amount: int | None
    fraction: float | None
    strategy: str
    reason: str | None


@dataclass
class PropagationStep:
    step_id: str
    src_unit_id: str
    dst_unit_id: str
    edge_type: str
    direction: str
    affected_dims: list[str]
    propagated_indices: list[int]
    status: str
    reason: str


@dataclass
class PruningPlan:
    plan_id: str
    model_name: str
    action: PruningAction
    affected_units: list[dict[str, Any]] = field(default_factory=list)
    propagation_steps: list[PropagationStep] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    manual_review_items: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ambiguous"
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def pruning_action_to_dict(action: PruningAction) -> dict[str, Any]:
    return asdict(action)


def pruning_plan_to_dict(plan: PruningPlan) -> dict[str, Any]:
    return asdict(plan)


def pruning_plan_from_dict(data: dict[str, Any]) -> PruningPlan:
    return PruningPlan(
        plan_id=data["plan_id"],
        model_name=data["model_name"],
        action=PruningAction(**data["action"]),
        affected_units=data.get("affected_units", []),
        propagation_steps=[PropagationStep(**step) for step in data.get("propagation_steps", [])],
        constraints=data.get("constraints", []),
        conflicts=data.get("conflicts", []),
        manual_review_items=data.get("manual_review_items", []),
        status=data.get("status", "ambiguous"),
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def load_pruning_action_json(path: Path) -> PruningAction:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PruningAction(**data)


def write_pruning_plan_json(plan: PruningPlan, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(pruning_plan_to_dict(plan), indent=2), encoding="utf-8")


def make_action_id(target_unit_id: str, prune_dim: str, indices: list[int], strategy: str) -> str:
    safe_target = target_unit_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    index_part = "-".join(str(index) for index in indices[:8])
    if len(indices) > 8:
        index_part = f"{index_part}-plus{len(indices) - 8}"
    return f"{strategy}__{safe_target}__{prune_dim}__{index_part}"
