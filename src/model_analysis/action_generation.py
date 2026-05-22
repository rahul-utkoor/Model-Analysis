"""Generate small deterministic dry-run pruning actions."""

from __future__ import annotations

from model_analysis.dependency_graph import DependencyGraph, PrunableUnit
from model_analysis.pruning_action import PruningAction, make_action_id


def _bound_for(unit: PrunableUnit, dim: str) -> int | None:
    if not unit.shape:
        return None
    if dim in {"out_features", "channel_out", "intermediate_dim"}:
        return unit.shape[0] if unit.shape else None
    if dim in {"in_features", "embedding_dim"}:
        return unit.shape[-1] if unit.shape else None
    return None


def _first_n_indices(unit: PrunableUnit, dim: str, preferred: int) -> list[int]:
    bound = _bound_for(unit, dim)
    if bound is None:
        count = preferred
    else:
        count = min(preferred, max(bound - 1, 0))
    return list(range(max(0, min(count, preferred))))


def _make_action(graph: DependencyGraph, unit: PrunableUnit, dim: str, indices: list[int], reason: str) -> PruningAction | None:
    if not indices:
        return None
    action_id = make_action_id(unit.unit_id, dim, indices, "first_n")
    return PruningAction(
        action_id=action_id,
        model_name=graph.model_name,
        target_unit_id=unit.unit_id,
        target_unit_name=unit.name,
        target_unit_type=unit.unit_type,
        prune_dim=dim,
        indices=indices,
        amount=len(indices),
        fraction=None,
        strategy="first_n",
        reason=reason,
    )


def generate_candidate_actions(graph: DependencyGraph, max_actions_per_unit: int = 3) -> list[PruningAction]:
    """Generate small candidate dry-run actions from graph units."""
    actions: list[PruningAction] = []

    for unit in graph.prunable_units:
        unit_actions: list[PruningAction] = []
        if unit.unit_type == "attention_qkv":
            unit_actions.append(
                _make_action(
                    graph,
                    unit,
                    "num_heads",
                    [0],
                    "Dry-run first attention head pruning; exact head mapping may require manual review.",
                )
            )
            if unit.shape:
                unit_actions.append(
                    _make_action(
                        graph,
                        unit,
                        "hidden_dim",
                        _first_n_indices(unit, "hidden_dim", 4),
                        "Dry-run first hidden-dimension chunk for attention QKV structure.",
                    )
                )
        elif unit.unit_type == "mlp_expansion":
            unit_actions.append(
                _make_action(
                    graph,
                    unit,
                    "out_features" if "out_features" in unit.prunable_dims else "intermediate_dim",
                    _first_n_indices(unit, "out_features", 4),
                    "Dry-run first intermediate channels in MLP expansion.",
                )
            )
        elif unit.unit_type == "mlp_projection":
            dim = "in_features" if "in_features" in unit.prunable_dims else "intermediate_dim"
            unit_actions.append(
                _make_action(
                    graph,
                    unit,
                    dim,
                    _first_n_indices(unit, "in_features", 4),
                    "Dry-run first intermediate input channels in MLP projection.",
                )
            )
        elif unit.unit_type in {"linear", "gemm", "matmul"}:
            unit_actions.append(
                _make_action(
                    graph,
                    unit,
                    "out_features" if "out_features" in unit.prunable_dims else unit.prunable_dims[0],
                    _first_n_indices(unit, "out_features", 4),
                    "Dry-run first output channels for a linear or matrix projection.",
                )
            )
        elif unit.unit_type == "conv":
            unit_actions.append(
                _make_action(
                    graph,
                    unit,
                    "channel_out",
                    _first_n_indices(unit, "channel_out", 1),
                    "Dry-run first output channel for a convolution or patch projection.",
                )
            )
        elif unit.unit_type == "embedding":
            unit_actions.append(
                _make_action(
                    graph,
                    unit,
                    "embedding_dim",
                    _first_n_indices(unit, "embedding_dim", 4),
                    "Dry-run first embedding dimensions; tied output head and vocabulary semantics require review.",
                )
            )

        actions.extend(action for action in unit_actions[:max_actions_per_unit] if action is not None)

    return actions
