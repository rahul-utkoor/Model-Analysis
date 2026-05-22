"""Detect conservative paired Linear repairs from pruning plans."""

from __future__ import annotations

from typing import Any

from model_analysis.dependency_graph import DependencyGraph
from model_analysis.pruning_action import PruningPlan, pruning_plan_from_dict
from model_analysis.repair_plan import RepairPlan, RepairSpec


EXECUTABLE_LINEAR_TYPES = {"linear", "mlp_expansion", "mlp_projection", "attention_output"}
AMBIGUOUS_PAIR_EDGE_TYPES = {"shape_dependency", "feeds", "propagation_only"}
MANUAL_EDGE_TYPES = {"head_dimension_coupling", "residual_coupling", "normalization_dependency", "embedding_tying"}


def _graph_dict(graph: DependencyGraph | dict[str, Any]) -> dict[str, Any]:
    return graph.to_dict() if isinstance(graph, DependencyGraph) else graph


def _plan_obj(plan: PruningPlan | dict[str, Any]) -> PruningPlan:
    return pruning_plan_from_dict(plan) if isinstance(plan, dict) else plan


def _units_by_id(graph: DependencyGraph | dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {unit.get("unit_id"): unit for unit in _graph_dict(graph).get("prunable_units", [])}


def _edges_from_graph(graph: DependencyGraph | dict[str, Any]) -> list[dict[str, Any]]:
    return list(_graph_dict(graph).get("dependency_edges", []))


def _unit_module(unit: dict[str, Any] | None) -> str | None:
    if not unit or unit.get("source") != "torch":
        return None
    return unit.get("module_or_node_name") or unit.get("name")


def _is_linear_unit(unit: dict[str, Any] | None) -> bool:
    return bool(unit and unit.get("source") == "torch" and unit.get("unit_type") in EXECUTABLE_LINEAR_TYPES)


def _looks_expansion(unit: dict[str, Any] | None) -> bool:
    if not unit:
        return False
    name = (unit.get("name") or unit.get("module_or_node_name") or "").lower()
    return unit.get("unit_type") == "mlp_expansion" or "fc1" in name or "intermediate" in name or "up_proj" in name


def _looks_projection(unit: dict[str, Any] | None) -> bool:
    if not unit:
        return False
    name = (unit.get("name") or unit.get("module_or_node_name") or "").lower()
    return unit.get("unit_type") == "mlp_projection" or "fc2" in name or "output.dense" in name or "down_proj" in name


def _edge_key(src: str, dst: str, edge_type: str) -> tuple[str, str, str]:
    return (src, dst, edge_type)


def _plan_steps(plan: PruningPlan) -> list[dict[str, Any]]:
    return [
        {
            "src": step.src_unit_id,
            "dst": step.dst_unit_id,
            "edge_type": step.edge_type,
            "affected_dims": step.affected_dims,
            "indices": step.propagated_indices,
            "confidence": "medium",
            "reason": step.reason,
            "source": "propagation_step",
        }
        for step in plan.propagation_steps
    ]


def _plan_relevant_edges(plan: PruningPlan, graph: DependencyGraph | dict[str, Any]) -> list[dict[str, Any]]:
    seen = set()
    items: list[dict[str, Any]] = []
    for item in _plan_steps(plan):
        key = _edge_key(item["src"], item["dst"], item["edge_type"])
        if key not in seen:
            seen.add(key)
            items.append(item)

    target = plan.action.target_unit_id
    for edge in _edges_from_graph(graph):
        src = edge.get("src")
        dst = edge.get("dst")
        if target not in {src, dst}:
            continue
        key = _edge_key(src, dst, edge.get("edge_type", ""))
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "src": src,
                "dst": dst,
                "edge_type": edge.get("edge_type"),
                "affected_dims": edge.get("affected_dims", []),
                "indices": list(plan.action.indices),
                "confidence": edge.get("confidence", "low"),
                "reason": edge.get("reason", "Dependency edge touches the target action."),
                "source": "dependency_edge",
            }
        )
    return items


def _make_repair_id(repair_type: str, source_module: str, target_module: str) -> str:
    safe_source = source_module.replace("/", "__").replace(":", "_").replace(" ", "_")
    safe_target = target_module.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"{repair_type}__{safe_source}__to__{safe_target}"


def _build_mlp_repair(edge: dict[str, Any], units: dict[str, dict[str, Any]]) -> RepairSpec | None:
    src_unit = units.get(edge["src"])
    dst_unit = units.get(edge["dst"])
    if not (_is_linear_unit(src_unit) and _is_linear_unit(dst_unit)):
        return None

    expansion = src_unit
    projection = dst_unit
    if _looks_projection(src_unit) and _looks_expansion(dst_unit):
        expansion = dst_unit
        projection = src_unit
    elif not (_looks_expansion(src_unit) and _looks_projection(dst_unit)):
        return None

    source_module = _unit_module(expansion)
    target_module = _unit_module(projection)
    if not source_module or not target_module or source_module == target_module:
        return None

    return RepairSpec(
        repair_id=_make_repair_id("mlp_pair", source_module, target_module),
        repair_type="mlp_pair",
        source_module=source_module,
        source_prune_dim="out_features",
        target_module=target_module,
        target_prune_dim="in_features",
        indices=list(edge.get("indices", [])),
        dependency_edge_type=edge.get("edge_type"),
        confidence="medium",
        reason="MLP hidden coupling requires pruning expansion outputs and projection inputs consistently.",
    )


def _build_ambiguous_linear_pair(edge: dict[str, Any], units: dict[str, dict[str, Any]]) -> RepairSpec | None:
    src_unit = units.get(edge["src"])
    dst_unit = units.get(edge["dst"])
    dims = set(edge.get("affected_dims", []))
    if not (_is_linear_unit(src_unit) and _is_linear_unit(dst_unit)):
        return None
    if not dims.intersection({"hidden_dim", "intermediate_dim", "in_features", "out_features"}):
        return None
    source_module = _unit_module(src_unit)
    target_module = _unit_module(dst_unit)
    if not source_module or not target_module or source_module == target_module:
        return None
    return RepairSpec(
        repair_id=_make_repair_id("linear_hidden_pair", source_module, target_module),
        repair_type="linear_hidden_pair",
        source_module=source_module,
        source_prune_dim="out_features",
        target_module=target_module,
        target_prune_dim="in_features",
        indices=list(edge.get("indices", [])),
        dependency_edge_type=edge.get("edge_type"),
        confidence="low",
        reason="Explicit edge suggests a Linear hidden-dimension pair, but mapping is not strong enough without ambiguity override.",
    )


def detect_linear_repair_plan(
    pruning_plan: PruningPlan | dict,
    dependency_graph: DependencyGraph | dict,
    allow_ambiguous: bool = False,
) -> RepairPlan:
    """Detect paired Linear repairs that are explicit in a pruning plan or dependency graph."""
    plan = _plan_obj(pruning_plan)
    units = _units_by_id(dependency_graph)
    repair_specs: list[RepairSpec] = []
    skipped: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []
    seen_repairs: set[str] = set()

    if plan.status == "rejected":
        return RepairPlan(
            repair_plan_id=f"repair__{plan.plan_id}",
            model_name=plan.model_name,
            action_id=plan.action.action_id,
            plan_id=plan.plan_id,
            status="rejected",
            manual_review_items=[{"type": "rejected_plan", "reason": "Cannot repair a rejected pruning plan."}],
            summary={"num_repairs": 0, "num_skipped": 0, "num_manual_review": 1},
        )

    for edge in _plan_relevant_edges(plan, dependency_graph):
        edge_type = edge.get("edge_type")
        if edge_type == "mlp_hidden_coupling":
            spec = _build_mlp_repair(edge, units)
            if spec and spec.repair_id not in seen_repairs:
                seen_repairs.add(spec.repair_id)
                repair_specs.append(spec)
            elif spec:
                continue
            else:
                skipped.append(
                    {
                        "edge_type": edge_type,
                        "src": edge.get("src"),
                        "dst": edge.get("dst"),
                        "confidence": edge.get("confidence", "low"),
                        "reason": "MLP coupling was present but could not be mapped to expansion/projection Linear modules.",
                    }
                )
            continue

        if edge_type in AMBIGUOUS_PAIR_EDGE_TYPES:
            spec = _build_ambiguous_linear_pair(edge, units)
            if spec and allow_ambiguous and spec.repair_id not in seen_repairs:
                seen_repairs.add(spec.repair_id)
                repair_specs.append(spec)
            elif spec:
                skipped.append(
                    {
                        "edge_type": edge_type,
                        "src": edge.get("src"),
                        "dst": edge.get("dst"),
                        "confidence": "low",
                        "reason": "Linear hidden pair requires allow_ambiguous=True before it becomes executable.",
                    }
                )
            continue

        if edge_type in MANUAL_EDGE_TYPES:
            manual_review.append(
                {
                    "type": "unsupported_repair_edge",
                    "edge_type": edge_type,
                    "src": edge.get("src"),
                    "dst": edge.get("dst"),
                    "reason": "This dependency type is intentionally not auto-repaired in Milestone 7.",
                }
            )

    if repair_specs and (skipped or manual_review):
        status = "partial"
    elif repair_specs and all(spec.confidence in {"high", "medium"} for spec in repair_specs):
        status = "executable"
    elif repair_specs:
        status = "ambiguous"
    else:
        status = "rejected"

    return RepairPlan(
        repair_plan_id=f"repair__{plan.plan_id}",
        model_name=plan.model_name,
        action_id=plan.action.action_id,
        plan_id=plan.plan_id,
        repair_specs=repair_specs,
        skipped_repairs=skipped,
        manual_review_items=manual_review,
        status=status,
        summary={
            "num_repairs": len(repair_specs),
            "num_skipped": len(skipped),
            "num_manual_review": len(manual_review),
            "allow_ambiguous": allow_ambiguous,
        },
        metadata={"source_plan_status": plan.status},
    )
