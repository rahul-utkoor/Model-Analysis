"""Execute conservative Linear-only pruning plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model_analysis.dependency_graph import DependencyGraph
from model_analysis.linear_pruning import get_module_by_name, prune_linear_layer, replace_module_by_name
from model_analysis.pruning_action import PruningPlan, pruning_plan_from_dict, pruning_plan_to_dict
from model_analysis.pruning_diff import compute_structural_diff
from model_analysis.pruning_execution import AppliedPruneRecord, LinearPruneSpec, PruningExecutionReport
from model_analysis.structural_inventory import summarize_torch_model


EXECUTABLE_UNIT_TYPES = {"linear", "mlp_expansion", "mlp_projection", "attention_output"}


def _graph_dict(graph: DependencyGraph | dict[str, Any]) -> dict[str, Any]:
    return graph.to_dict() if isinstance(graph, DependencyGraph) else graph


def _plan_obj(plan: PruningPlan | dict[str, Any]) -> PruningPlan:
    return pruning_plan_from_dict(plan) if isinstance(plan, dict) else plan


def _unit_by_id(graph: DependencyGraph | dict[str, Any]) -> dict[str, Any]:
    data = _graph_dict(graph)
    return {unit.get("unit_id"): unit for unit in data.get("prunable_units", [])}


def _affected_dim_to_linear_dim(affected_dim: str, unit: dict[str, Any]) -> str | None:
    prunable_dims = set(unit.get("prunable_dims", []))
    if affected_dim in {"out_features", "hidden_dim", "intermediate_dim", "channel_out"} and "out_features" in prunable_dims:
        return "out_features"
    if affected_dim in {"in_features", "intermediate_dim", "hidden_dim"} and "in_features" in prunable_dims:
        return "in_features"
    return None


def _module_name_for_unit(unit: dict[str, Any]) -> str | None:
    return unit.get("module_or_node_name") or unit.get("name")


def extract_linear_prune_specs_from_plan(
    plan: PruningPlan | dict,
    dependency_graph: DependencyGraph | dict,
    allow_ambiguous: bool = False,
    only_target: bool = False,
) -> list[LinearPruneSpec]:
    """Translate a dry-run pruning plan into executable Linear specs."""
    plan_obj = _plan_obj(plan)
    if plan_obj.status == "rejected":
        raise ValueError("Cannot execute a rejected pruning plan.")
    if plan_obj.status == "ambiguous" and not allow_ambiguous:
        raise ValueError("Cannot execute an ambiguous pruning plan without allow_ambiguous=True.")

    units = _unit_by_id(dependency_graph)
    candidates = [item for item in plan_obj.affected_units if item.get("unit_id") == plan_obj.action.target_unit_id] if only_target else plan_obj.affected_units
    specs: list[LinearPruneSpec] = []
    seen: set[tuple[str, str]] = set()

    for affected in candidates:
        unit = units.get(affected.get("unit_id"))
        if not unit or unit.get("source") != "torch":
            continue
        if unit.get("unit_type") not in EXECUTABLE_UNIT_TYPES:
            continue
        module_name = _module_name_for_unit(unit)
        prune_dim = _affected_dim_to_linear_dim(affected.get("affected_dim", plan_obj.action.prune_dim), unit)
        if not module_name or not prune_dim:
            continue
        key = (module_name, prune_dim)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            LinearPruneSpec(
                module_name=module_name,
                prune_dim=prune_dim,
                indices=list(affected.get("indices") or plan_obj.action.indices),
                original_shape=unit.get("shape"),
                new_shape=None,
                reason=affected.get("reason", "Translated from pruning plan affected unit."),
            )
        )
    return specs


def _empty_record(spec: LinearPruneSpec, status: str, reason: str) -> AppliedPruneRecord:
    return AppliedPruneRecord(
        module_name=spec.module_name,
        module_type="Linear",
        prune_dim=spec.prune_dim,
        indices=spec.indices,
        old_weight_shape=[],
        new_weight_shape=[],
        old_bias_shape=None,
        new_bias_shape=None,
        status=status,
        reason=reason,
    )


def _record_from_metadata(spec: LinearPruneSpec, metadata: dict[str, Any], status: str, reason: str) -> AppliedPruneRecord:
    return AppliedPruneRecord(
        module_name=spec.module_name,
        module_type="Linear",
        prune_dim=spec.prune_dim,
        indices=spec.indices,
        old_weight_shape=metadata["old_weight_shape"],
        new_weight_shape=metadata["new_weight_shape"],
        old_bias_shape=metadata["old_bias_shape"],
        new_bias_shape=metadata["new_bias_shape"],
        status=status,
        reason=reason,
    )


def execute_linear_pruning_plan(
    model: torch.nn.Module,
    model_name: str,
    source_model_dir: Path,
    output_model_dir: Path,
    plan: PruningPlan | dict,
    dependency_graph: DependencyGraph | dict,
    tokenizer_or_processor: object | None = None,
    allow_ambiguous: bool = False,
    only_target: bool = False,
    dry_run: bool = False,
) -> PruningExecutionReport:
    """Execute a Linear-only pruning plan against an in-memory model."""
    plan_obj = _plan_obj(plan)
    execution_id = f"{model_name}__{plan_obj.action.action_id}"
    config = {"name": model_name, "hf_id": model_name, "task": "pruning-execution"}
    before_summary = summarize_torch_model(model, model_name, config)
    applied: list[AppliedPruneRecord] = []
    skipped: list[AppliedPruneRecord] = []
    rejected: list[AppliedPruneRecord] = []

    try:
        specs = extract_linear_prune_specs_from_plan(plan_obj, dependency_graph, allow_ambiguous=allow_ambiguous, only_target=only_target)
    except ValueError as exc:
        report = PruningExecutionReport(
            execution_id=execution_id,
            model_name=model_name,
            source_model_dir=str(source_model_dir),
            output_model_dir=str(output_model_dir),
            action_id=plan_obj.action.action_id,
            plan_id=plan_obj.plan_id,
            status="rejected",
            rejected_records=[AppliedPruneRecord("", "Linear", "", [], [], [], None, None, "rejected", str(exc))],
            before_summary=before_summary,
            after_summary=before_summary,
            diff_summary=compute_structural_diff(before_summary, before_summary),
            metadata={"dry_run": dry_run, "only_target": only_target},
        )
        return report

    if not specs:
        rejected.append(AppliedPruneRecord("", "Linear", "", [], [], [], None, None, "rejected", "No executable Linear prune specs could be extracted."))

    for spec in specs:
        try:
            module = get_module_by_name(model, spec.module_name)
            if not isinstance(module, torch.nn.Linear):
                rejected.append(_empty_record(spec, "rejected", "Target module is not torch.nn.Linear."))
                continue
            if dry_run:
                metadata = {
                    "old_weight_shape": list(module.weight.shape),
                    "new_weight_shape": list(module.weight.shape),
                    "old_bias_shape": list(module.bias.shape) if module.bias is not None else None,
                    "new_bias_shape": list(module.bias.shape) if module.bias is not None else None,
                }
                skipped.append(_record_from_metadata(spec, metadata, "skipped", "Dry run: model was not modified."))
                continue
            new_module, metadata = prune_linear_layer(module, spec.prune_dim, spec.indices)
            replace_module_by_name(model, spec.module_name, new_module)
            applied.append(_record_from_metadata(spec, metadata, "applied", "Linear prune applied."))
        except Exception as exc:  # noqa: BLE001 - report all execution failures as rejected records.
            rejected.append(_empty_record(spec, "rejected", str(exc)))

    after_summary = summarize_torch_model(model, model_name, config)
    diff_summary = compute_structural_diff(before_summary, after_summary)
    status = "success" if applied and not rejected else "partial" if applied and rejected else "rejected" if rejected else "success"

    if not dry_run and applied and hasattr(model, "save_pretrained"):
        output_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_model_dir)
        if tokenizer_or_processor is not None and hasattr(tokenizer_or_processor, "save_pretrained"):
            tokenizer_or_processor.save_pretrained(output_model_dir)

    return PruningExecutionReport(
        execution_id=execution_id,
        model_name=model_name,
        source_model_dir=str(source_model_dir),
        output_model_dir=str(output_model_dir),
        action_id=plan_obj.action.action_id,
        plan_id=plan_obj.plan_id,
        status=status,
        applied_records=applied,
        skipped_records=skipped,
        rejected_records=rejected,
        before_summary=before_summary,
        after_summary=after_summary,
        diff_summary=diff_summary,
        metadata={"dry_run": dry_run, "only_target": only_target, "num_specs": len(specs)},
    )
