"""Atomic paired Linear pruning repairs."""

from __future__ import annotations

from typing import Any

import torch

from model_analysis.linear_pruning import get_module_by_name, make_keep_indices, prune_linear_layer, replace_module_by_name
from model_analysis.repair_plan import RepairPlan, RepairSpec, RepairTransactionRecord, repair_plan_from_dict


def _spec_obj(spec: RepairSpec | dict[str, Any]) -> RepairSpec:
    return RepairSpec(**spec) if isinstance(spec, dict) else spec


def _plan_obj(plan: RepairPlan | dict[str, Any]) -> RepairPlan:
    return repair_plan_from_dict(plan) if isinstance(plan, dict) else plan


def _shape(layer: torch.nn.Linear) -> list[int]:
    return list(layer.weight.shape)


def _new_source_shape(layer: torch.nn.Linear, indices: list[int]) -> list[int]:
    keep = make_keep_indices(layer.out_features, indices)
    return [len(keep), layer.in_features]


def _new_target_shape(layer: torch.nn.Linear, indices: list[int]) -> list[int]:
    keep = make_keep_indices(layer.in_features, indices)
    return [layer.out_features, len(keep)]


def _rejected_record(spec: RepairSpec, reason: str) -> RepairTransactionRecord:
    return RepairTransactionRecord(
        transaction_id=f"txn__{spec.repair_id}",
        repair_id=spec.repair_id,
        source_module=spec.source_module,
        target_module=spec.target_module,
        source_old_shape=None,
        source_new_shape=None,
        target_old_shape=None,
        target_new_shape=None,
        status="rejected",
        reason=reason,
    )


def apply_paired_linear_repair(
    model: torch.nn.Module,
    repair_spec: RepairSpec | dict[str, Any],
) -> RepairTransactionRecord:
    """Apply one atomic source out_features plus target in_features Linear repair."""
    spec = _spec_obj(repair_spec)
    try:
        if spec.source_module == spec.target_module:
            raise ValueError("Source and target modules must be different for paired repair.")
        if spec.source_prune_dim != "out_features":
            raise ValueError("Paired Linear repair requires source_prune_dim='out_features'.")
        if spec.target_prune_dim != "in_features":
            raise ValueError("Paired Linear repair requires target_prune_dim='in_features'.")

        source = get_module_by_name(model, spec.source_module)
        target = get_module_by_name(model, spec.target_module)
        if not isinstance(source, torch.nn.Linear):
            raise TypeError(f"Source module '{spec.source_module}' is not torch.nn.Linear.")
        if not isinstance(target, torch.nn.Linear):
            raise TypeError(f"Target module '{spec.target_module}' is not torch.nn.Linear.")

        source_old_shape = _shape(source)
        target_old_shape = _shape(target)
        source_new_shape = _new_source_shape(source, spec.indices)
        target_new_shape = _new_target_shape(target, spec.indices)

        source_replacement, _ = prune_linear_layer(source, "out_features", spec.indices)
        target_replacement, _ = prune_linear_layer(target, "in_features", spec.indices)

        replace_module_by_name(model, spec.source_module, source_replacement)
        replace_module_by_name(model, spec.target_module, target_replacement)

        return RepairTransactionRecord(
            transaction_id=f"txn__{spec.repair_id}",
            repair_id=spec.repair_id,
            source_module=spec.source_module,
            target_module=spec.target_module,
            source_old_shape=source_old_shape,
            source_new_shape=source_new_shape,
            target_old_shape=target_old_shape,
            target_new_shape=target_new_shape,
            status="applied",
            reason="Applied atomic paired Linear repair.",
        )
    except Exception as exc:  # noqa: BLE001 - callers need a structured transaction record.
        return _rejected_record(spec, str(exc))


def _dry_run_record(model: torch.nn.Module, spec: RepairSpec) -> RepairTransactionRecord:
    try:
        source = get_module_by_name(model, spec.source_module)
        target = get_module_by_name(model, spec.target_module)
        if not isinstance(source, torch.nn.Linear) or not isinstance(target, torch.nn.Linear):
            raise TypeError("Both repair endpoints must be torch.nn.Linear.")
        if spec.source_prune_dim != "out_features" or spec.target_prune_dim != "in_features":
            raise ValueError("Only source out_features to target in_features repairs are executable.")
        return RepairTransactionRecord(
            transaction_id=f"txn__{spec.repair_id}",
            repair_id=spec.repair_id,
            source_module=spec.source_module,
            target_module=spec.target_module,
            source_old_shape=_shape(source),
            source_new_shape=_new_source_shape(source, spec.indices),
            target_old_shape=_shape(target),
            target_new_shape=_new_target_shape(target, spec.indices),
            status="skipped",
            reason="Dry run: paired Linear repair was validated but not applied.",
        )
    except Exception as exc:  # noqa: BLE001
        return _rejected_record(spec, str(exc))


def apply_repair_plan(
    model: torch.nn.Module,
    repair_plan: RepairPlan | dict[str, Any],
    dry_run: bool = False,
    strict: bool = False,
) -> list[RepairTransactionRecord]:
    """Apply or validate all executable paired Linear repairs in a repair plan."""
    plan = _plan_obj(repair_plan)
    records: list[RepairTransactionRecord] = []
    for spec in plan.repair_specs:
        record = _dry_run_record(model, spec) if dry_run else apply_paired_linear_repair(model, spec)
        records.append(record)
        if strict and record.status == "rejected":
            break
    return records
