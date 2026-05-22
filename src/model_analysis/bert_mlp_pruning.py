"""BERT-style MLP block pruning for the intermediate feed-forward dimension."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from model_analysis.forward_validation import (
    ForwardSmokeResult,
    forward_smoke_result_to_dict,
    run_forward_smoke_test,
)
from model_analysis.linear_pruning import get_module_by_name
from model_analysis.paired_linear_pruning import apply_paired_linear_repair
from model_analysis.pruning_diff import compute_structural_diff
from model_analysis.repair_plan import RepairSpec, repair_transaction_record_to_dict
from model_analysis.structural_inventory import summarize_torch_model


@dataclass
class BertMlpBlockTarget:
    model_name: str
    layer_index: int
    intermediate_module: str
    output_module: str
    hidden_size: int | None
    intermediate_size: int | None
    confidence: str
    reason: str


@dataclass
class BertMlpPruneSpec:
    spec_id: str
    model_name: str
    layer_index: int
    intermediate_module: str
    output_module: str
    prune_indices: list[int]
    prune_count: int
    intermediate_size_before: int | None
    intermediate_size_after: int | None
    hidden_size: int | None
    strategy: str
    reason: str


@dataclass
class BertMlpPruningReport:
    execution_id: str
    model_name: str
    source_model_dir: str
    output_model_dir: str | None
    status: str
    spec: BertMlpPruneSpec
    target: BertMlpBlockTarget
    applied_records: list[dict[str, Any]] = field(default_factory=list)
    before_forward_smoke: dict[str, Any] | None = None
    after_forward_smoke: dict[str, Any] | None = None
    before_summary: dict[str, Any] = field(default_factory=dict)
    after_summary: dict[str, Any] | None = None
    diff_summary: dict[str, Any] | None = None
    rollback_manifest_path: str | None = None
    caveats: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


INTERMEDIATE_RE = re.compile(r"^(?P<prefix>(?:bert\.)?encoder\.layer\.(?P<layer>\d+))\.intermediate\.dense$")


def bert_mlp_target_to_dict(target: BertMlpBlockTarget) -> dict[str, Any]:
    return asdict(target)


def bert_mlp_prune_spec_to_dict(spec: BertMlpPruneSpec) -> dict[str, Any]:
    return asdict(spec)


def bert_mlp_pruning_report_to_dict(report: BertMlpPruningReport) -> dict[str, Any]:
    return asdict(report)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def bert_mlp_targets_to_markdown(model_name: str, targets: list[BertMlpBlockTarget]) -> str:
    rows = [bert_mlp_target_to_dict(target) for target in targets]
    return "\n".join(
        [
            f"# BERT MLP Block Targets: {model_name}",
            "",
            _markdown_table(
                rows,
                [
                    "layer_index",
                    "intermediate_module",
                    "output_module",
                    "hidden_size",
                    "intermediate_size",
                    "confidence",
                    "reason",
                ],
            ),
            "",
        ]
    )


def bert_mlp_pruning_report_to_markdown(report: BertMlpPruningReport) -> str:
    data = bert_mlp_pruning_report_to_dict(report)
    target = data["target"]
    spec = data["spec"]
    diff = data.get("diff_summary") or {}
    caveats = "\n".join(f"- {item}" for item in report.caveats)
    return "\n".join(
        [
            "# BERT MLP Block Pruning Report",
            "",
            "## Status",
            "",
            f"- `{report.status}`",
            "",
            "## Target Block",
            "",
            f"- Layer: `{target['layer_index']}`",
            f"- Intermediate module: `{target['intermediate_module']}`",
            f"- Output module: `{target['output_module']}`",
            f"- Hidden size: `{target['hidden_size']}`",
            f"- Intermediate size: `{target['intermediate_size']}`",
            f"- Confidence: `{target['confidence']}`",
            "",
            "## Prune Specification",
            "",
            f"- Spec ID: `{spec['spec_id']}`",
            f"- Strategy: `{spec['strategy']}`",
            f"- Indices: `{spec['prune_indices']}`",
            f"- Intermediate size before: `{spec['intermediate_size_before']}`",
            f"- Intermediate size after: `{spec['intermediate_size_after']}`",
            "",
            "## Applied Structural Changes",
            "",
            _markdown_table(report.applied_records, ["module_name", "prune_dim", "old_shape", "new_shape", "status", "reason"]),
            "",
            "## Before / After Forward Smoke",
            "",
            _markdown_table(
                [
                    {"phase": "before", **(report.before_forward_smoke or {})},
                    {"phase": "after", **(report.after_forward_smoke or {})},
                ],
                ["phase", "status", "input_kind", "error_type", "error_message"],
            ),
            "",
            "## Structural Diff",
            "",
            f"- Parameter delta: `{diff.get('parameter_delta')}`",
            f"- Changed Linear layers: `{len(diff.get('changed_linear_layers', []))}`",
            "",
            "## Output Model",
            "",
            f"- `{report.output_model_dir}`",
            "",
            "## Rollback",
            "",
            f"- `{report.rollback_manifest_path}`",
            "",
            "## Caveats",
            "",
            caveats,
            "",
        ]
    )


def _confidence_for_pair(intermediate: torch.nn.Linear, output: torch.nn.Linear) -> tuple[str, str]:
    if intermediate.out_features != output.in_features:
        return "low", "Intermediate output size does not match output projection input size."
    if intermediate.in_features != output.out_features:
        return "medium", "Intermediate and output hidden dimensions differ from standard BERT shape, but the intermediate pair is size-compatible."
    return "high", "BERT MLP intermediate/output Linear pair has matching intermediate and hidden dimensions."


def detect_bert_mlp_block_targets(model: torch.nn.Module, model_name: str) -> list[BertMlpBlockTarget]:
    """Detect BERT-style intermediate/output dense pairs."""
    modules = dict(model.named_modules())
    targets: list[BertMlpBlockTarget] = []
    for name, module in modules.items():
        match = INTERMEDIATE_RE.match(name)
        if not match or not isinstance(module, torch.nn.Linear):
            continue
        output_name = f"{match.group('prefix')}.output.dense"
        output = modules.get(output_name)
        if not isinstance(output, torch.nn.Linear):
            continue
        confidence, reason = _confidence_for_pair(module, output)
        targets.append(
            BertMlpBlockTarget(
                model_name=model_name,
                layer_index=int(match.group("layer")),
                intermediate_module=name,
                output_module=output_name,
                hidden_size=int(module.in_features),
                intermediate_size=int(module.out_features),
                confidence=confidence,
                reason=reason,
            )
        )
    return sorted(targets, key=lambda item: (0 if item.intermediate_module.startswith("bert.") else 1, item.layer_index))


def get_bert_mlp_block_target(
    model: torch.nn.Module,
    model_name: str,
    layer_index: int,
) -> BertMlpBlockTarget:
    targets = detect_bert_mlp_block_targets(model, model_name)
    for target in targets:
        if target.layer_index == layer_index:
            if target.confidence == "low":
                raise ValueError(f"Layer {layer_index} BERT MLP pair is invalid: {target.reason}")
            return target
    raise ValueError(f"BERT MLP block target for layer {layer_index} was not found.")


def _normalize_indices(indices: list[int]) -> list[int]:
    return sorted(set(indices))


def _indices_from_amount(size: int, count: int, strategy: str) -> list[int]:
    if count <= 0:
        raise ValueError("Prune count must be positive.")
    if strategy == "first_n":
        return list(range(count))
    if strategy == "last_n":
        return list(range(size - count, size))
    if strategy == "every_other":
        return list(range(0, size, 2))[:count]
    raise ValueError(f"Unsupported strategy '{strategy}'.")


def _validate_prune_indices(indices: list[int], intermediate_size: int | None) -> list[int]:
    normalized = _normalize_indices(indices)
    if not normalized:
        raise ValueError("At least one prune index is required.")
    if any(index < 0 for index in normalized):
        raise ValueError("Prune indices must be non-negative.")
    if intermediate_size is not None:
        if any(index >= intermediate_size for index in normalized):
            raise ValueError(f"Prune index out of bounds for intermediate size {intermediate_size}.")
        if len(normalized) >= intermediate_size:
            raise ValueError("Cannot prune all intermediate features.")
    return normalized


def make_bert_mlp_prune_spec(
    target: BertMlpBlockTarget,
    indices: list[int] | None = None,
    count: int | None = None,
    fraction: float | None = None,
    strategy: str = "first_n",
    reason: str | None = None,
) -> BertMlpPruneSpec:
    """Create a validated BERT MLP intermediate-dimension prune spec."""
    provided = sum(value is not None for value in (indices, count, fraction))
    if provided != 1:
        raise ValueError("Exactly one of indices, count, or fraction must be provided.")
    if target.intermediate_size is None:
        raise ValueError("Target intermediate size is unknown.")

    if fraction is not None:
        if fraction <= 0 or fraction >= 1:
            raise ValueError("Fraction must be greater than 0 and less than 1.")
        count = max(1, math.floor(target.intermediate_size * fraction))
    if count is not None:
        indices = _indices_from_amount(target.intermediate_size, count, strategy)
    elif indices is not None and strategy not in {"first_n", "last_n", "every_other", "manual_indices"}:
        raise ValueError(f"Unsupported strategy '{strategy}'.")

    prune_indices = _validate_prune_indices(list(indices or []), target.intermediate_size)
    after_size = target.intermediate_size - len(prune_indices)
    return BertMlpPruneSpec(
        spec_id=f"bert_mlp_layer_{target.layer_index}__{strategy}__{len(prune_indices)}",
        model_name=target.model_name,
        layer_index=target.layer_index,
        intermediate_module=target.intermediate_module,
        output_module=target.output_module,
        prune_indices=prune_indices,
        prune_count=len(prune_indices),
        intermediate_size_before=target.intermediate_size,
        intermediate_size_after=after_size,
        hidden_size=target.hidden_size,
        strategy=strategy,
        reason=reason or "BERT MLP intermediate-dimension pruning.",
    )


def _record_from_transaction(record: dict[str, Any], source_module: str, target_module: str) -> list[dict[str, Any]]:
    return [
        {
            "module_name": source_module,
            "prune_dim": "out_features",
            "old_shape": record.get("source_old_shape"),
            "new_shape": record.get("source_new_shape"),
            "status": record.get("status"),
            "reason": record.get("reason"),
        },
        {
            "module_name": target_module,
            "prune_dim": "in_features",
            "old_shape": record.get("target_old_shape"),
            "new_shape": record.get("target_new_shape"),
            "status": record.get("status"),
            "reason": record.get("reason"),
        },
    ]


def _smoke_dict(result: ForwardSmokeResult | None) -> dict[str, Any] | None:
    return forward_smoke_result_to_dict(result) if result is not None else None


def execute_bert_mlp_pruning(
    model: torch.nn.Module,
    model_name: str,
    source_model_dir: Path,
    output_model_dir: Path | None,
    spec: BertMlpPruneSpec,
    tokenizer_or_processor: object | None = None,
    model_config: dict | None = None,
    dry_run: bool = False,
    smoke_test_before: bool = False,
    smoke_test_after: bool = False,
    device: str | None = None,
) -> BertMlpPruningReport:
    """Execute architecture-specific BERT MLP intermediate pruning."""
    caveats = [
        "This prunes only the MLP intermediate dimension.",
        "It does not prune attention heads.",
        "It preserves hidden size, so residual and LayerNorm dimensions should remain unchanged.",
        "Passing smoke tests does not prove accuracy preservation.",
        "ONNX is not rewritten.",
        "A single-layer intermediate-size change is structurally valid in memory, but standard Hugging Face reload paths may need custom metadata support because BERT config stores one global intermediate_size.",
    ]
    execution_id = f"bert_mlp_layer_{spec.layer_index}__{spec.strategy}__{spec.prune_count}"
    config = dict(model_config or {"name": model_name, "hf_id": model_name, "task": "unit-test"})
    config.setdefault("name", model_name)
    config.setdefault("hf_id", model_name)
    config.setdefault("task", "unit-test")

    before_smoke = run_forward_smoke_test(model, config, tokenizer_or_processor, device=device) if smoke_test_before else None
    before_summary = summarize_torch_model(model, model_name, config)

    try:
        target = get_bert_mlp_block_target(model, model_name, spec.layer_index)
        if target.intermediate_module != spec.intermediate_module or target.output_module != spec.output_module:
            raise ValueError("Prune spec modules do not match the detected BERT MLP target.")
        _validate_prune_indices(spec.prune_indices, target.intermediate_size)
        if dry_run:
            return BertMlpPruningReport(
                execution_id=execution_id,
                model_name=model_name,
                source_model_dir=str(source_model_dir),
                output_model_dir=str(output_model_dir) if output_model_dir else None,
                status="dry_run",
                spec=spec,
                target=target,
                applied_records=[],
                before_forward_smoke=_smoke_dict(before_smoke),
                before_summary=before_summary,
                after_summary=None,
                diff_summary=None,
                caveats=caveats,
                metadata={"dry_run": True},
            )

        repair_spec = RepairSpec(
            repair_id=f"bert_mlp_layer_{spec.layer_index}",
            repair_type="mlp_pair",
            source_module=spec.intermediate_module,
            source_prune_dim="out_features",
            target_module=spec.output_module,
            target_prune_dim="in_features",
            indices=spec.prune_indices,
            dependency_edge_type="bert_mlp_block",
            confidence=target.confidence,
            reason=spec.reason,
        )
        transaction = apply_paired_linear_repair(model, repair_spec)
        transaction_dict = repair_transaction_record_to_dict(transaction)
        if transaction.status != "applied":
            raise RuntimeError(transaction.reason)

        after_summary = summarize_torch_model(model, model_name, config)
        diff_summary = compute_structural_diff(before_summary, after_summary)
        after_smoke = run_forward_smoke_test(model, config, tokenizer_or_processor, device=device) if smoke_test_after else None
        status = "failed" if after_smoke is not None and after_smoke.status == "failed" else "success"

        if output_model_dir is not None and hasattr(model, "save_pretrained"):
            output_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_model_dir)
            if tokenizer_or_processor is not None and hasattr(tokenizer_or_processor, "save_pretrained"):
                tokenizer_or_processor.save_pretrained(output_model_dir)

        return BertMlpPruningReport(
            execution_id=execution_id,
            model_name=model_name,
            source_model_dir=str(source_model_dir),
            output_model_dir=str(output_model_dir) if output_model_dir else None,
            status=status,
            spec=spec,
            target=target,
            applied_records=_record_from_transaction(transaction_dict, spec.intermediate_module, spec.output_module),
            before_forward_smoke=_smoke_dict(before_smoke),
            after_forward_smoke=_smoke_dict(after_smoke),
            before_summary=before_summary,
            after_summary=after_summary,
            diff_summary=diff_summary,
            caveats=caveats,
            metadata={"transaction": transaction_dict},
        )
    except Exception as exc:  # noqa: BLE001 - return a structured failed report for CLI/reporting.
        try:
            target = get_bert_mlp_block_target(model, model_name, spec.layer_index)
        except Exception:
            target = BertMlpBlockTarget(model_name, spec.layer_index, spec.intermediate_module, spec.output_module, spec.hidden_size, spec.intermediate_size_before, "low", str(exc))
        return BertMlpPruningReport(
            execution_id=execution_id,
            model_name=model_name,
            source_model_dir=str(source_model_dir),
            output_model_dir=str(output_model_dir) if output_model_dir else None,
            status="rejected",
            spec=spec,
            target=target,
            before_forward_smoke=_smoke_dict(before_smoke),
            before_summary=before_summary,
            after_summary=None,
            diff_summary=None,
            caveats=caveats,
            metadata={"error_type": exc.__class__.__name__, "error_message": str(exc)},
        )
