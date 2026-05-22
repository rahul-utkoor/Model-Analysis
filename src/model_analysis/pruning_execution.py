"""Data model and reports for executable pruning attempts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class LinearPruneSpec:
    module_name: str
    prune_dim: str
    indices: list[int]
    original_shape: list[int] | None
    new_shape: list[int] | None
    reason: str


@dataclass
class AppliedPruneRecord:
    module_name: str
    module_type: str
    prune_dim: str
    indices: list[int]
    old_weight_shape: list[int]
    new_weight_shape: list[int]
    old_bias_shape: list[int] | None
    new_bias_shape: list[int] | None
    status: str
    reason: str


@dataclass
class PruningExecutionReport:
    execution_id: str
    model_name: str
    source_model_dir: str
    output_model_dir: str
    action_id: str | None
    plan_id: str | None
    status: str
    applied_records: list[AppliedPruneRecord] = field(default_factory=list)
    skipped_records: list[AppliedPruneRecord] = field(default_factory=list)
    rejected_records: list[AppliedPruneRecord] = field(default_factory=list)
    before_summary: dict[str, Any] = field(default_factory=dict)
    after_summary: dict[str, Any] = field(default_factory=dict)
    diff_summary: dict[str, Any] = field(default_factory=dict)
    rollback_manifest_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def pruning_execution_report_to_dict(report: PruningExecutionReport) -> dict[str, Any]:
    return asdict(report)


def write_pruning_execution_report_json(report: PruningExecutionReport, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(pruning_execution_report_to_dict(report), indent=2), encoding="utf-8")


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def pruning_execution_report_to_markdown(report: PruningExecutionReport) -> str:
    data = pruning_execution_report_to_dict(report)
    before_params = data.get("before_summary", {}).get("parameter_summary", {}).get("total_parameters")
    after_params = data.get("after_summary", {}).get("parameter_summary", {}).get("total_parameters")
    diff = data.get("diff_summary", {})
    metadata = data.get("metadata", {})
    repair_plan = metadata.get("repair_plan")
    repair_transactions = metadata.get("repair_transactions", [])
    smoke_tests = metadata.get("forward_smoke_tests", {})
    lines = [
        f"# Pruning Execution Report: {report.execution_id}",
        "",
        "## Status",
        "",
        f"- `{report.status}`",
        "",
        "## Source Model",
        "",
        f"- `{report.source_model_dir}`",
        "",
        "## Output Model",
        "",
        f"- `{report.output_model_dir}`",
        "",
        "## Requested Plan / Action",
        "",
        f"- Plan ID: `{report.plan_id}`",
        f"- Action ID: `{report.action_id}`",
        "",
        "## Applied Pruning Records",
        "",
        _markdown_table(data["applied_records"], ["module_name", "prune_dim", "indices", "old_weight_shape", "new_weight_shape", "status", "reason"]),
        "",
        "## Skipped Records",
        "",
        _markdown_table(data["skipped_records"], ["module_name", "prune_dim", "indices", "status", "reason"]),
        "",
        "## Rejected Records",
        "",
        _markdown_table(data["rejected_records"], ["module_name", "prune_dim", "indices", "status", "reason"]),
        "",
        "## Before / After Summary",
        "",
        f"- Parameters before: `{before_params}`",
        f"- Parameters after: `{after_params}`",
        "",
        "## Structural Diff",
        "",
        f"- Parameter delta: `{diff.get('parameter_delta')}`",
        f"- Changed Linear layers: `{len(diff.get('changed_linear_layers', []))}`",
        "",
        "## Rollback",
        "",
        f"- Manifest: `{report.rollback_manifest_path}`",
        "- Rollback means deleting the generated output directory and using the original source model directory.",
        "",
        "## Repair Plan",
        "",
        f"- Status: `{repair_plan.get('status') if isinstance(repair_plan, dict) else None}`",
        f"- Repairs: `{len(repair_plan.get('repair_specs', [])) if isinstance(repair_plan, dict) else 0}`",
        "",
        "## Repair Transactions",
        "",
        _markdown_table(
            repair_transactions,
            ["repair_id", "source_module", "target_module", "source_old_shape", "source_new_shape", "target_old_shape", "target_new_shape", "status", "reason"],
        ),
        "",
        "## Forward Smoke Validation",
        "",
        _markdown_table(
            [
                {"phase": phase, **result}
                for phase, result in smoke_tests.items()
                if isinstance(result, dict)
            ],
            ["phase", "status", "input_kind", "error_type", "error_message"],
        ),
        "",
        "## Structural Consistency Notes",
        "",
        "- Paired repair metadata, when present, records source and target Linear dimension changes applied as repair transactions.",
        "- Forward smoke validation only checks that a minimal forward pass executes and returns summarizable outputs.",
        "",
        "## Caveats",
        "",
        "- This milestone only supports Linear module surgery.",
        "- It does not prove end-to-end model correctness.",
        "- Ambiguous transformer pruning may require coordinated updates to adjacent layers.",
        "- ONNX is not rewritten.",
        "",
    ]
    return "\n".join(lines)
