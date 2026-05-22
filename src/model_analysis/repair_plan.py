"""Repair-plan data structures for paired Linear pruning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RepairSpec:
    repair_id: str
    repair_type: str
    source_module: str
    source_prune_dim: str
    target_module: str
    target_prune_dim: str
    indices: list[int]
    dependency_edge_type: str | None
    confidence: str
    reason: str


@dataclass
class RepairPlan:
    repair_plan_id: str
    model_name: str
    action_id: str | None
    plan_id: str | None
    repair_specs: list[RepairSpec] = field(default_factory=list)
    skipped_repairs: list[dict[str, Any]] = field(default_factory=list)
    manual_review_items: list[dict[str, Any]] = field(default_factory=list)
    status: str = "rejected"
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairTransactionRecord:
    transaction_id: str
    repair_id: str
    source_module: str
    target_module: str
    source_old_shape: list[int] | None
    source_new_shape: list[int] | None
    target_old_shape: list[int] | None
    target_new_shape: list[int] | None
    status: str
    reason: str


def repair_plan_to_dict(plan: RepairPlan) -> dict[str, Any]:
    return asdict(plan)


def repair_plan_from_dict(data: dict[str, Any]) -> RepairPlan:
    return RepairPlan(
        repair_plan_id=data["repair_plan_id"],
        model_name=data["model_name"],
        action_id=data.get("action_id"),
        plan_id=data.get("plan_id"),
        repair_specs=[RepairSpec(**item) for item in data.get("repair_specs", [])],
        skipped_repairs=data.get("skipped_repairs", []),
        manual_review_items=data.get("manual_review_items", []),
        status=data.get("status", "rejected"),
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def repair_transaction_record_to_dict(record: RepairTransactionRecord) -> dict[str, Any]:
    return asdict(record)


def repair_transaction_records_to_dict(records: list[RepairTransactionRecord]) -> list[dict[str, Any]]:
    return [repair_transaction_record_to_dict(record) for record in records]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def repair_plan_to_markdown(plan: RepairPlan) -> str:
    data = repair_plan_to_dict(plan)
    return "\n".join(
        [
            f"# Repair Plan: {plan.repair_plan_id}",
            "",
            "## Status",
            "",
            f"- `{plan.status}`",
            "",
            "## Executable Repairs",
            "",
            _markdown_table(
                data["repair_specs"],
                [
                    "repair_id",
                    "repair_type",
                    "source_module",
                    "source_prune_dim",
                    "target_module",
                    "target_prune_dim",
                    "indices",
                    "confidence",
                    "reason",
                ],
            ),
            "",
            "## Skipped Repairs",
            "",
            _markdown_table(data["skipped_repairs"], ["edge_type", "src", "dst", "reason", "confidence"]),
            "",
            "## Manual Review Items",
            "",
            _markdown_table(data["manual_review_items"], ["type", "edge_type", "src", "dst", "reason"]),
            "",
            "## Interpretation",
            "",
            "- Executable repairs are limited to paired PyTorch `nn.Linear` dimension updates.",
            "- Attention, residual, normalization, and embedding dependencies remain manual-review items in this milestone.",
            "- A repair plan records structural consistency candidates; it does not claim semantic or accuracy preservation.",
            "",
        ]
    )


def repair_transaction_records_to_markdown(records: list[RepairTransactionRecord]) -> str:
    rows = repair_transaction_records_to_dict(records)
    return "\n".join(
        [
            "# Repair Transactions",
            "",
            _markdown_table(
                rows,
                [
                    "transaction_id",
                    "repair_id",
                    "source_module",
                    "target_module",
                    "source_old_shape",
                    "source_new_shape",
                    "target_old_shape",
                    "target_new_shape",
                    "status",
                    "reason",
                ],
            ),
            "",
        ]
    )
