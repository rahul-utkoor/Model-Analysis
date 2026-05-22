"""Markdown rendering for pruning action simulation outputs."""

from __future__ import annotations

from typing import Any

from model_analysis.pruning_action import PruningAction, PruningPlan, pruning_action_to_dict, pruning_plan_to_dict


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if not rows:
        return "_None detected._"
    selected = rows[:limit] if limit else rows
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if limit and len(rows) > limit:
        omitted = {columns[0]: "..."}
        if len(columns) > 1:
            omitted[columns[1]] = f"{len(rows) - limit} more rows omitted"
        lines.append("| " + " | ".join(str(omitted.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def pruning_plan_to_markdown(plan: PruningPlan) -> str:
    """Render a pruning plan as Markdown."""
    data = pruning_plan_to_dict(plan)
    action = data["action"]
    lines = [
        f"# Pruning Plan: {action['action_id']}",
        "",
        "## Status",
        "",
        f"- `{plan.status}`",
        "",
        "## Requested Action",
        "",
        f"- Model: `{action['model_name']}`",
        f"- Target unit: `{action['target_unit_id']}`",
        f"- Target name: `{action.get('target_unit_name')}`",
        f"- Target type: `{action.get('target_unit_type')}`",
        f"- Prune dimension: `{action['prune_dim']}`",
        f"- Indices: `{action['indices']}`",
        f"- Strategy: `{action['strategy']}`",
        f"- Reason: {action.get('reason') or '_None provided._'}",
        "",
        "## Affected Units",
        "",
        _markdown_table(data["affected_units"], ["unit_id", "name", "unit_type", "source", "affected_dim", "indices", "reason"], limit=200),
        "",
        "## Propagation Trace",
        "",
        _markdown_table(
            data["propagation_steps"],
            ["step_id", "src_unit_id", "dst_unit_id", "edge_type", "direction", "affected_dims", "propagated_indices", "status", "reason"],
            limit=300,
        ),
        "",
        "## Constraints",
        "",
        _markdown_table(data["constraints"], ["edge_type", "src", "dst", "affected_dims", "reason"], limit=200),
        "",
        "## Conflicts",
        "",
        _markdown_table(data["conflicts"], ["type", "target_unit_id", "prune_dim", "reason"], limit=100),
        "",
        "## Manual Review Items",
        "",
        _markdown_table(data["manual_review_items"], ["item", "reason", "confidence"], limit=200),
        "",
        "## Interpretation",
        "",
        _interpret_status(plan.status),
        "",
    ]
    return "\n".join(lines)


def _interpret_status(status: str) -> str:
    if status == "valid_global":
        return "The requested action propagated through all required graph constraints in this static dry run. It is still not executable pruning until later validation and transformation layers exist."
    if status == "valid_local":
        return "The requested action is locally valid for the target unit and no required coupling was found. This does not modify weights."
    if status == "rejected":
        return "The requested action is unsafe or malformed for the known graph evidence and should not be used as a pruning candidate."
    return "The requested action requires better mapping, shape analysis, or manual review before it can become an executable pruning transform."


def candidate_actions_to_markdown(actions: list[PruningAction]) -> str:
    """Render generated candidate actions as Markdown."""
    rows = [pruning_action_to_dict(action) for action in actions]
    return "\n".join(
        [
            "# Candidate Pruning Actions",
            "",
            _markdown_table(
                rows,
                ["action_id", "target_unit_id", "target_unit_type", "prune_dim", "indices", "reason"],
                limit=500,
            ),
            "",
            "These actions are dry-run candidates only. They do not prune weights and may produce ambiguous plans.",
            "",
        ]
    )
