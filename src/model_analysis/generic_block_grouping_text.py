"""Text rendering helpers for generic transformer block grouping."""

from __future__ import annotations

from typing import Any

from model_analysis.generic_block_grouping import GenericBlock, generic_block_to_dict


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def generic_block_to_markdown(block: GenericBlock | dict[str, Any]) -> str:
    data = generic_block_to_dict(block) if isinstance(block, GenericBlock) else block
    rows = [
        {
            "#": item.get("ordinal"),
            "group": item.get("display_name"),
            "kind": item.get("group_kind"),
            "category": item.get("semantic_category"),
            "ops": len(item.get("source_ops", [])),
            "class": item.get("pruning_class"),
            "plan": item.get("plan_status"),
            "validation": item.get("validation_status"),
        }
        for item in data.get("grouped_subgraphs", [])
    ]
    return "\n".join(
        [
            f"# Generic Block Grouping: {data.get('block_name')}",
            "",
            f"- Model: `{data.get('model_name')}`",
            f"- Family: `{data.get('family')}`",
            f"- Block kind: `{data.get('block_kind')}`",
            f"- Op range: `{data.get('op_range')}`",
            "",
            "## Groups",
            "",
            _table(rows, ["#", "group", "kind", "category", "ops", "class", "plan", "validation"]),
            "",
        ]
    )
