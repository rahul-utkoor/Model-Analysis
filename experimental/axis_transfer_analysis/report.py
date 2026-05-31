"""Render loop/access axis-transfer analysis reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from typing import Any

from experimental.axis_transfer_analysis.axis_relations import RegionAxisSummary
from experimental.axis_transfer_analysis.examples import Example
from experimental.axis_transfer_analysis.loop_ir import access_form
from experimental.axis_transfer_analysis.pattern_recognition import PatternMatch


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def render_relations(summary: RegionAxisSummary) -> str:
    lines = ["| op | source axis | target axis | relation | confidence | proof |", "| --- | --- | --- | --- | --- | --- |"]
    for op_summary in summary.op_summaries:
        for transfer in op_summary.transfers:
            target = f"{transfer.target_tensor}.{transfer.target_axis}" if transfer.target_tensor and transfer.target_axis else "-"
            lines.append(
                f"| {op_summary.op_id} | {transfer.source_tensor}.{transfer.source_axis} | {target} | "
                f"{transfer.relation.value} | {transfer.confidence} | {transfer.proof} |"
            )
    return "\n".join(lines)


def render_patterns(patterns: list[PatternMatch]) -> str:
    if not patterns:
        return "_None recognized._"
    lines = ["| pattern | ops | status | evidence | explanation |", "| --- | --- | --- | --- | --- |"]
    for pattern in patterns:
        lines.append(
            f"| {pattern.pattern_kind.value} | {', '.join(pattern.ops)} | {pattern.status} | "
            f"{'; '.join(pattern.evidence)} | {pattern.explanation} |"
        )
    return "\n".join(lines)


def render_markdown(example: Example, summary: RegionAxisSummary, patterns: list[PatternMatch]) -> str:
    lines = [
        f"# Axis Transfer Analysis: {example.name}",
        "",
        "## Region Summary",
        "",
        example.description,
        "",
        "## Loop / Access Form",
        "",
        "| op | access form |",
        "| --- | --- |",
        *[f"| {op.op_id} | `{access_form(op)}` |" for op in example.region.ops],
        "",
        "## Axis Transfer Summary",
        "",
        render_relations(summary),
        "",
        "## Reduced Axes",
        "",
        *(_render_axis_lists(summary, "reduced_axes") or ["_None._"]),
        "",
        "## Preserved Axes",
        "",
        *(_render_axis_lists(summary, "preserved_axes") or ["_None._"]),
        "",
        "## Protected / Blocked Axes",
        "",
        *(_render_protected_blocked(summary) or ["_None._"]),
        "",
        "## Recognized Pruning Patterns",
        "",
        render_patterns(patterns),
        "",
        "## Interpretation",
        "",
        example.interpretation,
        "",
        "Semantic roles should be derived from evidence: graph topology + axis roles + loop/access relations, not directly assigned from names.",
        "",
    ]
    return "\n".join(lines)


def _render_axis_lists(summary: RegionAxisSummary, field: str) -> list[str]:
    return [f"- `{op_summary.op_id}`: `{axis}`" for op_summary in summary.op_summaries for axis in getattr(op_summary, field)]


def _render_protected_blocked(summary: RegionAxisSummary) -> list[str]:
    lines: list[str] = []
    for op_summary in summary.op_summaries:
        lines.extend(f"- `{op_summary.op_id}` protected: `{axis}`" for axis in op_summary.protected_axes)
        lines.extend(f"- `{op_summary.op_id}` blocked: `{axis}`" for axis in op_summary.blocked_axes)
    return lines


def render_text(example: Example, summary: RegionAxisSummary, patterns: list[PatternMatch], *, show_relations: bool, show_patterns: bool) -> str:
    lines = [f"Axis Transfer Analysis: {example.name}", example.description, "", "Loop/access form:"]
    lines.extend(f"  - {op.op_id}: {access_form(op)}" for op in example.region.ops)
    if show_relations:
        lines.extend(["", "Axis relations:", render_relations(summary)])
    if show_patterns:
        lines.extend(["", "Recognized patterns:", render_patterns(patterns)])
    lines.extend(["", "Interpretation:", f"  {example.interpretation}", ""])
    return "\n".join(lines)


def render_json(example: Example, summary: RegionAxisSummary, patterns: list[PatternMatch]) -> str:
    payload = {
        "example": example.name,
        "description": example.description,
        "region": asdict(example.region),
        "axis_summary": asdict(summary),
        "patterns": [asdict(pattern) for pattern in patterns],
        "interpretation": example.interpretation,
    }
    return json.dumps(_jsonable(payload), indent=2)
