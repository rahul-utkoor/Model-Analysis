"""Render end-to-end axis-evidence-to-DFA bridge reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from typing import Any

from experimental.axis_transfer_analysis.loop_ir import access_form
from experimental.axis_transfer_analysis.report import render_patterns, render_relations
from experimental.dfa_pruning_propagation.report import render_final_facts, render_trace_table
from experimental.dfa_pruning_propagation.worklist import result_to_dict
from experimental.pruning_analysis_bridge.bridge_ir import BridgeResult


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def render_bridge_trace(result: BridgeResult) -> str:
    lines = ["| stage | message | evidence |", "| --- | --- | --- |"]
    for event in result.bridge_trace:
        lines.append(f"| {event.stage} | {event.message} | {'; '.join(event.evidence)} |")
    return "\n".join(lines)


def render_markdown(result: BridgeResult) -> str:
    lines = [
        f"# Pruning Analysis Bridge: {result.example_name}",
        "",
        "## Bridge Summary",
        "",
        "Semantic roles are not assigned directly. They are derived from axis-transfer evidence and then consumed by DFA transfer functions.",
        "",
        render_bridge_trace(result),
        "",
        "## Original Loop / Access Region",
        "",
        "| op | access form |",
        "| --- | --- |",
        *[f"| {op.op_id} | `{access_form(op)}` |" for op in result.region_spec.ops],
        "",
        "## Axis-Transfer Evidence",
        "",
        render_relations(result.axis_summary),
        "",
        "## Recognized Patterns",
        "",
        render_patterns(result.pattern_matches),
        "",
        "## Constructed DFA Graph",
        "",
        "```text",
        result.dfa_graph.pretty_print(),
        "```",
        "",
        "## Seed Facts",
        "",
        *[f"- `{fact.describe()}`" for fact in result.seed_facts],
        "",
        "## DFA Propagation Trace",
        "",
        render_trace_table(result.dfa_result),
        "",
        "## Final Facts",
        "",
        render_final_facts(result.dfa_result),
        "",
        "## Interpretation",
        "",
        result.summary["interpretation"],
        "",
        "Graph names are syntax. Loop/access relations provide evidence. Pattern recognition derives semantic roles. DFA computes propagation.",
        "",
    ]
    return "\n".join(lines)


def render_text(result: BridgeResult, *, show_axis_evidence: bool, show_dfa_trace: bool) -> str:
    lines = [
        f"Pruning Analysis Bridge: {result.example_name}",
        "Semantic roles are derived from axis-transfer evidence, not assigned directly.",
        "",
        f"Selected pattern: {result.summary['selected_pattern']}",
    ]
    if show_axis_evidence:
        lines.extend(["", "Axis-transfer evidence:", render_relations(result.axis_summary)])
    if show_dfa_trace:
        lines.extend(["", "DFA propagation trace:", render_trace_table(result.dfa_result)])
    lines.extend(["", "Final facts:", render_final_facts(result.dfa_result), "", "Interpretation:", f"  {result.summary['interpretation']}", ""])
    return "\n".join(lines)


def render_json(result: BridgeResult) -> str:
    payload = {
        "example_name": result.example_name,
        "region_spec": asdict(result.region_spec),
        "axis_summary": asdict(result.axis_summary),
        "pattern_matches": [asdict(pattern) for pattern in result.pattern_matches],
        "dfa": result_to_dict(result.dfa_result),
        "bridge_trace": [asdict(event) for event in result.bridge_trace],
        "summary": result.summary,
    }
    return json.dumps(_jsonable(payload), indent=2)
