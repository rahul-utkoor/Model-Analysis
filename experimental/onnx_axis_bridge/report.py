"""Render local ONNX-subgraph axis bridge reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from typing import Any

from experimental.axis_transfer_analysis.loop_ir import access_form
from experimental.axis_transfer_analysis.report import render_patterns, render_relations
from experimental.dfa_pruning_propagation.report import render_final_facts, render_trace_table
from experimental.dfa_pruning_propagation.worklist import result_to_dict
from experimental.onnx_axis_bridge.bridge_runner import LoweredRegionResult, OnnxAxisBridgeResult


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _render_hint_table(result: OnnxAxisBridgeResult) -> str:
    lines = ["| hint | confidence | nodes | evidence |", "| --- | --- | --- | --- |"]
    for hint in result.pattern_hints:
        lines.append(f"| {hint.kind.value} | {hint.confidence} | {', '.join(hint.nodes)} | {'; '.join(hint.evidence)} |")
    return "\n".join(lines)


def _render_lowered_region(item: LoweredRegionResult, ordinal: int) -> list[str]:
    lines = [
        f"### {ordinal}. {item.hint.kind.value}",
        "",
        f"Confidence: `{item.hint.confidence}`",
        "",
        "| op | loop/access form |",
        "| --- | --- |",
        *[f"| {op.op_id} | `{access_form(op)}` |" for op in item.region_spec.ops],
        "",
        "#### Axis-Transfer Summary",
        "",
        render_relations(item.axis_summary),
        "",
        "#### Pattern Recognition",
        "",
        render_patterns(item.pattern_matches),
        "",
    ]
    if item.bridge_result:
        lines.extend(
            [
                "#### DFA Propagation Result",
                "",
                "```text",
                item.bridge_result.dfa_graph.pretty_print(),
                "```",
                "",
                "Seed facts:",
                "",
                *[f"- `{fact.describe()}`" for fact in item.bridge_result.seed_facts],
                "",
                render_trace_table(item.bridge_result.dfa_result),
                "",
                "Final facts:",
                "",
                render_final_facts(item.bridge_result.dfa_result),
                "",
            ]
        )
    elif item.warning:
        lines.extend([f"_DFA propagation skipped: {item.warning}._", ""])
    return lines


def render_markdown(result: OnnxAxisBridgeResult) -> str:
    lines = [
        "# ONNX Subgraph Axis Bridge",
        "",
        "## ONNX Subgraph Summary",
        "",
        f"- path: `{result.onnx_path}`",
        f"- graph: `{result.subgraph.graph_name}`",
        f"- nodes: `{result.graph_summary.num_nodes}`",
        f"- op counts: `{result.graph_summary.op_type_counts}`",
        f"- inputs: `{list(result.subgraph.graph_inputs)}`",
        f"- outputs: `{list(result.subgraph.graph_outputs)}`",
        f"- initializers: `{list(result.subgraph.initializers)}`",
        "",
        "## Pattern Hints",
        "",
        _render_hint_table(result),
        "",
        "## Lowered Loop / Access Regions",
        "",
    ]
    if result.lowered_regions:
        for ordinal, item in enumerate(result.lowered_regions, start=1):
            lines.extend(_render_lowered_region(item, ordinal))
    else:
        lines.extend(["_No supported local pattern was lowered._", ""])
    lines.extend(["## Warnings", ""])
    lines.extend([f"- {warning}" for warning in result.warnings] or ["_None._"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This bridge does not infer semantics from ONNX names alone. It uses the ONNX subgraph to recover local structure and shape evidence, then lowers recognized patterns to loop/access summaries.",
            "",
            "This is not full ONNX-to-MLIR lowering. Future MLIR affine/linalg/scf extraction can replace the template-lowering step.",
            "",
        ]
    )
    return "\n".join(lines)


def render_text(result: OnnxAxisBridgeResult, *, show_summary: bool, show_hints: bool, show_axis: bool, show_dfa: bool) -> str:
    lines = ["ONNX Subgraph Axis Bridge", f"path: {result.onnx_path}"]
    if show_summary:
        lines.extend([f"nodes: {result.graph_summary.num_nodes}", f"op counts: {result.graph_summary.op_type_counts}"])
    if show_hints:
        lines.extend(["", "Pattern hints:", *[f"  - {hint.kind.value} ({hint.confidence}): {'; '.join(hint.evidence)}" for hint in result.pattern_hints]])
    for item in result.lowered_regions:
        lines.extend(["", f"Lowered region: {item.hint.kind.value}"])
        if show_axis:
            lines.append(render_relations(item.axis_summary))
        if show_dfa and item.bridge_result:
            lines.extend(["DFA propagation:", render_final_facts(item.bridge_result.dfa_result)])
    if result.warnings:
        lines.extend(["", "Warnings:", *[f"  - {warning}" for warning in result.warnings]])
    return "\n".join(lines) + "\n"


def _lowered_to_dict(item: LoweredRegionResult) -> dict[str, Any]:
    payload = {
        "hint": asdict(item.hint),
        "region_spec": asdict(item.region_spec),
        "axis_summary": asdict(item.axis_summary),
        "pattern_matches": [asdict(pattern) for pattern in item.pattern_matches],
        "warning": item.warning,
    }
    if item.bridge_result:
        payload["dfa_result"] = result_to_dict(item.bridge_result.dfa_result)
        payload["dfa_summary"] = item.bridge_result.summary
    return payload


def render_json(result: OnnxAxisBridgeResult) -> str:
    payload = {
        "onnx_path": result.onnx_path,
        "subgraph": asdict(result.subgraph),
        "graph_summary": asdict(result.graph_summary),
        "pattern_hints": [asdict(hint) for hint in result.pattern_hints],
        "lowered_regions": [_lowered_to_dict(item) for item in result.lowered_regions],
        "summary": result.summary,
        "warnings": result.warnings,
    }
    return json.dumps(_jsonable(payload), indent=2)
