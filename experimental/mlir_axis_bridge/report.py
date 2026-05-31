"""Render ONNX-MLIR local evidence bridge reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from typing import Any

from experimental.axis_transfer_analysis.loop_ir import access_form
from experimental.axis_transfer_analysis.report import render_patterns, render_relations
from experimental.dfa_pruning_propagation.report import render_final_facts, render_trace_table
from experimental.mlir_axis_bridge.bridge_runner import MlirAxisBridgeResult, MlirRegionResult


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _commands(result: MlirAxisBridgeResult) -> list[str]:
    lines = ["| stage | exit | command | generated files |", "| --- | --- | --- | --- |"]
    for command in result.lowering_result.commands:
        lines.append(f"| {command.stage} | {command.returncode} | `{' '.join(command.command)}` | {', '.join(command.generated_files) or '-'} |")
    return lines


def _artifacts(result: MlirAxisBridgeResult) -> list[str]:
    lines = ["| stage | artifact | bytes | dialect hints |", "| --- | --- | --- | --- |"]
    for artifact in result.artifacts:
        lines.append(f"| {artifact.stage} | `{artifact.path}` | {artifact.size_bytes} | {', '.join(artifact.dialect_hints) or '-'} |")
    return lines


def _region(item: MlirRegionResult, ordinal: int) -> list[str]:
    build = item.axis_build
    lines = [
        f"### {ordinal}. {item.hint.kind.value}",
        "",
        f"- MLIR artifact: `{item.mlir_summary.artifact_path}`",
        f"- evidence source: `{build.evidence_source}`",
        f"- extracted accesses: `{len(item.mlir_summary.access_records)}`",
        f"- dependence relations: `{len(item.mlir_summary.dependence_report.relations) if item.mlir_summary.dependence_report else 0}`",
        "",
    ]
    if build.region_spec and build.axis_summary:
        lines.extend(
            [
                "| op | lowered loop/access form |",
                "| --- | --- |",
                *[f"| {op.op_id} | `{access_form(op)}` |" for op in build.region_spec.ops],
                "",
                "#### Axis-Transfer Evidence",
                "",
                render_relations(build.axis_summary),
                "",
                "#### Recognized Pruning Patterns",
                "",
                render_patterns(build.pattern_matches),
                "",
            ]
        )
    if item.bridge_result:
        lines.extend(
            [
                "#### DFA Propagation Result",
                "",
                "```text",
                item.bridge_result.dfa_graph.pretty_print(),
                "```",
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
        lines.extend([f"_DFA skipped: {item.warning}._", ""])
    return lines


def render_markdown(result: MlirAxisBridgeResult) -> str:
    lines = [
        "# ONNX-MLIR Axis Bridge",
        "",
        "## Input ONNX Subgraph",
        "",
        f"- path: `{result.onnx_path}`",
        "",
        "## Toolchain Status",
        "",
        f"- onnx-mlir: `{result.toolchain_status.onnx_mlir_path}`",
        f"- mlir-opt: `{result.toolchain_status.mlir_opt_path}`",
        "",
        "## ONNX-MLIR Commands Run",
        "",
        *_commands(result),
        "",
        "## Generated MLIR Artifacts",
        "",
        *_artifacts(result),
        "",
        "## Extracted MLIR Operations / Accesses",
        "",
    ]
    for summary in result.mlir_access_summaries:
        lines.extend(
            [
                f"### `{summary.artifact_path}`",
                "",
                f"- operations: `{summary.operation_counts}`",
                f"- loop kinds: `{list(summary.loop_kinds)}`",
                f"- accesses: `{len(summary.access_records)}`",
                f"- dependence relations: `{len(summary.dependence_report.relations) if summary.dependence_report else 0}`",
                "",
            ]
        )
    if result.native_dependence_report:
        lines.extend(
            [
                "## Imported Native Dependence Evidence",
                "",
                f"- analysis tool: `{result.native_dependence_report.analysis_tool}`",
                f"- MLIR file: `{result.native_dependence_report.mlir_file}`",
                f"- relations: `{len(result.native_dependence_report.relations)}`",
                "",
            ]
        )
    if result.emitted_python_dependence_json:
        lines.extend(["## Python Dependence JSON", "", f"- emitted: `{result.emitted_python_dependence_json}`", ""])
    lines.extend(["## Axis Evidence and DFA Results", ""])
    for ordinal, item in enumerate(result.region_results, start=1):
        lines.extend(_region(item, ordinal))
    if not result.region_results:
        lines.extend(["_No supported local pruning pattern was lowered._", ""])
    lines.extend(["## Warnings and Limitations", ""])
    lines.extend([f"- {warning}" for warning in dict.fromkeys(result.warnings)] or ["_None._"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This prototype does not rewrite the pruning pipeline in MLIR. It uses ONNX-MLIR as a local evidence generator for selected ONNX subgraphs.",
            "",
            "Each lowered region reports whether evidence came from `native_mlir_dependence_evidence`, `actual_loop_access_evidence`, `high_level_mlir_dialect_evidence`, or `onnx_hint_fallback`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_text(result: MlirAxisBridgeResult, *, show_toolchain: bool, show_artifacts: bool, show_accesses: bool, show_axis: bool, show_dfa: bool) -> str:
    lines = ["ONNX-MLIR Axis Bridge", f"input: {result.onnx_path}", f"evidence source: {result.evidence_source}"]
    if show_toolchain:
        lines.extend([f"onnx-mlir: {result.toolchain_status.onnx_mlir_path}", f"mlir-opt: {result.toolchain_status.mlir_opt_path}"])
    if show_artifacts:
        lines.extend(["", "Artifacts:", *[f"  - {artifact.stage}: {artifact.path} ({', '.join(artifact.dialect_hints)})" for artifact in result.artifacts]])
    if show_accesses:
        lines.extend(["", "Access summaries:", *[f"  - {summary.artifact_path}: {len(summary.access_records)} accesses, {len(summary.dependence_report.relations) if summary.dependence_report else 0} dependence relations" for summary in result.mlir_access_summaries]])
    for item in result.region_results:
        lines.extend(["", f"{item.hint.kind.value}: {item.axis_build.evidence_source}"])
        if show_axis and item.axis_build.axis_summary:
            lines.append(render_relations(item.axis_build.axis_summary))
        if show_dfa and item.bridge_result:
            lines.extend(["DFA final facts:", render_final_facts(item.bridge_result.dfa_result)])
    return "\n".join(lines) + "\n"


def render_json(result: MlirAxisBridgeResult) -> str:
    payload = {
        "onnx_path": result.onnx_path,
        "toolchain_status": asdict(result.toolchain_status),
        "lowering_result": asdict(result.lowering_result),
        "artifacts": [asdict(artifact) for artifact in result.artifacts],
        "mlir_access_summaries": [asdict(summary) for summary in result.mlir_access_summaries],
        "evidence_source": result.evidence_source,
        "native_dependence_report": asdict(result.native_dependence_report) if result.native_dependence_report else None,
        "emitted_python_dependence_json": result.emitted_python_dependence_json,
        "regions": [
            {
                "hint": asdict(item.hint),
                "axis_build": asdict(item.axis_build),
                "dfa_summary": item.bridge_result.summary if item.bridge_result else None,
                "warning": item.warning,
            }
            for item in result.region_results
        ],
        "summary": result.summary,
        "warnings": result.warnings,
    }
    return json.dumps(_jsonable(payload), indent=2)
