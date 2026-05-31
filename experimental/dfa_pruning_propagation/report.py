"""Human-readable and JSON reporting for DFA worklist results."""

from __future__ import annotations

import json

from experimental.dfa_pruning_propagation.examples import Example
from experimental.dfa_pruning_propagation.lattice import FactKind
from experimental.dfa_pruning_propagation.worklist import AnalysisResult, result_to_dict


def render_semantic_roles(result: AnalysisResult) -> str:
    lines = ["| node label | op kind | semantic role | confidence | evidence |", "| --- | --- | --- | --- | --- |"]
    for node in result.graph.nodes.values():
        annotation = result.annotations.nodes[node.node_id]
        lines.append(
            f"| {node.name} | {node.op_kind} | {annotation.semantic_role.value} | "
            f"{annotation.confidence} | {'; '.join(annotation.evidence)} |"
        )
    return "\n".join(lines)


def render_patterns(result: AnalysisResult) -> str:
    if not result.annotations.patterns:
        return "_None detected._"
    return "\n".join(
        f"- `{pattern.pattern.value}`: `{', '.join(pattern.node_ids)}` ({'; '.join(pattern.evidence)})"
        for pattern in result.annotations.patterns
    )


def render_trace_table(result: AnalysisResult) -> str:
    lines = ["| step | node | action | output fact | explanation |", "| --- | --- | --- | --- | --- |"]
    for event in result.trace:
        lines.append(f"| {event.step} | {event.node} | {event.action} | {event.output_fact} | {event.explanation} |")
    return "\n".join(lines)


def render_final_facts(result: AnalysisResult) -> str:
    lines = ["| axis | fact | reason |", "| --- | --- | --- |"]
    for axis, fact in sorted(result.state.items()):
        if fact.kind != FactKind.UNKNOWN:
            lines.append(f"| {axis.label()} | {fact.kind.value} | {fact.reason} |")
    return "\n".join(lines)


def render_markdown(example: Example, result: AnalysisResult) -> str:
    blocked = [event for event in result.blocked_events]
    lines = [
        f"# DFA Pruning Propagation: {example.name}",
        "",
        "## Graph Summary",
        "",
        example.description,
        "",
        "```text",
        example.graph.pretty_print(),
        "```",
        "",
        "## Semantic Roles",
        "",
        render_semantic_roles(result),
        "",
        "## Detected Patterns",
        "",
        render_patterns(result),
        "",
        "## Seed Facts",
        "",
        *[f"- `{fact.describe()}`" for fact in example.seed_facts],
        "",
        "## Propagation Trace",
        "",
        render_trace_table(result),
        "",
        "## Final Facts",
        "",
        render_final_facts(result),
        "",
        "## Blocked Facts",
        "",
    ]
    if blocked:
        lines.extend(f"- `{event.output_fact}`: {event.explanation}" for event in blocked)
    else:
        lines.append("_None._")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            example.interpretation,
            "",
            "Sparse-weight pruning creates zeros. Structural pruning creates dead axes. Static pruning propagation proves how dead axes flow through the graph.",
            "",
        ]
    )
    return "\n".join(lines)


def render_text(example: Example, result: AnalysisResult, *, show_trace: bool = False) -> str:
    lines = [
        f"DFA Pruning Propagation: {example.name}",
        example.description,
        "",
        "Seed facts:",
        *[f"  - {fact.describe()}" for fact in example.seed_facts],
        "",
        "Semantic roles:",
        *[
            f"  - {node.name} [{node.op_kind}] -> {result.annotations.nodes[node.node_id].semantic_role.value}"
            for node in result.graph.nodes.values()
        ],
        "",
        "Detected patterns:",
        *[f"  - {pattern.pattern.value}: {', '.join(pattern.node_ids)}" for pattern in result.annotations.patterns],
        "",
    ]
    if show_trace:
        lines.append("Trace:")
        lines.extend(f"  {event.step:03d} {event.action:10s} {event.node}: {event.output_fact}" for event in result.trace)
        lines.append("")
    lines.append("Final facts:")
    lines.extend(
        f"  - {axis.label()} = {fact.kind.value}: {fact.reason}"
        for axis, fact in sorted(result.state.items())
        if fact.kind != FactKind.UNKNOWN
    )
    lines.extend(["", "Interpretation:", f"  {example.interpretation}", ""])
    return "\n".join(lines)


def render_json(example: Example, result: AnalysisResult) -> str:
    return json.dumps({"example": example.name, "description": example.description, "interpretation": example.interpretation, **result_to_dict(result)}, indent=2)
