"""Render the MLIR evidence coverage matrix as Markdown and JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from experimental.mlir_evidence_coverage.aggregate import CoverageAggregate, CoverageBreakdown, aggregate_to_dict
from experimental.mlir_evidence_coverage.coverage_case import CoverageResult


INTERPRETATION = (
    "Native MLIR dependence evidence currently covers selected important cases, not every propagation case.",
    "FFN/MLP propagation is native-proven for some models and fallback-proven for others.",
    "QK score blocking is native-proven where affine evidence is available.",
    "Attention context value-axis preservation can be native-proven locally, but full value-path propagation requires a seedable subgraph including out_proj.",
    "Residual/LayerNorm protection may still rely on high-level evidence/fallback.",
    "MLIR remains a local evidence generator, while pruning semantics and fixed-point propagation are handled by the analysis framework.",
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value.value if hasattr(value, "value") else value


def result_to_dict(result: CoverageResult) -> dict[str, Any]:
    return _jsonable(result)


def _key_relation(result: CoverageResult) -> str:
    if not result.axis_relations:
        return "-"
    relation = max(
        result.axis_relations,
        key=lambda item: (
            item.relation == "BLOCKED",
            "value_dim" in item.source or "value_context_dim" in item.target,
            item.relation == "REDUCED" and "intermediate_dim" in item.source,
            item.relation == "PROTECTED",
        ),
    )
    return f"{relation.source} -> {relation.target}: {relation.relation}"


def _dfa(result: CoverageResult) -> str:
    if not result.dfa_ran:
        return "not seedable"
    if result.dfa_final_dead_axes and result.dfa_blocked_axes:
        return "deadness proven; protected path blocked"
    if result.dfa_final_dead_axes:
        return "deadness proven"
    if result.dfa_blocked_axes:
        return "blocked"
    if result.dfa_protected_axes:
        return "protected"
    return "fixed point"


def _warnings(result: CoverageResult) -> str:
    if not result.warnings:
        return "-"
    return result.warnings[0].replace("|", "/")


def _breakdown_table(items: dict[str, CoverageBreakdown], heading: str) -> str:
    rows = [
        f"| {name} | {item.cases} | {item.native_proven} | {item.access_proven} | {item.fallback_proven} | "
        f"{item.blocked_as_expected} | {item.partial} | {item.missing} | {item.unknown} | {item.failed} |"
        for name, item in items.items()
    ]
    return f"""## {heading}

| Name | Cases | Native | Access | Fallback | Blocked | Partial | Missing | Unknown | Failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
"""


def render_case_markdown(result: CoverageResult) -> str:
    relations = "\n".join(
        f"- `{item.source}` -> `{item.target}`: `{item.relation}` ({item.confidence}) - {item.proof}"
        for item in result.axis_relations
    ) or "- No axis relations were proven."
    warnings = "\n".join(f"- {warning}" for warning in result.warnings) or "- None recorded."
    return f"""# {result.case.case_id}

- Model: `{result.case.model_name}`
- Layer: `{result.case.layer_index}`
- Pattern: `{result.case.pattern_kind.value}`
- Required for model: `{str(result.case.required_for_model).lower()}`
- ONNX path: `{result.case.onnx_path}`
- Found: `{str(result.found).lower()}`
- ONNX lowered: `{str(result.onnx_lowered).lower()}`
- MLIR artifacts: `{result.mlir_artifacts_count}`
- Dialects: `{", ".join(result.dialect_hints) or "-"}`
- Native pass ran: `{str(result.native_pass_ran).lower()}`
- Native pass return code: `{result.native_pass_returncode}`
- Evidence tier: `{result.evidence_tier.value}`
- Verdict: `{result.verdict.value}`

## Axis Relations

{relations}

## Recognized Patterns

{chr(10).join(f"- `{pattern}`" for pattern in result.recognized_patterns) or "- None."}

## DFA Result

- Ran: `{str(result.dfa_ran).lower()}`
- Final dead axes: `{", ".join(result.dfa_final_dead_axes) or "-"}`
- Blocked axes: `{", ".join(result.dfa_blocked_axes) or "-"}`
- Protected axes: `{", ".join(result.dfa_protected_axes) or "-"}`

## Warnings

{warnings}
"""


def render_index_markdown(results: list[CoverageResult], aggregate: CoverageAggregate) -> str:
    rows = [
        f"| {result.case.model_name} | {result.case.layer_index} | {result.case.pattern_kind.value} | "
        f"{result.case.subgraph_name} | {result.evidence_tier.value} | {result.verdict.value} | "
        f"{', '.join(result.recognized_patterns) or '-'} | {_key_relation(result)} | {_dfa(result)} | {_warnings(result)} |"
        for result in results
    ]
    details = "\n\n".join(render_case_markdown(result).replace(f"# {result.case.case_id}", f"### {result.case.case_id}", 1) for result in results)
    conclusions = "\n".join(f"{index}. {text}" for index, text in enumerate(INTERPRETATION, 1))
    return f"""# MLIR Evidence Coverage Study

## Executive Summary

- Total cases: `{aggregate.total_cases}`
- Found cases: `{aggregate.found_cases}`
- Missing cases: `{aggregate.missing_cases}`
- Native proven: `{aggregate.native_proven}`
- Access proven: `{aggregate.access_proven}`
- Fallback proven: `{aggregate.fallback_proven}`
- Blocked as expected: `{aggregate.blocked_as_expected}`
- Partial: `{aggregate.partial}`
- Unknown: `{aggregate.unknown}`
- Failed: `{aggregate.failed}`
- Evidence tier counts: `{json.dumps(aggregate.evidence_tier_counts, sort_keys=True)}`

## Evidence Tier Definitions

- `native_mlir_dependence_evidence`: the standalone MLIR-linked tool emitted pruning-relevant dependence facts.
- `actual_loop_access_evidence`: Python affine/access extraction reconstructed a supported local relation.
- `high_level_mlir_dialect_evidence`: emitted MLIR operations plus conservative local hints justified template lowering.
- `onnx_hint_fallback`: the local ONNX topology/shape bridge supplied the available proof.
- `unavailable`: no supported evidence was available.

## Coverage Matrix

| Model | Layer | Pattern | Subgraph | Evidence tier | Verdict | Recognized pattern | Key axis relation | DFA result | Warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

{_breakdown_table(aggregate.per_model, "Per-Model Summary")}
{_breakdown_table(aggregate.per_pattern, "Per-Pattern Summary")}
## Case Details

{details}

## Interpretation

{conclusions}
"""


def write_report_bundle(output_dir: str | Path, results: list[CoverageResult], aggregate: CoverageAggregate, output_format: str = "both") -> list[Path]:
    output = Path(output_dir)
    cases_dir = output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": aggregate_to_dict(aggregate), "cases": [result_to_dict(result) for result in results]}
    written: list[Path] = []
    if output_format in {"json", "both"}:
        path = output / "index.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    if output_format in {"markdown", "both"}:
        path = output / "index.md"
        path.write_text(render_index_markdown(results, aggregate), encoding="utf-8")
        written.append(path)
    for result in results:
        result.report_path = str(cases_dir / f"{result.case.case_id}.md")
        (cases_dir / f"{result.case.case_id}.json").write_text(json.dumps(result_to_dict(result), indent=2) + "\n", encoding="utf-8")
        (cases_dir / f"{result.case.case_id}.md").write_text(render_case_markdown(result), encoding="utf-8")
    return written
