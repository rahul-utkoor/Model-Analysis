"""Render a professor/student-facing cross-evidence pruning proof report."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from experimental.pruning_proof_report.aggregate import ProofAggregate, aggregate_to_dict
from experimental.pruning_proof_report.proof_case import AxisRelationRecord, ProofEvidence


TEACHING_CONCLUSIONS = (
    "Names are syntax; evidence comes from graph/shape/loop/access relations.",
    "FFN propagation is proven by intermediate-axis produced/preserved/consumed structure.",
    "Attention value context is proven by V.value_dim preservation into Context.value_context_dim.",
    "Q/K propagation is blocked because Q/K feature axes are reduced/mixed in the score contraction.",
    "MLIR is used as a selected-subgraph local evidence generator, not as the pruning framework itself.",
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def evidence_to_dict(evidence: ProofEvidence) -> dict[str, Any]:
    return _jsonable(evidence)


def _relation_summary(relations: list[AxisRelationRecord]) -> str:
    if not relations:
        return "-"
    relation = max(
        relations,
        key=lambda item: (
            item.relation == "BLOCKED",
            "value_dim" in item.source or "value_context_dim" in item.target,
            item.relation == "REDUCED" and "intermediate_dim" in item.source,
            "intermediate_dim" in item.source or "intermediate_dim" in item.target,
            item.relation == "PROTECTED",
        ),
    )
    suffix = f" (+{len(relations) - 1})" if len(relations) > 1 else ""
    return f"{relation.source} -> {relation.target}: {relation.relation}{suffix}"


def _dfa_result(evidence: ProofEvidence) -> str:
    dfa = evidence.dfa_summary
    if not dfa.get("ran"):
        return "not seedable"
    if dfa.get("final_dead_axes") and dfa.get("blocked_axes"):
        return "deadness proven; protected path blocked"
    if dfa.get("blocked_axes"):
        return "blocked"
    if dfa.get("final_dead_axes"):
        return "deadness proven"
    if dfa.get("protected_axes"):
        return "protected"
    return "fixed point"


def render_case_markdown(evidence: ProofEvidence) -> str:
    onnx = evidence.onnx_summary
    mlir = evidence.mlir_summary
    relations = "\n".join(
        f"- `{item.source}` -> `{item.target}`: `{item.relation}` ({item.confidence}) - {item.proof}"
        for item in evidence.axis_relations
    ) or "- No axis relations were proven."
    limitations = "\n".join(f"- {item}" for item in evidence.limitations) or "- None recorded."
    return f"""# {evidence.case_id}

- Model: `{evidence.model_name}`
- Layer: `{evidence.layer_index if evidence.layer_index is not None else "n/a"}`
- Subgraph: `{evidence.subgraph_name}`
- ONNX path: `{evidence.onnx_path}`
- Found: `{str(evidence.found).lower()}`
- Evidence source: `{evidence.evidence_source}`
- Verdict: `{evidence.verdict}`

## ONNX Summary

- Nodes: `{onnx.get("num_nodes", 0)}`
- Op counts: `{json.dumps(onnx.get("op_type_counts", {}), sort_keys=True)}`
- Pattern hints: `{", ".join(onnx.get("pattern_hints", [])) or "-"}`

## MLIR Summary

- Toolchain available: `{str(mlir.get("toolchain_available", False)).lower()}`
- Lowering succeeded: `{str(mlir.get("lowering_succeeded", False)).lower()}`
- Dialects: `{", ".join(mlir.get("dialect_hints", [])) or "-"}`
- Native pass available: `{str(mlir.get("native_pass_available", False)).lower()}`
- Native pass return code: `{mlir.get("native_pass_returncode")}`
- Native JSON path: `{mlir.get("native_json_path") or "-"}`

## Axis Relations

{relations}

## Recognized Patterns

{chr(10).join(f"- `{pattern}`" for pattern in evidence.recognized_patterns) or "- None."}

## DFA Summary

```json
{json.dumps(evidence.dfa_summary, indent=2, sort_keys=True)}
```

## Limitations

{limitations}
"""


def render_index_markdown(evidence: list[ProofEvidence], aggregate: ProofAggregate) -> str:
    rows = []
    for item in evidence:
        hints = item.onnx_summary.get("pattern_hints", [])
        dialects = item.mlir_summary.get("dialect_hints", [])
        rows.append(
            f"| {item.case_id} | {item.model_name} | {item.layer_index if item.layer_index is not None else '-'} | "
            f"{item.subgraph_name} | {', '.join(hints) or '-'} | {', '.join(dialects) or '-'} | "
            f"{item.evidence_source} | {_relation_summary(item.axis_relations)} | "
            f"{', '.join(item.recognized_patterns) or '-'} | {_dfa_result(item)} | {item.verdict} |"
        )
    case_details = "\n\n".join(render_case_markdown(item).replace(f"# {item.case_id}", f"### {item.case_id}", 1) for item in evidence)
    conclusions = "\n".join(f"{index}. {text}" for index, text in enumerate(TEACHING_CONCLUSIONS, 1))
    return f"""# Cross-Evidence Pruning Proof Report

## Executive Summary

- Total cases: `{aggregate.cases_total}`
- Cases found: `{aggregate.cases_found}`
- Cases missing: `{aggregate.cases_missing}`
- Proven: `{aggregate.proven}`
- Fallback proven: `{aggregate.fallback_proven}`
- Blocked: `{aggregate.blocked}`
- Partial: `{aggregate.partial}`
- Unknown: `{aggregate.unknown}`
- Failed: `{aggregate.failed}`
- Evidence source counts: `{json.dumps(aggregate.evidence_source_counts, sort_keys=True)}`

## Evidence Hierarchy

`ONNX subgraph -> ONNX-MLIR lowering -> native MLIR dependence evidence / Python affine evidence / fallback -> axis-transfer summary -> pruning pattern -> DFA propagation`

## Summary Table

| Case | Model | Layer | Subgraph | ONNX hint | MLIR dialects | Evidence source | Axis relation | Pattern | DFA result | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Case Details

{case_details}

## Key Teaching Conclusions

{conclusions}
"""


def write_report_bundle(output_dir: str | Path, evidence: list[ProofEvidence], aggregate: ProofAggregate, output_format: str = "both") -> list[Path]:
    output = Path(output_dir)
    cases_dir = output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    payload = {"summary": aggregate_to_dict(aggregate), "cases": [evidence_to_dict(item) for item in evidence]}
    if output_format in {"json", "both"}:
        index_json = output / "index.json"
        index_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(index_json)
    if output_format in {"markdown", "both"}:
        index_md = output / "index.md"
        index_md.write_text(render_index_markdown(evidence, aggregate), encoding="utf-8")
        written.append(index_md)
    for item in evidence:
        case_json = cases_dir / f"{item.case_id}.json"
        case_md = cases_dir / f"{item.case_id}.md"
        case_json.write_text(json.dumps(evidence_to_dict(item), indent=2) + "\n", encoding="utf-8")
        case_md.write_text(render_case_markdown(item), encoding="utf-8")
        written.extend((case_json, case_md))
    return written
