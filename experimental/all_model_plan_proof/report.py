"""Render all-model static pruning propagation proof reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from experimental.all_model_plan_proof.proof_model import AllModelPlanProof, ModelPlanProof, PlanProofCell


def _cell_row(cell: PlanProofCell) -> str:
    return (
        f"| {cell.layer_index} | {str(cell.found_artifact).lower()} | {cell.evidence_tier} | "
        f"{cell.recognized_pattern or '-'} | {str(cell.dfa_reached_fixed_point).lower()} | {cell.verdict} | "
        f"{'; '.join(cell.warnings[:1]) or '-'} |"
    )


def render_model_markdown(model: ModelPlanProof) -> str:
    summary = model.summary
    ffn_rows = "\n".join(_cell_row(cell) for cell in model.ffn_cells)
    attention_rows = "\n".join(_cell_row(cell) for cell in model.attention_value_cells)
    qk_rows = "\n".join(_cell_row(cell) for cell in model.qk_blocker_cells) or "| - | - | unavailable | - | false | not discovered | - |"
    return f"""# {model.model_name} Propagation Plan Proof

## Expected

- Layers evaluated: `{model.layer_count}`
- FFN plans: `{summary.ffn_expected}`
- Attention value-path plans: `{summary.attention_expected}`
- Total complete propagation plans: `{summary.total_expected}`
- Notes: {model.notes}

## Summary

- FFN proven: `{summary.ffn_proven}/{summary.ffn_expected}`
- Attention value paths proven: `{summary.attention_proven}/{summary.attention_expected}`
- QK blockers proven: `{summary.qk_blockers_proven}/{summary.qk_blockers_expected}`
- Total proven: `{summary.total_proven}/{summary.total_expected}`
- Native evidence plans: `{summary.native_evidence_count}`
- Fallback plans: `{summary.fallback_count}`
- Partial: `{summary.partial_count}`
- Missing: `{summary.missing_count}`
- Unsupported: `{summary.unsupported_count}`
- Failed: `{summary.failed_count}`
- Final verdict: `{model.final_verdict}`

## FFN Per Layer

| Layer | Artifact | Evidence tier | Pattern | DFA fixed point | Verdict | Warning |
| --- | --- | --- | --- | --- | --- | --- |
{ffn_rows}

## Attention Value Path Per Layer

| Layer | Artifact | Evidence tier | Pattern | DFA fixed point | Verdict | Warning |
| --- | --- | --- | --- | --- | --- | --- |
{attention_rows}

## QK Blockers Per Layer

QK score contractions are blockers, not pruning plans.

| Layer | Artifact | Evidence tier | Pattern | DFA fixed point | Verdict | Warning |
| --- | --- | --- | --- | --- | --- | --- |
{qk_rows}
"""


def render_index_markdown(proof: AllModelPlanProof) -> str:
    rows = [
        f"| {model.model_name} | {model.layer_count} | {model.summary.ffn_proven}/{model.summary.ffn_expected} | "
        f"{model.summary.attention_proven}/{model.summary.attention_expected} | "
        f"{model.summary.total_proven}/{model.summary.total_expected} | {model.summary.native_evidence_count} | "
        f"{model.summary.fallback_count} | {model.summary.partial_count} | {model.summary.missing_count} | "
        f"{model.summary.unsupported_count} | {model.final_verdict} |"
        for model in proof.models
    ]
    details = "\n\n".join(render_model_markdown(model).replace(f"# {model.model_name}", f"### {model.model_name}", 1) for model in proof.models)
    limitations = "\n".join(f"- {item}" for item in proof.limitations)
    aggregate = proof.aggregate
    return f"""# All-Model Static Pruning Propagation Plan Proof

## Executive Summary

| Model | Layers | FFN proven | Attention value proven | Total proven / expected | Native evidence | Fallback | Partial | Missing | Unsupported | Final verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

## Aggregate Summary

- Total expected plans: `{aggregate.total_expected}`
- Total proven plans: `{aggregate.total_proven}`
- Native evidence plans: `{aggregate.native_evidence_count}`
- Python affine/access evidence plans: `{aggregate.access_evidence_count}`
- Fallback plans: `{aggregate.fallback_count}`
- Partial plans: `{aggregate.partial_count}`
- Missing plans: `{aggregate.missing_count}`
- Unsupported plans: `{aggregate.unsupported_count}`
- Failed plans: `{aggregate.failed_count}`
- Evidence tier counts: `{json.dumps(aggregate.evidence_tier_counts, sort_keys=True)}`
- Verdict counts: `{json.dumps(aggregate.verdict_counts, sort_keys=True)}`

## Evidence Tier Definitions

- `native_mlir_dependence_evidence`: the MLIR-linked local tool emitted dependence facts.
- `actual_loop_access_evidence`: Python affine/access extraction proved the local relation.
- `high_level_mlir_dialect_evidence`: emitted MLIR and conservative topology evidence justified lowering.
- `onnx_hint_fallback`: local ONNX topology and shape evidence supplied the available proof.
- `unavailable`: no supported evidence exists yet.

## Model Details

{details}

## QK Blockers

QK score contractions are blockers, not pruning plans. The Q/K feature axis is reduced and mixed in `QK^T`, so simple one-to-one producer-output deadness propagation is invalid.

## Limitations

{limitations}

This is static evidence and proof reporting only. It does not execute pruning, choose channel indices, mutate model weights, or evaluate accuracy.
"""


def write_report_bundle(output_dir: str | Path, proof: AllModelPlanProof, output_format: str = "both") -> list[Path]:
    output = Path(output_dir)
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(proof)
    written: list[Path] = []
    if output_format in {"json", "both"}:
        path = output / "index.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    if output_format in {"markdown", "both"}:
        path = output / "index.md"
        path.write_text(render_index_markdown(proof), encoding="utf-8")
        written.append(path)
    for model in proof.models:
        stem = model.artifact_name
        (models_dir / f"{stem}.json").write_text(json.dumps(asdict(model), indent=2) + "\n", encoding="utf-8")
        (models_dir / f"{stem}.md").write_text(render_model_markdown(model), encoding="utf-8")
    return written
