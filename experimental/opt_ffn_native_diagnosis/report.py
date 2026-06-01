"""Render OPT FFN native-evidence diagnosis reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from experimental.opt_ffn_native_diagnosis.diagnosis_model import OptFfnNativeDiagnosis, OptFfnNativeDiagnosisReport


def render_layer_markdown(layer: OptFfnNativeDiagnosis) -> str:
    return f"""# OPT FFN Native MLIR Diagnosis: Layer {layer.layer_index}

- Source ONNX: `{layer.onnx_path}`
- FFN-core ONNX: `{layer.core_onnx_path}`
- Core lowering succeeded: `{str(layer.lowering_succeeded).lower()}`
- Dialects: `{", ".join(layer.dialect_hints) or "-"}`
- Native pass ran: `{str(layer.native_pass_ran).lower()}`
- Native pass return code: `{layer.native_pass_returncode}`
- Native relations: `{layer.native_relations_count}`
- Preserved / reduced / mixed: `{layer.preserved_relations}` / `{layer.reduced_relations}` / `{layer.mixed_relations}`
- Fallback detects FFN: `{str(layer.ffn_pattern_detected_by_fallback).lower()}`
- Native detects FFN: `{str(layer.ffn_pattern_detected_by_native).lower()}`
- Original blocker: `{layer.blocker_kind.value}`
- Fix applied: `{str(layer.fix_applied).lower()}`

## Native vs Fallback Evidence

{layer.blocker_explanation}

## Suggested Fix

{layer.suggested_fix}
"""


def render_index_markdown(report: OptFfnNativeDiagnosisReport) -> str:
    rows = [
        f"| {item.layer_index} | `{item.onnx_path}` | {', '.join(item.dialect_hints) or '-'} | {item.native_relations_count} | "
        f"{str(item.ffn_pattern_detected_by_fallback).lower()} | {str(item.ffn_pattern_detected_by_native).lower()} | "
        f"{item.blocker_kind.value} | {item.suggested_fix} |"
        for item in report.layers
    ]
    blocker_rows = "\n".join(f"- `{kind}`: `{count}`" for kind, count in sorted(report.blockers_by_kind.items()))
    return f"""# OPT FFN Native MLIR Evidence Diagnosis

## Executive Summary

- Layers analyzed: `{report.total_layers}`
- Native-proven after FFN-core extraction: `{report.native_proven}`
- Fallback-only: `{report.fallback_only}`
- Failed: `{report.failed}`

## Layer Table

| Layer | ONNX artifact | MLIR dialects | Native relations | Fallback detects FFN | Native detects FFN | Original blocker | Suggested fix |
| --- | --- | --- | ---: | --- | --- | --- | --- |
{chr(10).join(rows)}

## Native vs Fallback Evidence

The original OPT MLP-block artifact includes LayerNorm and residual boundary operations. ONNX-MLIR aborts before affine lowering because the LayerNorm activation input is `f32` while exported scale/bias parameters are `f16`. The high-level ONNX/MLIR topology still detects `fc1 -> activation -> fc2`, but the native pass receives no indexed accesses from that failed lowering.

The applied repair exports the topology-proven FFN core only: `fc1 -> activation -> fc2`. ONNX-MLIR lowers that local evidence unit into Krnl/Affine operations, and the native tool emits preserved and reduced relations sufficient for `FFN_INTERMEDIATE_CHAIN`.

## Blockers Observed on Original Artifacts

{blocker_rows or "- None."}

## Fixes Applied

- Exported read-only OPT FFN-core ONNX evidence artifacts under `artifacts/opt_ffn_native_subgraphs/`.
- Updated MLIR coverage discovery to prefer `mlp_native_core` artifacts over broader `mlp_block` artifacts.
- Kept the native proof criterion unchanged: native dependence JSON must justify preserved and reduced relations.

## Final Status

OPT FFN native evidence: `{report.native_proven}/{report.total_layers}`.

Recommendation: {report.final_recommendation}

This is static evidence diagnosis only. It does not execute pruning or mutate model weights.
"""


def write_report_bundle(output_dir: str | Path, report: OptFfnNativeDiagnosisReport) -> list[Path]:
    output = Path(output_dir)
    layers = output / "layers"
    layers.mkdir(parents=True, exist_ok=True)
    written = [output / "index.json", output / "index.md"]
    written[0].write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    written[1].write_text(render_index_markdown(report), encoding="utf-8")
    for layer in report.layers:
        (layers / f"layer_{layer.layer_index}.json").write_text(json.dumps(asdict(layer), indent=2) + "\n", encoding="utf-8")
        (layers / f"layer_{layer.layer_index}.md").write_text(render_layer_markdown(layer), encoding="utf-8")
    return written
