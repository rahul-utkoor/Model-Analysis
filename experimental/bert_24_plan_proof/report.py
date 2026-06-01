"""Render the BERT 24-plan proof as Markdown and JSON."""

from __future__ import annotations

import json
from pathlib import Path

from experimental.bert_24_plan_proof.runner import Bert24PlanProof, proof_to_dict


def render_markdown(proof: Bert24PlanProof) -> str:
    summary = proof.summary
    rows = [
        f"| {layer.layer_index} | {layer.ffn_plan_status}/{layer.ffn_validation_status} | {layer.ffn_evidence_tier} | "
        f"{layer.ffn_verdict} | {layer.attention_path_status}/{layer.attention_mapping_status} | "
        f"{layer.attention_evidence_tier} | {layer.attention_verdict} |"
        for layer in proof.layers
    ]
    limitations = "\n".join(f"- {item}" for item in proof.limitations) or "- None recorded."
    return f"""# BERT 24-Plan Static Pruning Propagation Proof

## Expected Plan Structure

Each BERT encoder layer has one `FFN_INTERMEDIATE_CHAIN` plan and one `ATTENTION_VALUE_PATH` plan.

Expected: `12 x 2 = 24` complete propagation plans.

## Aggregate

- Layers total: `{summary.layers_total}`
- Expected plans: `{summary.expected_plans}`
- FFN expected/found/proven: `{summary.ffn_expected}` / `{summary.ffn_found}` / `{summary.ffn_proven}`
- Attention expected/found/seedable/proven: `{summary.attention_expected}` / `{summary.attention_found}` / `{summary.attention_seedable}` / `{summary.attention_proven}`
- Total proven: `{summary.total_proven}`
- Missing/partial/failed: `{summary.missing}` / `{summary.partial}` / `{summary.failed}`
- Final verdict: `{summary.final_verdict}`

## Evidence Sources

| Layer | FFN plan status | FFN evidence tier | FFN verdict | Attention value-path status | Attention evidence tier | Attention verdict |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Interpretation

- FFN intermediate pruning is structurally propagated through activation into output dense input.
- Attention value-path pruning propagates from output projection input through context value axis to value projection output.
- QK score paths remain blockers and are not counted as pruning plans.
- MLIR dependence evidence is used as local axis-transfer proof, while DFA computes propagation.

## Limitations

{limitations}

This is static evidence and proof reporting only. It does not execute pruning or mutate model weights.
"""


def write_report_bundle(output_dir: str | Path, proof: Bert24PlanProof) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "index.json"
    md_path = output / "index.md"
    json_path.write_text(json.dumps(proof_to_dict(proof), indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(proof), encoding="utf-8")
    return json_path, md_path
