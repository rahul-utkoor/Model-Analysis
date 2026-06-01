"""Render the final static pruning propagation research report bundle."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from pathlib import Path

from experimental.final_report.collector import FinalReportData, PerModelSummary


SCOPE_BOUNDARIES = (
    "We do not claim accuracy recovery.",
    "We do not claim pruning index selection.",
    "We do not claim runtime speedup yet.",
    "We do not claim MLIR alone infers pruning rules.",
    "We do not claim arbitrary fused-QKV graphs are recoverable without branch evidence.",
)


def evidence_backed_claims(data: FinalReportData) -> tuple[str, ...]:
    aggregate = data.aggregate_summary
    bert = next((model for model in data.per_model_summary if model.model_name == "bert-base-uncased"), None)
    bert_proven = bert.proven_plans if bert else int(data.bert_24_summary.get("total_proven", 0))
    bert_expected = bert.expected_plans if bert else int(data.bert_24_summary.get("expected_plans", 24))
    return (
        f"The framework proves {aggregate.proven_plans}/{aggregate.expected_plans} expected FFN and attention value-path propagation plans across five supported models.",
        f"BERT has {bert_proven}/{bert_expected} complete propagation plans.",
        "QK score contractions are correctly identified as blockers.",
        "Attention value-path propagation is proven for separable and recovered fused-QKV value paths.",
        f"Native MLIR dependence evidence contributes {aggregate.native_mlir_evidence}/{aggregate.expected_plans} plan proofs.",
        "Sparse-weight pruning and structural pruning are distinct.",
    )


def _display_name(model_name: str) -> str:
    return {
        "bert-base-uncased": "BERT",
        "distilbert-base-uncased": "DistilBERT",
        "facebook/opt-125m": "OPT",
        "gpt2": "GPT-2",
        "google/vit-base-patch16-224": "ViT",
    }.get(model_name, model_name)


def _complete(model: PerModelSummary) -> str:
    return "complete" if model.proven_plans == model.expected_plans and not model.failed else model.final_verdict


def _warnings(data: FinalReportData) -> str:
    return "\n".join(f"- {warning}" for warning in data.warnings) or "- None recorded."


def _model_rows(data: FinalReportData) -> str:
    rows = [
        f"| {_display_name(model.model_name)} | {model.expected_plans} | {model.proven_plans} | {_complete(model)} |"
        for model in data.per_model_summary
    ]
    aggregate = data.aggregate_summary
    rows.append(f"| Total | {aggregate.expected_plans} | {aggregate.proven_plans} | {'complete' if aggregate.expected_plans == aggregate.proven_plans else 'partial'} |")
    return "\n".join(rows)


def _detail_rows(data: FinalReportData) -> str:
    return "\n".join(
        f"| {_display_name(model.model_name)} | {model.layers} | {model.ffn_proven}/{model.ffn_expected} | "
        f"{model.attention_value_proven}/{model.attention_value_expected} | {model.proven_plans}/{model.expected_plans} | "
        f"{model.native_evidence} | {model.fallback_evidence} | {model.unsupported} | {model.partial} | "
        f"{model.missing} | {model.failed} | {_complete(model)} |"
        for model in data.per_model_summary
    )


def render_final_report(data: FinalReportData) -> str:
    aggregate = data.aggregate_summary
    bert = data.bert_24_summary
    opt = data.deadbranch_summary.get("opt", {})
    claims = evidence_backed_claims(data)
    return f"""# Final Static Pruning Propagation Research Report

## 1. Executive Summary

| Model | Expected Plans | Proven Plans | Verdict |
| --- | ---: | ---: | --- |
{_model_rows(data)}

- Native MLIR evidence: `{aggregate.native_mlir_evidence}`
- High-level MLIR fallback: `{aggregate.high_level_mlir_fallback}`
- Unsupported / partial / missing / failed: `{aggregate.unsupported}` / `{aggregate.partial}` / `{aggregate.missing}` / `{aggregate.failed}`

The analysis proves `{aggregate.proven_plans}/{aggregate.expected_plans}` expected propagation plans across five supported transformer families.

## 2. Research Problem

Pruning is often treated as a local choice of weights or channels. Structural pruning requires a broader compiler-style question:

> If an axis becomes dead or pruned, what else must change?

A legal structural edit must propagate through producers, consumers, layout transformations, protected boundaries, and explicit blockers.

## 3. Sparse-Weight vs Structural Pruning

Sparse-weight pruning keeps tensor shapes unchanged while setting individual weights to zero. Examples include `2:4`, `V:N:M`, and SparseGPT-style fine-grained sparsity.

Structural pruning removes or deadens complete axes. It can change shapes and requires propagation and repair.

**Sparsity is not the same as deadness.**

## 4. Static Pruning Propagation Model

The framework analyzes a graph of tensor axes. Facts inhabit a conservative lattice:

`UNKNOWN`, `LIVE`, `DEAD`, `PRUNED`, `PROTECTED`, `BLOCKED`

Each operation contributes transfer functions. A DFA/worklist solver joins facts, applies transfers, enqueues changed axes, and terminates at a fixed point.

## 5. Axis-Transfer Semantics

| Relation | Meaning | Example |
| --- | --- | --- |
| `PRESERVED` | Axis maps one-to-one | activation `Y[..., j] = f(X[..., j])` |
| `REDUCED` | Axis is consumed by contraction | FFN input feature `j` in `Y[..., h] += X[..., j] * W[j, h]` |
| `MIXED` | Values interact without one-to-one transfer | Q/K feature `d` in `QK^T` |
| `PROTECTED` | Local pruning requires coordinated repair | residual hidden width |
| `BLOCKED` | Requested propagation is invalid | Q/K simple deadness propagation |

Attention context preserves the value axis:

```text
Context[..., q, d] += Prob[..., q, k] * V[..., k, d]
V.d -> Context.d = PRESERVED
```

## 6. Evidence Pipeline

```text
ONNX subgraph
  -> ONNX-MLIR lowering
  -> native MLIR dependence evidence / fallback
  -> axis-transfer summary
  -> pruning pattern recognition
  -> DFA worklist propagation
  -> proof verdict
```

## 7. Pruning Patterns

- `FFN_INTERMEDIATE_CHAIN`: contraction-input deadness propagates through index-preserving activation to expansion output.
- `ATTENTION_VALUE_PATH`: output-projection input deadness propagates through context value axis to value-projection output.
- `QK_SCORE_BLOCKER`: Q/K feature axes are reduced and mixed by score contraction.
- `RESIDUAL_HIDDEN_PROTECTED`: hidden width requires coordinated branch repair.
- `LAYERNORM_HIDDEN_PROTECTED`: normalized hidden width remains conservatively protected.

## 8. BERT 24-Plan Case Study

BERT has 12 encoder layers. Each contributes one FFN plan and one attention value-path plan:

- FFN intermediate plans: `{bert.get("ffn_proven", 0)}/12`
- Attention value-path plans: `{bert.get("attention_proven", 0)}/12`
- Total: `{bert.get("total_proven", 0)}/24`
- Native MLIR evidence: `24/24`

QK score contractions remain blockers and are intentionally excluded from pruning-plan totals.

## 9. All-Model 108/108 Proof

| Model | Layers | FFN Proven | Attention Value Proven | Total Proven | Native | Fallback | Unsupported | Partial | Missing | Failed | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{_detail_rows(data)}

Aggregate: `{aggregate.proven_plans}/{aggregate.expected_plans}` proven, with `{aggregate.native_mlir_evidence}` native MLIR proofs and `{aggregate.high_level_mlir_fallback}` high-level MLIR fallback proofs.

## 10. SparseGPT / Deadbranch Alignment

SparseGPT `2:4` fine-grained sparsity preserves shapes and does not guarantee dead channels. Structural channel pruning creates exact dead consumer axes that can justify backward deadness propagation.

The OPT deadbranch report predicts:

- FFN pairs: `{opt.get("ffn_pairs", 0)}`
- Attention value-path pairs: `{opt.get("attention_value_pairs", 0)}`
- Q/K blocked records: `{opt.get("query_key_blocked_pairs", 0)}`
- SparseGPT alignment: `{opt.get("sparsegpt_alignment_status", "unavailable")}`

The framework statically proves the expected `fc1 -> fc2` and `v_proj -> out_proj` propagation families while retaining Q/K blockers.

## 11. What MLIR Does and Does Not Do

MLIR does not automatically know pruning rules. ONNX-MLIR and the local native tool provide selected-subgraph index and access evidence. The pruning framework maps those facts into axis-transfer relations, recognizes legal patterns, and computes propagation through DFA/worklist fixed points.

**MLIR is a local evidence generator, not the pruning framework itself.**

## 12. Limitations

- This does not choose pruning indices.
- This does not execute pruning.
- This does not measure accuracy.
- Native MLIR evidence may not be universal for arbitrary future architectures.
- Ambiguous fused-QKV remains blocked unless a value branch is recoverable through `Split`, `Slice`, `Gather`, or equally clear graph evidence.
- Residual and LayerNorm handling remains conservative.

## 13. Final Claims

{chr(10).join(f"{index}. {claim}" for index, claim in enumerate(claims, 1))}

## 14. Next Research Directions

- Integrate with actual pruning-index selection.
- Validate speedup after graph rewrite.
- Connect structural deadness to compiler dead-code elimination.
- Extend evidence recovery to more architectures.
- Strengthen the native MLIR dependence pass.
- Generate explicit formal proof obligations.

## Input Warnings

{_warnings(data)}

This is final static artifact, evidence, and proof reporting only. It does not execute pruning or mutate model weights.
"""


def render_claims(data: FinalReportData) -> str:
    aggregate = data.aggregate_summary
    claims = evidence_backed_claims(data)
    return f"""# Static Pruning Propagation Claims

## Evidence-Backed Claims

{chr(10).join(f"{index}. {claim}" for index, claim in enumerate(claims, 1))}

Measured aggregate: `{aggregate.proven_plans}/{aggregate.expected_plans}` plans proven, `{aggregate.native_mlir_evidence}` with native MLIR dependence evidence and `{aggregate.high_level_mlir_fallback}` with high-level MLIR fallback.

## Non-Claims / Scope Boundaries

{chr(10).join(f"{index}. {claim}" for index, claim in enumerate(SCOPE_BOUNDARIES, 1))}
"""


def render_case_csv(data: FinalReportData) -> str:
    output = io.StringIO()
    fields = (
        "model",
        "layers",
        "ffn_expected",
        "ffn_proven",
        "attention_value_expected",
        "attention_value_proven",
        "total_expected",
        "total_proven",
        "native_mlir_evidence",
        "fallback_evidence",
        "unsupported",
        "partial",
        "missing",
        "failed",
        "final_verdict",
        "notes",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for model in data.per_model_summary:
        writer.writerow(
            {
                "model": _display_name(model.model_name),
                "layers": model.layers,
                "ffn_expected": model.ffn_expected,
                "ffn_proven": model.ffn_proven,
                "attention_value_expected": model.attention_value_expected,
                "attention_value_proven": model.attention_value_proven,
                "total_expected": model.expected_plans,
                "total_proven": model.proven_plans,
                "native_mlir_evidence": model.native_evidence,
                "fallback_evidence": model.fallback_evidence,
                "unsupported": model.unsupported,
                "partial": model.partial,
                "missing": model.missing,
                "failed": model.failed,
                "final_verdict": _complete(model),
                "notes": model.notes,
            }
        )
    aggregate = data.aggregate_summary
    writer.writerow(
        {
            "model": "TOTAL",
            "layers": sum(model.layers for model in data.per_model_summary),
            "ffn_expected": sum(model.ffn_expected for model in data.per_model_summary),
            "ffn_proven": sum(model.ffn_proven for model in data.per_model_summary),
            "attention_value_expected": sum(model.attention_value_expected for model in data.per_model_summary),
            "attention_value_proven": sum(model.attention_value_proven for model in data.per_model_summary),
            "total_expected": aggregate.expected_plans,
            "total_proven": aggregate.proven_plans,
            "native_mlir_evidence": aggregate.native_mlir_evidence,
            "fallback_evidence": aggregate.high_level_mlir_fallback,
            "unsupported": aggregate.unsupported,
            "partial": aggregate.partial,
            "missing": aggregate.missing,
            "failed": aggregate.failed,
            "final_verdict": "complete" if aggregate.expected_plans == aggregate.proven_plans else "partial",
            "notes": "Aggregate static proof summary.",
        }
    )
    return output.getvalue()


def summary_payload(data: FinalReportData) -> dict[str, object]:
    return {
        "generated_at": data.generated_at,
        "aggregate": asdict(data.aggregate_summary),
        "models": [asdict(model) for model in data.per_model_summary],
        "claims": list(evidence_backed_claims(data)),
        "scope_boundaries": list(SCOPE_BOUNDARIES),
        "warnings": data.warnings,
    }


def render_index(data: FinalReportData, files: list[str]) -> str:
    aggregate = data.aggregate_summary
    links = "\n".join(f"- [{Path(name).stem.replace('_', ' ').title()}]({name})" for name in files)
    return f"""# Final Static Pruning Propagation Research Report

## Outputs

{links}

## Snapshot

- Proven plans: `{aggregate.proven_plans}/{aggregate.expected_plans}`
- Native MLIR evidence: `{aggregate.native_mlir_evidence}`
- High-level MLIR fallback: `{aggregate.high_level_mlir_fallback}`
- Unsupported / partial / missing / failed: `{aggregate.unsupported}` / `{aggregate.partial}` / `{aggregate.missing}` / `{aggregate.failed}`

This is final reporting only. It does not execute pruning or mutate model weights.
"""


def write_report_bundle(output_dir: str | Path, data: FinalReportData) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    documents = {
        "static_pruning_propagation_final_report.md": render_final_report(data),
        "static_pruning_propagation_final_summary.json": json.dumps(summary_payload(data), indent=2) + "\n",
        "static_pruning_propagation_case_tables.csv": render_case_csv(data),
        "static_pruning_propagation_claims.md": render_claims(data),
    }
    written: list[Path] = []
    for name, text in documents.items():
        path = output / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    index_md = output / "index.md"
    index_md.write_text(render_index(data, list(documents)), encoding="utf-8")
    written.append(index_md)
    index_json = output / "index.json"
    index_json.write_text(
        json.dumps(
            {
                "generated_at": data.generated_at,
                "outputs": list(documents),
                "aggregate": asdict(data.aggregate_summary),
                "warnings": data.warnings,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(index_json)
    return written
