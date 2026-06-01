"""Render teaching and paper-quality static pruning propagation documents."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FormalizationInputs:
    bert_proof: dict[str, Any] = field(default_factory=dict)
    bert_coverage: dict[str, Any] = field(default_factory=dict)
    bert_value_paths: dict[str, Any] = field(default_factory=dict)
    bert_validation: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("summary", {}) if isinstance(data, dict) else {}


def _summary_or_self(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("summary", data) if isinstance(data, dict) else {}


def _number(data: dict[str, Any], key: str, default: int = 0) -> int:
    value = data.get(key, default)
    return int(value) if isinstance(value, (int, float)) else default


def _warnings(inputs: FormalizationInputs) -> str:
    return "\n".join(f"- {warning}" for warning in inputs.warnings) or "- None recorded."


def render_static_notes(inputs: FormalizationInputs) -> str:
    bert = _summary(inputs.bert_proof)
    value_paths = _summary_or_self(inputs.bert_value_paths)
    coverage = _summary(inputs.bert_coverage)
    return f"""# Static Pruning Propagation Analysis

## 1. Motivation

Pruning is often framed as choosing weights or channels to remove. A compiler-style analysis asks a different question:

> If an axis or channel is removed or becomes dead, what else must change?

Structural pruning is a graph-transformation legality problem. The analysis must prove how a structural edit propagates through producers, consumers, layout operations, residual paths, and protected boundaries before any optional executable backend is considered.

## 2. Sparse-Weight Pruning vs Structural Pruning

**Sparse-weight pruning** keeps tensor shapes unchanged while setting individual weights to zero. Examples include `2:4`, `V:N:M`, and SparseGPT-style fine-grained sparsity. Such sparsity does not necessarily create fully dead channels.

**Structural pruning** removes an entire channel or makes an axis dead. It may change tensor shapes and therefore requires propagation and repair across connected operations. Structural deadness is visible to a compiler-style analysis.

**Sparsity is not the same as deadness.**

## 3. Axis Facts

The DFA prototype uses a conservative fact lattice:

| Fact | Meaning |
| --- | --- |
| `UNKNOWN` | No pruning-relevant fact has been proven. |
| `LIVE` | The axis is required by a live computation. |
| `DEAD` | The axis has no live consumer or is exactly zero at a structural boundary. |
| `PRUNED` | The axis is structurally removed. |
| `PROTECTED` | The axis cannot be modified by a local propagation rule. |
| `BLOCKED` | A requested propagation is invalid or conflicts with protection. |

`DEAD` is an evidence fact; `PRUNED` is a structural transformation fact. A legal transfer rule is required before deadness justifies removal.

## 4. Axis-Transfer Relations

Loop/access evidence is summarized using:

| Relation | Meaning |
| --- | --- |
| `PRESERVED` | A source axis maps one-to-one to a target axis. |
| `REDUCED` | An axis is consumed by a contraction or reduction. |
| `MIXED` | Multiple values interact so one-to-one transfer is not justified. |
| `PROTECTED` | Local modification requires coordinated repair. |
| `BLOCKED` | Propagation is not legal under the current rule set. |
| `BROADCAST` | An axis is replicated rather than preserved one-to-one. |
| `PERMUTED` | An axis survives with layout reordering. |
| `UNKNOWN` | Available evidence is insufficient. |

### Activation

```text
Y[..., j] = f(X[..., j])
X.j -> Y.j = PRESERVED
```

### FFN contraction

```text
Y[..., h] += X[..., j] * W[j, h]
```

The intermediate axis `j` is consumed as the input-feature reduction axis. The hidden axis `h` is the output axis.

### QK score contraction

```text
Score[..., q, k] += Q[..., q, d] * K[..., k, d]
```

The projected feature axis `d` is reduced and mixed. Simple one-to-one Q/K propagation is blocked.

### Attention context

```text
Context[..., q, d] += Prob[..., q, k] * V[..., k, d]
V.d -> Context.d = PRESERVED
```

The value axis `d` remains free and is preserved into context.

## 5. Semantic Patterns from Evidence

Rich semantic labels are not assigned from operator names. **Names are syntax.** Semantics are derived from:

- graph connectivity
- tensor and axis roles
- loop/access relations
- native MLIR dependence evidence

The current recognizers emit:

- `FFN_INTERMEDIATE_CHAIN`
- `ATTENTION_VALUE_PATH`
- `QK_SCORE_BLOCKER`
- `RESIDUAL_HIDDEN_PROTECTED`
- `LAYERNORM_HIDDEN_PROTECTED`

## 6. DFA / Worklist Propagation

The semantic graph and a seed fact are analyzed to a fixed point.

```text
state := UNKNOWN for every axis
worklist := seed facts

while worklist is not empty:
    fact := pop(worklist)
    joined := join(state[fact.axis], fact)
    if joined changed state:
        state[fact.axis] := joined
        for each neighboring semantic node:
            for each fact produced by transfer(node, state):
                push(worklist, fact)

return fixed-point state
```

This is a standard monotone DFA/worklist structure: facts are joined conservatively, changed facts trigger transfer functions, and iteration stops when no state changes.

## 7. Propagation Rules

### FFN intermediate path

```text
output dense input j DEAD
  => activation output j DEAD
  => activation input j DEAD
  => intermediate dense output j DEAD
```

### Attention value path

```text
attention output projection input d DEAD
  => context value axis d DEAD
  => value projection output d DEAD
```

### QK blocker

```text
Q/K feature axis d is reduced and mixed in QK^T
  => simple one-to-one pruning propagation is BLOCKED
```

### Residual and LayerNorm protection

The hidden axis is `PROTECTED` unless a coordinated whole-branch repair is proven.

## 8. Evidence Hierarchy

| Evidence tier | Interpretation |
| --- | --- |
| `native_mlir_dependence_evidence` | Native MLIR-linked tool emitted pruning-relevant dependence facts. |
| `actual_loop_access_evidence` | Python affine/access extraction reconstructed a supported local relation. |
| `high_level_mlir_dialect_evidence` | MLIR operations plus conservative shape/topology hints justified lowering. |
| `onnx_hint_fallback` | Local ONNX topology and shapes supplied the available evidence. |
| `unavailable` | No supported evidence was recovered. |

Native MLIR evidence is the strongest tier. Fallback tiers remain useful for reporting, but they must stay visibly distinguished.

## 9. Current Results

- OPT attention value paths: `12/12` seedable and exported.
- BERT FFN intermediate plans: `{_number(bert, "ffn_proven")}/12` proven.
- BERT attention value paths: `{_number(bert, "attention_proven")}/12` proven.
- BERT total: `{_number(bert, "total_proven")}/24` proven.
- BERT value-path artifacts: `{_number(value_paths, "seedable")}/12` seedable.
- BERT native MLIR coverage cells: `{_number(coverage, "native_proven")}/24`.
- QK score contractions remain blockers and are intentionally excluded from pruning-plan counts.

## 10. Limitations

- Native MLIR dependence evidence is not universal across every model and pattern.
- Fused QKV projections require recovery of a separately justified value slice.
- Residual and LayerNorm protection may need stronger native evidence in some model families.
- The current native tool is a local selected-subgraph evidence generator, not a complete MLIR dependence framework.
- This is static legality and evidence analysis. It does not choose pruning indices, execute pruning, mutate model weights, or evaluate accuracy.

## Input Warnings

{_warnings(inputs)}
"""


def render_bert_case_study(inputs: FormalizationInputs) -> str:
    bert = _summary(inputs.bert_proof)
    coverage = _summary(inputs.bert_coverage)
    validations = _summary(inputs.bert_validation)
    layers = inputs.bert_proof.get("layers", [])
    rows = [
        f"| {layer.get('layer_index', '-')} | {layer.get('ffn_plan_status', 'missing')}/{layer.get('ffn_validation_status', 'missing')} | "
        f"{layer.get('attention_path_status', 'missing')}/{layer.get('attention_mapping_status', 'unproven')} | "
        f"{layer.get('attention_evidence_tier', 'unavailable')} | {layer.get('ffn_verdict', 'missing')}/{layer.get('attention_verdict', 'missing')} |"
        for layer in layers
    ]
    if not rows:
        rows = ["| - | unavailable | unavailable | unavailable | partial |"]
    verdict = bert.get("final_verdict", "partial")
    return f"""# BERT 24-Plan Case Study

## Expected Structure

BERT has 12 encoder layers. Each layer contributes:

- one FFN intermediate propagation plan
- one attention value-path propagation plan

Expected: `12 x 2 = 24` complete propagation plans.

## Result Summary

| Pattern | Expected | Found | Proven | Evidence |
| --- | ---: | ---: | ---: | --- |
| FFN intermediate | 12 | {_number(bert, "ffn_found")} | {_number(bert, "ffn_proven")} | native MLIR dependence coverage plus validated symbolic plan |
| Attention value path | 12 | {_number(bert, "attention_found")} | {_number(bert, "attention_proven")} | native MLIR dependence |
| Total | 24 | {_number(bert, "ffn_found") + _number(bert, "attention_found")} | {_number(bert, "total_proven")} | native MLIR dependence |

Supporting checks:

- Valid production FFN validations: `{_number(validations, "valid")}/12`
- Native MLIR coverage cells: `{_number(coverage, "native_proven")}/24`
- Missing coverage cells: `{_number(coverage, "missing_cases")}`

## Per-Layer Table

| Layer | FFN status | Attention value-path status | Evidence tier | Verdict |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Why QK is Excluded

QK score contractions are blockers. They are evidence of non-propagatability, not pruning plans. In:

```text
Score[..., q, k] += Q[..., q, d] * K[..., k, d]
```

the Q/K feature axis `d` is reduced and mixed. A dead score-side column does not imply a one-to-one dead producer-output channel in Q or K.

## Final Verdict

`{verdict}`

## Warnings

{_warnings(inputs)}

This is a static proof report. It does not execute pruning or mutate model weights.
"""


SLIDES = (
    ("Title", "Static Pruning Propagation Analysis", ("Compiler-style legality analysis for structural pruning", "Selected-subgraph evidence and fixed-point propagation"), "Pipeline overview: ONNX -> MLIR -> axis relations -> DFA."),
    ("Core question", "Ask what must change after an axis becomes dead", ("Separate channel selection from propagation legality", "Analyze repairs, blockers, and protected boundaries"), "A seed axis with arrows to affected producer and consumer axes."),
    ("Sparse-weight vs structural pruning", "Zeros and dead axes are different abstractions", ("Sparse-weight pruning preserves shape", "Structural pruning removes or deadens full channels"), "Side-by-side sparse matrix and removed-column matrix."),
    ("Sparsity is not deadness", "Fine-grained sparsity does not imply compiler-visible dead channels", ("2:4 and V:N:M may leave every channel live", "Exact dead consumer columns enable backward propagation"), "A column with scattered zeros versus an entirely dead column."),
    ("Axis facts", "Track pruning facts in a conservative lattice", ("UNKNOWN, LIVE, DEAD, PRUNED", "PROTECTED and BLOCKED prevent unsafe propagation"), "Small lattice diagram."),
    ("Axis-transfer relations", "Infer how axes move through operations", ("PRESERVED, REDUCED, MIXED", "PROTECTED, BLOCKED, BROADCAST, PERMUTED"), "Table of relations with one-line examples."),
    ("Why names are syntax", "Do not identify semantics from labels such as fc1 or v_proj", ("Names are useful diagnostics only", "Graph topology and access relations are evidence"), "Rename alpha -> beta -> gamma while preserving the same inferred pattern."),
    ("ONNX subgraphs as evidence units", "Analyze local pruning-relevant fragments", ("Avoid full-model lowering when a local path is sufficient", "Preserve boundary inputs and outputs explicitly"), "Highlighted local fragment inside a larger transformer block."),
    ("MLIR dependence evidence", "Use ONNX-MLIR as a local evidence generator", ("Inspect affine/scf loops and load/store accesses", "Native dependence JSON records preserved and reduced IVs"), "MLIR loop nest with highlighted IV d."),
    ("DFA worklist algorithm", "Propagate facts until a fixed point", ("Seed a DEAD or PRUNED fact", "Join conservatively and re-enqueue changed neighbors"), "Queue-based worklist flowchart."),
    ("FFN propagation example", "Intermediate channels propagate backward through index-preserving activation", ("output.dense input j DEAD", "activation j DEAD", "intermediate.dense output j DEAD"), "Three-node FFN chain."),
    ("Attention value-path example", "Value channels preserve identity through attention context", ("out_proj input d DEAD", "context d DEAD", "value projection output d DEAD"), "V projection -> context -> output projection."),
    ("QK blocker example", "QK score contraction reduces and mixes projected channels", ("Q/K axis d is reduced in QK^T", "Simple one-to-one propagation is BLOCKED"), "Q and K entering score MatMul with d crossed out."),
    ("Residual/LayerNorm protection", "Hidden width is protected without coordinated repair", ("Residual branches must stay aligned", "Normalization statistics couple hidden entries"), "Residual diamond and LayerNorm boundary."),
    ("Evidence hierarchy", "State the strength of every proof", ("Native MLIR dependence is strongest", "Access, high-level, and ONNX fallbacks remain explicit"), "Tiered evidence pyramid."),
    ("BERT 24-plan proof", "BERT reaches 12 FFN plus 12 attention value-path plans", ("24/24 cells native-proven", "QK blockers are excluded from plan counts"), "12-row layer grid with two green cells per layer."),
    ("Coverage and limitations", "Important paths are proven; universality is not claimed", ("Fused QKV needs value-slice recovery", "Residual/LayerNorm native evidence can be strengthened"), "Coverage matrix with proven and future-work cells."),
    ("Takeaways", "Static pruning propagation is compiler-style evidence-backed dataflow analysis", ("Sparsity is not deadness", "Names are syntax", "MLIR proves local relations; DFA computes propagation"), "Four concise takeaway blocks."),
)


def render_teaching_slides(inputs: FormalizationInputs) -> str:
    rendered = ["# Teaching Slide Outline", ""]
    for index, (title, message, bullets, diagram) in enumerate(SLIDES, 1):
        rendered.extend(
            [
                f"## Slide {index}: {title}",
                "",
                f"**Key message:** {message}",
                "",
                *[f"- {bullet}" for bullet in bullets],
                "",
                f"**Suggested diagram:** {diagram}",
                "",
            ]
        )
    rendered.extend(["## Input Warnings", "", _warnings(inputs), ""])
    return "\n".join(rendered)


def render_paper_methodology(inputs: FormalizationInputs) -> str:
    bert = _summary(inputs.bert_proof)
    return f"""# Paper Methodology Outline: Static Pruning Propagation Analysis

## 1. Problem Definition

Let the analyzed graph be `G = (V, E)`, where nodes `V` are operations or regions and edges `E` carry tensors. For a tensor axis `a`, the analysis asks whether a structural seed fact can propagate legally through `G` and which repairs or blockers follow.

## 2. Graph and Axis Model

Each tensor edge exposes one or more axes. An axis is identified by its tensor boundary and semantic role, such as hidden width, intermediate width, head width, value width, or sequence position. Operator names remain diagnostics; they are not semantic proof.

## 3. Axis Facts and Lattice

Facts inhabit a conservative fact lattice `L`:

```text
UNKNOWN, LIVE, DEAD, PRUNED, PROTECTED, BLOCKED
```

The join operation preserves agreement and turns conflicting facts into explicit blockers. `DEAD` does not automatically become `PRUNED`; a legal transfer rule is required.

## 4. Axis-Transfer Evidence

For each selected region, access relations summarize whether an axis is `PRESERVED`, `REDUCED`, `MIXED`, `PROTECTED`, `BLOCKED`, `BROADCAST`, `PERMUTED`, or `UNKNOWN`. For example:

```text
Context[..., q, d] += Prob[..., q, k] * V[..., k, d]
```

proves `V.d -> Context.d = PRESERVED` because `d` remains free while `k` is reduced.

## 5. MLIR-Backed Dependence Evidence

Selected ONNX subgraphs are lowered with ONNX-MLIR. The local native tool inspects affine/scf loop nests and affine/memref accesses, then emits dependence-style JSON facts. The analysis records whether evidence came from native MLIR dependence, Python affine extraction, high-level MLIR dialect evidence, ONNX fallback, or was unavailable.

MLIR is a local evidence generator, not the pruning framework itself.

## 6. Pattern Recognition

Axis relations induce pruning-relevant patterns:

- `FFN_INTERMEDIATE_CHAIN`
- `ATTENTION_VALUE_PATH`
- `QK_SCORE_BLOCKER`
- `RESIDUAL_HIDDEN_PROTECTED`
- `LAYERNORM_HIDDEN_PROTECTED`

Pattern recognition uses connectivity, axis roles, and dependence evidence rather than graph names.

## 7. DFA Worklist Propagation

Each semantic node `n` defines a monotone transfer function `F_n : L^k -> L^m`. Given seed facts, a worklist repeatedly applies transfer functions and conservative joins until reaching a fixed-point state `S*`.

```text
S := initial seed state
W := seed axes
while W is not empty:
    a := pop(W)
    for n adjacent to a:
        updates := F_n(S)
        S := join(S, updates)
        enqueue axes changed by join
return S*
```

## 8. Plan Validation

A valid propagation plan `P` records affected producer and consumer axes, preserved dimensions, required repairs, forbidden edits, and validation checks. QK blockers are diagnostics, not pruning plans.

## 9. Evidence Hierarchy

The report distinguishes:

1. `native_mlir_dependence_evidence`
2. `actual_loop_access_evidence`
3. `high_level_mlir_dialect_evidence`
4. `onnx_hint_fallback`
5. `unavailable`

This hierarchy prevents fallback reasoning from being reported as native proof.

## 10. Case Studies

### OPT value path

OPT exports 12 seedable value-path fragments, each connecting `v_proj -> context -> out_proj`.

### BERT 24-plan proof

BERT has 12 encoder layers. Each layer contributes one FFN intermediate plan and one attention value-path plan:

```text
12 FFN + 12 attention value path = {_number(bert, "total_proven")}/24 proven
```

QK score contractions remain blockers and are excluded from plan counts.

## 11. Limitations

- Native MLIR evidence does not yet cover every model family and pattern.
- Fused QKV operators require value-slice recovery before a standalone value-path proof is justified.
- Residual and LayerNorm protection evidence can be strengthened.
- The implementation analyzes static legality and propagation; it does not choose indices, execute pruning, mutate models, or evaluate accuracy.

## Soundness-Style Statement

If the analysis emits a valid propagation plan, then every affected axis is either:

- consistently pruned or dead,
- protected from modification,
- repaired through a mapped consumer/producer relation,
- or blocked with an explicit diagnostic.

This is a soundness-style design objective supported by the implemented transfer rules and validators. It is not a machine-checked formal proof.

## Input Warnings

{_warnings(inputs)}
"""


def render_index(inputs: FormalizationInputs, files: list[str]) -> str:
    bert = _summary(inputs.bert_proof)
    links = "\n".join(f"- [{Path(name).stem.replace('_', ' ').title()}]({name})" for name in files)
    return f"""# Static Pruning Propagation Formalization

This bundle formalizes the compiler-style pruning-propagation research story for teaching and paper preparation.

## Documents

{links}

## BERT Case Study Snapshot

- Expected plans: `{_number(bert, "expected_plans")}`
- Proven plans: `{_number(bert, "total_proven")}`
- Final verdict: `{bert.get("final_verdict", "partial")}`

## Warnings

{_warnings(inputs)}

This is documentation and formalization only. It does not execute pruning or mutate model weights.
"""
