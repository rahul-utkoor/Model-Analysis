# Axis Transfer Summary Prototype

## 1. Why This Exists

This directory is an independent compiler-style teaching prototype. It derives pruning-relevant axis behavior from loop variables and indexed tensor accesses:

```text
loop/access description
  -> axis-transfer summaries
  -> pruning-pattern recognition
```

It does not analyze or mutate model weights. It does not replace the production `src/model_analysis/` pipeline.

## 2. Graph Semantics Are Not Enough

Graph topology can show that one operation feeds another, but it does not prove how axes flow through a computation. The distinction matters:

- an activation preserves `j`
- an FFN contraction reduces input-feature axis `j`
- attention context preserves value axis `d`
- `QK^T` reduces projected feature axis `d`
- residual merges require hidden-axis agreement

These are access-relation facts, not naming conventions.

## 3. Loop / Access Relations as Pruning Evidence

The prototype uses a small MLIR-inspired IR. For example:

```text
Y[b,s,j] = gelu(X[b,s,j])
Y[b,s,h] += X[b,s,j] * W[j,h]
Score[b,head,q,k] += Q[b,head,q,d] * K[b,head,k,d]
Context[b,head,q,d] += P[b,head,q,k] * V[b,head,k,d]
```

Reusing an IV in a write proves that its axis remains free. An IV consumed only in reads is reduced. A protected merge records an obligation for coordinated pruning.

Semantic roles should be derived from evidence:

```text
graph topology + axis roles + loop/access relations
```

They should not be assigned directly from names.

## 4. Axis Relation Kinds

The relation lattice is intentionally small:

```text
UNKNOWN
PRESERVED
PERMUTED
RESHAPED_SPLIT
RESHAPED_MERGED
REDUCED
BROADCAST
MIXED
PROTECTED
BLOCKED
```

Each transfer carries source and target axes, confidence, and a proof string.

## 5. Examples

### Activation

`Y[b,s,j] = gelu(X[b,s,j])` proves `j` is `PRESERVED`.

### FFN

The intermediate axis is produced by the first projection, preserved by the activation, and consumed by the contraction. The recognizer emits `FFN_INTERMEDIATE_CHAIN`.

### QK Score Blocker

`Score[b,head,q,k] += Q[b,head,q,d] * K[b,head,k,d]` reduces `d`. The recognizer emits `QK_SCORE_BLOCKER` with:

```text
qk_score_contraction_mixes_channels
```

### Attention Value Path

`Context[b,head,q,d] += P[b,head,q,k] * V[b,head,k,d]` preserves value axis `d`. Combined with an output projection that consumes context `d`, this proves `ATTENTION_VALUE_PATH`.

### Residual Protection

`Y[b,s,h] = A[b,s,h] + B[b,s,h]` protects `h` unless both branches are repaired consistently.

## 6. Relationship to the DFA Prototype

`experimental/dfa_pruning_propagation/` starts from semantic graph annotations and runs a worklist to a fixed point. This prototype explores the evidence layer below those annotations: how loop/access relations justify preserved, reduced, protected, and blocked axes.

The two prototypes remain independent so each compiler concept stays teachable.

## 7. Future MLIR Connection

This IR is a bridge toward future extraction from MLIR affine, linalg, or scf lowering. A future frontend could translate iterator types and affine indexing maps into these summaries, then feed proven semantic patterns into pruning propagation.

## Run

```bash
python -m experimental.axis_transfer_analysis.cli --example activation --show-relations
python -m experimental.axis_transfer_analysis.cli --example ffn --show-patterns
python -m experimental.axis_transfer_analysis.cli --example qk-score --format markdown --show-relations
python -m experimental.axis_transfer_analysis.cli --example attention-context --show-relations
python -m experimental.axis_transfer_analysis.cli --example attention-value-path --show-patterns
python -m experimental.axis_transfer_analysis.cli --example residual --show-patterns
```

Run tests:

```bash
python -m pytest -q experimental/axis_transfer_analysis/tests
```
