# Bridge Axis-Transfer Evidence to DFA Propagation

## 1. Why This Bridge Exists

This directory connects two independent teaching prototypes:

```text
loop/access evidence
  -> axis-transfer summary
  -> pattern recognition
  -> semantic DFA graph construction
  -> worklist pruning propagation
```

The bridge demonstrates how compiler evidence can justify semantic pruning rules before a fixed-point analysis consumes them.

## 2. Problem With Directly Assigned Graph Semantics

A graph label such as `fc1`, `v_proj`, or `score_matmul` is syntax. It can help a reader, but it is not a proof.

The pruning analysis needs evidence:

- which loop IV remains free
- which IV is reduced
- which axes map one-to-one
- which axes are mixed by a contraction
- which hidden axes are protected by merges or normalization

## 3. Axis-Transfer Evidence as Semantic Proof

`experimental/axis_transfer_analysis/` derives `PRESERVED`, `REDUCED`, `MIXED`, `PROTECTED`, and `BLOCKED` axis relations from indexed accesses such as:

```text
Y[b,s,h] += X[b,s,j] * W[j,h]
Context[b,head,q,d] += P[b,head,q,k] * V[b,head,k,d]
Score[b,head,q,k] += Q[b,head,q,d] * K[b,head,k,d]
```

Pattern recognition then emits evidence-backed matches such as `FFN_INTERMEDIATE_CHAIN`, `ATTENTION_VALUE_PATH`, and `QK_SCORE_BLOCKER`.

## 4. Translation From Patterns to DFA Graphs

The bridge lowers each recognized pattern into a small semantic DFA graph. Lowered node labels are generic:

```text
producer_from_axis_summary
unary_from_axis_summary
consumer_from_axis_summary
```

The labels are for display only. DFA transfer dispatch uses semantic roles attached after pattern recognition.

## 5. FFN Example

The loop/access layer proves that an intermediate axis is produced, preserved through a unary operation, and consumed by a contraction.

The bridge lowers that proof and seeds:

```text
consumer intermediate input = DEAD
```

The DFA fixed point proves:

```text
producer intermediate output = DEAD
consumer hidden output = PROTECTED
```

## 6. Attention Value-Path Example

The context access relation proves:

```text
V.value_dim -> Context.value_context_dim = PRESERVED
```

The bridge lowers that proof and seeds output-projection input deadness. The DFA propagates backward to the value-producer output while preserving the hidden output width.

## 7. Q/K Blocker Example

The score contraction proves that the Q/K feature axis is `REDUCED`, `MIXED`, and `BLOCKED`:

```text
qk_score_contraction_mixes_channels
```

The bridge lowers that relation into a DFA score-contraction blocker. It does not infer simple one-to-one Q/K propagation.

## 8. Relationship to Other Layers

- `experimental/axis_transfer_analysis/` derives local axis evidence from loop/access behavior.
- `experimental/dfa_pruning_propagation/` applies semantic transfer functions until a fixed point.
- `experimental/pruning_analysis_bridge/` demonstrates the lowering between those layers.
- The production Model-Analysis pipeline remains independent and authoritative for real model reports.

## 9. How to Run

```bash
python -m experimental.pruning_analysis_bridge.cli --example ffn-from-access --format markdown --show-all
python -m experimental.pruning_analysis_bridge.cli --example attention-value-from-access --format markdown --show-all
python -m experimental.pruning_analysis_bridge.cli --example qk-blocked-from-access --format markdown --show-all
python -m experimental.pruning_analysis_bridge.cli --example residual-from-access --show-all
python -m experimental.pruning_analysis_bridge.cli --example layernorm-from-access --show-all
```

Run tests:

```bash
python -m pytest -q experimental/pruning_analysis_bridge/tests
```

Graph names are syntax. Loop/access relations provide evidence. Pattern recognition derives semantic roles. DFA computes propagation.
