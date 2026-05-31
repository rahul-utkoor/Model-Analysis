# DFA Worklist Prototype for Static Pruning Propagation

## 1. Why This Exists

This directory is a standalone teaching and research prototype. It formalizes structural pruning propagation as a compiler-style dataflow analysis:

```text
graph + seed facts + transfer functions + join lattice
  -> worklist iteration
  -> fixed-point pruning/deadness facts
```

It is intentionally separate from the production `src/model_analysis/` pipeline. The larger framework reads real model artifacts and emits model-level reports. This prototype uses tiny explicit graphs so the formal propagation story can be inspected without model-export or framework details.

## 2. Sparse-Weight Pruning vs Structural Pruning

Sparse-weight pruning creates zeros.

Structural pruning creates dead axes.

Static pruning propagation proves how dead axes flow through the graph.

SparseGPT-style `2:4` or `V:N:M` pruning can zero many individual weights while preserving channel shapes and liveness. Compiler-style propagation needs a stronger fact: an exact dead consumer input channel or a structurally pruned axis.

## 3. Dataflow Facts and Lattice

Each `Axis` names one symbolic tensor dimension, such as `hidden_dim`, `intermediate_dim`, or `value_dim`.

Each `Fact` assigns one lattice element:

```text
UNKNOWN
LIVE
DEAD
PRUNED
PROTECTED
BLOCKED
```

The join operation is conservative:

- `UNKNOWN join DEAD = DEAD`
- `DEAD join PRUNED = PRUNED`
- `PROTECTED join PRUNED = BLOCKED`
- `LIVE join DEAD = BLOCKED`

`BLOCKED` is an explicit diagnostic result, not an exception.

## 4. Transfer Functions

Transfer functions live in `transfer.py`.

- Transfer functions dispatch on inferred semantic roles, not node names or low-level operator strings.
- Activations preserve intermediate-axis indices in both directions.
- FFN contraction inputs can be dead while the contraction hidden output remains protected.
- Attention context propagates value-axis deadness only when `value_axis_mapping = proven`.
- Attention output-projection input deadness preserves the output hidden width.
- Residual additions and LayerNorm protect `hidden_dim`.
- `QK^T` score contraction blocks simple Q/K one-to-one propagation.

## Semantic Roles, Not Names

Names such as `fc1`, `fc2`, `v_proj`, and `out_proj` are display labels only. The prototype first runs a semantic annotation pass:

```text
raw graph node
  -> structural semantic-role inference
  -> transfer-rule selection
  -> DFA propagation
```

The pass uses low-level operator kinds, semantic axis roles, and graph connectivity. Semantic roles may also be provided explicitly when a frontend has stronger evidence.

For example, arbitrary labels still form an FFN chain:

```text
alpha -> beta -> gamma
```

when:

```text
alpha produces INTERMEDIATE
beta preserves INTERMEDIATE
gamma consumes INTERMEDIATE and produces HIDDEN
```

The inferred roles are:

```text
alpha = EXPANSION_PROJECTION
beta  = INDEX_PRESERVING_ACTIVATION
gamma = CONTRACTION_PROJECTION
```

Likewise, `producer_X -> bridge_Y -> consumer_Z` is treated as an attention value path when the axes carry `VALUE -> VALUE_CONTEXT -> HIDDEN`. A `HEAD + HEAD -> SCORE` contraction is blocked regardless of its label.

## 5. Worklist Algorithm

`worklist.py` implements a queue-based fixed-point solver:

```text
initialize state to UNKNOWN
enqueue seed facts
while queue is not empty:
    join incoming fact into axis state
    if the lattice kind changed:
        propagate across equivalent tensor-axis edges
        apply transfer functions for touching nodes
stop when the queue is empty
```

The result includes final facts and a trace of propagated, protected, joined, and blocked events.

## 6. FFN Example

```text
fc1 -> gelu -> fc2
```

Seed:

```text
fc2 input channel j = DEAD
```

Expected fixed point:

```text
fc2 input DEAD
gelu output DEAD
gelu input DEAD
fc1 output DEAD
fc2 output hidden_dim PROTECTED
```

This models the SparseGPT channel-pruning observation:

```text
fc2 dead input => fc1 dead output
```

## 7. Attention Value-Path Example

```text
v_proj -> attention_context -> out_proj
```

Seed:

```text
out_proj input channel j = DEAD
```

With proven value-axis mapping:

```text
out_proj input DEAD
attention context DEAD
v_proj output DEAD
out_proj output hidden_dim PROTECTED
```

This models:

```text
out_proj dead input => v_proj dead output
```

If the value-axis mapping is unproven, the transfer emits `BLOCKED`.

## 8. Q/K Blocked Example

```text
q_proj --\
         QK^T score_matmul
k_proj --/
```

Simple Q/K propagation is blocked:

```text
qk_score_contraction_mixes_channels
```

`QK^T` mixes projected dimensions, so consumer deadness does not prove one-to-one producer-output deadness.

## 9. Relationship to the Main Model-Analysis Pipeline

The production pipeline remains authoritative for real models. It provides Tensor IR, op semantics, region semantics, rankings, symbolic plans, validation, deadbranch reports, layer atlases, and browser/terminal exploration.

This experimental package does not replace those modules. It is a clean reference implementation for teaching the fixed-point semantics and comparing them against production reports such as OPT `fc1 -> fc2` and `v_proj -> out_proj`.

## 10. How to Run

```bash
python -m experimental.dfa_pruning_propagation.cli --example ffn --show-trace
python -m experimental.dfa_pruning_propagation.cli --example ffn-renamed --show-trace
python -m experimental.dfa_pruning_propagation.cli --example attention-value --show-trace
python -m experimental.dfa_pruning_propagation.cli --example attention-value-renamed --show-trace
python -m experimental.dfa_pruning_propagation.cli --example attention-qk --show-trace
python -m experimental.dfa_pruning_propagation.cli --example attention-qk-renamed --show-trace
python -m experimental.dfa_pruning_propagation.cli --example residual --show-trace
```

Write a Markdown report:

```bash
python -m experimental.dfa_pruning_propagation.cli \
  --example ffn \
  --format markdown \
  --output reports/dfa_pruning_propagation/ffn.md \
  --show-trace
```

Run prototype tests:

```bash
python -m pytest -q experimental/dfa_pruning_propagation/tests
```

This package performs static analysis only. It does not mutate models, execute pruning, select concrete indices, download models, or evaluate accuracy.
