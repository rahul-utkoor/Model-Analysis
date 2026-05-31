# ONNX Subgraph to Axis-Transfer Bridge Prototype

## 1. Why This Exists

This directory is an independent experimental bridge from selected ONNX subgraph artifacts into the compiler-style pruning prototypes:

```text
local ONNX subgraph
  -> topology and shape summary
  -> conservative pattern hint
  -> loop/access RegionSpec
  -> axis-transfer summary
  -> optional semantic DFA propagation
```

It does not execute ONNX models, mutate model weights, or modify the production analysis pipeline.

## 2. Why We Do Not Lower the Whole Model

Whole-model lowering introduces framework, control-flow, shape, and dialect concerns before the pruning proof is clear. The current goal is narrower: inspect small subgraphs already exported by the production reporting layer and recover local pruning evidence conservatively.

## 3. Selected ONNX Subgraph as the Unit

The production pipeline already emits evidence artifacts such as:

```text
artifacts/model_analysis_subgraphs/<model>/layers/layer_<N>/<subgraph>/subgraph.onnx
```

This prototype reads one such file without rewriting it. It extracts:

- nodes and operator types
- tensor producers and consumers
- graph inputs, outputs, and initializers
- static or symbolic shapes where available

## 4. ONNX Summary to Loop / Access Region

Hints are inferred from local structure and shape evidence:

- `FFN_LIKE`: parameterized projection -> elementwise/layout flow -> parameterized projection
- `QK_SCORE_LIKE`: rank-4 `MatMul` reducing projected feature width into query/key scores
- `ATTENTION_CONTEXT_LIKE`: rank-4 `MatMul` preserving value width into context output
- `ATTENTION_VALUE_PATH_LIKE`: value projection -> context -> output projection
- `RESIDUAL_LIKE`: shape-aligned `Add` with two non-initializer inputs
- `LAYERNORM_LIKE`: explicit `LayerNormalization`

Supported hints lower into template `RegionSpec` records from `experimental.axis_transfer_analysis`.

This bridge does not infer semantics from ONNX names alone. Names are retained only as weak diagnostics and provenance.

## 5. Relationship to Future MLIR Work

This is not full ONNX-to-MLIR lowering.

The template-lowering step is intentionally replaceable. A future MLIR bridge can lower ONNX subgraphs to affine/linalg/scf and extract iterator types plus indexing maps before constructing the same axis-transfer summaries.

## 6. Relationship to Axis Transfer Analysis

`experimental/axis_transfer_analysis/` receives the lowered `RegionSpec` and derives:

- `PRESERVED`
- `REDUCED`
- `MIXED`
- `PROTECTED`
- `BLOCKED`

It then recognizes patterns such as `FFN_INTERMEDIATE_CHAIN`, `ATTENTION_VALUE_PATH`, and `QK_SCORE_BLOCKER`.

## 7. Relationship to DFA Propagation

`experimental/pruning_analysis_bridge/` consumes supported access-derived patterns and constructs semantic DFA graphs. The DFA worklist then proves deadness propagation or a blocker at fixed point.

Standalone context subgraphs may lower to useful axis evidence without running DFA propagation because they do not contain a complete seedable path.

## 8. How to Run

Run on a generated GPT-2 MLP artifact:

```bash
python -m experimental.onnx_axis_bridge.cli \
  --onnx artifacts/model_analysis_subgraphs/gpt2/layers/layer_0/03_gpt_2_block_0_mlp_block/subgraph.onnx \
  --format markdown \
  --show-all
```

Run on BERT attention contractions:

```bash
python -m experimental.onnx_axis_bridge.cli \
  --onnx artifacts/model_analysis_subgraphs/bert-base-uncased/layers/layer_0/05_layer_0_attention_score_matmul/subgraph.onnx \
  --format markdown \
  --show-all

python -m experimental.onnx_axis_bridge.cli \
  --onnx artifacts/model_analysis_subgraphs/bert-base-uncased/layers/layer_0/08_layer_0_attention_context_matmul/subgraph.onnx \
  --format markdown \
  --show-all
```

Run tests:

```bash
python -m pytest -q experimental/onnx_axis_bridge/tests
```

## 9. Limitations

- Template lowering handles only recognized local motifs.
- Shape-free or ambiguous rank-4 contractions remain `UNKNOWN`.
- Decomposed LayerNorm and complex reshape/transpose value paths are only partially modeled.
- Fused attention and fused QKV projections require future rules.
- This is not a model executor and not a full ONNX-to-MLIR compiler.
