# ONNX-MLIR Axis Bridge Prototype

## Why This Exists

This experiment uses local ONNX-MLIR lowering as a semantic evidence source for selected pruning-analysis subgraphs. It connects real emitted MLIR text to the existing axis-transfer and DFA teaching prototypes.

## Why We Do Not Rewrite the Whole Analysis in MLIR

The production analysis already provides model-wide semantics, reports, and evidence packs. This prototype intentionally lowers only small ONNX subgraphs. It tests whether compiler IR can strengthen local axis proofs without replacing the broader analysis pipeline.

## Selected ONNX Subgraphs as Local Evidence Units

Input artifacts come from paths such as:

```text
artifacts/model_analysis_subgraphs/<model>/layers/layer_<N>/<subgraph>/subgraph.onnx
```

The subgraphs are read-only evidence artifacts. They are never executed or rewritten.

## ONNX-MLIR Lowering Stages

For each selected subgraph, the runner requests:

```bash
onnx-mlir subgraph.onnx --EmitONNXIR -o <prefix>_onnx
onnx-mlir subgraph.onnx --EmitMLIR --preserveMLIR -o <prefix>_lowered
```

Both command results and all emitted text artifacts are recorded.

## Dialects We May See

The artifact scanner records evidence for ONNX, Krnl, Linalg, SCF, Affine, and memref operations. A particular ONNX-MLIR build may expose only some of these stages.

## Access Extraction

The parser is intentionally lightweight. It extracts recognized operations plus `affine.load`, `affine.store`, `memref.load`, and `memref.store` accesses. It is not a complete MLIR parser.

## Axis-Transfer Summary Construction

Evidence sources are reported explicitly:

- `actual_loop_access_evidence`: indexed accesses prove a supported relation.
- `high_level_mlir_dialect_evidence`: emitted MLIR operations plus ONNX shape hints support conservative template lowering.
- `onnx_hint_fallback`: only the existing ONNX local hint is sufficient.

The bridge never presents fallback evidence as a loop-level proof.

## Relationship to Other Prototypes

- `experimental/onnx_axis_bridge/` supplies conservative local ONNX topology and shape hints.
- `experimental/axis_transfer_analysis/` summarizes preserved, reduced, and blocked axes.
- `experimental/pruning_analysis_bridge/` lowers complete axis patterns into DFA propagation.

## How to Run

```bash
python -m experimental.mlir_axis_bridge.cli \
  --onnx artifacts/model_analysis_subgraphs/gpt2/layers/layer_0/03_gpt_2_block_0_mlp_block/subgraph.onnx \
  --output-dir reports/mlir_axis_bridge/gpt2_layer0_mlp \
  --format markdown \
  --show-all \
  --verbose
```

## Limitations

This is not full ONNX-to-MLIR semantic lowering. The text parser is conservative, affine reconstruction is intentionally narrow, and unsupported layouts remain warnings. MLIR is used as a local semantic evidence source, not as the pruning framework itself.
