# MLIR Evidence Coverage Study

## Purpose

This experimental study measures how much selected-subgraph pruning evidence is covered by native MLIR dependence analysis and where weaker evidence tiers are still required.

It evaluates available local ONNX atlas artifacts across BERT, DistilBERT, OPT, GPT-2, and ViT. It does not lower full models, execute pruning, mutate weights, or evaluate accuracy.

## Evidence Tiers

1. `native_mlir_dependence_evidence`: the standalone MLIR-linked tool emitted pruning-relevant dependence facts.
2. `actual_loop_access_evidence`: Python affine/access extraction reconstructed a supported local relation.
3. `high_level_mlir_dialect_evidence`: emitted MLIR operations plus conservative local hints justified template lowering.
4. `onnx_hint_fallback`: the ONNX topology and shape bridge supplied the available proof.
5. `unavailable`: no supported evidence was found.

## Patterns

The matrix covers:

- FFN/MLP intermediate propagation
- Attention QK score blocking
- Attention context value-axis preservation
- Full attention value-path propagation
- Residual hidden-axis protection
- LayerNorm hidden-axis protection

Missing local atlas artifacts are recorded as coverage gaps rather than omitted.

## Run Layer 0

```bash
python -m experimental.mlir_evidence_coverage.cli \
  --models default \
  --layers layer0 \
  --output-dir reports/mlir_evidence_coverage \
  --format both \
  --verbose
```

## Run Focused All-Layer Coverage

```bash
python -m experimental.mlir_evidence_coverage.cli \
  --models default \
  --layers all \
  --patterns FFN_MLP_INTERMEDIATE,ATTENTION_QK_SCORE,ATTENTION_CONTEXT_VALUE_AXIS \
  --output-dir reports/mlir_evidence_coverage_all_layers \
  --format both \
  --verbose
```

## Interpretation

- `native_proven`: native MLIR dependence evidence proves the expected pattern.
- `access_proven`: Python affine/access evidence proves the expected pattern.
- `fallback_proven`: high-level MLIR or ONNX-local evidence proves the pattern.
- `blocked_as_expected`: the QK blocker was recovered.
- `partial`: a useful local mapping exists, but the selected subgraph is not sufficient for a full DFA path.
- `missing`: the local atlas does not expose a matching subgraph.

## Relationship to the Proof Report

`experimental/pruning_proof_report/` presents a curated teaching report. This module expands that idea into a systematic matrix over models, layers, and pruning-relevant pattern classes.

## Limitations

Coverage is coverage of available selected-subgraph artifacts, not proof that every model operation has been lowered or analyzed. Full attention value-path propagation requires a local artifact spanning value projection, context, and output projection.
