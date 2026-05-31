# Cross-Evidence Pruning Proof Report

## Why This Exists

This experimental report consolidates the selected-subgraph pruning proof story into one learner-facing artifact. It evaluates local ONNX subgraphs without executing pruning or changing model artifacts.

## Evidence Hierarchy

The runner records the strongest available evidence source:

1. `native_mlir_dependence_evidence`
2. `actual_loop_access_evidence`
3. `high_level_mlir_dialect_evidence`
4. `onnx_hint_fallback`
5. `unavailable`

The analysis flow is:

```text
ONNX subgraph
  -> ONNX topology and shape hints
  -> ONNX-MLIR local lowering
  -> native or Python affine dependence evidence
  -> axis-transfer summary
  -> pattern recognition
  -> DFA fixed-point propagation
```

## Run

```bash
python -m experimental.pruning_proof_report.cli \
  --models default \
  --output-dir reports/pruning_proof_report \
  --format both \
  --verbose
```

Run one configured case:

```bash
python -m experimental.pruning_proof_report.cli \
  --case bert_layer0_attention_score \
  --output-dir reports/pruning_proof_report \
  --format both \
  --verbose
```

## Cases

The default report includes available layer-0 FFN/MLP subgraphs for GPT-2, OPT, DistilBERT, and ViT; BERT attention score and context MatMuls; and BERT residual and LayerNorm examples. Missing optional artifacts are reported and do not stop the run.

## Relationship to the Experimental Stack

- `experimental/onnx_axis_bridge/` extracts local ONNX topology and shape hints.
- `experimental/mlir_axis_bridge/` uses ONNX-MLIR and the optional native dependence executable as selected-subgraph evidence generators.
- `experimental/axis_transfer_analysis/` describes preserved, reduced, protected, and blocked axes.
- `experimental/pruning_analysis_bridge/` lowers proven patterns into DFA graphs.
- `experimental/dfa_pruning_propagation/` computes the pruning/deadness fixed point.

## Limitations

This is a report and evaluation layer, not a production analysis stage. It does not lower full models, execute pruning, mutate weights, or evaluate accuracy. High-level and ONNX-only fallback proofs are labeled explicitly.
