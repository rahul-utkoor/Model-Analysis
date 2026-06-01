# BERT 24-Plan Static Pruning Propagation Proof

## Purpose

This report joins BERT's 12 validated FFN symbolic plans with 12 MLIR-backed attention value-path proofs.

Each encoder layer contributes:

- one `FFN_INTERMEDIATE_CHAIN` plan
- one `ATTENTION_VALUE_PATH` plan

The complete target is `12 x 2 = 24` static propagation plans.

## Evidence

FFN rows use production symbolic plan and validation reports. Attention rows use complete local ONNX value-path fragments, MLIR dependence evidence, axis-transfer recognition, and DFA fixed-point propagation.

QK score paths remain blockers and are not counted as pruning plans.

## Run

```bash
python -m experimental.bert_24_plan_proof.cli \
  --output-dir reports/bert_24_plan_proof \
  --verbose
```

This is static evidence and proof reporting only. It does not execute pruning or mutate model weights.
