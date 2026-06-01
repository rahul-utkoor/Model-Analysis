# Milestone 49: BERT 24-Plan Propagation Proof

This demo builds BERT attention value-path fragments and joins them with the existing 12 validated FFN plans.

Each encoder layer contributes:

```text
FFN: intermediate.dense -> GELU -> output.dense
Attention: attention.self.value -> context -> attention.output.dense
```

## Build BERT Value Paths

```bash
./conda-env/bin/python scripts/build_attention_value_path_subgraphs.py \
  --model bert-base-uncased \
  --layers all \
  --export-onnx \
  --render-svg \
  --verbose
```

## Run MLIR Coverage

```bash
./conda-env/bin/python -m experimental.mlir_evidence_coverage.cli \
  --models bert-base-uncased \
  --layers all \
  --patterns FFN_MLP_INTERMEDIATE,ATTENTION_VALUE_PATH \
  --output-dir reports/mlir_evidence_coverage_bert_24_plan \
  --format both \
  --verbose
```

## Generate Proof Report

```bash
./conda-env/bin/python -m experimental.bert_24_plan_proof.cli \
  --output-dir reports/bert_24_plan_proof \
  --verbose
```

QK score contractions remain blockers and are not counted as propagation plans. This is static evidence and proof reporting only.
