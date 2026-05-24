# Milestone 25.4: Pruning-Relevant Op Semantics

Milestone 25.4 adds a primitive Tensor IR annotation layer for pruning-relevant operation behavior.

Region Pruning Semantics says what an abstract region means. Op Semantics says what each primitive TensorOp does locally:

- learned projection MatMuls expose parameterized axes
- bias Adds follow projection output indices
- attention score/context MatMuls are contractions, not learned projections
- attention mask Adds and selects carry mask/broadcast metadata
- GELU decomposition ops preserve intermediate indices
- residual Adds require branch agreement
- reshape, transpose, shape, and constant ops carry axis or metadata flow

Build the report:

```bash
./conda-env/bin/python scripts/build_op_semantics.py \
  --model bert-base-uncased \
  --verbose
```

Explain attention contractions:

```bash
./conda-env/bin/python scripts/explain_op_semantics.py \
  --model bert-base-uncased \
  --semantic-kind attention_score_matmul \
  --limit 5
```

Explain learned projections:

```bash
./conda-env/bin/python scripts/explain_op_semantics.py \
  --model bert-base-uncased \
  --category parameterized_projection \
  --limit 10
```

Outputs:

```text
reports/op_semantics/bert-base-uncased.json
reports/op_semantics_dumps/bert-base-uncased.opsem
reports/op_semantics_explanations/bert-base-uncased.md
```

This is static reporting/analysis only. It does not modify models, execute pruning, rewrite ONNX, export ONNX, download models, or evaluate quality.

