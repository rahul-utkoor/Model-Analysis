# Milestone 60: Strict MLIR-Derived ONNX Axis Semantics

This demo generates debug-only ONNX axis-semantics annotations for selected local evidence units.

Key rule: semantic classes are derived only from MLIR access/dependence evidence. If ONNX-MLIR is missing, lowering fails, or no access relation is recovered, the sidecar records an explicit blocker rather than an ONNX op-schema fallback.

Run:

```bash
./conda-env/bin/python scripts/annotate_onnx_axis_semantics.py \
  --input artifacts/attention_value_path_subgraphs/bert-base-uncased/layers/layer_0/bert_layer_0_attention_value_path/subgraph.onnx \
  --output artifacts/annotated_onnx/bert-base-uncased/layer_0/attention_value_path.axis_annotated.onnx \
  --sidecar-json reports/onnx_axis_semantics/bert-base-uncased_layer0_attention_value_path.json \
  --dot artifacts/annotated_onnx/bert-base-uncased/layer_0/attention_value_path.axis_annotated.dot \
  --svg artifacts/annotated_onnx/bert-base-uncased/layer_0/attention_value_path.axis_annotated.svg \
  --mlir-output-dir reports/onnx_axis_semantics/bert-base-uncased_layer0_attention_value_path_mlir \
  --annotation-mode doc_string \
  --run-native-pass \
  --fallback-doc-string \
  --check-onnx \
  --verbose
```

Outputs:

- annotated ONNX: `artifacts/annotated_onnx/.../*.axis_annotated.onnx`
- sidecar JSON: `reports/onnx_axis_semantics/*.json`
- DOT/SVG graph: `artifacts/annotated_onnx/.../*.dot` and `.svg`
- MLIR evidence artifacts: `reports/onnx_axis_semantics/*_mlir/`

The original ONNX is not modified, and no pruning is executed.
