# Milestone 61: Compact ONNX Annotations and Leader Discovery

This demo reruns strict MLIR-derived ONNX axis-semantics annotation with compact node metadata and a leader-candidate report.

The ONNX node `doc_string` is visualization-friendly by default. Full MLIR relation/evidence details remain in the sidecar JSON.

Run:

```bash
./conda-env/bin/python scripts/annotate_onnx_axis_semantics.py \
  --input artifacts/attention_value_path_subgraphs/bert-base-uncased/layers/layer_0/bert_layer_0_attention_value_path/subgraph.onnx \
  --output artifacts/annotated_onnx/bert-base-uncased/layer_0/attention_value_path.axis_annotated.onnx \
  --sidecar-json reports/onnx_axis_semantics/bert-base-uncased_layer0_attention_value_path.json \
  --dot artifacts/annotated_onnx/bert-base-uncased/layer_0/attention_value_path.axis_annotated.dot \
  --svg artifacts/annotated_onnx/bert-base-uncased/layer_0/attention_value_path.axis_annotated.svg \
  --leader-report reports/onnx_axis_semantics/bert-base-uncased_layer0_attention_value_path.leaders.md \
  --mlir-output-dir reports/onnx_axis_semantics/bert-base-uncased_layer0_attention_value_path_mlir \
  --annotation-mode doc_string \
  --doc-string-format compact \
  --run-native-pass \
  --fallback-doc-string \
  --check-onnx \
  --verbose
```

Expected outputs:

- compact annotated ONNX
- full sidecar JSON
- readable DOT/SVG labels
- MLIR-derived leader report

This is annotation, visualization, and leader-candidate discovery only. It does not execute pruning.
