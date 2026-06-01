# Milestone 48: Attention Value-Path Subgraphs

This demo builds a connected attention value-path evidence artifact:

```text
value projection -> layout -> attention context -> layout -> output projection
```

The fragment makes the structural rule seedable from output-projection input deadness while keeping unrelated attention-score computation outside the local artifact.

## Build OPT Layer 0

```bash
./conda-env/bin/python scripts/build_attention_value_path_subgraphs.py \
  --model facebook/opt-125m \
  --layer 0 \
  --export-onnx \
  --render-svg \
  --verbose
```

## Inspect and Compare

```bash
./conda-env/bin/python scripts/explain_attention_value_path_subgraph.py \
  --model facebook/opt-125m \
  --layer 0 \
  --status seedable

./conda-env/bin/python scripts/compare_attention_value_path_subgraphs.py \
  --models facebook/opt-125m \
  --verbose
```

## Re-run Evidence Coverage

```bash
./conda-env/bin/python -m experimental.mlir_evidence_coverage.cli \
  --models facebook__opt-125m \
  --layers layer0 \
  --patterns ATTENTION_VALUE_PATH \
  --output-dir reports/mlir_evidence_coverage_opt_value_path \
  --format both \
  --verbose
```

This is static artifact and evidence generation only. It does not execute pruning or modify model weights.
