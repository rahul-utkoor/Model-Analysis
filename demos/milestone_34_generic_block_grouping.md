# Milestone 34: Generic Transformer Block Grouping

This demo builds learner-facing layer/block atlases for non-BERT transformer families using existing static artifacts.

```bash
./conda-env/bin/python scripts/build_layer_subgraph_validation_pack.py \
  --model facebook/opt-125m \
  --layer 0 \
  --export-onnx \
  --render-svg \
  --verbose

./conda-env/bin/python scripts/build_layer_subgraph_validation_pack.py \
  --model gpt2 \
  --layer 0 \
  --export-onnx \
  --render-svg \
  --verbose

./conda-env/bin/python scripts/build_full_model_analysis_report.py \
  --model google/vit-base-patch16-224 \
  --layers all \
  --export-onnx-subgraphs \
  --render-svg \
  --verbose
```

Inspect:

```text
reports/layer_subgraph_validation/facebook__opt-125m/layer_0/index.md
reports/layer_subgraph_validation/gpt2/layer_0/index.md
reports/model_analysis_reports/google__vit-base-patch16-224/index.md
```

The grouping is a report projection over full-model op semantics, region semantics, rankings, plans, and validation. ONNX subgraphs are visualization artifacts only.
