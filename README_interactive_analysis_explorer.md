# Interactive Static Analysis Explorer

## Purpose

`tools/interactive_analysis_explorer.py` is a guided CLI for exploring generated static pruning-analysis reports.

It is read-only. It does not choose pruning indices, execute pruning, modify models, rewrite ONNX, download models, or evaluate accuracy.

## Start

```bash
./conda-env/bin/python tools/interactive_analysis_explorer.py
```

Useful direct launch:

```bash
./conda-env/bin/python tools/interactive_analysis_explorer.py \
  --model bert-base-uncased \
  --layer 0 \
  --no-open
```

## Typical Walkthrough

1. Select a model.
2. Select a layer or block.
3. List abstract nodes/subgraphs.
4. Choose Feed Forward / MLP Block.
5. Inspect the ranking, symbolic plan, and validation.
6. Open or print the ONNX subgraph path.
7. Compare static coverage across models.

## Useful Commands

At model level:

```text
summary
layers
layer 0
pipeline
ranking
validation
find Feed Forward
compare
open
quit
```

At layer level:

```text
summary
nodes
subgraph Feed Forward
plans
find valid
back
quit
```

At subgraph level:

```text
explanation
json
ops
sem
rank
plan
validation
onnx
path
back
```

## Notes

- Reads `reports/model_analysis_reports/` and `reports/layer_subgraph_validation/`.
- Opens existing ONNX subgraphs from `artifacts/model_analysis_subgraphs/` or `artifacts/layer_subgraphs/`.
- With `--no-open`, the `onnx` command prints the path instead of launching Netron/system viewer.
- ONNX subgraphs are visualization/evidence artifacts, not standalone analysis inputs.
