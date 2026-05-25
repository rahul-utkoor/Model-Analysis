# Milestone 35: Interactive Static Analysis Explorer

This demo launches the read-only terminal explorer for generated static pruning-analysis reports.

```bash
./conda-env/bin/python tools/interactive_analysis_explorer.py \
  --model bert-base-uncased \
  --layer 0 \
  --no-open
```

Try this walkthrough:

```text
nodes
subgraph Feed Forward
plan
validation
path
back
back
compare
quit
```

The explorer reads existing reports and artifacts only. It does not execute pruning, modify models, rewrite ONNX, download models, or evaluate accuracy.
