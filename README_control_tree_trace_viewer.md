# Lightweight Control-Tree Trace Viewer

This local viewer browses the stepwise dataflow control-tree construction trace without loading the full trace JSON into the browser.

Start with an existing trace:

```bash
python scripts/build_control_tree_trace.py \
  --model bert-base-uncased \
  --format all \
  --max-dot-steps 20 \
  --verbose
```

Run the lazy API server:

```bash
python tools/control_tree_trace_api_server.py \
  --model bert-base-uncased \
  --port 8766
```

Open:

```text
http://127.0.0.1:8766/
```

The frontend fetches only the trace index, pages of step summaries, one selected step, and one selected step's local graph. It is an explanatory structural-analysis browser and does not modify models, pruning logic, Tensor IR, ONNX, or weights.
