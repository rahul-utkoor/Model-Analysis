# Ordered Dataflow Control-Tree Browser

This local browser shows the final Structural Region Tree as an ordered, expandable hierarchy.

It is different from the step trace viewer:

- the step trace viewer explains how individual collapses happened;
- the ordered tree browser shows the final hierarchy in Tensor IR/source order.

Start the server:

```bash
python tools/ordered_control_tree_api_server.py \
  --model bert-base-uncased \
  --port 8767
```

Open:

```text
http://127.0.0.1:8767/
```

The browser lazy-loads only the selected node, immediate children, search results, paths, and ordered primitive leaves. It does not load the full tree into the browser and does not modify models, pruning logic, Tensor IR, ONNX, or weights.
