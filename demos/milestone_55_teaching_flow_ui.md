# Milestone 55: Teaching Flow UI

Build the final report bundle and React teaching interface:

```bash
bash demo_scripts/run_demo_55_teaching_flow_ui.sh
```

Then start the local read-only server:

```bash
./conda-env/bin/python tools/analysis_ui_api_server.py \
  --host 127.0.0.1 \
  --port 8777 \
  --verbose
```

Open `http://127.0.0.1:8777/`.

Use `Teaching Flow` for the professor walkthrough, `Case Studies` for focused proofs, `Reports` for read-only Markdown previews, and `Models` for the existing layer/subgraph explorer.

This demo visualizes generated static evidence only. It does not execute pruning, mutate model weights, or evaluate accuracy.
