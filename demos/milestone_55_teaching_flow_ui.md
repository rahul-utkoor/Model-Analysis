# Milestone 55: Pipeline Walkthrough UI

Build the final report bundle and React pipeline interface:

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

Use `Pipeline Flow` for the compiler-stage walkthrough, `Case Studies` for focused proofs, `Reports` for read-only Markdown previews, and `Models` for the existing layer/subgraph explorer.

This demo visualizes generated static evidence only. It does not execute pruning, mutate model weights, or evaluate accuracy.
