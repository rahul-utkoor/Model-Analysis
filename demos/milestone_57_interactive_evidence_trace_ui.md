# Milestone 57: Interactive Evidence Trace UI

Build the React interface:

```bash
bash demo_scripts/run_demo_57_interactive_evidence_trace_ui.sh
```

Then start the local read-only server:

```bash
./conda-env/bin/python tools/analysis_ui_api_server.py \
  --host 127.0.0.1 \
  --port 8777 \
  --verbose
```

Open `http://127.0.0.1:8777/`.

Recommended walkthrough:

1. Pipeline Flow
2. Evidence Trace
3. FFN intermediate propagation
4. Attention value-path propagation
5. QK score blocker
6. Real Artifact panel
7. Models
8. Reports

This demo visualizes generated static evidence only. It does not execute pruning, mutate model weights, or evaluate accuracy.
