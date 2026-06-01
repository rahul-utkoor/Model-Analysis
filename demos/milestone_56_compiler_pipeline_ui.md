# Milestone 56: Compiler Pipeline Visualization UI

Build the React interface:

```bash
bash demo_scripts/run_demo_56_compiler_pipeline_ui.sh
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

1. Dashboard
2. Pipeline Flow
3. FFN propagation
4. Attention value path
5. QK blocker
6. All-model proof
7. Reports

This demo visualizes generated static evidence only. It does not execute pruning, mutate model weights, or evaluate accuracy.
