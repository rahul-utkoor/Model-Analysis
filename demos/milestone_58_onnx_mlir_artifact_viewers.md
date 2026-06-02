# Milestone 58: ONNX Graph and MLIR Artifact Viewers

Build the optional artifact inventory and React interface:

```bash
bash demo_scripts/run_demo_58_onnx_mlir_artifact_viewers.sh
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

1. Evidence Trace
2. FFN intermediate propagation
3. Real Artifact
4. ONNX Graph
5. MLIR
6. Dependence
7. Models -> BERT layer 0 -> Feed Forward -> Artifacts
8. Reports -> Final report

This demo reads generated artifacts only. It does not lower ONNX on request, execute pruning, mutate weights, or evaluate accuracy.
