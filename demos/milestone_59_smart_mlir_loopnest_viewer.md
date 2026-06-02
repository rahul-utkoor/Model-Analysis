# Milestone 59: Smart MLIR Loopnest Viewer

Build the React interface:

```bash
bash demo_scripts/run_demo_59_smart_mlir_loopnest_viewer.sh
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

1. Open `Evidence Trace`.
2. Select `FFN / MLP intermediate propagation`.
3. Scroll to `Real Artifact` and open `MLIR`.
4. Select `lowered affine`.
5. Inspect the default focused loop/access sections with original MLIR line numbers.
6. Use jump buttons for `affine.for`, loads, and stores.
7. Toggle `Full file` to inspect the raw generated artifact.

If an artifact has no affine loopnest, the UI reports that and shows available ONNX, Krnl, Linalg, or SCF fallback operations.

This demo reads generated artifacts only. It does not lower ONNX on request, execute pruning, mutate weights, or evaluate accuracy.
