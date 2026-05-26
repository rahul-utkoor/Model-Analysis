# Milestone 36: Pruning Analysis Web UI

Milestone 36 adds a local React + Vite browser interface over the generated static pruning-analysis reports.

Build the frontend:

```bash
cd ui/pruning-analysis-explorer
npm install
npm run build
cd ../..
```

Start the local API/server:

```bash
./conda-env/bin/python tools/analysis_ui_api_server.py --host 127.0.0.1 --port 8777
```

Open:

```text
http://127.0.0.1:8777/
```

Suggested inspection path:

1. Open the coverage dashboard and compare all configured models.
2. Choose `bert-base-uncased`, then `Layer 0`, then `Feed Forward`.
3. Inspect the symbolic plan, validation checks, and ONNX/SVG artifact links.
4. Choose `facebook/opt-125m`, then `Layer 0`, then the MLP block.
5. Use search for `Attention Score`, `LayerNorm`, `MLP`, or `valid`.

The UI is a read-only reporting layer. It uses existing reports and artifacts and does not execute pruning, modify models, rewrite ONNX, download models, or evaluate accuracy.
