# Pruning Analysis Web UI

## Purpose

The pruning analysis web UI is a browser interface for navigating generated static pruning-analysis reports. It presents models, layers/blocks, abstract subgraphs, ONNX/SVG evidence artifacts, op semantics, rankings, symbolic plans, and validation checks without requiring users to open dozens of files manually.

This is read-only visualization. It does not execute pruning, choose pruning indices, modify model weights, rewrite ONNX, download models, or evaluate accuracy.

## Build Frontend

```bash
cd ui/pruning-analysis-explorer
npm install
npm run build
cd ../..
```

## Start Backend

```bash
./conda-env/bin/python tools/analysis_ui_api_server.py --port 8777
```

Open:

```text
http://127.0.0.1:8777/
```

The backend uses Python stdlib `http.server`, reads existing files under `reports/` and `artifacts/`, and serves the built React app when `ui/pruning-analysis-explorer/dist/` exists.

## Dev Mode

Run the backend:

```bash
./conda-env/bin/python tools/analysis_ui_api_server.py --port 8777
```

Run Vite in another terminal:

```bash
cd ui/pruning-analysis-explorer
npm run dev -- --host 127.0.0.1 --port 5173
```

The Vite dev server proxies `/api` and `/artifact` requests to `http://127.0.0.1:8777`.

## What Can Be Inspected

- Cross-model static coverage
- Model summaries and pipeline counts
- Layer/block atlases
- Ordered abstract subgraphs
- ONNX, SVG, and DOT artifacts
- Primitive op lists
- Op and region semantics
- Opportunity ranking evidence
- Symbolic pruning plans
- Plan validation checks
- Rule-gap diagnosis and status reports

## Typical Walkthrough

1. Open the dashboard and choose `bert-base-uncased`.
2. Select `Layer 0`.
3. Click `Feed Forward`.
4. Inspect `Plan`, `Validation`, and `Artifacts`.
5. Switch to `facebook/opt-125m`, select `Layer 0`, and inspect the MLP block.
6. Return to the dashboard to compare valid plan counts across models.
