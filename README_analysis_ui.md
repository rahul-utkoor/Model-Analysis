# Pruning Analysis Web UI

## Purpose

The pruning analysis web UI is a browser interface for navigating generated static pruning-analysis reports. It presents a compiler-style pipeline flow alongside models, layers/blocks, abstract subgraphs, embedded ONNX/SVG graph previews, DOT source, generated MLIR text, dependence JSON, op semantics, rankings, symbolic plans, and validation checks.

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

- Pipeline flow from dead axes to DFA fixed-point proofs
- Evidence Trace laboratory with graph transitions, affine equations, axis states, and DFA timeline controls
- Real artifact bundles joining ONNX graph previews, DOT text, generated MLIR stages, native dependence JSON, and local proof summaries
- All-model `108 / 108` proof summary
- BERT, fused-QKV, OPT diagnosis, attention value-path, and QK blocker case studies
- Read-only final, formalization, methodology, and claims reports
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
- Deadbranch propagation reports
- Rule-gap diagnosis and status reports

## Typical Walkthrough

1. Open `Pipeline Flow` and establish why sparsity is not the same as deadness.
2. Open `Evidence Trace` and step through FFN, attention value-path, and QK blocker examples.
3. Switch the trace panels between pattern matching, MLIR evidence, axis state, and verdict.
4. Use `Case Studies` for the BERT `24 / 24`, all-model `108 / 108`, fused-QKV, and OPT diagnosis stories.
5. In `Evidence Trace`, open the real artifact panel to connect the explanatory trace to generated ONNX and MLIR files.
6. Open `Models`, choose `bert-base-uncased`, select `Layer 0`, and use the `Artifacts` tab to inspect the embedded graph, DOT, MLIR, dependence, and evidence panels.
7. Use `Reports` to preview the final report, claims, and formalization outlines.

## Optional Artifact Index

The backend discovers artifact bundles directly from existing generated files. For a compact offline inventory, run:

```bash
./conda-env/bin/python scripts/build_ui_mlir_artifact_index.py \
  --output reports/ui_artifact_index/index.json \
  --verbose
```

The indexer only reads generated artifacts. UI requests never invoke ONNX-MLIR.

## Read-only Teaching API

The server exposes:

- `/api/overview`
- `/api/proof-summary`
- `/api/teaching-flow`
- `/api/pipeline-flow`
- `/api/evidence-traces`
- `/api/evidence-artifact-map`
- `/api/artifact-bundle?model=<model>&layer=<layer>&subgraph=<slug>`
- `/api/artifact-text?path=<repo-relative-text-path>`
- `/api/case-studies`
- `/api/report-text?path=<relative-report-path>`

`/api/report-text` accepts only `.md`, `.json`, and `.csv` files beneath `reports/`.

`/api/artifact-text` accepts repository-confined `.mlir`, `.dot`, `.json`, `.md`, `.txt`, and `.csv` files and caps returned content at 1 MiB.
