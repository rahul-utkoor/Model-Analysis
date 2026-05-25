# Milestone 30: Full-Model Analysis Reports

Milestone 30 collects the static pruning-analysis pipeline into model-level and cross-model report trees.

It answers:

- Which layers expose safe validated FFN plans?
- Which attention structures are constrained or blocked?
- Which residual/LayerNorm paths protect hidden dimensions?
- How do generated reports compare across configured models?

Build the BERT report:

```bash
./conda-env/bin/python scripts/build_full_model_analysis_report.py \
  --model bert-base-uncased \
  --layers all \
  --export-onnx-subgraphs \
  --render-svg \
  --verbose
```

Build all available model reports without ONNX subgraph export:

```bash
./conda-env/bin/python scripts/build_all_model_analysis_reports.py \
  --models all \
  --layers all \
  --no-export-onnx-subgraphs \
  --verbose
```

Compare reports:

```bash
./conda-env/bin/python scripts/compare_model_analysis_reports.py --models all --verbose
```

Inspect:

- `reports/model_analysis_reports/bert-base-uncased/index.md`
- `reports/model_analysis_reports/bert-base-uncased/layers/layer_0/index.md`
- `reports/model_analysis_reports/cross_model/index.md`
- `artifacts/model_analysis_subgraphs/bert-base-uncased/`

This is static reporting/visualization only. ONNX fragments are evidence artifacts and are not standalone full-model analysis sources.
