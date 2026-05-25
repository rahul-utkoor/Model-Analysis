# Milestone 31: Cross-Model Static Coverage

Milestone 31 audits how far the static pruning-analysis pipeline gets for every configured model.

It answers:

- Which models are complete, partial, skipped, or failed?
- Which stages are present or missing?
- Which models expose safe candidates, plans, and valid plans?
- Where are new model-specific semantics needed?

Build BERT status:

```bash
./conda-env/bin/python scripts/build_static_pipeline_for_model.py \
  --model bert-base-uncased \
  --build-missing-analysis \
  --build-layer-packs \
  --verbose
```

Build all-model status and coverage:

```bash
./conda-env/bin/python scripts/build_static_pipeline_for_all_models.py \
  --models all \
  --build-missing-analysis \
  --build-layer-packs \
  --verbose
```

Regenerate coverage from manifests:

```bash
./conda-env/bin/python scripts/report_static_pipeline_coverage.py --models all --verbose
```

Inspect:

- `reports/static_pipeline_status/bert-base-uncased.md`
- `reports/static_pipeline_status/facebook__opt-125m.md`
- `reports/static_coverage_study/index.md`

This is static analysis/reporting only. It does not download models, execute pruning, rewrite full ONNX models, or evaluate accuracy.
