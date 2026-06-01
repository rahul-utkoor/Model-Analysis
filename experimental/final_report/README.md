# Final Static Pruning Propagation Research Report

This package generates the reproducible final reporting bundle for the static pruning propagation research pipeline.

It reads existing proof artifacts, formalization notes, MLIR coverage summaries, attention value-path summaries, deadbranch reports, symbolic plans, and validation reports. It does not execute analysis passes, pruning, model mutation, or accuracy evaluation.

## Run

```bash
./conda-env/bin/python scripts/build_final_static_pruning_report.py \
  --output-dir reports/final \
  --verbose
```

Use `--strict` to require `reports/all_model_plan_proof/index.json`.

Generated outputs include the final narrative report, claims boundary document, machine-readable summary, CSV case table, and index.
