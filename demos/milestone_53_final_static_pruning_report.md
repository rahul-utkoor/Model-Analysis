# Milestone 53: Final Static Pruning Propagation Research Report

Generate the final reproducible report bundle:

```bash
./conda-env/bin/python scripts/build_final_static_pruning_report.py \
  --output-dir reports/final \
  --verbose
```

Inspect:

- `reports/final/index.md`
- `reports/final/static_pruning_propagation_final_report.md`
- `reports/final/static_pruning_propagation_final_summary.json`
- `reports/final/static_pruning_propagation_case_tables.csv`
- `reports/final/static_pruning_propagation_claims.md`

Use `--strict` when the generated all-model proof must be present.

This demo performs final reporting only. It does not execute pruning or mutate model weights.
