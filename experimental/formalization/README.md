# Static Pruning Propagation Formalization

This module generates teaching and paper-preparation notes from the existing static evidence reports.

## Outputs

- formal notes
- BERT 24-plan case study
- teaching slide outline
- paper methodology outline
- Markdown and JSON index

## Run

```bash
python scripts/build_static_pruning_formalization.py \
  --output-dir reports/formalization \
  --verbose
```

Missing evidence inputs produce warnings and partial documentation rather than aborting generation.

This is documentation and formalization only. It does not execute pruning or mutate model weights.
