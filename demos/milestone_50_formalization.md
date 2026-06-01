# Milestone 50: Static Pruning Propagation Formalization

This demo generates teaching and paper-preparation material from existing static evidence reports.

## Build Formalization Reports

```bash
./conda-env/bin/python scripts/build_static_pruning_formalization.py \
  --output-dir reports/formalization \
  --verbose
```

## Generated Documents

- `reports/formalization/static_pruning_propagation_notes.md`
- `reports/formalization/bert_24_plan_case_study.md`
- `reports/formalization/teaching_slide_outline.md`
- `reports/formalization/paper_methodology_outline.md`
- `reports/formalization/index.md`
- `reports/formalization/index.json`

The generated material explains sparse-weight versus structural pruning, local ONNX evidence units, MLIR-backed axis-transfer proof, semantic pattern recognition, DFA worklist propagation, BERT's complete 24-plan result, QK blockers, and remaining limitations.

This is documentation and formalization only. It does not execute pruning or mutate model weights.
