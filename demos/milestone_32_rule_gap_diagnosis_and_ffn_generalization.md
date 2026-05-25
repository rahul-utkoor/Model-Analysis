# Milestone 32: Rule-Gap Diagnosis and FFN Generalization

Milestone 32 diagnoses why cross-model static analysis stops short of valid pruning plans, then applies generic FFN evidence matching where the existing artifacts expose expansion, activation, and contraction ops.

## Commands

```bash
./conda-env/bin/python scripts/diagnose_rule_gaps.py --models all --verbose
./conda-env/bin/python scripts/explain_rule_gap.py --model facebook/opt-125m
./conda-env/bin/python scripts/compare_rule_gaps.py --models all
./conda-env/bin/python scripts/build_static_pipeline_for_all_models.py --models all --build-missing-analysis --build-layer-packs --verbose
./conda-env/bin/python scripts/report_static_pipeline_coverage.py --models all --verbose
```

## Outputs

```text
reports/rule_gap_diagnosis/<model>.json
reports/rule_gap_diagnosis/<model>.md
reports/rule_gap_diagnosis_compare/index.json
reports/rule_gap_diagnosis_compare/index.md
reports/static_coverage_study/index.md
```

## What To Check

- BERT remains the complete reference model with valid FFN plans.
- OPT `fc1/fc2` evidence can bind to generic FFN plans when op semantics exposes learned projection roles.
- DistilBERT `ffn.lin1/lin2`, ViT `mlp.fc1/fc2`, and GPT-2 `mlp.c_fc/c_proj` are diagnosed rather than silently treated as BERT-only failures.
- Remaining gaps are reported as missing fusion, missing evidence, missing activation semantics, missing layer grouping, or unsupported family coverage.

This milestone is static diagnosis/reporting only. It does not choose pruning indices, modify models, execute pruning, rewrite ONNX, download models, or evaluate accuracy.
