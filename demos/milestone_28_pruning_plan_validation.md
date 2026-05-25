# Milestone 28: Pruning Plan Validation

Milestone 28 checks symbolic pruning plans before any optional executable pruning backend consumes them.

Inputs:

- Symbolic Pruning Plans
- Pruning Opportunity Ranking
- Region Pruning Semantics
- Op Semantics

The validator checks that a feed-forward `intermediate_dim` plan is internally consistent:

- the candidate is still safe in the ranking
- the plan is `ready_symbolic`
- producer output, intermediate bias, and consumer input actions are present
- producer, bias, consumer, and GELU op semantics agree
- required repairs are present
- hidden dimensions are preserved
- residual, LayerNorm, and output hidden-bias pruning are forbidden
- blockers and unknown critical ops are absent

Validate plans:

```bash
./conda-env/bin/python scripts/validate_pruning_plans.py \
  --model bert-base-uncased \
  --verbose
```

Inspect valid records:

```bash
./conda-env/bin/python scripts/explain_pruning_plan_validation.py \
  --model bert-base-uncased \
  --status valid \
  --limit 20
```

Inspect one layer:

```bash
./conda-env/bin/python scripts/explain_pruning_plan_validation.py \
  --model bert-base-uncased \
  --contains "Layer 0 Feed Forward"
```

Outputs:

```text
reports/pruning_plan_validation/bert-base-uncased.json
reports/pruning_plan_validation_dumps/bert-base-uncased.pvalid
reports/pruning_plan_validation_explanations/bert-base-uncased.md
```

This is static reporting/analysis only. It does not choose concrete indices, modify models, execute pruning, rewrite ONNX, export ONNX, download models, or evaluate quality.
