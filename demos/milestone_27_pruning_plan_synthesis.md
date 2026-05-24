# Milestone 27: Pruning Plan Synthesis

Milestone 27 turns the top safe pruning opportunities into symbolic repair/pruning plans.

Inputs:

- Pruning Opportunity Ranking for candidate selection
- Region Pruning Semantics for required repairs and protected dimensions
- Op Semantics for primitive TensorOp evidence

The current planner targets the safest class first: feed-forward `intermediate_dim` pruning. It does not choose concrete indices. Instead, each plan is parameterized by a symbolic index set such as `I_layer_0_intermediate`.

Build plans:

```bash
./conda-env/bin/python scripts/synthesize_pruning_plans.py \
  --model bert-base-uncased \
  --verbose
```

Inspect ready plans:

```bash
./conda-env/bin/python scripts/explain_pruning_plan.py \
  --model bert-base-uncased \
  --status ready_symbolic \
  --limit 20
```

Inspect one layer:

```bash
./conda-env/bin/python scripts/explain_pruning_plan.py \
  --model bert-base-uncased \
  --contains "Layer 0 Feed Forward"
```

Outputs:

```text
reports/pruning_plans/bert-base-uncased.json
reports/pruning_plan_dumps/bert-base-uncased.plan
reports/pruning_plan_explanations/bert-base-uncased.md
```

A ready FFN plan says to apply the same symbolic `intermediate_dim` index set to:

- the `intermediate.dense` projection output axis
- the `intermediate.dense` bias axis
- the GELU activation positions by no-index-change propagation
- the `output.dense` projection input axis

The plan also preserves the output hidden dimension and forbids residual or LayerNorm hidden-dimension pruning for this transformation.

This is static reporting/analysis only. It does not modify models, execute pruning, rewrite ONNX, export ONNX, download models, or evaluate quality.
