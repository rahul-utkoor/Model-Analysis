# Milestone 26: Pruning Opportunity Ranking

Milestone 26 turns static pruning semantics into a ranked candidate table.

Inputs:

- Region Pruning Semantics for region-level roles, repairs, and blockers
- Op Semantics for primitive TensorOp evidence

The ranking answers which opportunities should be inspected first:

- `safe`: clean opportunities such as feed-forward `intermediate_dim` pruning
- `constrained`: learned projections that need missing proofs, such as Q/K/V head-axis mapping
- `blocked`: residual hidden dimensions, LayerNorm hidden dimensions, and attention contractions
- `auxiliary`: shape, mask, axis, and metadata flow
- `unknown`: insufficient semantic evidence

Build the ranking:

```bash
./conda-env/bin/python scripts/rank_pruning_opportunities.py \
  --model bert-base-uncased \
  --verbose
```

Inspect safe candidates:

```bash
./conda-env/bin/python scripts/explain_pruning_opportunity.py \
  --model bert-base-uncased \
  --class safe \
  --limit 20
```

Inspect constrained and blocked cases:

```bash
./conda-env/bin/python scripts/explain_pruning_opportunity.py \
  --model bert-base-uncased \
  --class constrained \
  --limit 20

./conda-env/bin/python scripts/explain_pruning_opportunity.py \
  --model bert-base-uncased \
  --contains "Attention Score MatMul"
```

Outputs:

```text
reports/pruning_opportunity_rankings/bert-base-uncased.json
reports/pruning_opportunity_ranking_dumps/bert-base-uncased.rank
reports/pruning_opportunity_explanations/bert-base-uncased.md
```

This is static reporting/analysis only. It does not modify models, execute pruning, rewrite ONNX, export ONNX, download models, or evaluate quality.

