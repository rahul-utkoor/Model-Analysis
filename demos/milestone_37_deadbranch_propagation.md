# Milestone 37: Attention Value-Path Deadness Propagation

Milestone 37 adds a static deadbranch propagation report aligned with SparseGPT channel-pruning experiments.

Run:

```bash
./conda-env/bin/python scripts/analyze_deadbranch_propagation.py \
  --model facebook/opt-125m \
  --verbose

./conda-env/bin/python scripts/explain_deadbranch_propagation.py \
  --model facebook/opt-125m \
  --contains v_proj \
  --limit 5
```

Inspect:

```text
reports/deadbranch_propagation/facebook__opt-125m.json
reports/deadbranch_propagation_dumps/facebook__opt-125m.deadbranch
reports/deadbranch_propagation_explanations/facebook__opt-125m.md
```

The pass separates sparse weights from structural deadness:

- SparseGPT-style `2:4` / `V:N:M` pruning is shape-preserving and does not guarantee dead channels.
- Exact dead consumer columns permit backward deadness propagation when index mappings are proven.
- FFN propagation follows `fc1 -> fc2`.
- Attention value-path propagation follows `v_proj -> out_proj`.
- Query/key propagation remains blocked because `QK^T` score contraction mixes projected channels.

For OPT-125M, the report predicts `12` FFN pairs and `12` attention value-path pairs, matching the `24` experimentally observed propagation pairs.

This is static analysis/reporting only. It does not execute pruning or modify models.
