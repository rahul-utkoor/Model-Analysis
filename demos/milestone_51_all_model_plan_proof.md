# Milestone 51: All-Model Propagation Plan Proof

Generate a quick layer-0 report:

```bash
./conda-env/bin/python -m experimental.all_model_plan_proof.cli \
  --models all \
  --layers layer0 \
  --output-dir reports/all_model_plan_proof_layer0 \
  --format both \
  --build-missing-value-paths \
  --verbose
```

Generate the complete all-layer report:

```bash
./conda-env/bin/python -m experimental.all_model_plan_proof.cli \
  --models all \
  --layers all \
  --output-dir reports/all_model_plan_proof \
  --format both \
  --build-missing-value-paths \
  --verbose
```

Inspect:

- `reports/all_model_plan_proof/index.md`
- `reports/all_model_plan_proof/models/bert-base-uncased.md`
- `reports/all_model_plan_proof/models/facebook__opt-125m.md`
- `reports/all_model_plan_proof/models/gpt2.md`

The report counts QK score contractions as blockers rather than pruning plans. Milestone 52 adds GPT-2 fused-QKV value-slice recovery and confirms that the local ViT export exposes a recoverable separate value path.

This demo performs static evidence and proof reporting only.
