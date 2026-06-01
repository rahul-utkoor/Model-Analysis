# All-Model Propagation Plan Proof

This experimental reporting layer generalizes the BERT 24-plan case study across BERT, DistilBERT, OPT, GPT-2, and ViT.

For every transformer layer it evaluates one FFN intermediate propagation plan and one attention value-path plan. QK score contractions are reported separately as blockers because their reduced and mixed feature axis does not support simple one-to-one deadness propagation.

The report reuses the existing MLIR evidence coverage runner. Native MLIR dependence, Python affine/access evidence, high-level fallback, missing artifacts, and unsupported fused-QKV paths remain explicit.

GPT-2 and ViT currently expose a deliberate `fused_qkv_value_path_gap`: value-slice recovery has not yet been proven. DistilBERT can build separable `v_lin -> context -> out_lin` artifacts on demand.

## Run

```bash
./conda-env/bin/python -m experimental.all_model_plan_proof.cli \
  --models all \
  --layers all \
  --output-dir reports/all_model_plan_proof \
  --format both \
  --build-missing-value-paths \
  --verbose
```

This is static evidence and proof reporting only. It does not execute pruning, mutate models, or evaluate accuracy.
