# Milestone 52: Fused-QKV Value-Slice Recovery

Build GPT-2 fused-QKV attention value-path artifacts:

```bash
./conda-env/bin/python scripts/build_attention_value_path_subgraphs.py \
  --model gpt2 \
  --layers all \
  --export-onnx \
  --render-svg \
  --verbose
```

Build ViT attention value-path artifacts from the exported separate value projection:

```bash
./conda-env/bin/python scripts/build_attention_value_path_subgraphs.py \
  --model google/vit-base-patch16-224 \
  --layers all \
  --export-onnx \
  --render-svg \
  --verbose
```

Re-run the all-model proof:

```bash
./conda-env/bin/python -m experimental.all_model_plan_proof.cli \
  --models all \
  --layers all \
  --output-dir reports/all_model_plan_proof \
  --format both \
  --build-missing-value-paths \
  --verbose
```

GPT-2 recovery requires an explicit `Split`, `Slice`, or `Gather` branch that reaches attention context as the value operand. Ambiguous fused projections stay blocked.

This demo performs static evidence and proof generation only.
