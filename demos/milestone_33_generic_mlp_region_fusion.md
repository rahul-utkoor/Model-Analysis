# Milestone 33: Generic MLP Region Fusion

Milestone 33 recovers feed-forward/MLP pruning opportunities from op semantics when the structural tree does not already expose a native `FeedForwardRegion`.

## Core Pattern

```text
expansion projection: hidden_dim -> intermediate_dim
activation: intermediate_dim -> intermediate_dim, index-preserving
contraction projection: intermediate_dim -> hidden_dim
```

The synthesized `GenericMLPRegion` maps to the same symbolic plan shape used for BERT and OPT FFN pruning.

## Commands

```bash
./conda-env/bin/python scripts/build_region_pruning_semantics.py --model distilbert-base-uncased --verbose
./conda-env/bin/python scripts/rank_pruning_opportunities.py --model distilbert-base-uncased --verbose
./conda-env/bin/python scripts/synthesize_pruning_plans.py --model distilbert-base-uncased --verbose
./conda-env/bin/python scripts/validate_pruning_plans.py --model distilbert-base-uncased --verbose

./conda-env/bin/python scripts/build_static_pipeline_for_all_models.py --models all --build-missing-analysis --build-layer-packs --verbose
./conda-env/bin/python scripts/report_static_pipeline_coverage.py --models all --verbose
./conda-env/bin/python scripts/diagnose_rule_gaps.py --models all --verbose
```

## Expected Checks

- BERT remains at 12 valid FFN plans.
- OPT remains at 12 valid FFN plans.
- DistilBERT recovers 6 generic FFN plans when evidence is complete.
- ViT recovers 12 generic MLP plans when evidence is complete.
- GPT-2 recovers 12 generic MLP plans when evidence is complete.

This is static analysis/reporting only. It does not choose pruning indices, modify model weights, execute pruning, rewrite ONNX, download models, train, or evaluate accuracy.
