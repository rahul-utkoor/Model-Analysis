# Milestone 20: Semantic Fusion for Feed-Forward Regions

## What you learn

You learn why a model's semantic feed-forward block may not appear as a single frontend operator. In BERT exports, GELU is often lowered into `Div/Mul -> Erf -> Add -> Mul -> Mul`; semantic fusion recovers that activation and its enclosing projection-activation-projection structure in Tensor IR.

## Why this milestone exists

Region-aware legality analysis can only expose `intermediate_dim` when the Structural Region Tree recognizes a `FeedForwardRegion`. This pass recovers high-level meaning from decomposed operations, allowing BERT-style feed-forward structure to participate in region-scoped Dimension IR.

## Prerequisites

- Tensor IR from Milestone 16 under `reports/tensor_ir/<model>.json`

## Commands

```bash
bash demo_scripts/run_demo_20_semantic_fusion.sh
```

Equivalent commands:

```bash
python scripts/analyze_semantic_fusion.py --model bert-base-uncased --verbose
python scripts/build_structural_region_tree.py --model bert-base-uncased --verbose
python scripts/build_region_dimension_ir.py --model bert-base-uncased --verbose
python scripts/list_region_dimensions.py --model bert-base-uncased --contains intermediate --limit 20
```

## Main artifacts produced

- `reports/semantic_fusion/bert-base-uncased.md`
- `reports/fused_region_patterns/bert-base-uncased.md`
- `reports/structural_region_trees/bert-base-uncased.md`
- `reports/region_dimension_ir/bert-base-uncased.md`

## What to inspect

Open the semantic-fusion report and find `GeluActivation` and `FeedForward` rows. Then inspect the region tree for fused `FeedForwardRegion` entries and the region dimension listing for `intermediate_dim` producer/consumer variables.

## Expected interpretation

A high-confidence GELU fusion records graph evidence for the multiply-back activation structure. A feed-forward fusion surrounds it with two projections and exposes a same-index intermediate-dimension obligation. Activation-internal additions are not residual merges.

## Compiler analogy

This is semantic idiom recognition after frontend lowering: a sequence or DAG of primitive IR operations is lifted back into a higher-level region before symbolic data-flow and legality analysis.

## What this milestone does not prove

It does not prove numerical equivalence for every activation decomposition, execute pruning, modify model weights, rewrite ONNX, or evaluate accuracy.

## Connection to next milestone

Region-aware legality queries can now analyze BERT intermediate dimensions discovered through fused feed-forward regions. Future work can strengthen axis and shape evidence attached to those region variables.
