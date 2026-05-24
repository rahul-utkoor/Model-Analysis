# Milestone 25: Region Pruning Semantics

This demo answers a different question from the structure viewers:

> Given a learned abstract region tree, what pruning choices are legal, propagated, repaired, protected, or blocked?

The pass reads existing static artifacts. It does not modify models, execute pruning, rewrite ONNX, export ONNX, download models, or evaluate accuracy.

## Prerequisites

```bash
python scripts/build_tensor_ir.py --model bert-base-uncased --verbose
python scripts/analyze_semantic_fusion.py --model bert-base-uncased --verbose
python scripts/build_structural_region_tree.py --model bert-base-uncased --verbose
python scripts/build_region_dimension_ir.py --model bert-base-uncased --verbose
python tools/export_abstract_node_expansion_report.py --model bert-base-uncased --view main --max-leaf-names 30
```

## Build Semantics

```bash
python scripts/build_region_pruning_semantics.py \
  --model bert-base-uncased \
  --verbose
```

Inspect:

- `reports/region_pruning_semantics/bert-base-uncased.json`
- `reports/region_pruning_semantics_dumps/bert-base-uncased.rpsem`
- `reports/region_pruning_semantics_explanations/bert-base-uncased.md`

## Explain Key Cases

Feed-forward opportunity:

```bash
python scripts/explain_region_pruning_semantics.py \
  --model bert-base-uncased \
  --contains "Feed Forward" \
  --limit 5
```

Blocked/protected regions:

```bash
python scripts/explain_region_pruning_semantics.py \
  --model bert-base-uncased \
  --blocked-only \
  --limit 10
```

## Compare Models

```bash
python scripts/compare_region_pruning_semantics.py --models all
```

Milestone 25 is a static reporting layer over learner structural regions. It explains pruning information flow, repair obligations, and blockers without invoking an executable pruning backend.

Important interpretation:

- FFN `intermediate_dim` pruning is the clean directly prunable opportunity.
- Attention score/context MatMuls are contractions, not independent projection layers.
- Attention mask add applies score bias/masking, not residual hidden-state merging.
- Attention pruning remains blocked until head-axis mapping is proven.
- `source_region_type` says what the Structural Region Tree found; `semantic_category` says how pruning semantics interprets that region.
