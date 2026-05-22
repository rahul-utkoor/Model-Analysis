# Milestone 8: BERT MLP Block Pruning

## What you learn

You learn how an architecture-specific backend can prune only the BERT MLP intermediate dimension while preserving hidden size.

## Why this milestone exists

It demonstrates one controlled lowering experiment for a known structural pattern:
`intermediate.dense out_features` paired with `output.dense in_features`.

## Prerequisites

- Local BERT model
- This is optional and not part of the main analysis path

## Commands

List targets:

```bash
python scripts/list_bert_mlp_targets.py --model bert-base-uncased
```

Dry-run:

```bash
python scripts/prune_bert_mlp_block.py \
  --model bert-base-uncased \
  --layer 0 \
  --indices 0,1,2,3 \
  --dry-run \
  --smoke-test-before \
  --verbose
```

## Main artifacts produced

- `reports/block_pruning/`
- `reports/block_validation/`
- `reports/block_pruning_diffs/`

## What to inspect

Inspect detected MLP targets, applied structural changes in dry-run mode, caveats, and optional forward smoke reports.

## Expected interpretation

This backend reduces intermediate width while preserving hidden size. It should not be confused with a general pruning algorithm.

## Compiler analogy

This is architecture-specific lowering for one known IR pattern.

## What this milestone does not prove

It does not prune attention heads, rewrite ONNX, preserve accuracy, or generalize to every model.

## Connection to next milestone

Milestone 9 refocuses on the main research artifact: model-level pruning opportunity maps.

