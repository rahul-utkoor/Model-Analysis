# Milestone 13: k-Node and Join-Aware Subgraph Analysis

## What you learn

You learn how local ONNX regions expose pruning-relevant structure in two distinct forms: consecutive directed paths of one to five nodes, and join-centered regions where multiple branches merge.

## Why this milestone exists

A path such as `Gemm -> Gelu -> Gemm` can indicate an MLP-like intermediate dimension. A residual block cannot be understood from one path alone: an `Add` joins the skip and transformed branches and may impose equal hidden shapes. This pass preserves that distinction.

## Prerequisites

- ONNX graph summary from Milestone 2
- For the mainline pipeline, correspondence/shape evidence from Milestone 5 is useful context but not required by this command

## Commands

```bash
bash demo_scripts/run_demo_13_subgraph_analysis.sh
```

Equivalent direct command:

```bash
python scripts/analyze_subgraphs.py \
  --model bert-base-uncased \
  --max-nodes 5 \
  --branch-depth 2 \
  --post-join-depth 2 \
  --verbose
```

## Main artifacts produced

- `reports/subgraphs/bert-base-uncased.md`
- `reports/subgraph_patterns/bert-base-uncased.md`
- `reports/subgraph_pruning_analysis/bert-base-uncased.md`
- `reports/subgraph_dimension_evidence/bert-base-uncased.md`
- `reports/join_subgraphs/bert-base-uncased.md`
- `reports/residual_subgraphs/bert-base-uncased.md`

## What to inspect

In `reports/subgraphs/bert-base-uncased.md`, inspect path counts by size and the number of join-centered and residual-like regions. In `reports/join_subgraphs/bert-base-uncased.md`, inspect `Join(Add)` patterns, branch producers, and post-join normalization. In the pruning evidence report, inspect suggested constraint types such as `residual_equal_shape`, `mlp_same_intermediate_indices`, and `reshape_preservation`.

## Expected interpretation

Parameterized nodes such as `Gemm`, `MatMul`, and `Conv` expose local pruning surfaces. MLP-like paths identify possible intermediate-dimension coupling. Attention-like and reshape patterns require axis mapping. Residual-like joins require hidden-shape consistency across branches.

An `Add` is not automatically residual: `MatMul + initializer` is treated as a bias add, while an `Add` merging dataflow values and followed by `LayerNormalization` is a stronger residual candidate.

## Compiler analogy

This is local pattern matching over a graph IR, combined with dataflow join analysis. Sequential rewrite candidates and merge constraints are represented separately, as they would be in a compiler performing legality analysis.

## What this milestone does not prove

It does not rewrite Dimension IR automatically, execute pruning, modify models, or prove that any pruning candidate is legal.

## Connection to next milestone

The report-level residual and dimension evidence can later refine pruning maps and Dimension IR with more precise local constraint information.

