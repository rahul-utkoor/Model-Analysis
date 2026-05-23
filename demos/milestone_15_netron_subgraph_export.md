# Milestone 15: Netron ONNX Subgraph Export

## What you learn

You learn how abstract path, join, and DAG-region records become standalone ONNX visualization artifacts that can be opened in Netron.

## Why this milestone exists

Tables and JSON explain detected structure, but a visual graph makes branch boundaries, residual joins, fanout, and reconvergence much easier to inspect. This milestone makes selected report records visible without changing the source model.

## Prerequisites

- Original local ONNX model under `data/models/onnx/<safe-model>/model.onnx`
- Subgraph reports from Milestones 13 and 14

## Commands

Export a curated visualization set:

```bash
bash demo_scripts/run_demo_15_netron_subgraph_export.sh
```

Equivalent direct command:

```bash
python scripts/export_demo_subgraphs.py \
  --model bert-base-uncased \
  --max-per-category 3 \
  --verbose
```

Export only selected DAG regions:

```bash
python scripts/export_subgraph_onnx.py \
  --model bert-base-uncased \
  --kind dag_region \
  --max-exports 5 \
  --verbose
```

## Main artifacts produced

- `artifacts/subgraph_onnx/bert-base-uncased/demo/*.onnx`
- `reports/subgraph_exports/bert-base-uncased__demo.md`
- `reports/netron_subgraph_index/bert-base-uncased__demo.md`

## What to inspect

Open the Netron index and launch one exported file with its displayed `netron <file>` command. Inspect graph inputs and outputs as the artificial boundaries of the extracted region, then compare the visible nodes to the originating path, join, or DAG report.

## Expected interpretation

The exported files preserve selected nodes, required initializers, graph boundary tensors, available shapes, and provenance metadata. A residual join or reconvergent DAG region becomes visibly distinct from a linear sequence.

## Compiler analogy

This is a visualization/lifting aid for selected IR regions: it materializes a local graph view for human inspection, similar to dumping a compiler region around a matched pattern.

## What this milestone does not prove

The extracted files are not standalone semantically complete models. They introduce artificial input/output boundaries, do not evaluate correctness, do not execute pruning, and do not modify the original ONNX model.

## Connection to next milestone

Visual confirmation of region structure can guide future work that feeds multi-branch evidence into pruning maps and Dimension IR constraints.

