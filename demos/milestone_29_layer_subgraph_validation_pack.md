# Milestone 29: Layer Subgraph Validation Pack

Milestone 29 creates a local evidence pack for one encoder layer.

The pack is a projection of full-model analysis artifacts onto each learner-facing abstract node in the selected layer. It does not re-analyze each ONNX fragment as an independent model.

Inputs:

- Tensor IR
- Op Semantics
- Structural Region Tree
- Region Pruning Semantics
- Pruning Opportunity Ranking
- Symbolic Pruning Plans
- Pruning Plan Validation
- optional source ONNX for Netron visualization fragments

Build the layer 0 pack:

```bash
./conda-env/bin/python scripts/build_layer_subgraph_validation_pack.py \
  --model bert-base-uncased \
  --layer 0 \
  --export-onnx \
  --render-svg \
  --verbose
```

Inspect feed-forward evidence:

```bash
./conda-env/bin/python scripts/explain_layer_subgraph_validation.py \
  --model bert-base-uncased \
  --layer 0 \
  --contains "Feed Forward"
```

Inspect safe records:

```bash
./conda-env/bin/python scripts/explain_layer_subgraph_validation.py \
  --model bert-base-uncased \
  --layer 0 \
  --class safe
```

Outputs:

```text
reports/layer_subgraph_validation/bert-base-uncased/layer_0/index.json
reports/layer_subgraph_validation/bert-base-uncased/layer_0/index.md
reports/layer_subgraph_validation/bert-base-uncased/layer_0/<node>/explanation.md
artifacts/layer_subgraphs/bert-base-uncased/layer_0/<node>/subgraph.onnx
```

Interpretation:

- Feed Forward should show a safe candidate, symbolic plan, and valid plan validation.
- Query/Key/Value projections should show constrained attention-head mapping.
- Attention score/context MatMuls should show blocked attention contraction semantics.
- Attention mask add should show auxiliary/constraint-carrier semantics.

This is static analysis/reporting/visualization only. ONNX subgraphs are evidence artifacts for Netron, not standalone models for full re-analysis.
