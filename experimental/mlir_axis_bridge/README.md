# ONNX-MLIR Axis Bridge Prototype

## Why This Exists

This experiment uses local ONNX-MLIR lowering as a semantic evidence source for selected pruning-analysis subgraphs. It connects real emitted MLIR text to the existing axis-transfer and DFA teaching prototypes.

## Why We Do Not Rewrite the Whole Analysis in MLIR

The production analysis already provides model-wide semantics, reports, and evidence packs. This prototype intentionally lowers only small ONNX subgraphs. It tests whether compiler IR can strengthen local axis proofs without replacing the broader analysis pipeline.

## Selected ONNX Subgraphs as Local Evidence Units

Input artifacts come from paths such as:

```text
artifacts/model_analysis_subgraphs/<model>/layers/layer_<N>/<subgraph>/subgraph.onnx
```

The subgraphs are read-only evidence artifacts. They are never executed or rewritten.

## ONNX-MLIR Lowering Stages

For each selected subgraph, the runner requests:

```bash
onnx-mlir subgraph.onnx --EmitONNXIR -o <prefix>_onnx
onnx-mlir subgraph.onnx --EmitMLIR --preserveMLIR -o <prefix>_lowered
```

Both command results and all emitted text artifacts are recorded.

## Dialects We May See

The artifact scanner records evidence for ONNX, Krnl, Linalg, SCF, Affine, and memref operations. A particular ONNX-MLIR build may expose only some of these stages.

## Access Extraction

The parser is intentionally lightweight. It extracts recognized operations plus `affine.load`, `affine.store`, `memref.load`, and `memref.store` accesses. It is not a complete MLIR parser.

## Axis-Transfer Summary Construction

Evidence sources are reported explicitly:

- `native_mlir_dependence_evidence`: imported native-pass JSON proves a supported relation.
- `actual_loop_access_evidence`: indexed accesses prove a supported relation.
- `high_level_mlir_dialect_evidence`: emitted MLIR operations plus ONNX shape hints support conservative template lowering.
- `onnx_hint_fallback`: only the existing ONNX local hint is sufficient.

The bridge never presents fallback evidence as a loop-level proof.

## Native MLIR Dependence Evidence

The current bridge includes strengthened Python-side affine access extraction and an optional native dependence tool under `native/`. The Python extractor records enclosing loop IVs, affine/memref accesses, preserved IVs, reduced IVs, and conservative mixed relations. It can emit the same JSON contract as the native MLIR tool:

```json
{
  "mlir_file": "selected_subgraph.mlir",
  "analysis_tool": "native_mlir_pass",
  "dialects_seen": ["affine.for", "affine.load", "affine.store"],
  "relations": [
    {
      "relation_id": "context_value_preserved",
      "source_tensor": "V",
      "source_indices": ["b", "head", "k", "d"],
      "target_tensor": "Context",
      "target_indices": ["b", "head", "q", "d"],
      "loop_ivs": ["b", "head", "q", "k", "d"],
      "relation_kind": "preserved",
      "dependence_kind": "access_equivalence",
      "affine_evidence": ["affine.load ...", "affine.store ..."],
      "proof": "value IV d remains free from V into Context",
      "confidence": "high"
    }
  ],
  "reductions": ["k"],
  "preserved_axes": ["d"],
  "blocked_axes": [],
  "warnings": []
}
```

Import or emit dependence JSON with:

```bash
python -m experimental.mlir_axis_bridge.cli \
  --onnx <subgraph.onnx> \
  --output-dir reports/mlir_axis_bridge/example \
  --native-dependence-json experimental/mlir_axis_bridge/native/sample_expected_output.json \
  --prefer-native-dependence \
  --emit-python-dependence-json reports/mlir_axis_bridge/example/python_dependence.json \
  --format markdown
```

The native C++ analyzer is an optional standalone MLIR-linked tool. It is not compiled by the normal Python test suite and does not replace the Python bridge.

## Build and Run the Native Tool

```bash
bash experimental/mlir_axis_bridge/native/build_native_pass.sh

experimental/mlir_axis_bridge/native/build/pruning-axis-dependence \
  experimental/mlir_axis_bridge/native/samples/attention_context_affine.mlir \
  --output reports/mlir_axis_bridge/native_attention_context_sample.json
```

Run it automatically on the richest emitted MLIR artifact:

```bash
python -m experimental.mlir_axis_bridge.cli \
  --onnx <subgraph.onnx> \
  --output-dir reports/mlir_axis_bridge/example \
  --run-native-pass \
  --native-pass-tool experimental/mlir_axis_bridge/native/build/pruning-axis-dependence \
  --native-output-dir reports/mlir_axis_bridge/example/native \
  --format markdown
```

## Relationship to Other Prototypes

- `experimental/onnx_axis_bridge/` supplies conservative local ONNX topology and shape hints.
- `experimental/axis_transfer_analysis/` summarizes preserved, reduced, and blocked axes.
- `experimental/pruning_analysis_bridge/` lowers complete axis patterns into DFA propagation.

## How to Run

```bash
python -m experimental.mlir_axis_bridge.cli \
  --onnx artifacts/model_analysis_subgraphs/gpt2/layers/layer_0/03_gpt_2_block_0_mlp_block/subgraph.onnx \
  --output-dir reports/mlir_axis_bridge/gpt2_layer0_mlp \
  --format markdown \
  --show-all \
  --verbose
```

## Limitations

This is not full ONNX-to-MLIR semantic lowering. The native analyzer intentionally implements a minimal access-relation layer rather than a complete MLIR dependence solver. Unsupported layouts remain warnings. MLIR is used as a local semantic evidence source, not as the pruning framework itself.
