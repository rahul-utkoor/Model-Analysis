# Design Notes

Model Analysis is a research infrastructure repository for static structural analysis of neural networks. The long-term goal is pruning analysis with forward and backward propagation of pruning constraints. The current code intentionally stops before modifying weights.

## Pipeline Design

The pipeline is staged:

1. Model registry
2. Local model download
3. ONNX export
4. PyTorch structural inventory
5. ONNX graph summary
6. Conservative pruning hints
7. Dependency graph construction
8. Dependency graph analysis
9. Pruning action simulation
10. Propagation trace and validation diagnostics
11. PyTorch-to-ONNX correspondence
12. Static shape evidence and dependency validation

Each stage writes JSON and/or Markdown artifacts. JSON files are intended as machine-readable intermediate representation. Markdown files are intended for manual research review.

## Core Modules

`registry.py`
: Loads `configs/models.yaml`, lists models, and resolves configured names or Hugging Face IDs.

`paths.py`
: Centralizes project-root discovery and generated artifact paths.

`hf_utils.py`
: Maps task names to Hugging Face model, tokenizer, and processor classes.

`onnx_export.py`
: Exports local Hugging Face models to ONNX with conservative dummy inputs and metadata.

`structural_inventory.py`
: Builds PyTorch structural summaries from loaded models. It detects linear layers, embeddings, normalization layers, attention-like names, MLP-like names, and initial pruning-relevant groups.

`onnx_graph_analysis.py`
: Builds ONNX graph summaries from exported models. It records graph inputs, outputs, initializers, node types, tensor shapes where available, and pruning-relevant ONNX nodes.

`reporting.py`
: Writes JSON, Markdown, CSV, and renders human-readable inventory and pruning-hint reports.

`dependency_graph.py`
: Defines the conservative pruning dependency graph IR. It creates prunable units, dependency edges, coupled groups, independent-unit candidates, ambiguous review items, and optional ONNX evidence.

`dependency_analyzer.py`
: Summarizes dependency graphs into high-value pruning targets, manual review items, forward propagation paths, and backward constraints.

`pruning_action.py`
: Defines dry-run pruning action, propagation step, and pruning plan schemas.

`propagation_engine.py`
: Simulates pruning actions over the dependency graph. It validates target dimensions and indices, traverses dependency edges, records propagation steps, and assigns conservative plan status.

`action_generation.py`
: Generates small deterministic candidate dry-run actions from dependency graph units.

`pruning_plan_reporting.py`
: Renders pruning plans and candidate actions as Markdown reports.

`correspondence.py`
: Builds conservative parameter-to-initializer and module-to-node correspondence evidence from structural inventory and ONNX graph summaries.

`shape_evidence.py`
: Collects tensor and node shape evidence from ONNX graph metadata without running inference.

`dependency_validation.py`
: Uses correspondence and shape evidence to identify validated units, supported dependency edges, and remaining manual-review items.

## Dependency Graph IR

The dependency graph is the current compiler-like IR for pruning analysis.

### Prunable Units

A `PrunableUnit` represents a layer, module, ONNX node, or higher-level structure that may be prunable:

```text
linear
embedding
attention_qkv
attention_output
mlp_expansion
mlp_projection
conv
matmul
gemm
```

Each unit records likely prunable dimensions, confidence, shape evidence, parameter count where available, and a reason string.

### Dependency Edges

A `DependencyEdge` captures likely propagation or coupling constraints:

```text
feeds
shape_dependency
residual_coupling
qkv_coupling
head_dimension_coupling
mlp_hidden_coupling
embedding_tying
normalization_dependency
propagation_only
```

Edges carry affected dimensions, direction, confidence, and rationale. Bidirectional edges are used when pruning decisions must usually be coordinated both upstream and downstream.

## Heuristic Policy

The project is deliberately conservative:

- Naming evidence can identify candidates, not prove correctness.
- Q/K/V projections are coupled when grouped under common attention structure.
- MLP expansion and projection layers are coupled through the intermediate dimension.
- LayerNorm is treated as propagation-dependent, not directly prunable.
- ONNX MatMul, Gemm, and Conv nodes are high-interest but often unmapped to PyTorch modules.
- ONNX Add, Reshape, Transpose, Softmax, Gather, and normalization nodes are propagation-relevant.
- Ambiguous structures are reported for manual review instead of silently promoted to safe pruning targets.

## Generated Artifacts

Generated model data is ignored by git:

```text
data/models/hf/
data/models/onnx/
```

Generated reports are ignored by git except `.gitkeep` placeholders:

```text
reports/model_summaries/
reports/structural_inventory/
reports/onnx_graphs/
reports/pruning_hints/
reports/dependency_graphs/
reports/dependency_summaries/
reports/correspondence/
reports/shape_evidence/
reports/validated_dependency_graphs/
```

## Current Limitations

- PyTorch modules are not yet mapped directly to ONNX nodes.
- ONNX shape inference is limited to available graph metadata and exported value info.
- Dependency graph construction is static and heuristic-based.
- No weights are modified.
- No pruning masks or executable pruning plans are emitted yet.
- No post-pruning validation is implemented yet.
- Pruning plans are dry-run diagnostics only. They do not transform PyTorch modules or ONNX graphs.
- PyTorch-to-ONNX correspondence is heuristic and incomplete for models exported through fused or rewritten graph patterns.

## Pruning Action Simulation

Milestone 4 introduces a conservative dataflow simulation layer. A `PruningAction` specifies a target unit, prune dimension, indices, strategy, and rationale. The propagation engine checks local validity and traverses dependency edges according to edge-specific semantics:

- `qkv_coupling` propagates matching attention indices across Q/K/V candidates.
- `head_dimension_coupling` records attention-output constraints and is usually ambiguous until head-index mapping exists.
- `mlp_hidden_coupling` propagates intermediate-channel constraints between expansion and projection layers.
- `residual_coupling` is ambiguous and requires manual branch-shape review.
- `normalization_dependency` records LayerNorm hidden-dimension constraints.
- `embedding_tying` is ambiguous unless tied output heads are proven.
- `propagation_only`, `feeds`, and `shape_dependency` are traced conservatively.

Plan status values:

```text
valid_local   The target action is locally valid and no required coupling was found.
valid_global  Required propagation was resolved with no ambiguity or conflicts.
ambiguous     The action may be plausible but needs better shape/mapping evidence.
rejected      The action is malformed or conflicts with known graph evidence.
```

Ambiguity is expected for nontrivial transformer pruning until correspondence and shape propagation improve.

## Correspondence and Shape Evidence

Milestone 5 adds an evidence bridge between PyTorch structural summaries and ONNX graph summaries.

The correspondence layer uses deterministic heuristics:

- normalized exact name matches
- suffix/name containment matches
- module-name plus weight/bias token matches
- shape-only fallback when unique enough to be useful

The shape layer uses ONNX metadata only:

- graph inputs and outputs
- value_info shapes
- initializer dimensions
- node input/output tensor names

Dependency validation is conservative:

- Units are validated only with medium/high correspondence evidence.
- Edges are shape-supported only when source and destination evidence has compatible shapes.
- Q/K/V and MLP couplings gain confidence when corresponding projection nodes and dimensions line up.
- Residual, embedding tying, and unmapped ONNX operations remain manual-review surfaces.

This evidence can enrich pruning plans, but it does not make pruning executable by itself.
