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
```

## Current Limitations

- PyTorch modules are not yet mapped directly to ONNX nodes.
- ONNX shape inference is limited to available graph metadata and exported value info.
- Dependency graph construction is static and heuristic-based.
- No weights are modified.
- No pruning masks or executable pruning plans are emitted yet.
- No post-pruning validation is implemented yet.

## Suggested Milestone 4

Milestone 4 should add PyTorch-to-ONNX correspondence and stronger shape propagation:

- Map PyTorch module names to ONNX nodes where possible.
- Run or integrate ONNX shape inference more deeply.
- Track tensor producers and consumers explicitly.
- Convert dependency graph evidence into candidate pruning plans.
- Still avoid modifying weights until candidate plans can be validated.
