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
13. Reversible Linear-only pruning execution
14. Paired Linear structural repair and forward smoke validation
15. BERT MLP block-level executable pruning
16. Compiler-style pruning opportunity maps
17. Dimension-variable IR and symbolic propagation constraints

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

`linear_pruning.py`
: Performs validated row/column surgery on `torch.nn.Linear` modules while preserving dtype, device, bias presence, and `requires_grad`.

`pruning_plan_executor.py`
: Converts pruning plans into executable Linear prune specs and applies them to an in-memory model.

`pruning_execution.py`
: Defines execution report data structures and Markdown rendering.

`pruning_diff.py`
: Compares before/after structural summaries and reports changed Linear layers.

`rollback.py`
: Generates rollback manifests that describe created files and how to return to the original model directory.

`repair_plan.py`
: Defines paired Linear repair plans and transaction records.

`repair_detection.py`
: Detects explicit MLP and Linear hidden-dimension pair repairs from pruning plans and dependency edges.

`paired_linear_pruning.py`
: Applies atomic source `out_features` plus target `in_features` Linear repairs.

`forward_validation.py`
: Runs minimal forward smoke tests and summarizes output tensor structure.

`bert_mlp_pruning.py`
: Detects and executes the BERT-specific MLP pruning pattern where `intermediate.dense` output channels and `output.dense` input channels are pruned together.

`pruning_opportunity.py`
: Defines the compiler-style pruning opportunity IR: pruning dimensions, propagation constraints, opportunities, structural risks, and model pruning maps.

`pruning_map_compare.py`
: Aggregates and compares pruning map summaries across configured models.

`dimension_ir.py`
: Converts pruning maps into symbolic dimension variables, pruning-index variables, constraint equations, equivalence classes, blocked dimensions, and unresolved constraints.

`pruning_ir_text.py`
: Renders a deterministic MLIR-like `.pir` textual dump of the pruning Dimension IR.

`dimension_ir_compare.py`
: Compares Dimension IR summaries across configured models.

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
artifacts/pruned_models/
reports/pruning_execution/
reports/pruning_diffs/
reports/rollback_manifests/
reports/repair_plans/
reports/repair_transactions/
reports/forward_smoke_tests/
reports/block_pruning/
reports/block_validation/
reports/block_pruning_diffs/
reports/model_pruning_maps/
reports/pruning_opportunities/
reports/propagation_constraints/
reports/structural_risk_maps/
reports/dimension_ir/
reports/constraint_equations/
reports/dimension_equivalence/
reports/pruning_ir_dumps/
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
- Linear-only execution can create structurally edited checkpoints, but transformer-wide correctness is not guaranteed.
- Paired Linear repair is limited to explicit MLP/hidden-dimension pairs; attention-head, residual, LayerNorm, and embedding repairs are not automatic.
- BERT MLP block pruning reduces only the intermediate dimension; hidden-size and attention pruning are intentionally unsupported.

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

## Reversible Linear Pruning Prototype

Milestone 6 introduces a deliberately narrow executable transform:

- `nn.Linear` `out_features` pruning removes weight rows and matching bias entries.
- `nn.Linear` `in_features` pruning removes weight columns and leaves bias unchanged.
- The source model directory is never modified.
- Execution writes a new artifact directory and rollback manifest.
- Dry-run mode validates extraction and records skipped records without modifying the model.

This prototype is useful for validating structural surgery mechanics. It is not yet a full transformer pruning system because adjacent layers, residual paths, LayerNorm parameters, configs, and ONNX graphs may still need coordinated repair.

## Paired Linear Repair and Forward Smoke Validation

Milestone 7 adds atomic paired Linear repair for the first safe structural pattern:

- MLP expansion/intermediate `out_features` pruning
- matching MLP projection/output `in_features` pruning

The repair detector only emits executable repairs when the pruning plan or dependency graph explicitly represents the coupling. Attention output, residual, normalization, and embedding edges are recorded for manual review rather than rewritten.

Forward smoke tests run minimal synthetic inputs through a model before or after pruning. A passing smoke test means the forward call executed and produced summarizable outputs. It does not prove task accuracy, calibration, or semantic equivalence.

## BERT MLP Block-Level Pruning

Milestone 8 adds a direct architecture-specific pruning path for BERT encoder MLP blocks:

```text
bert.encoder.layer.<L>.intermediate.dense
bert.encoder.layer.<L>.output.dense
```

The executable transform prunes `intermediate.dense` `out_features` and applies the same indices to `output.dense` `in_features`. This reduces the feed-forward intermediate dimension while keeping the hidden dimension unchanged. Residual and LayerNorm dimensions should therefore remain unchanged, and attention modules are untouched.

This path is intentionally separate from the generic dependency graph executor. It relies on a known BERT MLP block structure rather than broad graph heuristics. It still does not rewrite ONNX, evaluate task quality, fine-tune, or support attention-head pruning. Because standard BERT config stores one global `intermediate_size`, single-layer pruning creates non-uniform MLP sizes that may require custom reload metadata in a later milestone.

Executable pruning modules are now treated as experimental validation backends. They help test whether a structural hypothesis can be carried into a concrete artifact, but they are not the main research direction.

## Compiler-Style Pruning Opportunity Maps

Milestone 9 returns the project to its core research objective: compiler-style structural analysis for pruning opportunities and pruning-information propagation.

The pruning opportunity IR contains:

- `PruningDimension`: a dimension variable attached to a unit, such as `out_features`, `intermediate_dim`, `num_heads`, or `embedding_dim`
- `PropagationConstraint`: an equality, same-index, reshape-preserving, residual, QKV, MLP, tied-parameter, or unknown-mapping relation between dimensions
- `PruningOpportunity`: a candidate pruning region with required constraints, affected units, propagation paths, risk level, and executability label
- `ModelPruningMap`: a model-level artifact that groups dimensions, constraints, opportunities, risks, independent regions, coupled regions, and blocked regions

The pruning map is the primary research artifact. It is intended to support later symbolic dimension analysis and legal pruning-space reasoning before any weight transformation is attempted.

## Dimension Variable IR

Milestone 10 turns descriptive pruning maps into a more analyzable symbolic IR.

The Dimension IR contains:

- `DimensionVariable`: a stable compiler-style variable for a model dimension such as `out_features`, `intermediate_dim`, `num_heads`, or `embedding_dim`
- `PruningIndexVariable`: a symbolic set of indices selected for pruning along a dimension
- `ConstraintEquation`: symbolic propagation rules such as `same_indices`, `eq`, `tied`, `reshape`, or `unknown`
- `DimensionEquivalenceClass`: groups of dimensions connected by equality, same-index, or tied-parameter constraints
- blocked dimensions and unresolved constraints for regions that cannot be legally transformed without stronger evidence

The `.pir` dump is inspired by MLIR:

```text
pruning.module @bert-base-uncased {
  pruning.dim %d0 owner("bert.encoder.layer.0.intermediate.dense") ...
  pruning.constraint %c0 same_indices(%d0, %d1) ...
  pruning.eq_class %e0 members(%d0, %d1) ...
}
```

This is a textual research artifact, not executable MLIR. It exists to make pruning legality, symbolic propagation, and blocked-region analysis easier to inspect and reason about.
