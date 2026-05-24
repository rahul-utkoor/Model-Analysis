# Design Notes

Model Analysis is a research infrastructure repository for static structural analysis of neural networks. The long-term goal is pruning analysis with forward and backward propagation of pruning constraints. ONNX is a frontend representation; Tensor Graph IR is the frontend-independent substrate, Structural Region Tree is its compiler-inspired semantic hierarchy, Region-Aware Dimension IR lowers region interfaces into symbolic equations, and region-aware legality analysis evaluates symbolic requests. The current analysis path intentionally stops before modifying weights.

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
18. Dimension-IR propagation analysis and legality checking
19. Demo track and research walkthrough
20. k-node and join-aware ONNX subgraph structural analysis
21. DAG motif and multi-join region structural analysis
22. Netron-visualizable ONNX subgraph extraction
23. Frontend-independent Tensor Graph IR import and reporting
24. Structural Region Tree construction over Tensor IR
25. Region-Aware Dimension IR construction from structural interfaces
26. Region-aware pruning propagation and legality analysis

Each stage writes JSON and/or Markdown artifacts. JSON files are intended as machine-readable intermediate representation. Markdown files are intended for manual research review.

## Demo Track

Milestone 12 adds a guided learning layer over the existing pipeline. It does not add pruning functionality or analysis algorithms. Instead, it explains how to run and interpret the existing artifacts.

The mainline demo ladder is:

```text
Model checkpoint
  -> ONNX frontend graph
  -> Structural inventory
  -> Tensor Graph IR
  -> Structural Region Tree
  -> Region-Aware Dimension IR
  -> Region-Aware Legality Analysis
  -> Dependency graph
  -> Correspondence and shape evidence
  -> k-node and join-aware subgraph evidence
  -> DAG motif and multi-join region evidence
  -> Netron visualization fragments
  -> Pruning opportunity map
  -> Dimension IR
  -> Legality check
```

Milestones 6-8 are documented as optional experimental backend demos. They are useful for validating lowering ideas, but Tensor IR, Structural Region Tree, Region-Aware Dimension IR, region-aware legality analysis, pruning maps, and Dimension IR remain the primary research artifacts.

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

`tensor_ir.py`
: Defines frontend-independent tensor values, canonical tensor operations, graph serialization, operation canonicalization, and human-readable Tensor IR reports.

`onnx_to_tensor_ir.py`
: Imports ONNX graph summaries into Tensor IR while retaining ONNX only as source provenance.

`tensor_ir_text.py`
: Renders deterministic `.tir` tensor-dataflow dumps for inspection and future structural-region work.

`tensor_ir_compare.py`
: Compares canonical operations, semantic roles, region hints, forks, and joins across Tensor IR graphs.

`structural_region_tree.py`
: Defines hierarchical semantic regions and preliminary region interfaces over Tensor IR.

`structural_region_detection.py`
: Detects conservative bounded region candidates, resolves overlap into a hierarchy, preserves primitive leaves, and infers structural interfaces.

`structural_region_tree_text.py`
: Renders readable `.srtree` compiler-style hierarchy dumps.

`structural_region_tree_compare.py`
: Compares semantic region types and pruning roles across constructed trees.

`region_dimension_ir.py`
: Derives region-scoped symbolic dimensions, constraints, equivalence classes, blocked dimensions, and unresolved mappings from Structural Region Tree interfaces.

`region_dimension_ir_text.py`
: Renders deterministic `.rdim` textual dumps for semantic-region-derived dimensions and equations.

`region_dimension_ir_compare.py`
: Compares axis roles, region types, constraints, blocked dimensions, and unresolved mappings across RegionDimensionIR reports.

`region_ir_graph.py`
: Builds conservative inferred-direction adjacency and forward/backward slices over region-scoped constraints.

`region_ir_analysis.py`
: Checks semantic-region pruning requests, computes repair obligations, and explains blocked/protected or unresolved region dimensions.

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

`ir_graph.py`
: Builds adjacency over Dimension IR constraints and extracts forward/backward propagation slices.

`ir_analysis.py`
: Performs static legality checks for symbolic pruning requests, computes minimal structural repair sets, and explains blocked pruning regions.

`subgraph_analysis.py`
: Enumerates directed ONNX paths of bounded length and join-centered branch-merge regions, distinguishing bias adds from residual-style Add candidates and emitting report-level pruning/propagation evidence.

`subgraph_compare.py`
: Aggregates local path, join, residual, risk, and evidence pattern summaries across model reports.

`dag_region_analysis.py`
: Detects bounded forks, reconvergent diamonds, and join-fork-join regions while preserving branch paths, internal edges, tensor boundaries, and suggested multi-branch constraints.

`dag_region_compare.py`
: Aggregates DAG motif patterns, region kinds, risks, and suggested constraints across model reports.

`onnx_subgraph_extractor.py`
: Converts selected path, join, or DAG-region records into derived ONNX fragments with artificial graph boundaries, consumed initializers, available tensor metadata, and extraction provenance for Netron inspection.

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
artifacts/subgraph_onnx/
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
reports/legality_checks/
reports/propagation_slices/
reports/repair_sets/
reports/ir_analysis/
reports/subgraphs/
reports/subgraph_patterns/
reports/subgraph_pruning_analysis/
reports/subgraph_dimension_evidence/
reports/join_subgraphs/
reports/residual_subgraphs/
reports/dag_regions/
reports/dag_region_patterns/
reports/dag_region_pruning_evidence/
reports/subgraph_exports/
reports/netron_subgraph_index/
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

## Region-Aware Dimension IR

Milestone 18 adds a semantic-region-derived path alongside the pruning-map-derived Dimension IR. Region interfaces now introduce scoped symbolic dimensions:

- projection regions expose output and propagated input feature dimensions
- feed-forward regions expose producer/consumer intermediate dimensions linked by same-index equations while protecting their hidden boundary
- residual merge and normalization regions protect hidden dimensions with equality constraints
- axis transform and attention skeleton regions retain unresolved mapping obligations conservatively
- fork and join regions expose fanout and branch-compatibility equations

The `.rdim` dump records these region variables, equations, and equivalence classes. This new path does not replace existing Dimension IR; it provides a structural-region semantic layer for later region-aware legality analysis. It does not modify models or execute pruning.

## Semantic Fusion for Feed-Forward Regions

Milestone 20 refines Structural Region Tree candidate detection with semantic idiom recovery over Tensor IR. Some frontend graphs, notably BERT ONNX-derived Tensor IR, express GELU as a small dataflow DAG rather than one activation op. The fusion pass recognizes an `Erf`-centered multiply-back form such as `Div -> Erf -> Add -> Mul -> Mul`, records it as a `GeluActivation`, and recognizes enclosing projection/GELU/projection motifs as `FeedForward` regions.

Fused regions are inserted as semantic region candidates before overlap resolution, while primitive operations remain leaves. Adds proven to be inside a GELU expression are not promoted to residual-merge blockers. Feed-forward fusion metadata flows into Region-Aware Dimension IR evidence so recovered BERT intermediate dimensions can be traced to their first and second projections. This is structural recovery only; no model or frontend graph is modified.

## Stepwise Control-Tree Construction Trace

Milestone 21 adds an explanatory construction trace for the Structural Region Tree. The trace initializes one active graph node per TensorOp, applies semantic-region evidence from fusion reports and the Structural Region Tree, collapses matched nodes into abstract region nodes, and records before/after graph summaries plus DOT snapshots.

The trace mirrors compiler structural analysis as a sequence of region reductions, but it is deliberately a teaching and debugging artifact. The final Structural Region Tree remains the authoritative hierarchy. The trace does not modify Tensor IR, ONNX, models, or pruning logic.

## Lightweight Control-Tree Trace Viewer

The trace viewer is API-backed because full traces can contain hundreds of steps and each step can contain graph snapshots. The browser loads only the trace index, a page of step summaries, one selected step, and a reduced local graph around the selected collapse. Moving to another step replaces the selected-step state rather than retaining complete historical snapshots in browser memory.

The local graph is intentionally small: created region, collapsed nodes, immediate incoming boundary nodes, immediate outgoing boundary nodes, and relevant dataflow/abstraction edges. It is an explanation tool, not a full graph renderer.

## Ordered Dataflow Control-Tree Browser

The ordered tree browser presents the final Structural Region Tree as the neural-model analogue of a compiler control tree. It preserves containment and Tensor IR/source order rather than grouping by region type or rendering a force-directed graph. Children are sorted by explicit region order when present, then by minimum source TensorOp index, then by stable numeric/id fallback.

The frontend lazy-loads root, immediate children, selected-node details, paths, search results, and primitive leaf lists. This makes the final hierarchy readable while keeping browser memory small for BERT-sized trees.

## Abstract Node Expansion Report

The abstract-node expansion report is a learner-facing companion to the ordered hierarchy and dataflow PDF views. It records two expansion notions explicitly:

- immediate expansion: direct abstract or primitive children in the learner hierarchy
- recursive primitive leaves: underlying source ONNX/TensorIR operations used as evidence

Root-like records (`ModelRegion`, virtual `SectionRegion`, and grouped `ShapeMotifRegion`) hide recursive primitive leaves by default so reports stay readable. Semantic regions such as feed-forward, attention, residual merge, layer norm, and linear projection still show primitive leaves as evidence. Auxiliary predicate, mask, and shape construction operations are grouped into shape motifs so the main compute report is not dominated by scalar and axis plumbing.

## Region Pruning Semantics

Region Pruning Semantics sits above the Structural Region Tree and Region-Aware Dimension IR. It does not decide concrete indices or transform weights. Instead, it assigns conservative pruning-flow meaning to each learner region:

- Feed-forward regions expose an `intermediate_dim` opportunity and same-index MLP repair obligations.
- GELU and elementwise activation regions propagate index sets without changing them.
- Residual merges and LayerNorm regions protect hidden dimensions and explain branch/parameter repair blockers.
- Attention skeletons recognize score/context structure but block executable head pruning until head and axis mappings are proven.
- Attention score/context MatMuls are treated as dataflow contractions (`Q x K^T` and `Softmax(scores) x V`), not directly prunable linear projections.
- Attention mask adds are treated as score bias/mask application, not residual hidden-state merges.
- Shape and mask regions carry axis/metadata propagation obligations rather than direct pruning choices.

The `.rpsem` text dump is intended to make these semantics readable as a compiler-style analysis artifact.

## Region-Aware Pruning Propagation Analysis

Milestone 19 queries RegionDimensionIR directly. Given a region dimension and symbolic or concrete index request, it determines:

- whether the requested region dimension is prunable, protected, or blocked
- which same-index or fanout constraints require propagated selections
- which residual and normalization constraints reject hidden-width changes
- which attention or axis-transform mappings remain unresolved
- forward propagation slices, backward constraint slices, and minimal repair obligations

Region equations currently omit explicit direction because their relation type encodes the intended semantics. This analysis conservatively infers bidirectional traversal for equality-like and unresolved mappings and forward traversal for fanout propagation. The result is a diagnostic legality oracle only; it does not modify models or invoke experimental backends.

## Region Tree Browsing and Export Tools

The Structural Region Tree can be too large for direct browser-side loading. The local API browsers treat reports as static analysis data and expose lazy GET endpoints for indexes, focused regions, direct children, blocked regions, search results, abstract structure catalogs, and paginated structure instances. Collapsed subtrees can be discarded from frontend memory and re-fetched later.

The abstract structure collector groups concrete regions by semantic signature: region type, pruning role, child-type multiset, op-type multiset, dimension roles, and constraint types. This supports browsing structure classes such as feed-forward, residual merge, fork, layer norm, axis transform, attention skeleton, and primitive regions before drilling into instances.

The MindNode exporter walks from `root_region_id` and preserves structural child order where available, falling back to source/topological operation order. Primitive leaves are omitted by default so the outline remains readable. These tools do not change Tensor IR, region trees, Dimension IR, model weights, ONNX files, or pruning behavior.

## Dimension-IR Legality Analysis

Milestone 11 adds static analysis over the Dimension IR. Given a root dimension and symbolic or concrete pruning request, the analyzer determines whether the request is locally legal, legal with structural repairs, ambiguous, or rejected.

The legality layer derives:

- equivalent dimensions that must share pruning decisions
- forward propagation slices
- backward constraint slices
- constraint satisfaction states
- minimal structural repair sets
- blocking reasons and unresolved mappings

This layer does not modify weights or execute pruning. It is intended to support future constraint solving and pruning-legality proofs over the symbolic IR.

## Join-Aware ONNX Subgraph Analysis

Milestone 13 adds local pattern analysis over saved ONNX graph summaries:

- directed simple paths of one through five nodes expose local projection, MLP, attention, normalization, and shape-transform patterns
- join-centered subgraphs preserve branch-merge semantics around `Add`, `Sum`, and `Concat`
- initializer-backed `Add` operations are kept as bias additions instead of being incorrectly promoted to residual joins
- dataflow joins followed by `LayerNormalization` provide stronger residual hidden-shape evidence

This evidence is report-level input for future pruning-map and Dimension-IR precision improvements. It does not alter models or automatically rewrite existing IR artifacts.

## DAG Motif and Multi-Join Region Analysis

Milestone 14 extends local analysis from paths and single joins to bounded DAG motifs:

- a `fork` records one producer feeding multiple consumers
- a `diamond` records fanout followed by reconvergence
- a `join_fork_join` records a node that first merges branches, then fans out into branches that merge again

The canonical example is:

```text
A -> C
B -> C
C -> D
C -> E
D -> F
E -> F
```

In this region, pruning a value associated with `C`, `D`, or `E` may impose compatibility at both `C` and `F`. The report records `fanout_same_indices`, `branch_output_compatibility`, `residual_equal_shape`, and reshape-related constraints conservatively. It does not modify models or rewrite Dimension IR automatically.

## Netron ONNX Subgraph Export

Milestone 15 converts selected structural-analysis records into derived ONNX graphs for visual inspection:

- source node order is preserved
- boundary inputs and outputs become explicit graph boundaries
- consumed initializers and available `value_info` metadata are copied
- opset imports and IR version are preserved
- metadata identifies the source graph, subgraph record, pattern, and extraction reason

The Netron index places the canonical original ONNX file under `data/models/onnx/<model>/model.onnx` first as a full-graph comparison baseline; it is referenced in place rather than duplicated. The files under `artifacts/subgraph_onnx/` are visualization artifacts. They make local and multi-branch evidence easier to inspect in Netron, but they are not standalone semantically complete models and do not modify the source ONNX file.

## Static-Shape ONNX Export for Netron

`scripts/export_static_shape_onnx.py` creates independent fixed-shape visualization exports under `data/models/onnx_static/`. Text exports trace a wrapper that reconstructs keyword model inputs, preventing Hugging Face positional-signature differences from changing semantics; tokenizer fields unsupported by a model are recorded and dropped. Image exports likewise trace a named `pixel_values` wrapper.

The exporter retains static axes, runs ONNX shape inference and validation, and writes metadata beside each artifact. These files support Netron visualization with concrete shapes only. The dynamic export under `data/models/onnx/` remains the source for analysis passes.
