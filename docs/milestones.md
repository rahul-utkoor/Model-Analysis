# Milestones

## Milestone 1: Repository Infrastructure

Status: complete.

Implemented:

- Git-backed Python project scaffold
- Model registry in `configs/models.yaml`
- Download scripts for Hugging Face models
- ONNX export script
- Lightweight inspection script
- Initial tests for paths and registry
- README setup and usage instructions

Primary commands:

```bash
python scripts/download_models.py --model all
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/inspect_model.py --model bert-base-uncased
```

## Milestone 2: Structural Inventory Generation

Status: complete.

Implemented:

- PyTorch structural inventory summaries
- ONNX graph summaries
- Conservative pruning hints
- Report writing helpers
- Structural inventory CLI
- Tests for inventory, ONNX analysis, and reporting

Primary commands:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model all --require-onnx
```

## Milestone 3: Pruning Dependency Graph Construction

Status: complete.

Implemented:

- Prunable unit and dependency edge data model
- Dependency graph construction from PyTorch structural summaries
- Optional ONNX graph augmentation
- Coupled group and independent-unit detection
- Ambiguous/manual-review item detection
- Dependency graph analyzer
- Dependency graph CLI
- Tests for dependency graph and analyzer behavior

Primary commands:

```bash
python scripts/build_dependency_graph.py --model bert-base-uncased
python scripts/build_dependency_graph.py --model all --require-onnx --verbose
```

## Milestone 4: Pruning Action Simulation and Dimension Propagation

Status: complete.

Implemented:

- Dry-run pruning action schema
- Propagation step and pruning plan schema
- Conservative propagation engine over dependency graphs
- Edge-specific semantics for QKV, MLP, residual, normalization, embedding, feeds, shape, and propagation-only edges
- Candidate action generation
- Manual and candidate-action CLI scripts
- Markdown and JSON pruning plan reports
- Tests for action serialization, propagation, candidate generation, and reporting

Primary commands:

```bash
python scripts/generate_candidate_actions.py --model bert-base-uncased --simulate --limit 5
python scripts/simulate_pruning_action.py --model bert-base-uncased --target-unit <unit_id> --dim out_features --indices 0,1,2,3 --verbose
```

## Milestone 5: PyTorch-to-ONNX Correspondence and Shape Evidence

Status: complete.

Implemented:

- Parameter-to-initializer evidence
- Module-to-node correspondence reports
- Static ONNX tensor and node shape evidence
- Dependency graph validation with correspondence/shape support
- Evidence-enriched pruning action simulation
- Correspondence CLI and docs

Primary commands:

```bash
python scripts/build_correspondence.py --model bert-base-uncased --require-dependency-graph --verbose
python scripts/simulate_pruning_action.py --model bert-base-uncased --target-unit <unit_id> --dim out_features --indices 0,1,2,3 --use-evidence --verbose
```

## Milestone 6: Proposed Next Step

Status: complete.

Implemented:

- Reversible PyTorch `nn.Linear` out_features pruning
- Reversible PyTorch `nn.Linear` in_features pruning
- Linear-only pruning plan executor
- Dry-run and only-target execution modes
- Pruning execution reports
- Structural pruning diffs
- Rollback manifests
- Tests for Linear surgery, execution, diffs, and rollback

Primary commands:

```bash
python scripts/execute_pruning_plan.py --model bert-base-uncased --target-unit torch:linear:bert.encoder.layer.0.attention.self.query --dim out_features --indices 0,1,2,3 --only-target --dry-run --verbose
python scripts/execute_pruning_plan.py --model bert-base-uncased --target-unit torch:linear:bert.encoder.layer.0.attention.self.query --dim out_features --indices 0,1,2,3 --only-target --allow-ambiguous --verbose
```

## Milestone 7: Paired Linear Structural Repair and Forward Smoke Validation

Status: complete.

Implemented:

- Paired Linear repair plan data model
- MLP expansion/projection repair detection from explicit `mlp_hidden_coupling`
- Atomic source `out_features` plus target `in_features` Linear repair
- Repair plan and repair transaction reports
- Forward smoke validation module and standalone CLI
- Optional before/after smoke validation during pruning execution
- Tests for repair detection, paired pruning atomicity, and smoke validation

Primary commands:

```bash
python scripts/execute_pruning_plan.py --model bert-base-uncased --target-unit torch:linear:bert.encoder.layer.0.intermediate.dense --dim out_features --indices 0,1,2,3 --repair-pairs --write-repair-plan-only --allow-ambiguous --verbose
python scripts/execute_pruning_plan.py --model bert-base-uncased --target-unit torch:linear:bert.encoder.layer.0.intermediate.dense --dim out_features --indices 0,1,2,3 --repair-pairs --dry-run --allow-ambiguous --verbose
python scripts/run_forward_smoke_test.py --model bert-base-uncased --device cpu --verbose
```

## Milestone 8: BERT MLP Block-Level Executable Pruning

Status: complete.

Implemented:

- BERT-style MLP block target detection
- Architecture-specific prune specs for intermediate-dimension pruning
- Executable `intermediate.dense` `out_features` plus `output.dense` `in_features` pruning
- Block-level reports, diffs, validation reports, and rollback manifests
- Target-list CLI and BERT MLP pruning CLI
- Tiny BERT-like tests for detection, dry-run, execution, forward pass, and rejection cases

Primary commands:

```bash
python scripts/list_bert_mlp_targets.py --model bert-base-uncased
python scripts/prune_bert_mlp_block.py --model bert-base-uncased --layer 0 --indices 0,1,2,3 --dry-run --smoke-test-before --verbose
python scripts/prune_bert_mlp_block.py --model bert-base-uncased --layer 0 --indices 0,1,2,3 --smoke-test-before --smoke-test-after --verbose
```

## Milestone 9: Compiler-Style Pruning Opportunity Analysis

Status: complete.

Implemented:

- Pruning opportunity IR for dimensions, propagation constraints, opportunities, and model pruning maps
- Dimension extraction from dependency graphs
- Constraint extraction from dependency edges and optional validation evidence
- Opportunity inference for local Linear outputs, MLP intermediate dimensions, attention QKV structures, embeddings, ONNX MatMul/Gemm candidates, and blocked residual hidden-size regions
- Structural risk maps
- Per-model pruning map CLI
- Cross-model pruning map comparison CLI
- Tests for opportunity extraction, constraint typing, risk detection, and comparison matrices

Primary commands:

```bash
python scripts/build_pruning_map.py --model bert-base-uncased --verbose
python scripts/build_pruning_map.py --model all --verbose
python scripts/compare_pruning_maps.py --models all
```

## Milestone 10: Dimension Variable IR and Symbolic Propagation Constraints

Status: complete.

Implemented:

- Symbolic Dimension IR data model
- Dimension variables from model pruning maps
- Pruning-index variables for symbolic pruning selections
- Constraint equations for MLP, QKV, residual, LayerNorm, tied-parameter, reshape, and unknown mappings
- Union-find equivalence classes for equality, same-index, and tied dimensions
- Blocked-dimension and unresolved-constraint tracking
- MLIR-like `.pir` textual dump
- Dimension IR build and comparison CLIs
- Tests for Dimension IR construction, text dumps, and cross-model comparison

Primary commands:

```bash
python scripts/build_dimension_ir.py --model bert-base-uncased --verbose
python scripts/build_dimension_ir.py --model all --verbose
python scripts/compare_dimension_irs.py --models all
```

## Milestone 11: Proposed Next Step

Status: complete.

Implemented:

- Symbolic pruning request model
- Constraint adjacency and slicing helpers over Dimension IR
- Static legality checker for symbolic/concrete pruning requests
- Constraint satisfaction classification
- Forward and backward propagation slices
- Minimal structural repair-set computation
- Blocked-region explanations
- Dimension listing, legality check, and blocked-region CLIs
- Tests for graph traversal, legality statuses, repair sets, invalid requests, and blocked-region mitigation

Primary commands:

```bash
python scripts/list_pruning_dimensions.py --model bert-base-uncased --contains intermediate.dense
python scripts/check_pruning_legality.py --model bert-base-uncased --dimension-var <dimension_var_id> --count 4 --verbose
python scripts/explain_blocked_regions.py --model bert-base-uncased
```

## Milestone 12: Demo Track and Research Walkthrough

Status: complete.

Implemented:

- Guided demo documentation under `demos/`
- Milestone-by-milestone walkthroughs for Milestones 1 through 11
- Full research pipeline walkthrough
- Research glossary for pruning/compiler terms
- Demo shell scripts for the main analysis path
- Formal demo-track documentation under `docs/demo_track.md`
- Lightweight tests that verify demo documentation entry points

Primary commands:

```bash
bash demo_scripts/run_demo_01_setup_check.sh
PYTHON=python MODEL=bert-base-uncased bash demo_scripts/run_full_analysis_pipeline.sh
```

The demo track makes pruning maps and Dimension IR the main learning path. Executable pruning modules are documented as optional experimental backends.

## Milestone 13: k-Node and Join-Aware ONNX Subgraph Structural Analysis

Status: complete.

Implemented:

- Directed ONNX path enumeration for lengths one through five
- Join-centered subgraph extraction for `Add`, `Sum`, and `Concat` branch merges
- Bias-add versus residual-add classification
- Residual-like join and post-join normalization evidence
- Local pattern classification for projection, MLP, attention, shape-transform, join, and residual structures
- Report-level pruning/dimension evidence
- Cross-model subgraph comparison
- Guided demo documentation and synthetic tests

Primary commands:

```bash
python scripts/analyze_subgraphs.py --model bert-base-uncased --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose
python scripts/analyze_subgraphs.py --model all --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose
python scripts/compare_subgraphs.py --models all
```

This pass performs structural analysis over saved ONNX summaries only. Residual and join evidence is intended to refine future pruning maps and Dimension IR constraints.

## Milestone 14: DAG Motif and Multi-Join Region Subgraph Analysis

Status: complete.

Implemented:

- Fork-region detection for producer fanout
- Bounded reconvergence search and diamond-region extraction
- Join-fork-join detection for multi-join motifs such as `A,B -> C -> D,E -> F`
- Canonical op-type patterns for multi-branch regions
- Suggested fanout, branch compatibility, residual, and reshape constraints
- DAG region report and cross-model comparison CLIs
- Demo documentation and synthetic exact-motif tests

Primary commands:

```bash
python scripts/analyze_dag_regions.py --model bert-base-uncased --max-branch-depth 4 --verbose
python scripts/analyze_dag_regions.py --model all --max-branch-depth 4 --verbose
python scripts/compare_dag_regions.py --models all
```

This pass captures fork, join, diamond, and join-fork-join regions but does not modify models.

## Milestone 15: Proposed Next Step

Recommended focus:

- Improve precision of the Dimension IR
- Better extraction of tensor axis semantics
- Map ONNX reshape/transpose axes into symbolic dimension equations
- Distinguish batch, sequence, hidden, intermediate, and head axes
- Richer textual IR diagnostics
