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

Recommended focus:

- Propagation analysis over the Dimension IR
- Constraint solving
- Legality checks for symbolic pruning actions
- Blocked-region explanation
- Forward/backward slice extraction
- Minimal repair set computation
