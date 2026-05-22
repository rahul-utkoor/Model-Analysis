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

Recommended focus:

- Executable but reversible pruning transforms for a small subset first
- Linear `out_features` and `in_features` pruning in PyTorch
- Before/after structural checks
- Rollback-safe artifacts
- No broad ONNX graph rewrite until PyTorch behavior is validated

Milestone 6 should introduce executable pruning only for a very small, well-understood layer type first.
