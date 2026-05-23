# Glossary

## Structural inventory

A front-end report describing PyTorch modules, parameters, Linear layers, embeddings, normalization layers, attention-like modules, MLP-like modules, and initial pruning hints.

## ONNX graph summary

A graph-level report describing ONNX inputs, outputs, initializers, nodes, op type counts, tensor shapes where available, and pruning-relevant graph operations.

## Tensor Graph IR

A frontend-independent tensor-dataflow representation of operations and tensor values, including producer/consumer links, forks, joins, semantic roles, and region hints. ONNX is currently one importer into this IR.

## Structural Region Tree

A planned hierarchy that collapses recognized Tensor IR regions, such as sequential projections, forks, joins, and reconvergent motifs, into nested structural nodes for propagation analysis.

## Prunable unit

A module, graph node, or higher-level structure that might expose a prunable dimension, such as a Linear layer, embedding matrix, Conv node, MLP pair, or Q/K/V attention group.

## Dependency graph

A conservative graph of prunable units and dependency edges that records how pruning information may need to propagate.

## Propagation edge

A directed or bidirectional relationship between units, such as MLP hidden coupling, Q/K/V coupling, residual shape coupling, normalization dependency, or shape dependency.

## Pruning action

A dry-run request to prune a target unit along a dimension using concrete or generated indices. It produces a plan; it does not modify weights.

## Correspondence

Heuristic evidence linking PyTorch modules or parameters to ONNX nodes and initializers.

## Shape evidence

Static tensor and node shape information extracted from ONNX metadata without running inference.

## Pruning opportunity

A model-level candidate pruning region with required constraints, affected units, risk level, and executability label.

## Structural risk

A region where pruning may be blocked or dangerous because of residual paths, LayerNorm hidden dependencies, attention reshapes, tied parameters, unknown shapes, or unmapped ONNX nodes.

## Dimension variable

A compiler-style symbolic variable representing a model dimension such as `out_features`, `in_features`, `intermediate_dim`, `hidden_dim`, `num_heads`, or `embedding_dim`.

## Index variable

A symbolic set of indices selected for pruning along a dimension variable.

## Constraint equation

A symbolic rule over dimension or index variables, such as equality, same-index propagation, tied parameters, reshape preservation, or unknown mapping.

## Equivalence class

A group of dimensions connected by equality, same-index, or tied-parameter constraints that must be reasoned about together.

## Legality check

A static analysis query over the Dimension IR that classifies a pruning request as legal, legal with repairs, ambiguous, or rejected.

## Forward slice

The set of downstream dimensions and constraints reached from a requested pruning dimension.

## Backward slice

The set of upstream dimensions and constraints that can impose requirements on a requested pruning dimension.

## Minimal repair set

The smallest conservative set of structural updates implied by the legality analysis, such as applying the same indices to a coupled consumer dimension.

## Executable backend

An experimental lowering path that can modify local model artifacts. In this repository, executable backends are secondary to Tensor IR, pruning maps, and Dimension IR.
