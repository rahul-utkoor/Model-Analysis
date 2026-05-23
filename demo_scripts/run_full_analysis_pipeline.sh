#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

echo "=== Full Model-Analysis Research Pipeline ==="
echo "Model: ${MODEL}"
echo
echo "This pipeline may download a Hugging Face model and export ONNX."
echo "It does not execute pruning or modify model weights."
echo

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

run_cmd "${PYTHON_BIN}" scripts/download_models.py --model "${MODEL}"
run_cmd "${PYTHON_BIN}" scripts/export_to_onnx.py --model "${MODEL}"
run_cmd "${PYTHON_BIN}" scripts/generate_structural_inventory.py --model "${MODEL}" --require-onnx
run_cmd "${PYTHON_BIN}" scripts/build_tensor_ir.py --model "${MODEL}" --verbose
run_cmd "${PYTHON_BIN}" scripts/build_dependency_graph.py --model "${MODEL}" --require-onnx --verbose
run_cmd "${PYTHON_BIN}" scripts/build_correspondence.py --model "${MODEL}" --require-dependency-graph --verbose
run_cmd "${PYTHON_BIN}" scripts/analyze_subgraphs.py --model "${MODEL}" --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose
run_cmd "${PYTHON_BIN}" scripts/analyze_dag_regions.py --model "${MODEL}" --max-branch-depth 4 --verbose
run_cmd "${PYTHON_BIN}" scripts/export_demo_subgraphs.py --model "${MODEL}" --max-per-category 3 --verbose
run_cmd "${PYTHON_BIN}" scripts/build_pruning_map.py --model "${MODEL}" --verbose
run_cmd "${PYTHON_BIN}" scripts/build_dimension_ir.py --model "${MODEL}" --verbose
run_cmd "${PYTHON_BIN}" scripts/list_pruning_dimensions.py --model "${MODEL}" --contains intermediate.dense --limit 10
run_cmd "${PYTHON_BIN}" scripts/explain_blocked_regions.py --model "${MODEL}"

echo
echo "Main artifacts:"
echo "  reports/structural_inventory/${MODEL}.md"
echo "  reports/tensor_ir/${MODEL}.md"
echo "  reports/tensor_ir_dumps/${MODEL}.tir"
echo "  reports/dependency_graphs/${MODEL}.md"
echo "  reports/correspondence/${MODEL}.md"
echo "  reports/join_subgraphs/${MODEL}.md"
echo "  reports/subgraph_pruning_analysis/${MODEL}.md"
echo "  reports/dag_regions/${MODEL}.md"
echo "  reports/dag_region_pruning_evidence/${MODEL}.md"
echo "  reports/netron_subgraph_index/${MODEL}__demo.md"
echo "  reports/model_pruning_maps/${MODEL}.md"
echo "  reports/dimension_ir/${MODEL}.md"
echo "  reports/pruning_ir_dumps/${MODEL}.pir"
echo "  reports/ir_analysis/${MODEL}__dimension_list.md"
echo
echo "Next step:"
echo "  Pick a dimension var_id from the dimension list and run:"
echo "  ${PYTHON_BIN} scripts/check_pruning_legality.py --model ${MODEL} --dimension-var <dimension_var_id> --count 4 --verbose"
