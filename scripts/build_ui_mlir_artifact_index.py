#!/usr/bin/env python3
"""Index existing ONNX, graph, MLIR, and dependence artifacts for the read-only UI."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOTS = [
    ROOT / "artifacts" / "model_analysis_subgraphs",
    ROOT / "artifacts" / "attention_value_path_subgraphs",
    ROOT / "artifacts" / "opt_ffn_native_subgraphs",
]
MLIR_ROOTS = [
    ROOT / "reports" / "mlir_axis_bridge",
    ROOT / "reports" / "mlir_evidence_coverage",
    ROOT / "reports" / "mlir_evidence_coverage_bert_24_plan",
    ROOT / "reports" / "mlir_evidence_coverage_opt_ffn_native_diagnosis",
    ROOT / "reports" / "all_model_plan_proof",
    ROOT / "reports" / "opt_ffn_native_diagnosis",
]


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def artifact_entry(root: Path, onnx_path: Path) -> dict[str, Any]:
    layer_dir = onnx_path.parents[1]
    layer = int(layer_dir.name.removeprefix("layer_"))
    subgraph_dir = onnx_path.parent
    paths = {
        suffix.removeprefix("."): relative(path)
        for suffix in [".onnx", ".svg", ".dot"]
        if (path := subgraph_dir / f"subgraph{suffix}").exists()
    }
    return {
        "artifact_root": relative(root),
        "model": onnx_path.parents[3].name,
        "layer": layer,
        "subgraph": subgraph_dir.name,
        "paths": paths,
    }


def build_index() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for root in ARTIFACT_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/layers/layer_*/*/subgraph.onnx")):
            entries.append(artifact_entry(root, path))
    mlir_artifacts = [
        relative(path)
        for root in MLIR_ROOTS
        if root.exists()
        for path in sorted(root.rglob("*.mlir"))
    ]
    dependence_json = [
        relative(path)
        for root in MLIR_ROOTS
        if root.exists()
        for path in sorted(root.rglob("*dependence*.json"))
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "mlir_artifacts": mlir_artifacts,
        "dependence_json": dependence_json,
        "summary": {
            "artifact_bundles": len(entries),
            "mlir_artifacts": len(mlir_artifacts),
            "dependence_json": len(dependence_json),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index existing artifacts for the read-only pruning analysis UI.")
    parser.add_argument("--output", default="reports/ui_artifact_index/index.json")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = ROOT / args.output
    index = build_index()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    if args.verbose:
        print(f"wrote {output}")
        print(json.dumps(index["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
