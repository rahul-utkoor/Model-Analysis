"""Compare layer subgraph validation packs."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _model_key(pack: dict[str, Any]) -> str:
    return f"{pack.get('model_name', 'model')}::layer_{pack.get('layer_index', 0)}"


def _matrix(packs: list[dict[str, Any]], counter_fn) -> dict[str, dict[str, int]]:
    models = [_model_key(pack) for pack in packs]
    observed = sorted({key for pack in packs for key in counter_fn(pack)})
    out = {key: {} for key in observed}
    for pack in packs:
        model = _model_key(pack)
        counts = counter_fn(pack)
        for key in observed:
            out[key][model] = counts.get(key, 0)
    for row in out.values():
        for model in models:
            row.setdefault(model, 0)
    return out


def _class_counts(pack: dict[str, Any]) -> Counter:
    return Counter(item.get("classification", {}).get("pruning_class", "unknown") for item in pack.get("subgraphs", []))


def _category_counts(pack: dict[str, Any]) -> Counter:
    return Counter(item.get("semantic_category", "unknown") for item in pack.get("subgraphs", []))


def compare_layer_subgraph_validation_packs(packs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_packs": len(packs),
        "packs": [_model_key(pack) for pack in packs],
        "pruning_class_matrix": _matrix(packs, _class_counts),
        "semantic_category_matrix": _matrix(packs, _category_counts),
        "summary": {
            "total_subgraphs": sum(len(pack.get("subgraphs", [])) for pack in packs),
            "total_onnx_exported": sum(pack.get("summary", {}).get("onnx_exported", 0) for pack in packs),
            "total_onnx_failed": sum(pack.get("summary", {}).get("onnx_failed", 0) for pack in packs),
            "total_valid_plan_subgraphs": sum(pack.get("summary", {}).get("valid_plan_subgraphs", 0) for pack in packs),
        },
    }


def comparison_to_markdown(comparison: dict[str, Any]) -> str:
    def table(matrix: dict[str, dict[str, int]]) -> str:
        if not matrix:
            return "_None._"
        packs = comparison.get("packs", [])
        lines = ["| item | " + " | ".join(packs) + " |", "|---|" + "|".join("---" for _ in packs) + "|"]
        for item, row in sorted(matrix.items()):
            lines.append("| " + item + " | " + " | ".join(str(row.get(pack, 0)) for pack in packs) + " |")
        return "\n".join(lines)

    summary = comparison.get("summary", {})
    return "\n".join(
        [
            "# Layer Subgraph Validation Comparison",
            "",
            f"- Packs: `{comparison.get('num_packs', 0)}`",
            f"- Total subgraphs: `{summary.get('total_subgraphs', 0)}`",
            f"- ONNX exported: `{summary.get('total_onnx_exported', 0)}`",
            f"- ONNX failed: `{summary.get('total_onnx_failed', 0)}`",
            f"- Valid plan subgraphs: `{summary.get('total_valid_plan_subgraphs', 0)}`",
            "",
            "## Pruning Classes",
            "",
            table(comparison.get("pruning_class_matrix", {})),
            "",
            "## Semantic Categories",
            "",
            table(comparison.get("semantic_category_matrix", {})),
            "",
            "This comparison summarizes static layer subgraph validation packs.",
            "",
        ]
    )
