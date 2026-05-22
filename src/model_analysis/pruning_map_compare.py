"""Compare compiler-style pruning maps across models."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _matrix(maps: list[dict[str, Any]], summary_key: str) -> dict[str, dict[str, int]]:
    return {
        item.get("model_name", f"model_{index}"): dict(item.get("summary", {}).get(summary_key, {}))
        for index, item in enumerate(maps)
    }


def _types_for_map(model_map: dict[str, Any]) -> set[str]:
    return {item.get("opportunity_type") for item in model_map.get("opportunities", []) if item.get("opportunity_type")}


def _risks_for_map(model_map: dict[str, Any]) -> set[str]:
    return {item.get("risk_type") for item in model_map.get("structural_risks", []) if item.get("risk_type")}


def compare_model_pruning_maps(maps: list[dict]) -> dict:
    """Build aggregate matrices and common/model-specific opportunity sets."""
    model_names = [item.get("model_name", f"model_{index}") for index, item in enumerate(maps)]
    opportunity_sets = {name: _types_for_map(item) for name, item in zip(model_names, maps)}
    risk_sets = {name: _risks_for_map(item) for name, item in zip(model_names, maps)}
    all_opportunities = set().union(*opportunity_sets.values()) if opportunity_sets else set()
    common_opportunities = set.intersection(*opportunity_sets.values()) if opportunity_sets else set()
    common_risks = set.intersection(*risk_sets.values()) if risk_sets else set()
    model_specific = {
        name: sorted(types - set().union(*(other for other_name, other in opportunity_sets.items() if other_name != name)))
        for name, types in opportunity_sets.items()
    }
    total_opportunity_counts = Counter()
    for model_map in maps:
        total_opportunity_counts.update(item.get("opportunity_type") for item in model_map.get("opportunities", []) if item.get("opportunity_type"))

    return {
        "num_models": len(maps),
        "models": model_names,
        "opportunity_type_matrix": _matrix(maps, "opportunity_type_counts"),
        "risk_level_matrix": _matrix(maps, "risk_level_counts"),
        "executability_matrix": _matrix(maps, "executability_counts"),
        "common_opportunity_types": sorted(common_opportunities),
        "model_specific_opportunity_types": model_specific,
        "common_risks": sorted(common_risks),
        "summary": {
            "all_opportunity_types": sorted(all_opportunities),
            "total_opportunity_type_counts": dict(total_opportunity_counts),
        },
    }


def _matrix_to_markdown(matrix: dict[str, dict[str, int]]) -> str:
    columns = sorted({key for row in matrix.values() for key in row})
    if not columns:
        return "_None._"
    lines = [
        "| model | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---" for _ in columns) + " |",
    ]
    for model, row in sorted(matrix.items()):
        lines.append("| " + model + " | " + " | ".join(str(row.get(column, 0)) for column in columns) + " |")
    return "\n".join(lines)


def pruning_map_comparison_to_markdown(comparison: dict[str, Any]) -> str:
    specific_lines = []
    for model, values in comparison.get("model_specific_opportunity_types", {}).items():
        specific_lines.append(f"- `{model}`: {', '.join(values) if values else 'None'}")
    return "\n".join(
        [
            "# Pruning Map Comparison",
            "",
            "## Summary",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Model names: `{comparison.get('models', [])}`",
            "",
            "## Opportunity Type Matrix",
            "",
            _matrix_to_markdown(comparison.get("opportunity_type_matrix", {})),
            "",
            "## Risk Level Matrix",
            "",
            _matrix_to_markdown(comparison.get("risk_level_matrix", {})),
            "",
            "## Executability Matrix",
            "",
            _matrix_to_markdown(comparison.get("executability_matrix", {})),
            "",
            "## Common Opportunities",
            "",
            "\n".join(f"- `{item}`" for item in comparison.get("common_opportunity_types", [])) or "_None._",
            "",
            "## Model-Specific Opportunities",
            "",
            "\n".join(specific_lines) or "_None._",
            "",
            "## Common Risks",
            "",
            "\n".join(f"- `{item}`" for item in comparison.get("common_risks", [])) or "_None._",
            "",
            "## Interpretation",
            "",
            "This comparison is static analysis over pruning maps. It compares structural opportunity classes and risks across models, not pruning quality or model accuracy.",
            "",
        ]
    )
