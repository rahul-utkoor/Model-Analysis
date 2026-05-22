"""Structural diffs for pruning execution reports."""

from __future__ import annotations

from typing import Any


def _linear_by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {layer["name"]: layer for layer in summary.get("linear_layers", [])}


def _layer_shape(layer: dict[str, Any]) -> list[int | None]:
    return [layer.get("out_features"), layer.get("in_features")]


def _layer_params(layer: dict[str, Any]) -> int:
    return int(layer.get("parameters") or 0)


def compute_structural_diff(before_summary: dict, after_summary: dict) -> dict[str, Any]:
    before_params = int(before_summary.get("parameter_summary", {}).get("total_parameters", 0))
    after_params = int(after_summary.get("parameter_summary", {}).get("total_parameters", 0))
    before_linear = _linear_by_name(before_summary)
    after_linear = _linear_by_name(after_summary)
    changed = []
    warnings = []

    for name, before_layer in before_linear.items():
        after_layer = after_linear.get(name)
        if not after_layer:
            warnings.append(f"Linear layer '{name}' is missing after pruning.")
            continue
        old_shape = _layer_shape(before_layer)
        new_shape = _layer_shape(after_layer)
        if old_shape != new_shape:
            old_parameters = _layer_params(before_layer)
            new_parameters = _layer_params(after_layer)
            changed.append(
                {
                    "module_name": name,
                    "old_shape": old_shape,
                    "new_shape": new_shape,
                    "old_parameters": old_parameters,
                    "new_parameters": new_parameters,
                    "parameter_delta": new_parameters - old_parameters,
                }
            )

    if after_params > before_params:
        warnings.append("Total parameter count increased after pruning.")

    delta = after_params - before_params
    return {
        "total_parameters_before": before_params,
        "total_parameters_after": after_params,
        "parameter_delta": delta,
        "parameter_delta_percent": (delta / before_params * 100.0) if before_params else 0.0,
        "linear_layer_count_before": len(before_linear),
        "linear_layer_count_after": len(after_linear),
        "changed_linear_layers": changed,
        "warnings": warnings,
    }


def pruning_diff_to_markdown(diff: dict[str, Any]) -> str:
    rows = diff.get("changed_linear_layers", [])
    if rows:
        table = "\n".join(
            [
                "| module_name | old_shape | new_shape | old_parameters | new_parameters | parameter_delta |",
                "| --- | --- | --- | --- | --- | --- |",
                *[
                    f"| {row['module_name']} | {row['old_shape']} | {row['new_shape']} | {row['old_parameters']} | {row['new_parameters']} | {row['parameter_delta']} |"
                    for row in rows
                ],
            ]
        )
    else:
        table = "_No changed Linear layers detected._"

    warnings = "\n".join(f"- {warning}" for warning in diff.get("warnings", [])) or "_None._"
    return "\n".join(
        [
            "# Pruning Structural Diff",
            "",
            f"- Total parameters before: `{diff.get('total_parameters_before')}`",
            f"- Total parameters after: `{diff.get('total_parameters_after')}`",
            f"- Parameter delta: `{diff.get('parameter_delta')}`",
            f"- Parameter delta percent: `{diff.get('parameter_delta_percent'):.6f}`",
            f"- Linear layers before: `{diff.get('linear_layer_count_before')}`",
            f"- Linear layers after: `{diff.get('linear_layer_count_after')}`",
            "",
            "## Changed Linear Layers",
            "",
            table,
            "",
            "## Warnings",
            "",
            warnings,
            "",
        ]
    )
