"""Markdown renderers for attention value-path evidence artifacts."""

from __future__ import annotations

from typing import Any

from model_analysis.attention_value_path_subgraph import attention_value_path_report_to_dict


def attention_value_path_report_to_markdown(value: dict[str, Any]) -> str:
    data = attention_value_path_report_to_dict(value)
    rows = []
    for path in data.get("paths", []):
        mapping = path.get("axis_mapping", {})
        rows.append(
            f"| {path.get('layer_index')} | {path.get('path_name')} | {path.get('analysis_status')} | "
            f"{mapping.get('mapping_status')} | {path.get('export_status')} | {path.get('artifact_paths', {}).get('onnx', '-')} |"
        )
    return "\n".join(
        [
            f"# Attention Value-Path Subgraphs: {data.get('model_name')}",
            "",
            "## Summary",
            "",
            f"- Total paths: `{data.get('total_paths', 0)}`",
            f"- Seedable: `{data.get('seedable', 0)}`",
            f"- Partial: `{data.get('partial', 0)}`",
            f"- Blocked: `{data.get('blocked', 0)}`",
            f"- Exported/skipped/failed: `{data.get('exported', 0)}` / `{data.get('skipped', 0)}` / `{data.get('failed', 0)}`",
            "",
            "## Paths",
            "",
            "| Layer | Path | Analysis status | Mapping | Export | ONNX |",
            "| --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "## Propagation Rule",
            "",
            "The value projection feeds attention context and then the output projection. When the layout mapping is proven, `out_proj` input deadness propagates backward through `V.value_dim -> Context.value_context_dim` to the value-projection output.",
            "",
            "This is static artifact/evidence generation only. It does not execute pruning or modify model weights.",
            "",
        ]
    )
