"""Text rendering for layer subgraph validation packs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_analysis.layer_subgraph_validation_pack import LayerSubgraphValidationPack, layer_subgraph_pack_to_dict
from model_analysis.paths import ensure_dir


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def layer_subgraph_pack_to_text(pack: LayerSubgraphValidationPack | dict[str, Any]) -> str:
    data = layer_subgraph_pack_to_dict(pack) if isinstance(pack, LayerSubgraphValidationPack) else pack
    lines = [f'layer_subgraph_validation @{_escape(data.get("model_name", "model"))} layer={data.get("layer_index", 0)} {{']
    for item in data.get("subgraphs", []):
        cls = item.get("classification", {})
        lines.append(f'  subgraph "{_escape(item.get("display_name", ""))}" {{')
        lines.append(f'    ordinal = {item.get("ordinal")}')
        lines.append(f'    semantic_category = {item.get("semantic_category", "unknown")}')
        lines.append(f'    pruning_class = {cls.get("pruning_class", "unknown")}')
        lines.append(f'    plan_status = {cls.get("plan_status", "unknown")}')
        lines.append(f'    validation_status = {cls.get("validation_status", "unknown")}')
        lines.append(f'    onnx_status = {item.get("onnx_export", {}).get("status", "skipped")}')
        lines.append("    primitive_ops {")
        for op in item.get("primitive_ops", []):
            lines.append(f'      {op.get("topological_index")} "{_escape(op.get("source_name", ""))}"')
        lines.append("    }")
        lines.append("  }")
        lines.append("")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_layer_subgraph_pack_text(pack: LayerSubgraphValidationPack | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(layer_subgraph_pack_to_text(pack), encoding="utf-8")
