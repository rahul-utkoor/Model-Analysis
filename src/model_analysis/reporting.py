"""Report writing and Markdown rendering helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


def write_json(data: dict, path: Path) -> None:
    """Write a dictionary as formatted JSON."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_markdown(text: str, path: Path) -> None:
    """Write Markdown text."""
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_csv_rows(rows: list[dict], path: Path) -> None:
    """Write rows to CSV using the union of row keys as headers."""
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if not rows:
        return "_None detected._"

    selected_rows = rows[:limit] if limit else rows
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in selected_rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")

    if limit and len(rows) > limit:
        omitted = {"name": "...", columns[0]: "..."}
        if len(columns) > 1:
            omitted[columns[1]] = f"{len(rows) - limit} more rows omitted"
        body.append("| " + " | ".join(str(omitted.get(column, "")) for column in columns) + " |")
    return "\n".join([header, separator, *body])


def _counts_table(counts: dict[str, int], limit: int = 40) -> str:
    rows = [
        {"type": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]
    return _markdown_table(rows, ["type", "count"])


def structural_inventory_to_markdown(summary: dict) -> str:
    """Render a PyTorch structural inventory summary as Markdown."""
    params = summary["parameter_summary"]
    modules = summary["module_summary"]
    lines = [
        f"# Structural Inventory: {summary['model_name']}",
        "",
        "## Metadata",
        "",
        f"- Hugging Face ID: `{summary.get('hf_id')}`",
        f"- Task: `{summary.get('task')}`",
        "",
        "## Parameter Summary",
        "",
        f"- Total parameters: `{params['total_parameters']:,}`",
        f"- Trainable parameters: `{params['trainable_parameters']:,}`",
        f"- Non-trainable parameters: `{params['non_trainable_parameters']:,}`",
        "",
        "## Module Summary",
        "",
        f"- Total modules: `{modules['total_modules']:,}`",
        "",
        "### Module Type Counts",
        "",
        _counts_table(modules["module_type_counts"]),
        "",
        "### Parameter Distribution By Module Type",
        "",
        _counts_table(modules.get("parameter_distribution_by_module_type", {})),
        "",
        "## Linear Layers",
        "",
        _markdown_table(summary["linear_layers"], ["name", "in_features", "out_features", "bias", "parameters"], limit=200),
        "",
        "## Embedding Layers",
        "",
        _markdown_table(summary["embedding_layers"], ["name", "num_embeddings", "embedding_dim", "parameters"], limit=100),
        "",
        "## Normalization Layers",
        "",
        _markdown_table(summary["normalization_layers"], ["name", "type", "parameters"], limit=150),
        "",
        "## Attention-Like Modules",
        "",
        _markdown_table(summary["attention_like_modules"], ["name", "type", "reason"], limit=200),
        "",
        "## MLP-Like Modules",
        "",
        _markdown_table(summary["mlp_like_modules"], ["name", "type", "reason"], limit=200),
        "",
        "## Pruning-Relevant Groups",
        "",
        _markdown_table(summary["pruning_relevant_groups"], ["group_name", "group_type", "members", "confidence", "reason"], limit=200),
        "",
    ]
    return "\n".join(lines)


def onnx_summary_to_markdown(summary: dict) -> str:
    """Render an ONNX graph summary as Markdown."""
    graph = summary["graph_summary"]
    lines = [
        f"# ONNX Graph Summary: {summary['model_name']}",
        "",
        "## Metadata",
        "",
        f"- Hugging Face ID: `{summary.get('hf_id')}`",
        f"- Task: `{summary.get('task')}`",
        f"- ONNX path: `{summary.get('onnx_path')}`",
        "",
        "## Graph Summary",
        "",
        f"- Nodes: `{graph['num_nodes']:,}`",
        f"- Initializers: `{graph['num_initializers']:,}`",
        f"- Inputs: `{graph['num_inputs']:,}`",
        f"- Outputs: `{graph['num_outputs']:,}`",
        "",
        _counts_table(graph["op_type_counts"]),
        "",
        "## Inputs",
        "",
        _markdown_table(summary["inputs"], ["name", "shape", "data_type"]),
        "",
        "## Outputs",
        "",
        _markdown_table(summary["outputs"], ["name", "shape", "data_type"]),
        "",
        "## Initializers",
        "",
        _markdown_table(summary["initializers"], ["name", "dims", "data_type"], limit=200),
        "",
        "## Nodes",
        "",
        _markdown_table(summary["nodes"], ["name", "op_type", "inputs", "outputs"], limit=300),
        "",
        "## Pruning-Relevant Nodes",
        "",
        _markdown_table(summary["pruning_relevant_nodes"], ["name", "op_type", "confidence", "reason"], limit=200),
        "",
    ]
    return "\n".join(lines)


def _group_members_by_type(groups: list[dict], group_type: str) -> list[str]:
    members: list[str] = []
    for group in groups:
        if group.get("group_type") == group_type:
            members.extend(group.get("members", []))
    return members


def pruning_hints_to_markdown(torch_summary: dict, onnx_summary: dict | None) -> str:
    """Render conservative pruning hints from static PyTorch and optional ONNX evidence."""
    model_name = torch_summary["model_name"]
    groups = torch_summary.get("pruning_relevant_groups", [])
    linear_count = len(torch_summary.get("linear_layers", []))
    embedding_count = len(torch_summary.get("embedding_layers", []))
    attention_members = _group_members_by_type(groups, "attention_qkv")
    mlp_members = _group_members_by_type(groups, "mlp_projection_pair")
    output_projection_members = _group_members_by_type(groups, "attention_output_projection")

    onnx_relevant = []
    if onnx_summary:
        onnx_relevant = onnx_summary.get("pruning_relevant_nodes", [])

    direct_onnx = [node for node in onnx_relevant if node.get("op_type") in {"MatMul", "Gemm", "Conv"}]
    propagation_onnx = [node for node in onnx_relevant if node.get("op_type") not in {"MatMul", "Gemm", "Conv"}]

    lines = [
        f"# Pruning Hints: {model_name}",
        "",
        "## What appears structurally prunable",
        "",
        f"- Linear layers: `{linear_count}` detected. These are candidate projection matrices, subject to dependency checks.",
        f"- Attention projections: `{len(attention_members)}` Q/K/V members and `{len(output_projection_members)}` output projections detected by naming heuristics.",
        f"- MLP projections: `{len(mlp_members)}` members detected by MLP/FFN naming heuristics.",
        f"- Embedding matrices: `{embedding_count}` detected. These require vocabulary and possible output-tying checks before pruning.",
        f"- Vision patch projection, if applicable: ONNX Conv candidates detected: `{sum(1 for node in direct_onnx if node.get('op_type') == 'Conv')}`.",
        "",
        "## What requires dependency propagation",
        "",
        "- Residual connections must preserve compatible hidden dimensions across branches.",
        "- LayerNorm parameters usually need to follow the hidden dimension they normalize.",
        "- Attention head reshape and transpose operations can couple projection width, head count, and head dimension.",
        "- Q/K/V consistency is required when pruning attention heads or shared hidden dimensions.",
        "- MLP hidden dimension consistency is required between expansion and projection layers.",
        "- Embedding-to-output tying, if present, requires shared pruning decisions across tied matrices.",
        "",
        "## Forward propagation considerations",
        "",
        "Downstream tensors that consume pruned projections may need updated hidden sizes, attention head dimensions, reshape constants, concatenation widths, and residual branch dimensions. ONNX propagation-relevant nodes can help locate reshape, transpose, normalization, and residual paths that must remain shape-compatible.",
        "",
        "## Backward propagation considerations",
        "",
        "Upstream tensors that feed pruned layers may need corresponding constraints so input channels, token embeddings, previous block outputs, and tied parameters remain aligned with the pruned structure.",
        "",
        "## ONNX Evidence",
        "",
    ]

    if onnx_summary:
        lines.extend(
            [
                f"- Direct high-interest ONNX nodes: `{len(direct_onnx)}`.",
                f"- Propagation-relevant ONNX nodes: `{len(propagation_onnx)}`.",
            ]
        )
    else:
        lines.append("- ONNX graph summary was not available for this model.")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "This report is static structural evidence, not a correctness proof. It identifies likely pruning surfaces and dependency paths, but any pruning plan still needs graph-aware validation and numerical checks.",
            "",
        ]
    )
    return "\n".join(lines)
