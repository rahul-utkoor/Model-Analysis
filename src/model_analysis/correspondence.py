"""PyTorch-to-ONNX correspondence evidence."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class ParameterEvidence:
    torch_param_name: str
    torch_module_name: str | None
    torch_shape: list[int] | None
    onnx_initializer_name: str | None
    onnx_shape: list[int] | None
    match_type: str
    confidence: str
    reason: str


@dataclass
class ModuleNodeCorrespondence:
    torch_module_name: str
    torch_module_type: str | None
    torch_unit_id: str | None
    onnx_node_names: list[str] = field(default_factory=list)
    onnx_op_types: list[str] = field(default_factory=list)
    onnx_initializer_names: list[str] = field(default_factory=list)
    input_tensors: list[str] = field(default_factory=list)
    output_tensors: list[str] = field(default_factory=list)
    input_shapes: dict[str, list[int] | str] = field(default_factory=dict)
    output_shapes: dict[str, list[int] | str] = field(default_factory=dict)
    parameter_evidence: list[ParameterEvidence] = field(default_factory=list)
    confidence: str = "low"
    reason: str = ""


@dataclass
class CorrespondenceReport:
    model_name: str
    hf_id: str
    task: str
    module_node_correspondences: list[ModuleNodeCorrespondence] = field(default_factory=list)
    parameter_evidence: list[ParameterEvidence] = field(default_factory=list)
    unmatched_torch_modules: list[dict[str, Any]] = field(default_factory=list)
    unmatched_onnx_nodes: list[dict[str, Any]] = field(default_factory=list)
    unmatched_onnx_initializers: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_name(name: str) -> str:
    """Normalize names for deterministic PyTorch/ONNX comparison."""
    normalized = name.lower().replace("onnx::", "")
    normalized = re.sub(r"[/\.\:\-\s]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = re.sub(r"^(model_|module_|base_model_)+", "", normalized)
    return normalized.strip("_")


def name_similarity_score(torch_name: str, onnx_name: str) -> float:
    torch_norm = normalize_name(torch_name)
    onnx_norm = normalize_name(onnx_name)
    if not torch_norm or not onnx_norm:
        return 0.0
    if torch_norm == onnx_norm:
        return 1.0
    if torch_norm.endswith(onnx_norm) or onnx_norm.endswith(torch_norm):
        return 0.85
    if torch_norm in onnx_norm or onnx_norm in torch_norm:
        return 0.70
    torch_tokens = set(torch_norm.split("_"))
    onnx_tokens = set(onnx_norm.split("_"))
    if not torch_tokens or not onnx_tokens:
        return 0.0
    overlap = len(torch_tokens & onnx_tokens)
    union = len(torch_tokens | onnx_tokens)
    return round(overlap / union, 3) if union else 0.0


def _module_parameters(torch_summary: dict[str, Any]) -> list[dict[str, Any]]:
    params = []
    for section, module_type in (
        ("linear_layers", "Linear"),
        ("embedding_layers", "Embedding"),
        ("normalization_layers", "Normalization"),
    ):
        for layer in torch_summary.get(section, []):
            module_name = layer.get("name", "")
            weight_name = layer.get("weight_name") or (f"{module_name}.weight" if module_name else None)
            if weight_name:
                params.append(
                    {
                        "param_name": weight_name,
                        "module_name": module_name,
                        "module_type": module_type,
                        "shape": layer.get("weight_shape") or _fallback_weight_shape(layer, module_type),
                    }
                )
            bias_name = layer.get("bias_name")
            if bias_name:
                params.append(
                    {
                        "param_name": bias_name,
                        "module_name": module_name,
                        "module_type": module_type,
                        "shape": layer.get("bias_shape") or _fallback_bias_shape(layer, module_type),
                    }
                )
    return params


def _fallback_weight_shape(layer: dict[str, Any], module_type: str) -> list[int] | None:
    if module_type == "Linear" and layer.get("out_features") is not None and layer.get("in_features") is not None:
        return [layer["out_features"], layer["in_features"]]
    if module_type == "Embedding" and layer.get("num_embeddings") is not None and layer.get("embedding_dim") is not None:
        return [layer["num_embeddings"], layer["embedding_dim"]]
    return None


def _fallback_bias_shape(layer: dict[str, Any], module_type: str) -> list[int] | None:
    if module_type == "Linear" and layer.get("out_features") is not None:
        return [layer["out_features"]]
    return None


def _initializer_entries(onnx_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"name": item.get("name", ""), "shape": item.get("dims")}
        for item in onnx_summary.get("initializers", [])
    ]


def _match_param(param: dict[str, Any], initializers: list[dict[str, Any]], used: set[str]) -> ParameterEvidence:
    param_name = param["param_name"]
    param_norm = normalize_name(param_name)
    param_shape = param.get("shape")

    candidates: list[tuple[float, str, str, dict[str, Any]]] = []
    for initializer in initializers:
        init_name = initializer["name"]
        init_norm = normalize_name(init_name)
        score = name_similarity_score(param_name, init_name)
        match_type = "unmatched"
        confidence = "low"
        if score == 1.0:
            match_type, confidence = "exact_name", "high"
        elif score >= 0.85:
            match_type, confidence = "suffix_name", "high"
        elif score >= 0.70:
            match_type, confidence = "normalized_name", "medium"
        elif param_shape and initializer.get("shape") == param_shape:
            match_type, confidence, score = "shape_only", "low", 0.25
        elif param_norm.split("_")[-2:] == init_norm.split("_")[-2:]:
            match_type, confidence, score = "suffix_name", "medium", 0.65
        if match_type != "unmatched":
            candidates.append((score, match_type, confidence, initializer))

    if not candidates:
        return ParameterEvidence(param_name, param.get("module_name"), param_shape, None, None, "unmatched", "low", "No compatible ONNX initializer found.")

    candidates.sort(key=lambda item: (item[0], item[2] == "high", item[3]["name"] not in used), reverse=True)
    _, match_type, confidence, initializer = candidates[0]
    used.add(initializer["name"])
    reason = f"Matched by {match_type.replace('_', ' ')}."
    if match_type == "shape_only":
        same_shape = [item for item in initializers if item.get("shape") == param_shape]
        if len(same_shape) > 1:
            reason = "Shape-compatible fallback is non-unique; treat as weak evidence."
    return ParameterEvidence(param_name, param.get("module_name"), param_shape, initializer["name"], initializer.get("shape"), match_type, confidence, reason)


def build_parameter_evidence(torch_summary: dict, onnx_summary: dict) -> list[ParameterEvidence]:
    used: set[str] = set()
    initializers = _initializer_entries(onnx_summary)
    return [_match_param(param, initializers, used) for param in _module_parameters(torch_summary)]


def _torch_modules(torch_summary: dict[str, Any]) -> list[dict[str, Any]]:
    modules = []
    for section, module_type in (
        ("linear_layers", "Linear"),
        ("embedding_layers", "Embedding"),
        ("normalization_layers", "Normalization"),
    ):
        for layer in torch_summary.get(section, []):
            modules.append({"name": layer.get("name", ""), "type": module_type, "raw": layer})
    return modules


def _unit_id_for_module(module_name: str, dependency_graph: dict | None) -> str | None:
    if not dependency_graph:
        return None
    for unit in dependency_graph.get("prunable_units", []):
        if unit.get("module_or_node_name") == module_name or unit.get("name") == module_name:
            return unit.get("unit_id")
    return None


def _nodes_consuming_initializer(nodes: list[dict[str, Any]], initializer_name: str | None, op_types: set[str]) -> list[dict[str, Any]]:
    if not initializer_name:
        return []
    return [
        node
        for node in nodes
        if node.get("op_type") in op_types and initializer_name in node.get("inputs", [])
    ]


def _node_shapes(node: dict[str, Any], tensor_shape_map: dict[str, Any], key: str) -> dict[str, list[int] | str]:
    return {
        tensor: tensor_shape_map.get(tensor, "unknown")
        for tensor in node.get(key, [])
    }


def _build_correspondence_for_module(
    module: dict[str, Any],
    nodes: list[dict[str, Any]],
    tensor_shape_map: dict[str, Any],
    param_evidence: list[ParameterEvidence],
    dependency_graph: dict | None,
) -> ModuleNodeCorrespondence:
    module_name = module["name"]
    module_type = module["type"]
    module_params = [item for item in param_evidence if item.torch_module_name == module_name]
    matched_initializers = [item.onnx_initializer_name for item in module_params if item.onnx_initializer_name]
    preferred_ops = {"Gemm", "MatMul"} if module_type == "Linear" else {"Gather"} if module_type == "Embedding" else {"LayerNormalization", "SkipLayerNormalization"}
    if "conv" in module_name.lower():
        preferred_ops.add("Conv")

    matched_nodes: list[dict[str, Any]] = []
    for initializer in matched_initializers:
        matched_nodes.extend(_nodes_consuming_initializer(nodes, initializer, preferred_ops))
    matched_nodes = list({node.get("name", ""): node for node in matched_nodes}.values())
    if not matched_nodes:
        scored = [
            (name_similarity_score(module_name, node.get("name", "")), node)
            for node in nodes
            if node.get("op_type") in preferred_ops
        ]
        scored = [(score, node) for score, node in scored if score >= 0.45]
        scored.sort(key=lambda item: item[0], reverse=True)
        matched_nodes = [node for _, node in scored[:2]]

    high_param = any(item.confidence == "high" and item.onnx_initializer_name for item in module_params)
    medium_param = any(item.confidence in {"high", "medium"} and item.onnx_initializer_name for item in module_params)
    if matched_nodes and high_param:
        confidence, reason = "high", "ONNX node consumes a high-confidence matched initializer."
    elif matched_nodes and medium_param:
        confidence, reason = "medium", "ONNX node consumes a matched initializer."
    elif matched_nodes:
        confidence, reason = "low", "ONNX node matched by name similarity only."
    else:
        confidence, reason = "low", "No ONNX node correspondence found."

    input_tensors = sorted({tensor for node in matched_nodes for tensor in node.get("inputs", [])})
    output_tensors = sorted({tensor for node in matched_nodes for tensor in node.get("outputs", [])})
    return ModuleNodeCorrespondence(
        torch_module_name=module_name,
        torch_module_type=module_type,
        torch_unit_id=_unit_id_for_module(module_name, dependency_graph),
        onnx_node_names=[node.get("name", "") for node in matched_nodes],
        onnx_op_types=sorted({node.get("op_type", "") for node in matched_nodes}),
        onnx_initializer_names=sorted(set(matched_initializers)),
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        input_shapes={tensor: tensor_shape_map.get(tensor, "unknown") for tensor in input_tensors},
        output_shapes={tensor: tensor_shape_map.get(tensor, "unknown") for tensor in output_tensors},
        parameter_evidence=module_params,
        confidence=confidence,
        reason=reason,
    )


def build_module_node_correspondence(
    torch_summary: dict,
    onnx_summary: dict,
    parameter_evidence: list[ParameterEvidence],
    dependency_graph: dict | None = None,
) -> CorrespondenceReport:
    nodes = onnx_summary.get("nodes", [])
    tensor_shape_map = onnx_summary.get("tensor_shape_map") or _fallback_tensor_shape_map(onnx_summary)
    modules = _torch_modules(torch_summary)
    correspondences = [
        _build_correspondence_for_module(module, nodes, tensor_shape_map, parameter_evidence, dependency_graph)
        for module in modules
    ]
    matched_nodes = {node for corr in correspondences for node in corr.onnx_node_names}
    matched_initializers = {name for evidence in parameter_evidence if evidence.onnx_initializer_name for name in [evidence.onnx_initializer_name]}
    relevant_ops = {"Gemm", "MatMul", "Conv", "Gather", "Add", "Reshape", "Transpose", "Softmax", "LayerNormalization", "SkipLayerNormalization"}
    unmatched_modules = [
        {"name": corr.torch_module_name, "type": corr.torch_module_type, "reason": corr.reason, "confidence": corr.confidence}
        for corr in correspondences
        if not corr.onnx_node_names
    ]
    unmatched_nodes = [
        {"name": node.get("name"), "op_type": node.get("op_type"), "reason": "Relevant ONNX node was not mapped to a PyTorch module."}
        for node in nodes
        if node.get("op_type") in relevant_ops and node.get("name") not in matched_nodes
    ]
    unmatched_initializers = [
        {"name": item.get("name"), "dims": item.get("dims"), "reason": "Initializer was not matched to a PyTorch parameter."}
        for item in onnx_summary.get("initializers", [])
        if item.get("name") not in matched_initializers
    ]
    confidence_counts = Counter(corr.confidence for corr in correspondences)
    param_counts = Counter(item.confidence for item in parameter_evidence)
    report = CorrespondenceReport(
        model_name=torch_summary.get("model_name", onnx_summary.get("model_name", "")),
        hf_id=torch_summary.get("hf_id", onnx_summary.get("hf_id", "")),
        task=torch_summary.get("task", onnx_summary.get("task", "")),
        module_node_correspondences=correspondences,
        parameter_evidence=parameter_evidence,
        unmatched_torch_modules=unmatched_modules,
        unmatched_onnx_nodes=unmatched_nodes,
        unmatched_onnx_initializers=unmatched_initializers,
        summary={
            "num_module_correspondences": len(correspondences),
            "confidence_counts": dict(confidence_counts),
            "parameter_confidence_counts": dict(param_counts),
            "matched_parameters": sum(1 for item in parameter_evidence if item.onnx_initializer_name),
            "unmatched_torch_modules": len(unmatched_modules),
            "unmatched_onnx_nodes": len(unmatched_nodes),
            "unmatched_onnx_initializers": len(unmatched_initializers),
        },
        metadata={"source": "static_name_shape_correspondence"},
    )
    return report


def _fallback_tensor_shape_map(onnx_summary: dict[str, Any]) -> dict[str, Any]:
    shape_map = {}
    for item in onnx_summary.get("inputs", []) + onnx_summary.get("outputs", []):
        shape_map[item.get("name", "")] = item.get("shape", [])
    shape_map.update(onnx_summary.get("value_info_shapes", {}))
    shape_map.update(onnx_summary.get("initializer_shapes", {}))
    return shape_map


def correspondence_report_to_dict(report: CorrespondenceReport) -> dict[str, Any]:
    return asdict(report)


def write_correspondence_json(report: CorrespondenceReport, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(correspondence_report_to_dict(report), indent=2), encoding="utf-8")


def load_correspondence_json(path: Path) -> CorrespondenceReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CorrespondenceReport(
        model_name=data["model_name"],
        hf_id=data.get("hf_id", ""),
        task=data.get("task", ""),
        module_node_correspondences=[
            ModuleNodeCorrespondence(
                **{
                    **item,
                    "parameter_evidence": [ParameterEvidence(**evidence) for evidence in item.get("parameter_evidence", [])],
                }
            )
            for item in data.get("module_node_correspondences", [])
        ],
        parameter_evidence=[ParameterEvidence(**item) for item in data.get("parameter_evidence", [])],
        unmatched_torch_modules=data.get("unmatched_torch_modules", []),
        unmatched_onnx_nodes=data.get("unmatched_onnx_nodes", []),
        unmatched_onnx_initializers=data.get("unmatched_onnx_initializers", []),
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if not rows:
        return "_None detected._"
    selected = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if limit and len(rows) > limit:
        lines.append("| " + " | ".join(["..."] + [f"{len(rows) - limit} more rows omitted"] + [""] * max(0, len(columns) - 2)) + " |")
    return "\n".join(lines)


def correspondence_report_to_markdown(report: CorrespondenceReport) -> str:
    data = correspondence_report_to_dict(report)
    summary = data["summary"]
    module_rows = [
        {
            "torch_module": item["torch_module_name"],
            "torch_type": item["torch_module_type"],
            "onnx_nodes": item["onnx_node_names"],
            "op_types": item["onnx_op_types"],
            "initializers": item["onnx_initializer_names"],
            "confidence": item["confidence"],
            "reason": item["reason"],
        }
        for item in data["module_node_correspondences"]
    ]
    parameter_rows = [
        {
            "torch_parameter": item["torch_param_name"],
            "torch_shape": item["torch_shape"],
            "onnx_initializer": item["onnx_initializer_name"],
            "onnx_shape": item["onnx_shape"],
            "match_type": item["match_type"],
            "confidence": item["confidence"],
            "reason": item["reason"],
        }
        for item in data["parameter_evidence"]
    ]
    lines = [
        f"# PyTorch-to-ONNX Correspondence: {report.model_name}",
        "",
        "## Summary",
        "",
        f"- Module correspondences: `{summary.get('num_module_correspondences', 0)}`",
        f"- Confidence counts: `{summary.get('confidence_counts', {})}`",
        f"- Matched parameters: `{summary.get('matched_parameters', 0)}`",
        f"- Unmatched PyTorch modules: `{summary.get('unmatched_torch_modules', 0)}`",
        f"- Unmatched ONNX nodes: `{summary.get('unmatched_onnx_nodes', 0)}`",
        f"- Unmatched ONNX initializers: `{summary.get('unmatched_onnx_initializers', 0)}`",
        "",
        "## Module-to-Node Matches",
        "",
        _markdown_table(module_rows, ["torch_module", "torch_type", "onnx_nodes", "op_types", "initializers", "confidence", "reason"], limit=250),
        "",
        "## Parameter Evidence",
        "",
        _markdown_table(parameter_rows, ["torch_parameter", "torch_shape", "onnx_initializer", "onnx_shape", "match_type", "confidence", "reason"], limit=250),
        "",
        "## Unmatched / Manual Review",
        "",
        "### Unmatched PyTorch Modules",
        "",
        _markdown_table(data["unmatched_torch_modules"], ["name", "type", "confidence", "reason"], limit=150),
        "",
        "### Unmatched ONNX Nodes",
        "",
        _markdown_table(data["unmatched_onnx_nodes"], ["name", "op_type", "reason"], limit=150),
        "",
        "### Unmatched ONNX Initializers",
        "",
        _markdown_table(data["unmatched_onnx_initializers"], ["name", "dims", "reason"], limit=150),
        "",
    ]
    return "\n".join(lines)
