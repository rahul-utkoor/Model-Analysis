"""Semantic fusion detection over frontend-independent Tensor IR."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FusedSemanticRegion:
    fusion_id: str
    model_name: str
    fusion_type: str
    op_ids: list[str]
    input_values: list[str]
    output_values: list[str]
    pattern: str
    confidence: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticFusionReport:
    model_name: str
    fusions: list[FusedSemanticRegion]
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def fused_semantic_region_to_dict(region: FusedSemanticRegion) -> dict[str, Any]:
    return asdict(region)


def semantic_fusion_report_to_dict(report: SemanticFusionReport) -> dict[str, Any]:
    return asdict(report)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def semantic_fusion_report_to_markdown(report: SemanticFusionReport | dict) -> str:
    data = semantic_fusion_report_to_dict(report) if isinstance(report, SemanticFusionReport) else report
    summary = data.get("summary", {})
    lines = [
        f"# Semantic Fusion Report: {data.get('model_name', '')}",
        "",
        "## Summary",
        "",
        f"- Fusions: `{summary.get('num_fusions', 0)}`",
        f"- GELU activation fusions: `{summary.get('num_gelu_fusions', 0)}`",
        f"- Feed-forward fusions: `{summary.get('num_feedforward_fusions', 0)}`",
        f"- Confidence counts: `{summary.get('confidence_counts', {})}`",
        "",
        "## Fused Semantic Regions",
        "",
        "| fusion_id | type | ops | pattern | confidence | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in data.get("fusions", []):
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    item.get("fusion_id", ""),
                    item.get("fusion_type", ""),
                    len(item.get("op_ids", [])),
                    item.get("pattern", ""),
                    item.get("confidence", ""),
                    item.get("reason", ""),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Semantic fusions recover high-level activation and feed-forward structure from decomposed Tensor IR operations. They are structural analysis evidence only and do not execute pruning or modify models.",
            "",
        ]
    )
    return "\n".join(lines)


def _ops(tensor_graph: dict) -> dict[str, dict]:
    return {item["op_id"]: item for item in tensor_graph.get("ops", [])}


def _values(tensor_graph: dict) -> dict[str, dict]:
    return {item["value_id"]: item for item in tensor_graph.get("values", [])}


def _raw_type(op: dict | None) -> str:
    return str((op or {}).get("op_type", "")).lower()


def _ordered_op_ids(tensor_graph: dict, ids: set[str] | list[str]) -> list[str]:
    selected = set(ids)
    return [item["op_id"] for item in tensor_graph.get("ops", []) if item["op_id"] in selected]


def _producer(value_id: str | None, values: dict[str, dict], ops: dict[str, dict]) -> dict | None:
    producer_id = values.get(value_id or "", {}).get("producer")
    return ops.get(producer_id)


def _is_constant_value(value_id: str, values: dict[str, dict], ops: dict[str, dict]) -> bool:
    value = values.get(value_id, {})
    if value.get("is_initializer") or value.get("semantic_role") in {"parameter", "constant"}:
        return True
    producer = _producer(value_id, values, ops)
    return bool(producer and producer.get("canonical_op_type") == "constant")


def _data_inputs(op: dict, values: dict[str, dict], ops: dict[str, dict]) -> list[str]:
    return [value_id for value_id in op.get("inputs", []) if not _is_constant_value(value_id, values, ops)]


def _output(op: dict | None) -> str | None:
    outputs = (op or {}).get("outputs", [])
    return outputs[0] if outputs else None


def _successors(op: dict, ops: dict[str, dict]) -> list[dict]:
    return [ops[item] for item in op.get("successor_ops", []) if item in ops]


def _first_successor(op: dict, ops: dict[str, dict], raw_types: set[str]) -> dict | None:
    candidates = [item for item in _successors(op, ops) if _raw_type(item) in raw_types]
    return sorted(candidates, key=lambda item: item["op_id"])[0] if candidates else None


def detect_gelu_fusions(tensor_graph: dict) -> list[FusedSemanticRegion]:
    """Recover graph-structured GELU motifs centered on an Erf operation."""
    ops = _ops(tensor_graph)
    values = _values(tensor_graph)
    model_name = tensor_graph.get("model_name", "")
    fusions: list[FusedSemanticRegion] = []
    for erf in tensor_graph.get("ops", []):
        if _raw_type(erf) != "erf":
            continue
        matched = {erf["op_id"]}
        transform = None
        transform_value = erf.get("inputs", [None])[0] if erf.get("inputs") else None
        producer = _producer(transform_value, values, ops)
        if producer and _raw_type(producer) in {"div", "mul"}:
            transform = producer
            matched.add(transform["op_id"])
            source_candidates = _data_inputs(transform, values, ops)
            source_value = source_candidates[0] if source_candidates else transform_value
        else:
            source_value = transform_value

        add = _first_successor(erf, ops, {"add"})
        first_mul = _first_successor(add, ops, {"mul"}) if add else None
        multiply_back = bool(first_mul and source_value in _data_inputs(first_mul, values, ops))
        final_mul = _first_successor(first_mul, ops, {"mul"}) if first_mul else None
        if add:
            matched.add(add["op_id"])
        if first_mul:
            matched.add(first_mul["op_id"])
        if final_mul:
            matched.add(final_mul["op_id"])

        if add and first_mul and final_mul and multiply_back:
            confidence = "high"
            reason = "Erf branch rejoins the original activation through multiply-back and scalar scaling, matching decomposed GELU."
        elif add and first_mul and multiply_back:
            confidence = "medium"
            reason = "Erf branch multiplies back with its source activation, matching a partial GELU-like structure."
        else:
            confidence = "low"
            reason = "Erf-centered activation fragment found without a proven GELU multiply-back chain."
        last_op = final_mul or first_mul or add or erf
        pattern_ops = _ordered_op_ids(tensor_graph, matched)
        fusions.append(
            FusedSemanticRegion(
                fusion_id=f"fusion::gelu::{len(fusions) + 1:06d}",
                model_name=model_name,
                fusion_type="GeluActivation",
                op_ids=pattern_ops,
                input_values=[source_value] if source_value else [],
                output_values=[_output(last_op)] if _output(last_op) else [],
                pattern=" -> ".join(ops[op_id].get("op_type", "Unknown") for op_id in pattern_ops),
                confidence=confidence,
                reason=reason,
                metadata={
                    "activation_kind": "gelu",
                    "source_value": source_value,
                    "transform_op": transform["op_id"] if transform else None,
                    "erf_op": erf["op_id"],
                    "add_op": add["op_id"] if add else None,
                    "multiply_back_op": first_mul["op_id"] if first_mul else None,
                    "scale_op": final_mul["op_id"] if final_mul else None,
                    "multiply_back_confirmed": multiply_back,
                },
            )
        )
    return fusions


def _linear_projection_before(
    value_id: str | None,
    values: dict[str, dict],
    ops: dict[str, dict],
) -> tuple[list[str], dict | None]:
    producer = _producer(value_id, values, ops)
    if not producer:
        return [], None
    if producer.get("canonical_op_type") == "bias_add":
        parents = [
            ops[item]
            for item in producer.get("predecessor_ops", [])
            if item in ops and ops[item].get("canonical_op_type") in {"linear", "matmul"}
        ]
        if parents:
            return [parents[0]["op_id"], producer["op_id"]], parents[0]
    if producer.get("canonical_op_type") in {"linear", "matmul"}:
        return [producer["op_id"]], producer
    return [], None


def _linear_projection_after(
    value_id: str | None,
    values: dict[str, dict],
    ops: dict[str, dict],
) -> tuple[list[str], dict | None, str | None]:
    consumers = [
        ops[item]
        for item in values.get(value_id or "", {}).get("consumers", [])
        if item in ops and ops[item].get("canonical_op_type") in {"linear", "matmul"}
    ]
    if not consumers:
        return [], None, None
    projection = sorted(consumers, key=lambda item: item["op_id"])[0]
    matched = [projection["op_id"]]
    output_value = _output(projection)
    bias_consumers = [
        ops[item]
        for item in values.get(output_value or "", {}).get("consumers", [])
        if item in ops and ops[item].get("canonical_op_type") == "bias_add"
    ]
    if bias_consumers:
        bias = sorted(bias_consumers, key=lambda item: item["op_id"])[0]
        matched.append(bias["op_id"])
        output_value = _output(bias)
    return matched, projection, output_value


def detect_feedforward_fusions(
    tensor_graph: dict,
    gelu_fusions: list[FusedSemanticRegion],
) -> list[FusedSemanticRegion]:
    """Recover projection/GELU/projection feed-forward motifs."""
    ops = _ops(tensor_graph)
    values = _values(tensor_graph)
    model_name = tensor_graph.get("model_name", "")
    fusions: list[FusedSemanticRegion] = []
    for activation in gelu_fusions:
        if activation.confidence == "low" or not activation.input_values or not activation.output_values:
            continue
        first_ops, first_projection = _linear_projection_before(activation.input_values[0], values, ops)
        second_ops, second_projection, hidden_output = _linear_projection_after(activation.output_values[0], values, ops)
        if not first_projection or not second_projection:
            continue
        both_biased = len(first_ops) == 2 and len(second_ops) == 2
        confidence = "high" if both_biased and activation.confidence == "high" else "medium"
        reason = (
            "Two projection regions surround a decomposed GELU activation, exposing a coupled feed-forward intermediate dimension."
            if confidence == "high"
            else "Linear-like projections surround a GELU fusion; feed-forward coupling is plausible but lacks complete bias evidence."
        )
        matched = _ordered_op_ids(tensor_graph, set([*first_ops, *activation.op_ids, *second_ops]))
        hidden_inputs = _data_inputs(first_projection, values, ops)
        fusions.append(
            FusedSemanticRegion(
                fusion_id=f"fusion::feedforward::{len(fusions) + 1:06d}",
                model_name=model_name,
                fusion_type="FeedForward",
                op_ids=matched,
                input_values=hidden_inputs,
                output_values=[hidden_output] if hidden_output else [],
                pattern="LinearProjection -> GeluActivation -> LinearProjection",
                confidence=confidence,
                reason=reason,
                metadata={
                    "activation_kind": "gelu",
                    "activation_fusion_id": activation.fusion_id,
                    "first_projection_ops": first_ops,
                    "second_projection_ops": second_ops,
                    "candidate_intermediate_value": activation.input_values[0],
                    "candidate_hidden_input": hidden_inputs[0] if hidden_inputs else None,
                    "candidate_hidden_output": hidden_output,
                },
            )
        )
    return fusions


def build_semantic_fusion_report(tensor_graph: dict) -> SemanticFusionReport:
    gelu = detect_gelu_fusions(tensor_graph)
    feedforward = detect_feedforward_fusions(tensor_graph, gelu)
    fusions = [*gelu, *feedforward]
    type_counts = Counter(item.fusion_type for item in fusions)
    confidence_counts = Counter(item.confidence for item in fusions)
    return SemanticFusionReport(
        model_name=tensor_graph.get("model_name", ""),
        fusions=fusions,
        summary={
            "num_fusions": len(fusions),
            "num_gelu_fusions": type_counts.get("GeluActivation", 0),
            "num_feedforward_fusions": type_counts.get("FeedForward", 0),
            "fusion_type_counts": dict(type_counts),
            "confidence_counts": dict(confidence_counts),
        },
        metadata={
            "source_frontend": tensor_graph.get("source_frontend", "unknown"),
            "analysis_note": "Semantic fusions are structural evidence over Tensor IR and do not modify model artifacts.",
        },
    )
