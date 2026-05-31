"""Static deadbranch propagation analysis for structural channel deadness."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.generic_mlp_fusion import detect_generic_mlp_matches
from model_analysis.paths import ensure_dir


@dataclass
class DeadbranchPropagationPair:
    pair_id: str
    model_name: str
    layer_index: int
    family: str
    pair_kind: str
    producer_region_name: str
    producer_op_name: str
    producer_op_type: str
    producer_axis: str
    consumer_region_name: str
    consumer_op_name: str
    consumer_op_type: str
    consumer_axis: str
    deadness_rule: str
    required_mapping: str
    mapping_status: str
    status: str
    confidence: str
    evidence_ops: list[dict[str, Any]]
    explanation: str


@dataclass
class BlockedDeadbranchPair:
    pair_id: str
    layer_index: int
    pair_kind: str
    producer_region_name: str
    consumer_region_name: str
    blocker_type: str
    status: str
    explanation: str


@dataclass
class DeadbranchPropagationReport:
    model_name: str
    generated_at: str
    summary: dict[str, Any]
    pairs: list[DeadbranchPropagationPair] = field(default_factory=list)
    blocked_pairs: list[BlockedDeadbranchPair] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _norm(value: Any) -> str:
    text = str(value or "").lower().replace("\\", "/").replace("__", "/").replace(".", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text


def _index(op: dict[str, Any] | None) -> int:
    if not op:
        return 10**12
    try:
        return int(op.get("topological_index"))
    except (TypeError, ValueError):
        return 10**12


def _source(op: dict[str, Any] | None) -> str:
    return str((op or {}).get("source_name", ""))


def _op_summary(op: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_name": op.get("source_name", ""),
        "op_type": op.get("op_type", ""),
        "topological_index": op.get("topological_index"),
        "semantic_kind": op.get("semantic_kind", ""),
        "semantic_category": op.get("semantic_category", ""),
        "dimension_roles": op.get("dimension_roles", {}),
    }


def _dedup_ops(ops: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for op in sorted((item for item in ops if item), key=_index):
        key = str(op.get("op_id") or op.get("source_name"))
        if key not in seen:
            seen.add(key)
            out.append(_op_summary(op))
    return out


def _family(model_name: str, source_name: str = "") -> str:
    text = _norm(f"{model_name} {source_name}")
    if "distilbert" in text:
        return "distilbert"
    if "opt" in text or "/decoder/layers/" in text:
        return "opt"
    if "gpt2" in text or "/transformer/h/" in text:
        return "gpt2"
    if "vit" in text or "/vit/layers/" in text:
        return "vit"
    if "bert" in text or "/encoder/layer/" in text:
        return "bert"
    return "unknown"


def _layer(source_name: str) -> int | None:
    source = _norm(source_name)
    for pattern in (
        r"/encoder/layer/(\d+)/",
        r"/transformer/layer/(\d+)/",
        r"/decoder/layers/(\d+)/",
        r"/vit/layers/(\d+)/",
        r"/transformer/h/(\d+)/",
    ):
        match = re.search(pattern, source)
        if match:
            return int(match.group(1))
    return None


def _block_prefix(source_name: str, layer_index: int) -> str:
    source = _norm(source_name)
    for pattern in (
        rf"(.*/encoder/layer/{layer_index})(?:/|$)",
        rf"(.*/transformer/layer/{layer_index})(?:/|$)",
        rf"(.*/decoder/layers/{layer_index})(?:/|$)",
        rf"(.*/vit/layers/{layer_index})(?:/|$)",
        rf"(.*/transformer/h/{layer_index})(?:/|$)",
    ):
        match = re.search(pattern, source)
        if match:
            return match.group(1)
    return ""


def _inside_prefix(op: dict[str, Any], prefix: str) -> bool:
    source = _norm(_source(op))
    return not prefix or source == prefix or source.startswith(prefix.rstrip("/") + "/")


def _is_projection_op(op: dict[str, Any]) -> bool:
    return str(op.get("op_type", "")).lower() in {"matmul", "gemm"}


def _attention_role(op: dict[str, Any]) -> str | None:
    if not _is_projection_op(op):
        return None
    source = _norm(_source(op))
    if "/self_attn/q_proj/" in source or "/attention/self/query/" in source or "/attention/q_lin/" in source:
        return "query"
    if "/self_attn/k_proj/" in source or "/attention/self/key/" in source or "/attention/k_lin/" in source:
        return "key"
    if "/self_attn/v_proj/" in source or "/attention/self/value/" in source or "/attention/v_lin/" in source:
        return "value"
    if "/self_attn/out_proj/" in source or "/attention/output/dense/" in source or "/attention/out_lin/" in source:
        return "output"
    if "/attn/c_proj/" in source:
        return "output"
    if "/attention/" in source and any(token in source for token in ("/value/", "/v_proj/", "/v_lin/")):
        return "value"
    if "/attention/" in source and any(token in source for token in ("/output/", "/out_proj/", "/out_lin/")):
        return "output"
    return None


def _between(ops: list[dict[str, Any]], start: dict[str, Any], end: dict[str, Any]) -> list[dict[str, Any]]:
    low, high = sorted((_index(start), _index(end)))
    return [op for op in ops if low < _index(op) < high]


def _attention_groups(model_name: str, ops: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    groups: dict[tuple[str, int, str], dict[str, Any]] = {}
    for op in ops:
        role = _attention_role(op)
        layer_index = _layer(_source(op))
        if role is None or layer_index is None:
            continue
        family = _family(model_name, _source(op))
        prefix = _block_prefix(_source(op), layer_index)
        group = groups.setdefault((family, layer_index, prefix), {})
        group.setdefault(role, op)
    return groups


def _attention_mapping(
    ops: list[dict[str, Any]], value_op: dict[str, Any], output_op: dict[str, Any]
) -> tuple[str, str, list[dict[str, Any]]]:
    between = _between(ops, value_op, output_op)
    axis_ops = [op for op in between if str(op.get("op_type", "")).lower() in {"reshape", "transpose"}]
    softmax_indices = [_index(op) for op in between if str(op.get("op_type", "")).lower() == "softmax"]
    context_ops = [
        op
        for op in between
        if str(op.get("op_type", "")).lower() == "matmul"
        and softmax_indices
        and _index(op) > max(softmax_indices)
    ]
    evidence = [value_op, *axis_ops, *context_ops, output_op]
    if axis_ops and context_ops:
        return "proven", "high", evidence
    return "assumed_by_pattern", "medium", evidence


def _ffn_pairs(model_name: str, op_semantics: dict[str, Any]) -> list[DeadbranchPropagationPair]:
    pairs: list[DeadbranchPropagationPair] = []
    for match in detect_generic_mlp_matches(model_name, op_semantics):
        if not match.expansion_op or not match.contraction_op:
            continue
        status = "propagatable" if match.evidence_status == "complete" else "constrained"
        mapping_status = "proven" if status == "propagatable" else "unproven"
        pairs.append(
            DeadbranchPropagationPair(
                pair_id=f"deadbranch::{match.family}::{match.layer_index:03d}::ffn",
                model_name=model_name,
                layer_index=match.layer_index,
                family=match.family,
                pair_kind="ffn_intermediate_deadness",
                producer_region_name=match.block_name + " Expansion Projection",
                producer_op_name=_source(match.expansion_op),
                producer_op_type=str(match.expansion_op.get("op_type", "")),
                producer_axis="intermediate_dim",
                consumer_region_name=match.block_name + " Contraction Projection",
                consumer_op_name=_source(match.contraction_op),
                consumer_op_type=str(match.contraction_op.get("op_type", "")),
                consumer_axis="intermediate_dim",
                deadness_rule="consumer_input_zero_implies_producer_output_dead",
                required_mapping="same_intermediate_index",
                mapping_status=mapping_status,
                status=status,
                confidence=match.confidence,
                evidence_ops=_dedup_ops([match.expansion_op, *match.activation_ops, match.contraction_op]),
                explanation="A dead contraction input channel makes the same index in the MLP expansion output dead.",
            )
        )
    return pairs


def _attention_pairs(
    model_name: str, ops: list[dict[str, Any]]
) -> tuple[list[DeadbranchPropagationPair], list[BlockedDeadbranchPair], list[str]]:
    pairs: list[DeadbranchPropagationPair] = []
    blocked: list[BlockedDeadbranchPair] = []
    notes: list[str] = []
    groups = _attention_groups(model_name, ops)
    for (family, layer_index, prefix), group in sorted(groups.items(), key=lambda item: item[0][1]):
        value_op = group.get("value")
        output_op = group.get("output")
        query_op = group.get("query")
        key_op = group.get("key")
        if value_op and output_op:
            mapping_status, confidence, evidence = _attention_mapping(ops, value_op, output_op)
            status = "propagatable" if mapping_status == "proven" else "constrained"
            pairs.append(
                DeadbranchPropagationPair(
                    pair_id=f"deadbranch::{family}::{layer_index:03d}::attention_value",
                    model_name=model_name,
                    layer_index=layer_index,
                    family=family,
                    pair_kind="attention_value_deadness",
                    producer_region_name=f"{family.upper()} Layer {layer_index} Value Projection",
                    producer_op_name=_source(value_op),
                    producer_op_type=str(value_op.get("op_type", "")),
                    producer_axis="value_channel",
                    consumer_region_name=f"{family.upper()} Layer {layer_index} Attention Output Projection",
                    consumer_op_name=_source(output_op),
                    consumer_op_type=str(output_op.get("op_type", "")),
                    consumer_axis="value_context_channel",
                    deadness_rule="consumer_input_zero_implies_producer_output_dead",
                    required_mapping="reshape_transpose_value_axis_mapping",
                    mapping_status=mapping_status,
                    status=status,
                    confidence=confidence,
                    evidence_ops=_dedup_ops(evidence),
                    explanation=(
                        "A dead attention output-projection input channel makes the corresponding context "
                        "channel dead and propagates backward to the same value-projection output channel."
                    ),
                )
            )
        elif value_op or output_op:
            notes.append(f"{family} layer {layer_index}: value-path projection pair is incomplete.")
        for role, op, pair_kind in (
            ("query", query_op, "query_score_deadness"),
            ("key", key_op, "key_score_deadness"),
        ):
            if not op:
                continue
            blocked.append(
                BlockedDeadbranchPair(
                    pair_id=f"deadbranch::{family}::{layer_index:03d}::{role}_blocked",
                    layer_index=layer_index,
                    pair_kind=pair_kind,
                    producer_region_name=f"{family.upper()} Layer {layer_index} {role.title()} Projection",
                    consumer_region_name=f"{family.upper()} Layer {layer_index} Attention Score MatMul",
                    blocker_type="qk_score_contraction_mixes_channels",
                    status="blocked",
                    explanation=(
                        "Q and K feed QK^T score contraction. Consumer-column deadness does not imply "
                        "one-to-one producer-output deadness for this mixed channel path."
                    ),
                )
            )
        if family == "gpt2" and not value_op:
            notes.append(f"gpt2 layer {layer_index}: fused QKV projection does not prove an independent value split.")
    return pairs, blocked, sorted(set(notes))


def _summary(model_name: str, pairs: list[DeadbranchPropagationPair], blocked: list[BlockedDeadbranchPair]) -> dict[str, Any]:
    ffn_pairs = sum(pair.pair_kind == "ffn_intermediate_deadness" for pair in pairs)
    value_pairs = sum(pair.pair_kind == "attention_value_deadness" for pair in pairs)
    propagatable = sum(pair.status == "propagatable" for pair in pairs)
    constrained = sum(pair.status == "constrained" for pair in pairs)
    expected = 24 if _family(model_name) == "opt" else len(pairs)
    if _family(model_name) == "opt":
        alignment = "matches_expected" if ffn_pairs == 12 and value_pairs == 12 and len(pairs) == 24 else "partial"
    else:
        alignment = "matches_expected" if len(pairs) == expected else "mismatch"
    return {
        "total_pairs": len(pairs),
        "propagatable_pairs": propagatable,
        "constrained_pairs": constrained,
        "blocked_pairs": len(blocked),
        "ffn_pairs": ffn_pairs,
        "attention_value_pairs": value_pairs,
        "query_key_blocked_pairs": len(blocked),
        "expected_sparsegpt_pairs": expected,
        "sparsegpt_alignment_status": alignment,
    }


def analyze_deadbranch_propagation(model_name: str, op_semantics: dict[str, Any]) -> DeadbranchPropagationReport:
    """Build static deadbranch propagation records from existing op semantics."""
    ops = list(op_semantics.get("ops", []))
    ffn_pairs = _ffn_pairs(model_name, op_semantics)
    attention_pairs, blocked, notes = _attention_pairs(model_name, ops)
    pairs = sorted([*ffn_pairs, *attention_pairs], key=lambda pair: (pair.layer_index, pair.pair_kind, pair.pair_id))
    return DeadbranchPropagationReport(
        model_name=model_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        summary=_summary(model_name, pairs, blocked),
        pairs=pairs,
        blocked_pairs=sorted(blocked, key=lambda pair: (pair.layer_index, pair.pair_kind)),
        notes=[
            "SparseGPT 2:4 fine-grained sparsity is shape-preserving and does not guarantee dead channels.",
            "Structural channel deadness can propagate only when a consumer input channel is exactly zero/dead.",
            *notes,
        ],
    )


def deadbranch_report_to_dict(report: DeadbranchPropagationReport | dict[str, Any]) -> dict[str, Any]:
    return report if isinstance(report, dict) else asdict(report)


def write_deadbranch_report(report: DeadbranchPropagationReport | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(deadbranch_report_to_dict(report), indent=2), encoding="utf-8")
