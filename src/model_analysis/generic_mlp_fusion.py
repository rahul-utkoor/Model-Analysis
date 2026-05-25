"""Detect generic transformer MLP blocks from primitive op semantics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GenericMLPMatch:
    model_name: str
    family: str
    layer_index: int
    block_name: str
    expansion_op: dict[str, Any] | None
    expansion_bias_op: dict[str, Any] | None
    activation_ops: list[dict[str, Any]]
    contraction_op: dict[str, Any] | None
    contraction_bias_op: dict[str, Any] | None
    residual_op: dict[str, Any] | None
    layernorm_op: dict[str, Any] | None
    source_ops: list[dict[str, Any]]
    op_range: str
    confidence: str
    evidence_status: str
    missing_evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def generic_mlp_match_to_dict(value: GenericMLPMatch) -> dict[str, Any]:
    return asdict(value)


def _norm(value: Any) -> str:
    normalized = str(value or "").lower().replace("\\", "/").replace("__", "/").replace(".", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def _source(op: dict[str, Any] | None) -> str:
    return str((op or {}).get("source_name", ""))


def _index(op: dict[str, Any] | None) -> int:
    if not op:
        return 10**12
    value = op.get("topological_index")
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        digits = re.findall(r"\d+", str(op.get("op_id", "")))
        return int(digits[-1]) if digits else 10**12


def _family(model_name: str, source: str) -> str:
    text = _norm(f"{model_name} {source}")
    if "distilbert" in text or "/ffn/lin1/" in text:
        return "distilbert"
    if "opt" in text or "/decoder/layers/" in text:
        return "opt"
    if "gpt2" in text or "/transformer/h/" in text or "/mlp/c_fc/" in text:
        return "gpt2"
    if "vit" in text or "/vit/layers/" in text or "/mlp/fc1/" in text:
        return "vit"
    if "/encoder/layer/" in text or "/intermediate/dense/" in text:
        return "bert"
    return "unknown"


def _layer(source: str) -> int | None:
    normalized = _norm(source)
    for pattern in (
        r"/encoder/layer/(\d+)/",
        r"/transformer/layer/(\d+)/",
        r"/decoder/layers/(\d+)/",
        r"/vit/layers/(\d+)/",
        r"/layers/(\d+)/",
        r"/transformer/h/(\d+)/",
        r"/h/(\d+)/",
    ):
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))
    return None


def _is_parameterized_projection(op: dict[str, Any]) -> bool:
    roles = op.get("dimension_roles", {})
    return (
        op.get("semantic_category") == "parameterized_projection"
        and op.get("parameterized") is True
        and roles.get("input") in {"hidden_dim", "intermediate_dim"}
        and roles.get("output") in {"hidden_dim", "intermediate_dim"}
    )


def _is_expansion(op: dict[str, Any]) -> bool:
    roles = op.get("dimension_roles", {})
    source = _norm(_source(op))
    return (
        _is_parameterized_projection(op)
        and roles.get("input") == "hidden_dim"
        and roles.get("output") == "intermediate_dim"
        and not any(token in source for token in ("/attention/", "/attn/", "/self_attn/"))
    )


def _is_contraction(op: dict[str, Any]) -> bool:
    roles = op.get("dimension_roles", {})
    source = _norm(_source(op))
    if any(token in source for token in ("/attention/", "/self_attn/", "/attn/c_proj/")):
        return False
    return (
        _is_parameterized_projection(op)
        and roles.get("input") == "intermediate_dim"
        and roles.get("output") == "hidden_dim"
    )


def _is_activation(op: dict[str, Any]) -> bool:
    roles = op.get("dimension_roles", {})
    return (
        op.get("semantic_category") == "elementwise_index_preserving"
        and roles.get("input") == "intermediate_dim"
        and roles.get("output") == "intermediate_dim"
    )


def _is_bias(op: dict[str, Any]) -> bool:
    return op.get("semantic_kind") == "linear_bias_add" and op.get("semantic_category") == "parameterized_projection"


def _prefix_and_side(source: str) -> tuple[str, str] | None:
    normalized = _norm(source)
    patterns = [
        ("/intermediate/dense/", "bert", "expansion"),
        ("/output/dense/", "bert", "contraction"),
        ("/ffn/lin1/", "distilbert", "expansion"),
        ("/ffn/lin2/", "distilbert", "contraction"),
        ("/fc1/", "opt", "expansion"),
        ("/fc2/", "opt", "contraction"),
        ("/mlp/fc1/", "vit", "expansion"),
        ("/mlp/fc2/", "vit", "contraction"),
        ("/mlp/c_fc/", "gpt2", "expansion"),
        ("/mlp/c_proj/", "gpt2", "contraction"),
    ]
    for token, _family_name, side in patterns:
        if token in normalized:
            prefix = normalized.split(token, 1)[0]
            return prefix, side
    return None


def _activation_prefix(source: str) -> str | None:
    normalized = _norm(source)
    for token in (
        "/intermediate/intermediate_act_fn/",
        "/ffn/activation/",
        "/activation_fn/",
        "/mlp/activation_fn/",
        "/mlp/act/",
    ):
        if token in normalized:
            return normalized.split(token, 1)[0]
    return None


def _support_op(ops: list[dict[str, Any]], prefix: str, kinds: set[str], tokens: tuple[str, ...]) -> dict[str, Any] | None:
    for op in sorted(ops, key=_index):
        source = _norm(_source(op))
        if not source.startswith(prefix):
            continue
        if op.get("semantic_kind") not in kinds:
            continue
        if any(token in source for token in tokens):
            return op
    return None


def _display_name(family: str, layer: int) -> str:
    if family == "distilbert":
        return f"DistilBERT Layer {layer} FFN"
    if family == "vit":
        return f"ViT Layer {layer} MLP"
    if family == "gpt2":
        return f"GPT2 Block {layer} MLP"
    if family == "opt":
        return f"OPT Layer {layer} FFN"
    if family == "bert":
        return f"BERT Layer {layer} Feed Forward"
    return f"Layer {layer} Generic MLP"


def _op_range(ops: list[dict[str, Any]]) -> str:
    if not ops:
        return "-"
    indices = [_index(op) for op in ops if _index(op) < 10**12]
    return f"{min(indices)}-{max(indices)}" if indices else "-"


def detect_generic_mlp_matches(model_name: str, op_semantics: dict[str, Any]) -> list[GenericMLPMatch]:
    ops = list(op_semantics.get("ops", []))
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for op in ops:
        source = _source(op)
        layer = _layer(source)
        if layer is None:
            continue
        prefix_side = _prefix_and_side(source)
        if not prefix_side:
            continue
        prefix, side = prefix_side
        family = _family(model_name, source)
        key = (family, layer, prefix)
        item = grouped.setdefault(key, {"expansion": None, "contraction": None, "expansion_bias": None, "contraction_bias": None, "activation": []})
        if side == "expansion" and _is_expansion(op):
            item["expansion"] = op
        elif side == "contraction" and _is_contraction(op):
            item["contraction"] = op
        elif side == "expansion" and _is_bias(op):
            item["expansion_bias"] = op
        elif side == "contraction" and _is_bias(op):
            item["contraction_bias"] = op

    for op in ops:
        if not _is_activation(op):
            continue
        layer = _layer(_source(op))
        prefix = _activation_prefix(_source(op))
        if layer is None or prefix is None:
            continue
        family = _family(model_name, _source(op))
        key = (family, layer, prefix)
        if key in grouped:
            grouped[key]["activation"].append(op)

    matches: list[GenericMLPMatch] = []
    for (family, layer, prefix), item in sorted(grouped.items(), key=lambda row: (row[0][1], row[0][0], row[0][2])):
        expansion = item.get("expansion")
        contraction = item.get("contraction")
        activations = sorted(item.get("activation") or [], key=_index)
        missing: list[str] = []
        warnings: list[str] = []
        if not expansion:
            missing.append("missing_expansion_projection")
        if not activations:
            missing.append("missing_activation_evidence")
        if not contraction:
            missing.append("missing_contraction_projection")
        if not item.get("expansion_bias") and not (expansion and str(expansion.get("op_type", "")).lower() == "gemm"):
            warnings.append("no_expansion_bias")
        if not item.get("contraction_bias") and not (contraction and str(contraction.get("op_type", "")).lower() == "gemm"):
            warnings.append("no_contraction_bias")
        residual = _support_op(ops, prefix, {"residual_add"}, ("/output/add", "/add"))
        layernorm = _support_op(ops, prefix, {"layernorm"}, ("layernorm", "layer_norm", "ln_2", "ln/2"))
        if not residual:
            warnings.append("missing_residual_evidence")
        if not layernorm:
            warnings.append("missing_layernorm_evidence")
        source_ops = [op for op in [expansion, item.get("expansion_bias"), *activations, contraction, item.get("contraction_bias"), residual, layernorm] if op]
        required_missing = [value for value in missing if value in {"missing_expansion_projection", "missing_activation_evidence", "missing_contraction_projection"}]
        status = "complete" if not required_missing else "partial" if source_ops else "missing"
        confidence = "high" if status == "complete" else "medium" if status == "partial" else "low"
        matches.append(
            GenericMLPMatch(
                model_name=model_name,
                family=family,
                layer_index=layer,
                block_name=_display_name(family, layer),
                expansion_op=expansion,
                expansion_bias_op=item.get("expansion_bias"),
                activation_ops=activations,
                contraction_op=contraction,
                contraction_bias_op=item.get("contraction_bias"),
                residual_op=residual,
                layernorm_op=layernorm,
                source_ops=source_ops,
                op_range=_op_range(source_ops),
                confidence=confidence,
                evidence_status=status,
                missing_evidence=missing,
                warnings=warnings,
            )
        )
    return matches
