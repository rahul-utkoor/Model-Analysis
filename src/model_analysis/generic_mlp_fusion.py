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


def _in_prefix(source: str, prefix: str) -> bool:
    """Return true only when source is inside the exact normalized layer/block prefix.

    A plain startswith(prefix) is unsafe for GPT-2 because
    /transformer/h/1 also matches /transformer/h/10 and /transformer/h/11.
    Require a path boundary after the prefix.
    """
    normalized_source = _norm(source)
    normalized_prefix = _norm(prefix).rstrip("/")
    if not normalized_prefix:
        return True
    return normalized_source == normalized_prefix or normalized_source.startswith(normalized_prefix + "/")


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
        if not _in_prefix(source, prefix):
            continue
        if op.get("semantic_kind") not in kinds:
            continue
        if any(token in source for token in tokens):
            return op
    return None


def _support_op_near(
    ops: list[dict[str, Any]],
    *,
    prefix: str,
    anchor_index: int,
    kinds: set[str] | None = None,
    op_types: set[str] | None = None,
    tokens: tuple[str, ...] = (),
    before: bool = True,
) -> dict[str, Any] | None:
    """Return the nearest support op around an anchor.

    This is mainly used for decoder MLP evidence. GPT-2 blocks contain two
    LayerNorms; choosing the first LayerNorm in the block gives the attention
    pre-norm, not the MLP pre-norm. For MLP evidence we want the nearest
    LayerNorm before c_fc, and the nearest residual Add after c_proj.
    """
    candidates: list[dict[str, Any]] = []
    for op in ops:
        source = _norm(_source(op))
        if prefix and not _in_prefix(source, prefix):
            continue
        if kinds is not None and op.get("semantic_kind") not in kinds:
            continue
        if op_types is not None and str(op.get("op_type", "")) not in op_types:
            continue
        if tokens and not any(token in source for token in tokens):
            continue
        idx = _index(op)
        if before and idx < anchor_index:
            candidates.append(op)
        elif not before and idx > anchor_index:
            candidates.append(op)
    if not candidates:
        return None
    if before:
        return max(candidates, key=_index)
    return min(candidates, key=_index)


def _path_ops(
    ops: list[dict[str, Any]],
    *,
    prefix: str,
    token: str,
    op_types: set[str],
) -> list[dict[str, Any]]:
    """Return compact visual ops from one MLP path segment.

    ONNX exports of GPT-2 Conv1D-style projections often contain reshape nodes
    around Gemm. If the report selects only Gemm + activation ops, the exported
    visualization can look disconnected even though the semantic plan is valid.
    Keep the learner-facing path compact but connected by including the reshape
    nodes that are directly in the c_fc/c_proj path.
    """
    out: list[dict[str, Any]] = []
    normalized_token = _norm(token)
    for op in sorted(ops, key=_index):
        source = _norm(_source(op))
        if prefix and not _in_prefix(source, prefix):
            continue
        if normalized_token not in source:
            continue
        if str(op.get("op_type", "")) in op_types:
            out.append(op)
    return out


def _activation_path_ops(ops: list[dict[str, Any]], *, prefix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op in sorted(ops, key=_index):
        source = _norm(_source(op))
        if prefix and not _in_prefix(source, prefix):
            continue
        if "/mlp/act/" not in source and "/activation" not in source and "/intermediate/intermediate_act_fn/" not in source and "/ffn/activation/" not in source:
            continue
        if str(op.get("op_type", "")) in {"Pow", "Mul", "Add", "Tanh", "Erf", "Div", "Relu", "Gelu"}:
            out.append(op)
    return out


def _dedup_full_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for op in sorted([item for item in ops if item], key=_index):
        key = str(op.get("op_id") or op.get("source_name") or _index(op))
        if key in seen:
            continue
        seen.add(key)
        out.append(op)
    return out


def _visual_mlp_source_ops(
    *,
    family: str,
    ops: list[dict[str, Any]],
    prefix: str,
    expansion: dict[str, Any] | None,
    expansion_bias: dict[str, Any] | None,
    activations: list[dict[str, Any]],
    contraction: dict[str, Any] | None,
    contraction_bias: dict[str, Any] | None,
    residual: dict[str, Any] | None,
    layernorm: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if family == "gpt2":
        source_ops: list[dict[str, Any]] = []
        if layernorm:
            source_ops.append(layernorm)
        # Keep the exported GPT-2 MLP evidence connected:
        # ln_2 -> Reshape -> c_fc/Gemm -> Reshape -> NewGELU -> Reshape -> c_proj/Gemm -> Reshape -> residual Add.
        source_ops.extend(_path_ops(ops, prefix=prefix, token="/mlp/c_fc/", op_types={"Reshape", "Gemm"}))
        source_ops.extend(_activation_path_ops(ops, prefix=prefix))
        source_ops.extend(_path_ops(ops, prefix=prefix, token="/mlp/c_proj/", op_types={"Reshape", "Gemm"}))
        if residual:
            source_ops.append(residual)
        return _dedup_full_ops(source_ops)

    # Other families already export compact connected paths with the learned
    # projection, optional bias, activation, and contraction projection.
    return _dedup_full_ops([expansion, expansion_bias, *activations, contraction, contraction_bias, residual, layernorm])


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
        expansion_index = _index(expansion)
        contraction_index = _index(contraction)
        if family == "gpt2":
            layernorm = _support_op_near(
                ops,
                prefix=prefix,
                anchor_index=expansion_index,
                kinds={"layernorm"},
                tokens=("ln/2", "ln_2", "layernormalization"),
                before=True,
            )
            # GPT-2 residual Add is currently often categorized as unknown in
            # op semantics, so select the nearest Add after c_proj by topology.
            residual = _support_op_near(
                ops,
                prefix=prefix,
                anchor_index=contraction_index,
                op_types={"Add"},
                before=False,
            )
        elif family == "opt":
            # OPT has both self_attn_layer_norm and final_layer_norm in the same
            # decoder block.  The MLP evidence path starts at the nearest
            # LayerNorm before fc1, not the earlier self-attention LayerNorm.
            layernorm = _support_op_near(
                ops,
                prefix=prefix,
                anchor_index=expansion_index,
                kinds={"layernorm"},
                before=True,
            )
            # The post-fc2 residual add is not always semantically categorized,
            # so use the nearest Add after fc2 as visual evidence.
            residual = _support_op_near(
                ops,
                prefix=prefix,
                anchor_index=contraction_index,
                op_types={"Add"},
                before=False,
            )
        else:
            # For encoder families, the nearest hidden-dim LayerNorm before the
            # expansion projection gives a compact connected visual path.
            layernorm = _support_op_near(
                ops,
                prefix=prefix,
                anchor_index=expansion_index,
                kinds={"layernorm"},
                before=True,
            ) or _support_op(ops, prefix, {"layernorm"}, ("layernorm", "layer_norm", "ln_2", "ln/2"))
            residual = _support_op(ops, prefix, {"residual_add"}, ("/output/add", "/add"))
        if not residual:
            warnings.append("missing_residual_evidence")
        if not layernorm:
            warnings.append("missing_layernorm_evidence")
        source_ops = _visual_mlp_source_ops(
            family=family,
            ops=ops,
            prefix=prefix,
            expansion=expansion,
            expansion_bias=item.get("expansion_bias"),
            activations=activations,
            contraction=contraction,
            contraction_bias=item.get("contraction_bias"),
            residual=residual,
            layernorm=layernorm,
        )
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
