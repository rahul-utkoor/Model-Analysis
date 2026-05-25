"""Family-aware transformer block grouping for learner-facing reports."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from model_analysis.generic_mlp_fusion import GenericMLPMatch, detect_generic_mlp_matches, generic_mlp_match_to_dict


@dataclass
class GenericSubgraphGroup:
    group_id: str
    ordinal: int
    display_name: str
    group_kind: str
    semantic_category: str
    source_ops: list[dict[str, Any]]
    op_range: str
    pruning_class: str
    plan_status: str
    validation_status: str
    why_no_plan: str
    explanation: str


@dataclass
class GenericBlock:
    model_name: str
    family: str
    block_index: int
    block_kind: str
    block_name: str
    path_prefixes: list[str]
    op_range: str
    primitive_ops: list[dict[str, Any]]
    grouped_subgraphs: list[GenericSubgraphGroup] = field(default_factory=list)
    mlp_match: dict[str, Any] | None = None
    attention_groups: list[GenericSubgraphGroup] = field(default_factory=list)
    residual_groups: list[GenericSubgraphGroup] = field(default_factory=list)
    layernorm_groups: list[GenericSubgraphGroup] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def generic_block_to_dict(value: GenericBlock) -> dict[str, Any]:
    return asdict(value)


def generic_subgraph_group_to_dict(value: GenericSubgraphGroup) -> dict[str, Any]:
    return asdict(value)


def normalize_source(value: Any) -> str:
    text = str(value or "").lower().replace("\\", "/").replace("__", "/").replace(".", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text


def detect_family(model_name: str, source: str = "") -> str:
    text = normalize_source(f"{model_name} {source}")
    if "distilbert" in text or "/ffn/lin" in text:
        return "distilbert"
    if "opt" in text or "/decoder/layers/" in text:
        return "opt"
    if "gpt2" in text or "/transformer/h/" in text or "/mlp/c_fc/" in text:
        return "gpt2"
    if "vit" in text or "/vit/layers/" in text or "/mlp/fc1/" in text:
        return "vit"
    if "bert" in text or "/encoder/layer/" in text:
        return "bert"
    return "unknown"


def block_index_from_source(source: str) -> int | None:
    normalized = normalize_source(source)
    for pattern in (
        r"/encoder/layer/(\d+)(?:/|$)",
        r"/transformer/layer/(\d+)(?:/|$)",
        r"/decoder/layers/(\d+)(?:/|$)",
        r"/transformer/h/(\d+)(?:/|$)",
        r"/vit/layers/(\d+)(?:/|$)",
        r"\bencoder layer\s+(\d+)\b",
        r"\bdecoder block\s+(\d+)\b",
        r"\bvit layer\s+(\d+)\b",
        r"\bblock\s+(\d+)\b",
        r"\blayer\s+(\d+)\b",
    ):
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))
    return None


def _index(op: dict[str, Any] | None) -> int:
    if not op:
        return 10**12
    for key in ("topological_index", "source_index"):
        value = op.get(key)
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except Exception:
            pass
    digits = re.findall(r"\d+", str(op.get("op_id", "")))
    return int(digits[-1]) if digits else 10**12


def _source(op: dict[str, Any] | None) -> str:
    return str((op or {}).get("source_name", ""))


def _op_ref(op: dict[str, Any]) -> dict[str, Any]:
    return {
        "op_id": op.get("op_id", ""),
        "source_name": op.get("source_name", ""),
        "op_type": op.get("op_type", ""),
        "topological_index": op.get("topological_index"),
    }


def _dedup_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for op in sorted([item for item in ops if item], key=_index):
        key = op.get("op_id") or op.get("source_name")
        if key and key not in seen:
            seen.add(key)
            out.append(_op_ref(op))
    return out


def _op_range(ops: list[dict[str, Any]]) -> str:
    indices = [_index(op) for op in ops if _index(op) < 10**12]
    return f"{min(indices)}-{max(indices)}" if indices else ""


def _block_name(family: str, index: int) -> str:
    if family == "gpt2":
        return f"GPT-2 Block {index}"
    if family == "opt":
        return f"OPT Decoder Block {index}"
    if family == "vit":
        return f"ViT Layer {index}"
    if family == "distilbert":
        return f"DistilBERT Layer {index}"
    if family == "bert":
        return f"BERT Layer {index}"
    return f"Transformer Block {index}"


def _block_kind(family: str) -> str:
    if family in {"bert", "distilbert"}:
        return "encoder_layer"
    if family in {"opt", "gpt2"}:
        return "decoder_layer"
    if family == "vit":
        return "vit_encoder_layer"
    return "unknown"


def _kind_order(family: str) -> dict[str, int]:
    if family in {"bert", "distilbert"}:
        order = [
            "attention_query_projection",
            "attention_key_projection",
            "attention_value_projection",
            "attention_skeleton",
            "attention_score_matmul",
            "attention_mask_add",
            "attention_softmax",
            "attention_context_matmul",
            "attention_output_projection",
            "residual_merge",
            "layer_norm",
            "mlp_block",
            "mlp_expansion_projection",
            "mlp_activation",
            "mlp_contraction_projection",
        ]
    elif family == "gpt2":
        order = [
            "layer_norm",
            "attention_qkv_projection",
            "attention_score_matmul",
            "attention_mask_add",
            "attention_softmax",
            "attention_context_matmul",
            "attention_output_projection",
            "residual_merge",
            "mlp_block",
            "mlp_expansion_projection",
            "mlp_activation",
            "mlp_contraction_projection",
        ]
    elif family == "vit":
        order = [
            "layer_norm",
            "attention_query_projection",
            "attention_key_projection",
            "attention_value_projection",
            "attention_score_matmul",
            "attention_softmax",
            "attention_context_matmul",
            "attention_output_projection",
            "residual_merge",
            "mlp_block",
            "mlp_expansion_projection",
            "mlp_activation",
            "mlp_contraction_projection",
        ]
    else:
        order = [
            "attention_query_projection",
            "attention_key_projection",
            "attention_value_projection",
            "attention_qkv_projection",
            "attention_score_matmul",
            "attention_mask_add",
            "attention_softmax",
            "attention_context_matmul",
            "attention_output_projection",
            "residual_merge",
            "layer_norm",
            "mlp_block",
            "mlp_expansion_projection",
            "mlp_activation",
            "mlp_contraction_projection",
        ]
    return {name: idx for idx, name in enumerate(order)}


def _contains_any(source: str, tokens: tuple[str, ...]) -> bool:
    return any(token in source for token in tokens)


def _attention_kind(op: dict[str, Any]) -> tuple[str, str, str] | None:
    source = normalize_source(_source(op))
    category = str(op.get("semantic_category", ""))
    kind = str(op.get("semantic_kind", ""))
    if category == "query_projection" or _contains_any(source, ("/attention/self/query/", "/self_attn/q_proj/", "/attention/attention/query/")):
        return "attention_query_projection", "query_projection", "Query Projection"
    if category == "key_projection" or _contains_any(source, ("/attention/self/key/", "/self_attn/k_proj/", "/attention/attention/key/")):
        return "attention_key_projection", "key_projection", "Key Projection"
    if category == "value_projection" or _contains_any(source, ("/attention/self/value/", "/self_attn/v_proj/", "/attention/attention/value/")):
        return "attention_value_projection", "value_projection", "Value Projection"
    if ("/attn/c_attn/" in source or "/attention/c_attn/" in source) and op.get("semantic_category") == "parameterized_projection":
        return "attention_qkv_projection", "attention_qkv_projection", "Attention QKV Projection"
    if category == "attention_output_projection" or (
        "attention" in source and _contains_any(source, ("/output/dense/", "/self_attn/out_proj/", "/attn/c_proj/"))
    ):
        return "attention_output_projection", "attention_output_projection", "Attention Output Projection"
    if category == "attention_score_matmul" or kind == "attention_score_matmul":
        return "attention_score_matmul", "attention_score_matmul", "Attention Score MatMul"
    if category == "attention_context_matmul" or kind == "attention_context_matmul":
        return "attention_context_matmul", "attention_context_matmul", "Attention Context MatMul"
    if category == "attention_softmax" or kind == "attention_softmax":
        return "attention_softmax", "attention_softmax", "Attention Softmax"
    if category == "attention_mask_add" or kind == "attention_mask_add":
        return "attention_mask_add", "attention_mask_add", "Attention Mask Add"
    if category == "attention_skeleton":
        return "attention_skeleton", "attention_skeleton", "Attention"
    return None


def _simple_group(
    *,
    block_name: str,
    block_index: int,
    family: str,
    group_kind: str,
    semantic_category: str,
    title: str,
    ops: list[dict[str, Any]],
    pruning_class: str,
    why_no_plan: str,
    explanation: str,
) -> GenericSubgraphGroup:
    display = f"{block_name} {title}"
    return GenericSubgraphGroup(
        group_id=f"{family}::{block_index}::{group_kind}::{semantic_category}",
        ordinal=0,
        display_name=display,
        group_kind=group_kind,
        semantic_category=semantic_category,
        source_ops=_dedup_ops(ops),
        op_range=_op_range(ops),
        pruning_class=pruning_class,
        plan_status="no_plan_expected",
        validation_status="not_applicable",
        why_no_plan=why_no_plan,
        explanation=explanation,
    )


def _mlp_candidate(candidates: list[dict[str, Any]], match: GenericMLPMatch) -> dict[str, Any] | None:
    source_names = {_source(op) for op in match.source_ops}
    for candidate in candidates:
        if candidate.get("candidate_kind") != "feedforward_intermediate_pruning":
            continue
        evidence = {item.get("source_name", "") for item in candidate.get("op_semantics_evidence", [])}
        if evidence & source_names:
            return candidate
        if str(match.layer_index) in str(candidate.get("region_name", "")) and candidate.get("semantic_category") in {"feed_forward_block", "mlp_block"}:
            return candidate
    return None


def _plan_for_candidate(plans: list[dict[str, Any]], candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    candidate_id = candidate.get("candidate_id")
    for plan in plans:
        if plan.get("candidate_id") == candidate_id:
            return plan
    return None


def _validation_for_plan(validations: list[dict[str, Any]], plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if not plan:
        return None
    plan_id = plan.get("plan_id")
    for validation in validations:
        if validation.get("plan_id") == plan_id:
            return validation
    return None


def _mlp_block_group(
    block_name: str,
    family: str,
    match: GenericMLPMatch,
    candidates: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    validations: list[dict[str, Any]],
) -> GenericSubgraphGroup:
    candidate = _mlp_candidate(candidates, match)
    plan = _plan_for_candidate(plans, candidate)
    validation = _validation_for_plan(validations, plan)
    plan_status = "valid_plan" if validation and validation.get("validation_status") == "valid" else "no_plan_but_expected" if not plan else "unknown"
    validation_status = validation.get("validation_status", "unknown") if validation else "unknown"
    pruning_class = candidate.get("pruning_class", "safe" if match.evidence_status == "complete" else "constrained") if candidate else ("safe" if match.evidence_status == "complete" else "constrained")
    return GenericSubgraphGroup(
        group_id=f"{family}::{match.layer_index}::mlp_block",
        ordinal=0,
        display_name=f"{block_name} MLP Block",
        group_kind="mlp_block",
        semantic_category="feed_forward_block",
        source_ops=_dedup_ops(match.source_ops),
        op_range=match.op_range,
        pruning_class=pruning_class,
        plan_status=plan_status,
        validation_status=validation_status,
        why_no_plan="" if plan else "expected FFN/MLP symbolic plan is missing from current artifacts.",
        explanation="Generic MLP block: expansion projection, index-preserving activation, and contraction projection share the intermediate_dim index set.",
    )


def _mlp_component_groups(block_name: str, family: str, match: GenericMLPMatch) -> list[GenericSubgraphGroup]:
    groups: list[GenericSubgraphGroup] = []
    if match.expansion_op:
        ops = [op for op in [match.expansion_op, match.expansion_bias_op] if op]
        groups.append(
            _simple_group(
                block_name=block_name,
                block_index=match.layer_index,
                family=family,
                group_kind="mlp_expansion_projection",
                semantic_category="ffn_intermediate_projection",
                title="MLP Expansion Projection",
                ops=ops,
                pruning_class="safe",
                why_no_plan="safe component; full plan belongs to the enclosing MLP block.",
                explanation="Expansion projection creates the prunable intermediate_dim axis.",
            )
        )
    if match.activation_ops:
        groups.append(
            _simple_group(
                block_name=block_name,
                block_index=match.layer_index,
                family=family,
                group_kind="mlp_activation",
                semantic_category="gelu_activation",
                title="MLP Activation",
                ops=match.activation_ops,
                pruning_class="auxiliary",
                why_no_plan="index-preserving activation propagation, not a standalone pruning target.",
                explanation="Activation preserves selected intermediate_dim indices through the MLP.",
            )
        )
    if match.contraction_op:
        ops = [op for op in [match.contraction_op, match.contraction_bias_op] if op]
        groups.append(
            _simple_group(
                block_name=block_name,
                block_index=match.layer_index,
                family=family,
                group_kind="mlp_contraction_projection",
                semantic_category="ffn_output_projection",
                title="MLP Contraction Projection",
                ops=ops,
                pruning_class="constrained",
                why_no_plan="consumer input repair belongs to the enclosing MLP plan; output hidden_dim remains preserved.",
                explanation="Contraction projection consumes the same intermediate_dim index set and preserves hidden_dim output.",
            )
        )
    return groups


def _residual_or_norm_groups(block_name: str, family: str, block_index: int, ops: list[dict[str, Any]]) -> tuple[list[GenericSubgraphGroup], list[GenericSubgraphGroup]]:
    residuals = [op for op in ops if op.get("semantic_kind") == "residual_add" or op.get("semantic_category") == "residual_merge"]
    norms = [op for op in ops if op.get("semantic_kind") == "layernorm" or op.get("semantic_category") == "normalization"]
    residual_groups: list[GenericSubgraphGroup] = []
    norm_groups: list[GenericSubgraphGroup] = []
    for idx, op in enumerate(sorted(residuals, key=_index), start=1):
        residual_groups.append(
            _simple_group(
                block_name=block_name,
                block_index=block_index,
                family=family,
                group_kind="residual_merge",
                semantic_category="residual_merge",
                title=f"Residual Merge {idx}" if len(residuals) > 1 else "Residual Merge",
                ops=[op],
                pruning_class="blocked",
                why_no_plan="semantic blocker; residual hidden_dim agreement is protected.",
                explanation="Residual branch merge requires hidden_dim agreement and is not a direct pruning target.",
            )
        )
    for idx, op in enumerate(sorted(norms, key=_index), start=1):
        norm_groups.append(
            _simple_group(
                block_name=block_name,
                block_index=block_index,
                family=family,
                group_kind="layer_norm",
                semantic_category="layer_norm",
                title=f"LayerNorm {idx}" if len(norms) > 1 else "LayerNorm",
                ops=[op],
                pruning_class="blocked",
                why_no_plan="semantic blocker; LayerNorm hidden_dim and gamma/beta semantics are protected.",
                explanation="LayerNorm is protected under hidden_dim pruning unless a future backend proves coordinated parameter repair.",
            )
        )
    return residual_groups, norm_groups


def detect_generic_blocks(
    model_name: str,
    op_semantics: dict[str, Any],
    region_semantics: dict[str, Any] | None = None,
    ranking: dict[str, Any] | None = None,
    plans: dict[str, Any] | None = None,
    validations: dict[str, Any] | None = None,
) -> list[GenericBlock]:
    del region_semantics
    ops = [op for op in op_semantics.get("ops", []) if block_index_from_source(_source(op)) is not None]
    ops_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for op in ops:
        layer = block_index_from_source(_source(op))
        if layer is not None:
            ops_by_layer[layer].append(op)
    matches_by_layer = {match.layer_index: match for match in detect_generic_mlp_matches(model_name, op_semantics)}
    candidates = list((ranking or {}).get("candidates", []))
    plan_list = list((plans or {}).get("plans", []))
    validation_list = list((validations or {}).get("validations", []))

    blocks: list[GenericBlock] = []
    for layer, layer_ops in sorted(ops_by_layer.items()):
        family = detect_family(model_name, _source(layer_ops[0]) if layer_ops else "")
        block_name = _block_name(family, layer)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for op in layer_ops:
            attn = _attention_kind(op)
            if attn:
                group_kind, semantic_category, _title = attn
                grouped[(group_kind, semantic_category)].append(op)

        groups: list[GenericSubgraphGroup] = []
        for (group_kind, semantic_category), group_ops in grouped.items():
            title = _attention_kind(group_ops[0])[2] if _attention_kind(group_ops[0]) else group_kind.replace("_", " ").title()
            cls = "blocked" if semantic_category in {"attention_score_matmul", "attention_context_matmul"} else "auxiliary" if semantic_category in {"attention_mask_add", "attention_softmax"} else "constrained"
            if semantic_category in {"query_projection", "key_projection", "value_projection", "attention_qkv_projection", "attention_output_projection"}:
                why = "attention projection requires head-axis mapping proof before a symbolic plan is emitted."
                explanation = f"{title} is a learned attention projection but remains constrained by attention/head-axis semantics."
            elif semantic_category in {"attention_score_matmul", "attention_context_matmul"}:
                why = "semantic blocker; attention contractions are not learned parameter projections."
                explanation = f"{title} is an attention contraction, not an independent learned pruning axis."
            else:
                why = "auxiliary attention probability or mask flow, not a direct pruning target."
                explanation = f"{title} carries attention metadata/probabilities and is not directly pruned."
            groups.append(
                _simple_group(
                    block_name=block_name,
                    block_index=layer,
                    family=family,
                    group_kind=group_kind,
                    semantic_category=semantic_category,
                    title=title,
                    ops=group_ops,
                    pruning_class=cls,
                    why_no_plan=why,
                    explanation=explanation,
                )
            )

        residual_groups, norm_groups = _residual_or_norm_groups(block_name, family, layer, layer_ops)
        groups.extend(residual_groups)
        groups.extend(norm_groups)

        match = matches_by_layer.get(layer)
        warnings: list[str] = []
        if match:
            groups.append(_mlp_block_group(block_name, family, match, candidates, plan_list, validation_list))
            groups.extend(_mlp_component_groups(block_name, family, match))
            warnings.extend(match.warnings)
        else:
            warnings.append("missing_generic_mlp_match")

        order = _kind_order(family)
        groups.sort(key=lambda item: (order.get(item.group_kind, 10_000), _index(item.source_ops[0]) if item.source_ops else 10**12, item.display_name))
        for ordinal, group in enumerate(groups, start=1):
            group.ordinal = ordinal

        prefixes = sorted({re.sub(r"/(?:attention|self_attn|attn|mlp|ffn|intermediate|output).*$", "", normalize_source(_source(op))) for op in layer_ops if _source(op)})
        blocks.append(
            GenericBlock(
                model_name=model_name,
                family=family,
                block_index=layer,
                block_kind=_block_kind(family),
                block_name=block_name,
                path_prefixes=prefixes,
                op_range=_op_range(layer_ops),
                primitive_ops=_dedup_ops(layer_ops),
                grouped_subgraphs=groups,
                mlp_match=generic_mlp_match_to_dict(match) if match else None,
                attention_groups=[group for group in groups if group.group_kind.startswith("attention_")],
                residual_groups=residual_groups,
                layernorm_groups=norm_groups,
                warnings=warnings,
            )
        )
    return blocks


def generic_block_layer_indices(model_name: str, op_semantics: dict[str, Any]) -> list[int]:
    return [block.block_index for block in detect_generic_blocks(model_name, op_semantics)]


def generic_group_to_expansion_record(block: GenericBlock, group: GenericSubgraphGroup) -> dict[str, Any]:
    return {
        "region_id": group.group_id,
        "region_type": "GenericSubgraphGroup",
        "source_region_type": "GenericSubgraphGroup",
        "semantic_category": group.semantic_category,
        "name": group.display_name,
        "section": block.block_name,
        "op_range": group.op_range,
        "recursive_primitive_leaves": [
            {
                "id": op.get("op_id", ""),
                "source_name": op.get("source_name", ""),
                "op_type": op.get("op_type", ""),
                "op_index": op.get("topological_index"),
            }
            for op in group.source_ops
        ],
        "generic_group": generic_subgraph_group_to_dict(group),
        "reason": group.explanation,
    }


def generic_layer_records(
    model_name: str,
    layer_index: int,
    op_semantics: dict[str, Any],
    region_semantics: dict[str, Any] | None = None,
    ranking: dict[str, Any] | None = None,
    plans: dict[str, Any] | None = None,
    validations: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    for block in detect_generic_blocks(model_name, op_semantics, region_semantics, ranking, plans, validations):
        if block.block_index == layer_index:
            return [generic_group_to_expansion_record(block, group) for group in block.grouped_subgraphs]
    return []
