"""Template-lower supported ONNX pattern hints into loop/access RegionSpec records."""

from __future__ import annotations

from copy import deepcopy

from experimental.axis_transfer_analysis.examples import (
    attention_context_example,
    attention_value_path_example,
    ffn_example,
    layernorm_example,
    qk_score_example,
    residual_example,
)
from experimental.axis_transfer_analysis.loop_ir import RegionSpec
from experimental.onnx_axis_bridge.onnx_loader import OnnxSubgraph
from experimental.onnx_axis_bridge.pattern_hints import OnnxPatternHint, OnnxPatternHintKind


def _metadata(subgraph: OnnxSubgraph, hint: OnnxPatternHint) -> dict[str, object]:
    return {
        "source": "onnx_axis_bridge",
        "onnx_path": subgraph.path,
        "hint_kind": hint.kind.value,
        "evidence": list(hint.evidence),
        "confidence": hint.confidence,
        "source_onnx_nodes": list(hint.nodes),
    }


def _decorate(region: RegionSpec, subgraph: OnnxSubgraph, hint: OnnxPatternHint) -> RegionSpec:
    metadata = _metadata(subgraph, hint)
    region.region_id = f"onnx::{hint.kind.value.lower()}"
    region.label = f"ONNX lowered {hint.kind.value}"
    region.attrs.update(metadata)
    source_nodes = list(hint.nodes)
    for index, op in enumerate(region.ops):
        if source_nodes:
            if len(region.ops) == 1:
                selected = source_nodes
            elif index == 0:
                selected = [source_nodes[0]]
            elif index == len(region.ops) - 1:
                selected = [source_nodes[-1]]
            else:
                selected = source_nodes[1:-1] or source_nodes
        else:
            selected = []
        op.attrs.update(metadata)
        op.attrs["source_onnx_nodes"] = selected
        op.label = f"{op.op_id} lowered from {', '.join(selected) if selected else 'local ONNX evidence'}"
    return region


def lower_onnx_hint_to_region_spec(subgraph: OnnxSubgraph, hint: OnnxPatternHint) -> RegionSpec:
    """Lower one supported local hint into the internal loop/access template IR."""
    templates = {
        OnnxPatternHintKind.FFN_LIKE: ffn_example,
        OnnxPatternHintKind.ATTENTION_CONTEXT_LIKE: attention_context_example,
        OnnxPatternHintKind.QK_SCORE_LIKE: qk_score_example,
        OnnxPatternHintKind.ATTENTION_VALUE_PATH_LIKE: attention_value_path_example,
        OnnxPatternHintKind.RESIDUAL_LIKE: residual_example,
        OnnxPatternHintKind.LAYERNORM_LIKE: layernorm_example,
    }
    try:
        region = deepcopy(templates[hint.kind]().region)
    except KeyError as exc:
        raise ValueError(f"unsupported ONNX pattern hint: {hint.kind.value}") from exc
    return _decorate(region, subgraph, hint)
