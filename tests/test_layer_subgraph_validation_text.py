from __future__ import annotations

from model_analysis.layer_subgraph_validation_text import layer_subgraph_pack_to_text
from test_layer_subgraph_validation_pack import build_pack


def test_text_dump_contains_subgraphs_and_classes() -> None:
    text = layer_subgraph_pack_to_text(build_pack())

    assert "layer_subgraph_validation @bert-base-uncased" in text
    assert "Layer 0 Feed Forward" in text
    assert "pruning_class = safe" in text
