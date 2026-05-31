"""End-to-end loop/access-to-DFA bridge examples."""

from __future__ import annotations

from experimental.axis_transfer_analysis.examples import (
    ffn_example,
    layernorm_example,
    attention_value_path_example,
    qk_score_example,
    residual_example,
)
from experimental.pruning_analysis_bridge.bridge_ir import BridgeInput, BridgeSeedPolicy


def ffn_from_access_example() -> BridgeInput:
    source = ffn_example()
    return BridgeInput(
        "ffn-from-access",
        source.region,
        BridgeSeedPolicy(
            "ffn_consumer_input_dead",
            "consumer intermediate input",
            "seed: consumer intermediate input channel j is exactly dead",
        ),
        "The loop/access analysis proves that the intermediate axis is produced, preserved, and consumed. The bridge lowers this evidence into an FFN DFA graph and proves producer-output deadness from consumer-input deadness.",
    )


def attention_value_from_access_example() -> BridgeInput:
    source = attention_value_path_example()
    return BridgeInput(
        "attention-value-from-access",
        source.region,
        BridgeSeedPolicy(
            "attention_output_input_dead",
            "attention output-projection value-context input",
            "seed: output-projection value-context input channel j is exactly dead",
        ),
        "The loop/access analysis proves V.value_dim is preserved into Context.value_context_dim. The bridge lowers this into an attention value-path DFA graph and proves value-producer output deadness from output-projection input deadness.",
    )


def qk_blocked_from_access_example() -> BridgeInput:
    source = qk_score_example()
    return BridgeInput(
        "qk-blocked-from-access",
        source.region,
        BridgeSeedPolicy(
            "qk_query_output_pruned",
            "query projection output head_dim",
            "seed: attempt to prune query output feature channel j",
        ),
        "The loop/access analysis proves the Q/K feature axis is reduced and mixed by the score contraction. The bridge lowers this evidence into a DFA score-contraction blocker.",
    )


def residual_from_access_example() -> BridgeInput:
    source = residual_example()
    return BridgeInput(
        "residual-from-access",
        source.region,
        BridgeSeedPolicy(
            "residual_hidden_pruned",
            "residual hidden input",
            "seed: attempt to prune residual hidden channel j",
        ),
        "The loop/access analysis proves residual hidden-axis alignment. The bridge lowers this into a protected DFA boundary.",
    )


def layernorm_from_access_example() -> BridgeInput:
    source = layernorm_example()
    return BridgeInput(
        "layernorm-from-access",
        source.region,
        BridgeSeedPolicy(
            "layernorm_hidden_pruned",
            "normalized hidden input",
            "seed: attempt to prune normalized hidden channel j",
        ),
        "The loop/access analysis proves that the normalized hidden axis is protected. The bridge lowers this into a protected DFA normalization boundary.",
    )


def get_example(name: str) -> BridgeInput:
    examples = {
        "ffn-from-access": ffn_from_access_example,
        "attention-value-from-access": attention_value_from_access_example,
        "qk-blocked-from-access": qk_blocked_from_access_example,
        "residual-from-access": residual_from_access_example,
        "layernorm-from-access": layernorm_from_access_example,
    }
    try:
        return examples[name]()
    except KeyError as exc:
        raise ValueError(f"unknown example: {name}") from exc
