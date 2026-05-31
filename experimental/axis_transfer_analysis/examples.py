"""Loop/access examples for the axis-transfer analysis prototype."""

from __future__ import annotations

from dataclasses import dataclass

from experimental.axis_transfer_analysis.loop_ir import LoopIV, OperationSpec, RegionSpec, Tensor, TensorAccess


@dataclass(frozen=True)
class Example:
    name: str
    description: str
    region: RegionSpec
    interpretation: str


def _tensor(name: str, *axes: str) -> Tensor:
    return Tensor(name, len(axes), tuple(axes))


def _read(tensor: str, *indices: str) -> TensorAccess:
    return TensorAccess(tensor, tuple(indices), "read")


def _write(tensor: str, *indices: str, update: bool = False) -> TensorAccess:
    return TensorAccess(tensor, tuple(indices), "update" if update else "write")


def _loops(*names: str) -> list[LoopIV]:
    return [LoopIV(name) for name in names]


def activation_example() -> Example:
    tensors = {
        "X": _tensor("X", "batch", "sequence", "intermediate_dim"),
        "Y": _tensor("Y", "batch", "sequence", "intermediate_dim"),
    }
    op = OperationSpec("activation", "Elementwise activation", "unary_elementwise", _loops("b", "s", "j"), [_read("X", "b", "s", "j")], [_write("Y", "b", "s", "j")])
    return Example(
        "activation",
        "Elementwise activation: Y[b,s,j] = gelu(X[b,s,j]).",
        RegionSpec("activation", "Index-preserving activation", tensors, [op]),
        "The intermediate axis j is preserved because the same IV indexes the input and output.",
    )


def ffn_example() -> Example:
    tensors = {
        "X": _tensor("X", "batch", "sequence", "hidden_dim"),
        "W_expand": _tensor("W_expand", "hidden_dim", "intermediate_dim"),
        "Intermediate": _tensor("Intermediate", "batch", "sequence", "intermediate_dim"),
        "Activated": _tensor("Activated", "batch", "sequence", "intermediate_dim"),
        "W_contract": _tensor("W_contract", "intermediate_dim", "hidden_dim"),
        "Y": _tensor("Y", "batch", "sequence", "hidden_dim"),
    }
    ops = [
        OperationSpec(
            "expand",
            "First projection",
            "matmul",
            _loops("b", "s", "h", "j"),
            [_read("X", "b", "s", "h"), _read("W_expand", "h", "j")],
            [_write("Intermediate", "b", "s", "j", update=True)],
        ),
        OperationSpec(
            "activation",
            "Index-preserving activation",
            "unary_elementwise",
            _loops("b", "s", "j"),
            [_read("Intermediate", "b", "s", "j")],
            [_write("Activated", "b", "s", "j")],
        ),
        OperationSpec(
            "contract",
            "Second projection",
            "matmul",
            _loops("b", "s", "j", "h"),
            [_read("Activated", "b", "s", "j"), _read("W_contract", "j", "h")],
            [_write("Y", "b", "s", "h", update=True)],
        ),
    ]
    return Example(
        "ffn",
        "MLP chain expressed as indexed projection, activation, and contraction accesses.",
        RegionSpec("ffn", "FFN intermediate chain", tensors, ops),
        "The intermediate axis is produced, preserved by the unary operation, and consumed as the second projection input-feature axis.",
    )


def qk_score_example() -> Example:
    tensors = {
        "Q": _tensor("Q", "batch", "head", "query_position", "head_dim"),
        "K": _tensor("K", "batch", "head", "key_position", "head_dim"),
        "Score": _tensor("Score", "batch", "head", "query_position", "key_position"),
    }
    op = OperationSpec(
        "qk_score",
        "QK score contraction",
        "contraction",
        _loops("b", "head", "q", "k", "d"),
        [_read("Q", "b", "head", "q", "d"), _read("K", "b", "head", "k", "d")],
        [_write("Score", "b", "head", "q", "k", update=True)],
    )
    return Example(
        "qk-score",
        "Attention scores: Score[b,head,q,k] += Q[b,head,q,d] * K[b,head,k,d].",
        RegionSpec("qk-score", "QK score contraction", tensors, [op]),
        "The feature axis d is reduced inside QK^T. Simple one-to-one pruning propagation through Q/K is blocked.",
    )


def attention_context_example() -> Example:
    tensors = {
        "P": _tensor("P", "batch", "head", "query_position", "key_position"),
        "V": _tensor("V", "batch", "head", "key_position", "value_dim"),
        "Context": _tensor("Context", "batch", "head", "query_position", "value_context_dim"),
    }
    op = OperationSpec(
        "context",
        "Attention context contraction",
        "contraction",
        _loops("b", "head", "q", "k", "d"),
        [_read("P", "b", "head", "q", "k"), _read("V", "b", "head", "k", "d")],
        [_write("Context", "b", "head", "q", "d", update=True)],
    )
    return Example(
        "attention-context",
        "Attention context: Context[b,head,q,d] += P[b,head,q,k] * V[b,head,k,d].",
        RegionSpec("attention-context", "Attention value-axis mapping", tensors, [op]),
        "The value axis d appears in V and Context and is not reduced. This proves the value-axis mapping.",
    )


def attention_value_path_example() -> Example:
    tensors = {
        "ValueInput": _tensor("ValueInput", "batch", "head", "key_position", "hidden_dim"),
        "W_value": _tensor("W_value", "hidden_dim", "value_dim"),
        "V": _tensor("V", "batch", "head", "key_position", "value_dim"),
        "P": _tensor("P", "batch", "head", "query_position", "key_position"),
        "Context": _tensor("Context", "batch", "head", "query_position", "value_context_dim"),
        "W_output": _tensor("W_output", "value_context_dim", "hidden_dim"),
        "Output": _tensor("Output", "batch", "head", "query_position", "hidden_dim"),
    }
    ops = [
        OperationSpec(
            "value_projection",
            "Value-producing projection",
            "matmul",
            _loops("b", "head", "k", "h", "d"),
            [_read("ValueInput", "b", "head", "k", "h"), _read("W_value", "h", "d")],
            [_write("V", "b", "head", "k", "d", update=True)],
        ),
        OperationSpec(
            "context",
            "Attention context contraction",
            "contraction",
            _loops("b", "head", "q", "k", "d"),
            [_read("P", "b", "head", "q", "k"), _read("V", "b", "head", "k", "d")],
            [_write("Context", "b", "head", "q", "d", update=True)],
        ),
        OperationSpec(
            "output_projection",
            "Attention output projection",
            "matmul",
            _loops("b", "head", "q", "d", "h"),
            [_read("Context", "b", "head", "q", "d"), _read("W_output", "d", "h")],
            [_write("Output", "b", "head", "q", "h", update=True)],
        ),
    ]
    return Example(
        "attention-value-path",
        "Value projection, attention context contraction, and output projection expressed through indexed accesses.",
        RegionSpec("attention-value-path", "Attention value path", tensors, ops),
        "The access relations prove that output-projection input deadness can propagate backward through Context to V.",
    )


def residual_example() -> Example:
    tensors = {
        "A": _tensor("A", "batch", "sequence", "hidden_dim"),
        "B": _tensor("B", "batch", "sequence", "hidden_dim"),
        "Y": _tensor("Y", "batch", "sequence", "hidden_dim"),
    }
    op = OperationSpec("residual", "Residual add", "residual_add", _loops("b", "s", "h"), [_read("A", "b", "s", "h"), _read("B", "b", "s", "h")], [_write("Y", "b", "s", "h")])
    return Example(
        "residual",
        "Residual merge: Y[b,s,h] = A[b,s,h] + B[b,s,h].",
        RegionSpec("residual", "Residual hidden-axis protection", tensors, [op]),
        "The hidden axis is protected because both residual branches must remain aligned.",
    )


def layernorm_example() -> Example:
    tensors = {
        "X": _tensor("X", "batch", "sequence", "hidden_dim"),
        "Y": _tensor("Y", "batch", "sequence", "hidden_dim"),
    }
    op = OperationSpec("layernorm", "LayerNorm", "layernorm", _loops("b", "s", "h"), [_read("X", "b", "s", "h")], [_write("Y", "b", "s", "h")])
    return Example(
        "layernorm",
        "LayerNorm over hidden width.",
        RegionSpec("layernorm", "LayerNorm hidden-axis protection", tensors, [op]),
        "The normalized hidden axis is protected because normalization statistics couple its elements.",
    )


def get_example(name: str) -> Example:
    examples = {
        "activation": activation_example,
        "ffn": ffn_example,
        "qk-score": qk_score_example,
        "attention-context": attention_context_example,
        "attention-value-path": attention_value_path_example,
        "residual": residual_example,
        "layernorm": layernorm_example,
    }
    try:
        return examples[name]()
    except KeyError as exc:
        raise ValueError(f"unknown example: {name}") from exc
