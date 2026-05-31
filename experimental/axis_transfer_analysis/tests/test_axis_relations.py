from experimental.axis_transfer_analysis.access_analysis import analyze_region
from experimental.axis_transfer_analysis.axis_relations import AxisRelationKind
from experimental.axis_transfer_analysis.examples import attention_context_example, layernorm_example, residual_example


def test_attention_context_preserves_value_axis_into_context() -> None:
    summary = analyze_region(attention_context_example().region)

    assert any(
        transfer.source_tensor == "V"
        and transfer.source_axis == "value_dim"
        and transfer.target_tensor == "Context"
        and transfer.target_axis == "value_context_dim"
        and transfer.relation == AxisRelationKind.PRESERVED
        for transfer in summary.op_summaries[0].transfers
    )


def test_residual_hidden_axis_is_protected() -> None:
    summary = analyze_region(residual_example().region)

    assert any(transfer.source_axis == "hidden_dim" and transfer.relation == AxisRelationKind.PROTECTED for transfer in summary.op_summaries[0].transfers)


def test_layernorm_hidden_axis_is_protected() -> None:
    summary = analyze_region(layernorm_example().region)

    assert any(transfer.source_axis == "hidden_dim" and transfer.relation == AxisRelationKind.PROTECTED for transfer in summary.op_summaries[0].transfers)
