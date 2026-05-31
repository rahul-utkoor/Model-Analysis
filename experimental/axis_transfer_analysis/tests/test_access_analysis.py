from experimental.axis_transfer_analysis.access_analysis import analyze_region
from experimental.axis_transfer_analysis.axis_relations import AxisRelationKind
from experimental.axis_transfer_analysis.examples import activation_example, qk_score_example


def test_activation_preserves_intermediate_axis() -> None:
    summary = analyze_region(activation_example().region)

    assert any(
        transfer.source_axis == "intermediate_dim"
        and transfer.target_axis == "intermediate_dim"
        and transfer.relation == AxisRelationKind.PRESERVED
        for transfer in summary.op_summaries[0].transfers
    )


def test_qk_score_marks_feature_axis_reduced_and_blocked() -> None:
    summary = analyze_region(qk_score_example().region)
    transfers = summary.op_summaries[0].transfers

    assert any(transfer.source_axis == "head_dim" and transfer.relation == AxisRelationKind.REDUCED for transfer in transfers)
    assert any(transfer.source_axis == "head_dim" and transfer.relation == AxisRelationKind.MIXED for transfer in transfers)
    assert any(transfer.relation == AxisRelationKind.BLOCKED and transfer.proof == "qk_score_contraction_mixes_channels" for transfer in transfers)
