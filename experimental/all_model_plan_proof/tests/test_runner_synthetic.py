from experimental.all_model_plan_proof.config import model_expectations
from experimental.all_model_plan_proof.proof_model import PlanProofCell
from experimental.all_model_plan_proof.runner import summarize_model
from experimental.all_model_plan_proof.config import PlanFamily


def _cell(model, layer, family, verdict="proven"):
    return PlanProofCell(model.model_name, model.artifact_name, layer, family, found_artifact=True, verdict=verdict)


def test_summarize_complete_model():
    model = model_expectations("bert")[0]
    ffn = [_cell(model, layer, PlanFamily.FFN_INTERMEDIATE) for layer in range(12)]
    attention = [_cell(model, layer, PlanFamily.ATTENTION_VALUE_PATH) for layer in range(12)]
    summary, verdict = summarize_model(model, ffn, attention, [])
    assert summary.total_expected == 24
    assert summary.total_proven == 24
    assert verdict == "complete_plan_proof"


def test_summarize_fused_qkv_gap():
    model = model_expectations("gpt2")[0]
    ffn = [_cell(model, layer, PlanFamily.FFN_INTERMEDIATE) for layer in range(12)]
    attention = [_cell(model, layer, PlanFamily.ATTENTION_VALUE_PATH, "unsupported") for layer in range(12)]
    summary, verdict = summarize_model(model, ffn, attention, [])
    assert summary.unsupported_count == 12
    assert verdict == "unsupported_attention_value_path"
