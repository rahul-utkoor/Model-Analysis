from experimental.all_model_plan_proof.aggregate import aggregate_model_proofs
from experimental.all_model_plan_proof.config import PlanFamily, model_expectations
from experimental.all_model_plan_proof.proof_model import ModelPlanProof, PlanProofCell
from experimental.all_model_plan_proof.runner import summarize_model


def _proof(short_name, attention_verdict="proven"):
    model = model_expectations(short_name)[0]
    ffn = [PlanProofCell(model.model_name, model.artifact_name, layer, PlanFamily.FFN_INTERMEDIATE, verdict="proven") for layer in range(model.layer_count)]
    attention = [PlanProofCell(model.model_name, model.artifact_name, layer, PlanFamily.ATTENTION_VALUE_PATH, verdict=attention_verdict) for layer in range(model.layer_count)]
    summary, verdict = summarize_model(model, ffn, attention, [])
    return ModelPlanProof(model.model_name, model.artifact_name, model.layer_count, ffn, attention, [], summary, verdict)


def test_aggregate_synthetic_complete():
    aggregate = aggregate_model_proofs([_proof("bert"), _proof("opt")])
    assert aggregate.total_expected == 48
    assert aggregate.total_proven == 48


def test_aggregate_synthetic_partial():
    aggregate = aggregate_model_proofs([_proof("gpt2", "unsupported"), _proof("vit", "unsupported")])
    assert aggregate.total_expected == 48
    assert aggregate.total_proven == 24
    assert aggregate.unsupported_count == 24
