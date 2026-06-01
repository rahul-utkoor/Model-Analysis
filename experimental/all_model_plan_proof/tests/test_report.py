from experimental.all_model_plan_proof.aggregate import aggregate_model_proofs
from experimental.all_model_plan_proof.config import PlanFamily, model_expectations
from experimental.all_model_plan_proof.proof_model import AllModelPlanProof, ModelPlanProof, PlanProofCell
from experimental.all_model_plan_proof.report import render_index_markdown
from experimental.all_model_plan_proof.runner import summarize_model


def test_report_contains_all_models_and_qk_explanation():
    models = []
    for expectation in model_expectations("all"):
        ffn = [PlanProofCell(expectation.model_name, expectation.artifact_name, 0, PlanFamily.FFN_INTERMEDIATE, verdict="proven")]
        attention = [PlanProofCell(expectation.model_name, expectation.artifact_name, 0, PlanFamily.ATTENTION_VALUE_PATH, verdict="unsupported")]
        summary, verdict = summarize_model(expectation, ffn, attention, [])
        models.append(ModelPlanProof(expectation.model_name, expectation.artifact_name, 1, ffn, attention, [], summary, verdict))
    report = render_index_markdown(AllModelPlanProof.create(models, aggregate_model_proofs(models), []))
    for expectation in model_expectations("all"):
        assert expectation.model_name in report
    assert "QK score contractions are blockers, not pruning plans" in report
