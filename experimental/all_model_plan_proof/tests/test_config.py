from experimental.all_model_plan_proof.config import AttentionValuePolicy, model_expectations


def test_config_has_five_models():
    assert len(model_expectations("all")) == 5


def test_expected_counts():
    models = {model.short_name: model for model in model_expectations("all")}
    assert models["bert"].total_expected == 24
    assert models["opt"].total_expected == 24
    assert models["distilbert"].total_expected == 12
    assert models["gpt2"].attention_value_policy == AttentionValuePolicy.REQUIRED_IF_SEPARABLE
    assert models["vit"].attention_value_policy == AttentionValuePolicy.REQUIRED_IF_SEPARABLE
