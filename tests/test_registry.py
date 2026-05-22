from model_analysis.registry import get_model_config, load_model_registry, resolve_model_name


def test_registry_loads_five_models():
    models = load_model_registry()
    assert len(models) == 5


def test_resolve_model_name_accepts_short_name_and_hf_id():
    assert resolve_model_name("opt-125m") == "opt-125m"
    assert resolve_model_name("facebook/opt-125m") == "opt-125m"


def test_each_model_has_required_fields():
    for model in load_model_registry():
        assert model["hf_id"]
        assert model["task"]
        assert model["local_dir"]
        assert model["onnx_dir"]


def test_get_model_config_resolves_hf_id():
    config = get_model_config("facebook/opt-125m")
    assert config["name"] == "opt-125m"
