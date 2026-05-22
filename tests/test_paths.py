from model_analysis.paths import get_hf_model_dir, get_onnx_model_dir, get_project_root, safe_model_name


def test_safe_model_name_replaces_slash():
    assert safe_model_name("facebook/opt-125m") == "facebook__opt-125m"


def test_model_paths_are_under_data_models():
    root = get_project_root()

    hf_path = get_hf_model_dir("facebook/opt-125m")
    onnx_path = get_onnx_model_dir("facebook/opt-125m")

    assert hf_path.is_relative_to(root / "data" / "models" / "hf")
    assert onnx_path.is_relative_to(root / "data" / "models" / "onnx")
