from onnx import TensorProto, helper

from model_analysis.attention_value_path_subgraph import bind_path_to_onnx, detect_attention_value_paths


def _pair(consumer: str = "/block/self_attn/out_proj/MatMul", producer: str = "/block/self_attn/v_proj/MatMul") -> dict:
    return {
        "pair_id": "deadbranch::opt::000::attention_value",
        "model_name": "facebook/opt-125m",
        "layer_index": 0,
        "family": "opt",
        "pair_kind": "attention_value_deadness",
        "producer_op_name": producer,
        "producer_op_type": "MatMul",
        "consumer_op_name": consumer,
        "consumer_op_type": "MatMul",
        "mapping_status": "proven",
        "status": "propagatable",
        "evidence_ops": [
            {"source_name": producer, "op_type": "MatMul"},
            {"source_name": "/block/self_attn/context/MatMul", "op_type": "MatMul"},
            {"source_name": consumer, "op_type": "MatMul"},
        ],
    }


def _model():
    nodes = [
        helper.make_node("MatMul", ["X", "Wv"], ["v_raw"], name="/block/self_attn/v_proj/MatMul"),
        helper.make_node("Add", ["v_raw", "Bv"], ["v_add"], name="/block/self_attn/v_proj/Add"),
        helper.make_node("Reshape", ["v_add", "shape_v"], ["v_reshape"], name="/block/self_attn/Reshape"),
        helper.make_node("Transpose", ["v_reshape"], ["v_layout"], name="/block/self_attn/Transpose"),
        helper.make_node("MatMul", ["Prob", "v_layout"], ["context"], name="/block/self_attn/context/MatMul"),
        helper.make_node("Transpose", ["context"], ["context_layout"], name="/block/self_attn/Transpose_1"),
        helper.make_node("Reshape", ["context_layout", "shape_context"], ["out_input"], name="/block/self_attn/Reshape_1"),
        helper.make_node("MatMul", ["out_input", "Wo"], ["Y"], name="/block/self_attn/out_proj/MatMul"),
    ]
    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 4]),
        helper.make_tensor_value_info("Prob", TensorProto.FLOAT, [1, 2, 2]),
    ]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 4])]
    initializers = [
        helper.make_tensor("Wv", TensorProto.FLOAT, [4, 4], [0.0] * 16),
        helper.make_tensor("Bv", TensorProto.FLOAT, [4], [0.0] * 4),
        helper.make_tensor("shape_v", TensorProto.INT64, [3], [1, 2, 4]),
        helper.make_tensor("shape_context", TensorProto.INT64, [3], [1, 2, 4]),
        helper.make_tensor("Wo", TensorProto.FLOAT, [4, 4], [0.0] * 16),
    ]
    return helper.make_model(helper.make_graph(nodes, "value_path", inputs, outputs, initializer=initializers))


def _bert_pair() -> dict:
    producer = "/model/bert/encoder/layer.0/attention/self/value/MatMul"
    context = "/model/bert/encoder/layer.0/attention/self/MatMul_1"
    consumer = "/model/bert/encoder/layer.0/attention/output/dense/MatMul"
    return {
        **_pair(consumer, producer),
        "pair_id": "deadbranch::bert::000::attention_value",
        "model_name": "bert-base-uncased",
        "family": "bert",
        "evidence_ops": [
            {"source_name": producer, "op_type": "MatMul"},
            {"source_name": context, "op_type": "MatMul"},
            {"source_name": consumer, "op_type": "MatMul"},
        ],
    }


def _bert_model():
    nodes = [
        helper.make_node("MatMul", ["X", "Wv"], ["v_raw"], name="/model/bert/encoder/layer.0/attention/self/value/MatMul"),
        helper.make_node("Reshape", ["v_raw", "shape_v"], ["v_reshape"], name="/model/bert/encoder/layer.0/attention/self/Reshape_2"),
        helper.make_node("Transpose", ["v_reshape"], ["v_layout"], name="/model/bert/encoder/layer.0/attention/self/Transpose_1"),
        helper.make_node("MatMul", ["Prob", "v_layout"], ["context"], name="/model/bert/encoder/layer.0/attention/self/MatMul_1"),
        helper.make_node("Transpose", ["context"], ["context_layout"], name="/model/bert/encoder/layer.0/attention/self/Transpose_3"),
        helper.make_node("Reshape", ["context_layout", "shape_context"], ["out_input"], name="/model/bert/encoder/layer.0/attention/self/Reshape_3"),
        helper.make_node("MatMul", ["out_input", "Wo"], ["Y"], name="/model/bert/encoder/layer.0/attention/output/dense/MatMul"),
    ]
    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 4]),
        helper.make_tensor_value_info("Prob", TensorProto.FLOAT, [1, 2, 2]),
    ]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 4])]
    initializers = [
        helper.make_tensor("Wv", TensorProto.FLOAT, [4, 4], [0.0] * 16),
        helper.make_tensor("shape_v", TensorProto.INT64, [3], [1, 2, 4]),
        helper.make_tensor("shape_context", TensorProto.INT64, [3], [1, 2, 4]),
        helper.make_tensor("Wo", TensorProto.FLOAT, [4, 4], [0.0] * 16),
    ]
    return helper.make_model(helper.make_graph(nodes, "bert_value_path", inputs, outputs, initializer=initializers))


def test_synthetic_opt_value_path_is_seedable() -> None:
    path = detect_attention_value_paths("facebook/opt-125m", {"pairs": [_pair()]})[0]
    bind_path_to_onnx(path, _model())
    names = {op["source_name"] for op in path.source_ops}
    assert path.analysis_status == "seedable"
    assert path.axis_mapping["mapping_status"] == "proven"
    assert "/block/self_attn/v_proj/MatMul" in names
    assert "/block/self_attn/context/MatMul" in names
    assert "/block/self_attn/out_proj/MatMul" in names


def test_synthetic_bert_value_path_is_seedable() -> None:
    path = detect_attention_value_paths("bert-base-uncased", {"pairs": [_bert_pair()]})[0]
    bind_path_to_onnx(path, _bert_model())
    names = {op["source_name"] for op in path.source_ops}
    assert path.analysis_status == "seedable"
    assert path.axis_mapping["mapping_status"] == "proven"
    assert "/model/bert/encoder/layer.0/attention/self/value/MatMul" in names
    assert "/model/bert/encoder/layer.0/attention/self/MatMul_1" in names
    assert "/model/bert/encoder/layer.0/attention/output/dense/MatMul" in names


def test_missing_output_projection_is_partial() -> None:
    path = detect_attention_value_paths("facebook/opt-125m", {"pairs": [_pair("/missing/out_proj/MatMul")]})[0]
    bind_path_to_onnx(path, _model())
    assert path.analysis_status == "partial"
    assert "anchor was not found" in path.explanation


def test_fused_qkv_is_blocked() -> None:
    path = detect_attention_value_paths("gpt2", {"pairs": [_pair(producer="/block/attn/c_attn/MatMul")]})[0]
    assert path.analysis_status == "blocked"
    assert path.axis_mapping["mapping_status"] == "unproven"
    assert "Fused QKV" in path.explanation
