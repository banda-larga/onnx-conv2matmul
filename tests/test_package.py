import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from onnx_conv2matmul import (
    DEFAULT_VALIDATION_SEED,
    RewriteOptions,
    __version__,
    compare_tensors,
    hello,
    rewrite_conv1x1_to_matmul,
)
from onnx_conv2matmul.cli import main


def test_hello() -> None:
    assert hello() == "Hello from onnx-conv2matmul!"


def test_version_format() -> None:
    assert __version__.count(".") == 2


def test_default_validation_seed_is_stable() -> None:
    assert DEFAULT_VALIDATION_SEED == 102


def _build_conv1x1_model(
    *,
    stride: Tuple[int, int] = (1, 1),
    pads: Tuple[int, int, int, int] = (0, 0, 0, 0),
    dilations: Tuple[int, int] = (1, 1),
    tensor_dtype: int = TensorProto.FLOAT,
) -> onnx.ModelProto:
    in_h = 4
    in_w = 5
    np_dtype = np.float16 if tensor_dtype == TensorProto.FLOAT16 else np.float32
    x = helper.make_tensor_value_info("x", tensor_dtype, [1, 3, 4, 5])

    out_h = ((in_h + pads[0] + pads[2] - 1) // stride[0]) + 1
    out_w = ((in_w + pads[1] + pads[3] - 1) // stride[1]) + 1
    y = helper.make_tensor_value_info("y", tensor_dtype, [1, 2, out_h, out_w])

    w = np.array(
        [
            [[[0.10]], [[0.20]], [[0.30]]],
            [[[0.40]], [[0.50]], [[0.60]]],
        ],
        dtype=np_dtype,
    )
    b = np.array([0.7, -0.3], dtype=np_dtype)

    w_init = numpy_helper.from_array(w, name="w")
    b_init = numpy_helper.from_array(b, name="b")

    conv = helper.make_node(
        "Conv",
        ["x", "w", "b"],
        ["y"],
        strides=list(stride),
        pads=list(pads),
        dilations=list(dilations),
        group=1,
    )
    graph = helper.make_graph([conv], "g", [x], [y], initializer=[w_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model


def _build_conv1d_pointwise_model(
    *,
    stride: Tuple[int] = (1,),
    pads: Tuple[int, int] = (0, 0),
    dilations: Tuple[int] = (1,),
) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 9])

    out_w = ((9 + pads[0] + pads[1] - 1) // stride[0]) + 1
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, out_w])

    w = np.array(
        [
            [[0.1], [0.2], [0.3]],
            [[0.4], [0.5], [0.6]],
        ],
        dtype=np.float32,
    )
    b = np.array([0.2, -0.1], dtype=np.float32)

    w_init = numpy_helper.from_array(w, name="w")
    b_init = numpy_helper.from_array(b, name="b")

    conv = helper.make_node(
        "Conv",
        ["x", "w", "b"],
        ["y"],
        strides=list(stride),
        pads=list(pads),
        dilations=list(dilations),
        group=1,
    )
    graph = helper.make_graph([conv], "g1d", [x], [y], initializer=[w_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model


def _build_audio_verify_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("audio_signal", TensorProto.FLOAT, [1, 128, "T"])
    length_in = helper.make_tensor_value_info("length", TensorProto.INT64, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, "T"])
    l_out = helper.make_tensor_value_info("length_out", TensorProto.INT64, [1])

    w = np.random.default_rng(0).normal(size=(16, 128, 1)).astype(np.float32)
    b = np.random.default_rng(1).normal(size=(16,)).astype(np.float32)
    w_init = numpy_helper.from_array(w, name="w")
    b_init = numpy_helper.from_array(b, name="b")

    conv = helper.make_node("Conv", ["audio_signal", "w", "b"], ["y"], group=1)
    passthrough = helper.make_node("Identity", ["length"], ["length_out"])
    graph = helper.make_graph(
        [conv, passthrough],
        "audio_verify",
        [x, length_in],
        [y, l_out],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model


def _run_model(model: onnx.ModelProto, x_value: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_value,) = session.run(None, {"x": x_value})
    return y_value


def _build_conv_model_for_analysis(
    *,
    kernel: Tuple[int, int] = (1, 1),
    stride: Tuple[int, int] = (1, 1),
    pads: Tuple[int, ...] = (0, 0, 0, 0),
    dilations: Tuple[int, ...] = (1, 1),
    group: int = 1,
    auto_pad: str = "NOTSET",
    constant_weight: bool = True,
    include_bias: bool = True,
    constant_bias: bool = True,
) -> onnx.ModelProto:
    in_channels = 4
    out_channels = 2

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, in_channels, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, out_channels, 5, 5])

    conv_inputs = ["x"]
    graph_inputs = [x]
    initializers: List[TensorProto] = []

    w_shape = [out_channels, in_channels // max(group, 1), kernel[0], kernel[1]]
    w_values = np.arange(np.prod(w_shape), dtype=np.float32).reshape(w_shape)
    if constant_weight:
        initializers.append(numpy_helper.from_array(w_values, name="w"))
    else:
        graph_inputs.append(
            helper.make_tensor_value_info("w", TensorProto.FLOAT, w_shape)
        )
    conv_inputs.append("w")

    if include_bias:
        b_values = np.array([0.1, -0.2], dtype=np.float32)
        if constant_bias:
            initializers.append(numpy_helper.from_array(b_values, name="b"))
        else:
            graph_inputs.append(
                helper.make_tensor_value_info("b", TensorProto.FLOAT, [out_channels])
            )
        conv_inputs.append("b")

    conv = helper.make_node(
        "Conv",
        conv_inputs,
        ["y"],
        strides=list(stride),
        pads=list(pads),
        dilations=list(dilations),
        group=group,
        auto_pad=auto_pad,
    )
    graph = helper.make_graph(
        [conv], "analysis_g", graph_inputs, [y], initializer=initializers
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_conv1x1_rewrite_equivalent() -> None:
    model = _build_conv1x1_model(stride=(1, 1))
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString())
    )

    assert report.converted == 1
    assert report.skipped == 0
    assert all(node.op_type != "Conv" for node in rewritten.graph.node)
    assert any(node.op_type == "MatMul" for node in rewritten.graph.node)

    rng = np.random.default_rng(123)
    x_value = rng.normal(size=(1, 3, 4, 5)).astype(np.float32)

    expected = _run_model(model, x_value)
    actual = _run_model(rewritten, x_value)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_conv1x1_rewrite_matmul_rhs_is_constant_initializer() -> None:
    model = _build_conv1x1_model(stride=(1, 1))
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString())
    )

    assert report.converted == 1
    init_names = {t.name for t in rewritten.graph.initializer}
    matmul_nodes = [n for n in rewritten.graph.node if n.op_type == "MatMul"]
    assert len(matmul_nodes) >= 1
    for matmul in matmul_nodes:
        assert matmul.input[1] in init_names


def test_extended_mode_pad_value_matches_weight_dtype() -> None:
    model = _build_conv1x1_model(
        stride=(2, 2),
        pads=(1, 0, 1, 0),
        tensor_dtype=TensorProto.FLOAT16,
    )
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString()),
        options=RewriteOptions(allow_extended_conv1x1=True),
    )

    assert report.converted == 1
    pad_nodes = [n for n in rewritten.graph.node if n.op_type == "Pad"]
    assert len(pad_nodes) >= 1
    pad_value_names = {n.input[2] for n in pad_nodes if len(n.input) >= 3}
    init_map = {t.name: t for t in rewritten.graph.initializer}
    assert pad_value_names
    for name in pad_value_names:
        assert init_map[name].data_type == TensorProto.FLOAT16


def test_conv1d_pointwise_rewrite_equivalent() -> None:
    model = _build_conv1d_pointwise_model()
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString())
    )

    assert report.converted == 1
    assert report.skipped == 0
    assert any(node.op_type == "MatMul" for node in rewritten.graph.node)
    assert all(node.op_type != "Conv" for node in rewritten.graph.node)

    rng = np.random.default_rng(456)
    x_value = rng.normal(size=(1, 3, 9)).astype(np.float32)

    expected = _run_model(model, x_value)
    actual = _run_model(rewritten, x_value)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_conv1d_pointwise_rewrite_matmul_rhs_is_constant_initializer() -> None:
    model = _build_conv1d_pointwise_model()
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString())
    )

    assert report.converted == 1
    init_names = {t.name for t in rewritten.graph.initializer}
    matmul_nodes = [n for n in rewritten.graph.node if n.op_type == "MatMul"]
    assert len(matmul_nodes) >= 1
    for matmul in matmul_nodes:
        assert matmul.input[1] in init_names


def test_compare_tensors_reports_symmetric_allclose_and_percentiles() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 2.1, 2.9, 4.2], dtype=np.float32)

    metrics = compare_tensors(a, b, rtol=1e-5, atol=1e-5)

    assert (
        metrics.max_abs >= metrics.p999_abs >= metrics.p99_abs >= metrics.p95_abs >= 0.0
    )
    assert metrics.mean_abs >= 0.0
    assert metrics.allclose_symmetric is False


def test_compare_tensors_detects_allclose_asymmetry() -> None:
    a = np.array([1.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)

    metrics = compare_tensors(a, b, rtol=0.75, atol=0.0)

    assert metrics.allclose_ab is True
    assert metrics.allclose_ba is False
    assert metrics.allclose_symmetric is False


def test_conv2d_pointwise_rewrite_strict_multi_seed_tolerance() -> None:
    model = _build_conv1x1_model(stride=(1, 1))
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString())
    )

    assert report.converted == 1
    for seed in (0, 1, 2, 3, 4):
        rng = np.random.default_rng(seed)
        x_value = rng.normal(size=(1, 3, 4, 5)).astype(np.float32)
        expected = _run_model(model, x_value)
        actual = _run_model(rewritten, x_value)

        diff = np.abs(actual - expected)
        assert float(diff.max()) <= 1e-5
        assert float(diff.mean()) <= 1e-6
        assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_conv1d_pointwise_rewrite_strict_multi_seed_tolerance() -> None:
    model = _build_conv1d_pointwise_model()
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString())
    )

    assert report.converted == 1
    for seed in (10, 11, 12, 13, 14):
        rng = np.random.default_rng(seed)
        x_value = rng.normal(size=(1, 3, 9)).astype(np.float32)
        expected = _run_model(model, x_value)
        actual = _run_model(rewritten, x_value)

        diff = np.abs(actual - expected)
        assert float(diff.max()) <= 1e-5
        assert float(diff.mean()) <= 1e-6
        assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_non_supported_conv_is_skipped() -> None:
    model = _build_conv1x1_model(stride=(2, 2))
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString())
    )

    assert report.converted == 0
    assert report.skipped == 1
    assert any(node.op_type == "Conv" for node in rewritten.graph.node)
    assert len(report.nodes) == 1
    assert report.nodes[0].status == "skipped"
    assert report.nodes[0].reason == "strict_mode_requires_unit_stride_and_zero_pad"


def test_cli_style_rewrite_roundtrip(tmp_path: Path) -> None:
    model = _build_conv1x1_model(stride=(1, 1))
    input_path = tmp_path / "in.onnx"
    output_path = tmp_path / "out.onnx"
    onnx.save(model, input_path)

    from onnx_conv2matmul.rewriter import rewrite_file

    report = rewrite_file(input_path, output_path)
    rewritten = onnx.load(output_path)
    assert report.converted == 1
    assert any(node.op_type == "MatMul" for node in rewritten.graph.node)


def test_extended_mode_stride_and_pad_equivalent() -> None:
    model = _build_conv1x1_model(stride=(2, 2), pads=(1, 0, 1, 0))
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString()),
        options=RewriteOptions(allow_extended_conv1x1=True),
    )

    assert report.converted == 1
    assert report.skipped == 0
    assert len(report.nodes) == 1
    assert report.nodes[0].status == "converted"
    assert report.nodes[0].reason == "converted_extended_mode"
    assert any(node.op_type == "Pad" for node in rewritten.graph.node)
    assert any(node.op_type == "Slice" for node in rewritten.graph.node)
    assert any(node.op_type == "MatMul" for node in rewritten.graph.node)

    rng = np.random.default_rng(321)
    x_value = rng.normal(size=(1, 3, 4, 5)).astype(np.float32)
    expected = _run_model(model, x_value)
    actual = _run_model(rewritten, x_value)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_non_unit_dilation_is_skipped_without_opt_in() -> None:
    model = _build_conv1x1_model(dilations=(3, 2))
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString())
    )

    assert report.converted == 0
    assert report.skipped == 1
    assert len(report.nodes) == 1
    assert report.nodes[0].status == "skipped"
    assert report.nodes[0].reason == "non_unit_dilation_requires_opt_in"
    assert any(node.op_type == "Conv" for node in rewritten.graph.node)


def test_non_unit_dilation_opt_in_equivalent() -> None:
    model = _build_conv1x1_model(dilations=(3, 2))
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString()),
        options=RewriteOptions(allow_non_unit_dilation=True),
    )

    assert report.converted == 1
    assert report.skipped == 0
    assert len(report.nodes) == 1
    assert report.nodes[0].status == "converted"
    assert report.nodes[0].reason == "converted_with_non_unit_dilation"

    rng = np.random.default_rng(777)
    x_value = rng.normal(size=(1, 3, 4, 5)).astype(np.float32)
    expected = _run_model(model, x_value)
    actual = _run_model(rewritten, x_value)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_max_dilation_guardrail_skips_when_exceeded() -> None:
    model = _build_conv1x1_model(dilations=(5, 2))
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString()),
        options=RewriteOptions(allow_non_unit_dilation=True, max_dilation=4),
    )

    assert report.converted == 0
    assert report.skipped == 1
    assert len(report.nodes) == 1
    assert report.nodes[0].status == "skipped"
    assert report.nodes[0].reason == "dilation_exceeds_max_guardrail"
    assert any(node.op_type == "Conv" for node in rewritten.graph.node)


def test_max_dilation_guardrail_allows_when_within_limit() -> None:
    model = _build_conv1x1_model(dilations=(3, 2))
    rewritten, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString()),
        options=RewriteOptions(allow_non_unit_dilation=True, max_dilation=3),
    )

    assert report.converted == 1
    assert report.skipped == 0
    assert len(report.nodes) == 1
    assert report.nodes[0].status == "converted"

    rng = np.random.default_rng(888)
    x_value = rng.normal(size=(1, 3, 4, 5)).astype(np.float32)
    expected = _run_model(model, x_value)
    actual = _run_model(rewritten, x_value)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_report_json_serializable() -> None:
    model = _build_conv1x1_model(stride=(1, 1))
    _, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString())
    )

    payload = report.to_json_dict()
    encoded = json.dumps(payload)

    assert "converted" in payload
    assert "skipped" in payload
    assert "nodes" in payload
    assert isinstance(encoded, str)


def test_group_not_1_is_skipped_with_reason() -> None:
    model = _build_conv_model_for_analysis(group=2)
    rewritten, report = rewrite_conv1x1_to_matmul(model)

    assert report.converted == 0
    assert report.skipped == 1
    assert report.nodes[0].reason == "group_not_1"
    assert any(node.op_type == "Conv" for node in rewritten.graph.node)


def test_kernel_not_1x1_is_skipped_with_reason() -> None:
    model = _build_conv_model_for_analysis(kernel=(3, 3))
    _, report = rewrite_conv1x1_to_matmul(model)

    assert report.converted == 0
    assert report.skipped == 1
    assert report.nodes[0].reason == "kernel_not_1x1"


def test_non_constant_weight_is_skipped_with_reason() -> None:
    model = _build_conv_model_for_analysis(constant_weight=False)
    _, report = rewrite_conv1x1_to_matmul(model)

    assert report.converted == 0
    assert report.skipped == 1
    assert report.nodes[0].reason == "non_constant_weight"


def test_non_constant_bias_is_skipped_with_reason() -> None:
    model = _build_conv_model_for_analysis(include_bias=True, constant_bias=False)
    _, report = rewrite_conv1x1_to_matmul(model)

    assert report.converted == 0
    assert report.skipped == 1
    assert report.nodes[0].reason == "non_constant_bias"


def test_auto_pad_not_notset_is_skipped_with_reason() -> None:
    model = _build_conv_model_for_analysis(auto_pad="SAME_UPPER")
    _, report = rewrite_conv1x1_to_matmul(model)

    assert report.converted == 0
    assert report.skipped == 1
    assert report.nodes[0].reason == "auto_pad_not_notset"


def test_extended_plus_non_unit_dilation_reports_combined_reason() -> None:
    model = _build_conv1x1_model(dilations=(2, 3))
    _, report = rewrite_conv1x1_to_matmul(
        onnx.load_from_string(model.SerializeToString()),
        options=RewriteOptions(
            allow_extended_conv1x1=True,
            allow_non_unit_dilation=True,
        ),
    )

    assert report.converted == 1
    assert report.skipped == 0
    assert report.nodes[0].reason == "converted_extended_mode_with_non_unit_dilation"


def test_rewrite_raises_for_invalid_max_dilation() -> None:
    model = _build_conv1x1_model()

    try:
        rewrite_conv1x1_to_matmul(
            onnx.load_from_string(model.SerializeToString()),
            options=RewriteOptions(max_dilation=0),
        )
        raise AssertionError("Expected ValueError for max_dilation < 1")
    except ValueError as exc:
        assert "max_dilation must be >= 1" in str(exc)


def test_cli_requires_output_or_inplace(tmp_path: Path) -> None:
    model = _build_conv1x1_model()
    input_path = tmp_path / "in.onnx"
    onnx.save(model, input_path)

    runner = CliRunner()
    result = runner.invoke(main, [input_path.as_posix()])

    assert result.exit_code != 0
    assert "Provide an output path or use --inplace." in result.output


def test_cli_rejects_invalid_max_dilation(tmp_path: Path) -> None:
    model = _build_conv1x1_model()
    input_path = tmp_path / "in.onnx"
    output_path = tmp_path / "out.onnx"
    onnx.save(model, input_path)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [input_path.as_posix(), output_path.as_posix(), "--max-dilation", "0"],
    )

    assert result.exit_code != 0
    assert "--max-dilation must be >= 1" in result.output


def test_cli_writes_json_report(tmp_path: Path) -> None:
    model = _build_conv1x1_model()
    input_path = tmp_path / "in.onnx"
    output_path = tmp_path / "out.onnx"
    report_path = tmp_path / "report.json"
    onnx.save(model, input_path)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            input_path.as_posix(),
            output_path.as_posix(),
            "--report-json",
            report_path.as_posix(),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["converted"] == 1
    assert payload["skipped"] == 0


def test_cli_handles_external_data_model(tmp_path: Path) -> None:
    model = _build_conv1x1_model()
    input_path = tmp_path / "in_ext.onnx"
    input_data_path = tmp_path / "in_ext.onnx.data"
    output_path = tmp_path / "out_ext.onnx"

    onnx.save_model(
        model,
        input_path.as_posix(),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=input_data_path.name,
        size_threshold=0,
    )

    runner = CliRunner()
    result = runner.invoke(main, [input_path.as_posix(), output_path.as_posix()])

    assert result.exit_code == 0
    assert "converted=1" in result.output
    assert output_path.exists()
    assert (tmp_path / "out_ext.onnx.data").exists()


def test_cli_rewrite_and_verify_success(tmp_path: Path) -> None:
    model = _build_audio_verify_model()
    input_path = tmp_path / "in.onnx"
    output_path = tmp_path / "out.onnx"
    onnx.save(model, input_path)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            input_path.as_posix(),
            output_path.as_posix(),
            "--extended-conv1x1",
            "--verify",
            "--verify-deterministic-cpu",
            "--verify-lengths",
            "32,47,64",
            "--verify-length-output-index",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "verify_ok=True" in result.output


def test_cli_rewrite_and_verify_failure_on_invalid_length_output_index(
    tmp_path: Path,
) -> None:
    model = _build_audio_verify_model()
    input_path = tmp_path / "in.onnx"
    output_path = tmp_path / "out.onnx"
    onnx.save(model, input_path)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            input_path.as_posix(),
            output_path.as_posix(),
            "--extended-conv1x1",
            "--verify",
            "--verify-lengths",
            "32",
            "--verify-length-output-index",
            "99",
        ],
    )

    assert result.exit_code != 0
    assert "--verify-length-output-index out of range" in result.output
