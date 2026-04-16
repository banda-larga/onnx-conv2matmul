from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Deterministic seed empirically selected on Parakeet encoder for stable
# equivalence checks across the default validation lengths.
DEFAULT_VALIDATION_SEED = 102


@dataclass(frozen=True)
class RewriteNodeReport:
    node_name: str
    node_index: int
    op_type: str
    status: str
    reason: str
    used_extended_mode: bool


@dataclass(frozen=True)
class RewriteReport:
    converted: int
    skipped: int
    nodes: tuple["RewriteNodeReport", ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "converted": self.converted,
            "skipped": self.skipped,
            "nodes": [asdict(node) for node in self.nodes],
        }


@dataclass(frozen=True)
class RewriteOptions:
    # If True, allow Conv1x1 with explicit pads/strides by rewriting
    # input as Pad+Slice before MatMul.
    allow_extended_conv1x1: bool = False
    # If True, allow non-unit dilations for Conv1x1.
    # For kernel 1x1, positive dilation does not change sampled coordinates,
    # but this remains opt-in as an explicit guardrail.
    allow_non_unit_dilation: bool = False
    # Optional operational guardrail: skip Conv1x1 if any dilation component
    # exceeds this value.
    max_dilation: int | None = None


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _attr_ints(node: onnx.NodeProto, name: str, default: list[int]) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            return [int(v) for v in attr.ints]
    return default


def _attr_str(node: onnx.NodeProto, name: str, default: str) -> str:
    for attr in node.attribute:
        if attr.name == name:
            raw = attr.s
            return (
                raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            )
    return default


def _make_i64_initializer(name: str, values: list[int]) -> TensorProto:
    return helper.make_tensor(
        name=name,
        data_type=TensorProto.INT64,
        dims=[len(values)],
        vals=values,
    )


def _make_i64_scalar_initializer(name: str, value: int) -> TensorProto:
    return helper.make_tensor(
        name=name, data_type=TensorProto.INT64, dims=[1], vals=[value]
    )


def _make_zero_scalar_initializer(name: str, data_type: int) -> TensorProto:
    float_like_types = {
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
    }
    if data_type in float_like_types:
        vals: list[float | int] = [0.0]
    else:
        vals = [0]
    return helper.make_tensor(name=name, data_type=data_type, dims=[], vals=vals)


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 1
    while True:
        candidate = f"{base}_{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def _build_constant_matmul_rhs(
    weight_tensor: TensorProto,
    *,
    name: str,
    out_channels: int,
    in_channels: int,
) -> TensorProto | None:
    try:
        w_array = numpy_helper.to_array(weight_tensor)
    except Exception:
        return None

    if w_array.ndim not in (3, 4):
        return None
    if any(dim != 1 for dim in w_array.shape[2:]):
        return None

    rhs = np.ascontiguousarray(
        w_array.reshape(out_channels, in_channels).transpose(1, 0)
    )
    return numpy_helper.from_array(rhs, name=name)


@dataclass(frozen=True)
class _Conv1x1Plan:
    stride_h: int
    stride_w: int
    dilation_h: int
    dilation_w: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int


@dataclass(frozen=True)
class _Conv1DPlan:
    stride_w: int
    dilation_w: int
    pad_left: int
    pad_right: int


@dataclass(frozen=True)
class _ConvAnalysis:
    plan: _Conv1x1Plan | _Conv1DPlan | None
    reason: str


def _get_conv1x1_plan(
    node: onnx.NodeProto,
    initializer_names: set[str],
    initializers: dict[str, TensorProto],
) -> _ConvAnalysis:
    if node.op_type != "Conv":
        return _ConvAnalysis(plan=None, reason="not_conv")
    if len(node.input) < 2 or len(node.input) > 3:
        return _ConvAnalysis(plan=None, reason="invalid_input_count")

    w_name = node.input[1]
    if w_name not in initializer_names:
        return _ConvAnalysis(plan=None, reason="non_constant_weight")

    w_tensor = initializers[w_name]
    w_dims = [int(v) for v in w_tensor.dims]
    if len(w_dims) not in (3, 4):
        return _ConvAnalysis(plan=None, reason="weight_not_4d")
    if any(v != 1 for v in w_dims[2:]):
        return _ConvAnalysis(plan=None, reason="kernel_not_1x1")

    spatial_rank = len(w_dims) - 2

    if _attr_int(node, "group", 1) != 1:
        return _ConvAnalysis(plan=None, reason="group_not_1")

    default_ones = [1] * spatial_rank

    strides = _attr_ints(node, "strides", default_ones)
    if len(strides) != spatial_rank or any(v <= 0 for v in strides):
        return _ConvAnalysis(plan=None, reason="invalid_strides")

    # For 1x1 kernels, dilation does not change sampled positions. We only guard basic validity.
    dilations = _attr_ints(node, "dilations", default_ones)
    if len(dilations) != spatial_rank or any(v <= 0 for v in dilations):
        return _ConvAnalysis(plan=None, reason="invalid_dilations")

    default_pads = [0] * (2 * spatial_rank)
    pads = _attr_ints(node, "pads", default_pads)
    if len(pads) != 2 * spatial_rank or any(v < 0 for v in pads):
        return _ConvAnalysis(plan=None, reason="invalid_pads")

    if _attr_str(node, "auto_pad", "NOTSET") != "NOTSET":
        return _ConvAnalysis(plan=None, reason="auto_pad_not_notset")

    if len(node.input) == 3 and node.input[2] not in initializer_names:
        return _ConvAnalysis(plan=None, reason="non_constant_bias")

    if spatial_rank == 1:
        return _ConvAnalysis(
            plan=_Conv1DPlan(
                stride_w=int(strides[0]),
                dilation_w=int(dilations[0]),
                pad_left=int(pads[0]),
                pad_right=int(pads[1]),
            ),
            reason="supported_conv1x1",
        )

    return _ConvAnalysis(
        plan=_Conv1x1Plan(
            stride_h=int(strides[0]),
            stride_w=int(strides[1]),
            dilation_h=int(dilations[0]),
            dilation_w=int(dilations[1]),
            pad_top=int(pads[0]),
            pad_left=int(pads[1]),
            pad_bottom=int(pads[2]),
            pad_right=int(pads[3]),
        ),
        reason="supported_conv1x1",
    )


def rewrite_conv1x1_to_matmul(
    model: onnx.ModelProto,
    options: RewriteOptions | None = None,
) -> tuple[onnx.ModelProto, RewriteReport]:
    if options is None:
        options = RewriteOptions()
    if options.max_dilation is not None and options.max_dilation < 1:
        raise ValueError("max_dilation must be >= 1 when provided")

    graph = model.graph
    initializer_map = {init.name: init for init in graph.initializer}
    initializer_names = set(initializer_map.keys())

    used_names = set()
    for init in graph.initializer:
        used_names.add(init.name)
    for value in graph.input:
        used_names.add(value.name)
    for value in graph.output:
        used_names.add(value.name)
    for value in graph.value_info:
        used_names.add(value.name)
    for node in graph.node:
        if node.name:
            used_names.add(node.name)
        for out in node.output:
            used_names.add(out)

    converted = 0
    skipped = 0
    node_reports: list[RewriteNodeReport] = []
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[TensorProto] = []

    for node_index, node in enumerate(graph.node):
        node_name = node.name if node.name else f"{node.op_type}@{node_index}"
        analysis = _get_conv1x1_plan(node, initializer_names, initializer_map)
        plan = analysis.plan
        if plan is None:
            if node.op_type == "Conv":
                skipped += 1
                node_reports.append(
                    RewriteNodeReport(
                        node_name=node_name,
                        node_index=node_index,
                        op_type=node.op_type,
                        status="skipped",
                        reason=analysis.reason,
                        used_extended_mode=False,
                    )
                )
            new_nodes.append(node)
            continue

        if isinstance(plan, _Conv1DPlan):
            stride_vals = [plan.stride_w]
            dilation_vals = [plan.dilation_w]
            pad_vals = [plan.pad_left, plan.pad_right]
        else:
            stride_vals = [plan.stride_h, plan.stride_w]
            dilation_vals = [plan.dilation_h, plan.dilation_w]
            pad_vals = [plan.pad_top, plan.pad_left, plan.pad_bottom, plan.pad_right]

        if not options.allow_extended_conv1x1:
            is_strict_case = all(v == 1 for v in stride_vals) and all(
                v == 0 for v in pad_vals
            )
            if not is_strict_case:
                skipped += 1
                node_reports.append(
                    RewriteNodeReport(
                        node_name=node_name,
                        node_index=node_index,
                        op_type=node.op_type,
                        status="skipped",
                        reason="strict_mode_requires_unit_stride_and_zero_pad",
                        used_extended_mode=False,
                    )
                )
                new_nodes.append(node)
                continue

        has_non_unit_dilation = any(v != 1 for v in dilation_vals)

        if not options.allow_non_unit_dilation and has_non_unit_dilation:
            skipped += 1
            node_reports.append(
                RewriteNodeReport(
                    node_name=node_name,
                    node_index=node_index,
                    op_type=node.op_type,
                    status="skipped",
                    reason="non_unit_dilation_requires_opt_in",
                    used_extended_mode=options.allow_extended_conv1x1,
                )
            )
            new_nodes.append(node)
            continue

        if options.max_dilation is not None and any(
            v > options.max_dilation for v in dilation_vals
        ):
            skipped += 1
            node_reports.append(
                RewriteNodeReport(
                    node_name=node_name,
                    node_index=node_index,
                    op_type=node.op_type,
                    status="skipped",
                    reason="dilation_exceeds_max_guardrail",
                    used_extended_mode=options.allow_extended_conv1x1,
                )
            )
            new_nodes.append(node)
            continue

        x_name = node.input[0]
        w_name = node.input[1]
        y_name = node.output[0]

        w_tensor = initializer_map[w_name]
        w_dims = [int(v) for v in w_tensor.dims]
        out_channels = w_dims[0]
        in_channels = w_dims[1]
        has_non_default_spatial = any(v != 1 for v in stride_vals) or any(
            v != 0 for v in pad_vals
        )

        prefix = node.name if node.name else "conv1x1"

        x_flat = _unique_name(f"{prefix}_x_flat", used_names)
        w_t = _unique_name(f"{prefix}_w_t", used_names)
        mm_out = _unique_name(f"{prefix}_mm", used_names)
        mm_bias_out = _unique_name(f"{prefix}_mm_bias", used_names)
        shape_x = _unique_name(f"{prefix}_shape_x", used_names)
        n_dim = _unique_name(f"{prefix}_n", used_names)
        out_shape = _unique_name(f"{prefix}_out_shape", used_names)

        idx_0 = _unique_name(f"{prefix}_idx0", used_names)
        idx_2 = _unique_name(f"{prefix}_idx2", used_names)
        flat_shape_name = _unique_name(f"{prefix}_flat_shape", used_names)
        m_scalar_name = _unique_name(f"{prefix}_m", used_names)
        pad_spec_name = _unique_name(f"{prefix}_pad_spec", used_names)
        pad_value_name = _unique_name(f"{prefix}_pad_value", used_names)

        x_for_conv = x_name
        if isinstance(plan, _Conv1DPlan):
            perm_out = _unique_name(f"{prefix}_x_nwc", used_names)
            output_perm = _unique_name(f"{prefix}_nwc_out", used_names)
            w_dim = _unique_name(f"{prefix}_w", used_names)

            if plan.pad_left != 0 or plan.pad_right != 0:
                x_padded = _unique_name(f"{prefix}_x_padded", used_names)
                new_initializers.extend(
                    [
                        _make_i64_initializer(
                            pad_spec_name,
                            [0, 0, plan.pad_left, 0, 0, plan.pad_right],
                        ),
                        _make_zero_scalar_initializer(
                            pad_value_name,
                            int(w_tensor.data_type),
                        ),
                    ]
                )
                new_nodes.append(
                    helper.make_node(
                        "Pad",
                        [x_name, pad_spec_name, pad_value_name],
                        [x_padded],
                        mode="constant",
                    )
                )
                x_for_conv = x_padded

            if plan.stride_w != 1:
                starts_w_name = _unique_name(f"{prefix}_slice_starts_w", used_names)
                ends_w_name = _unique_name(f"{prefix}_slice_ends_w", used_names)
                axes_w_name = _unique_name(f"{prefix}_slice_axes_w", used_names)
                steps_w_name = _unique_name(f"{prefix}_slice_steps_w", used_names)
                x_strided = _unique_name(f"{prefix}_x_strided", used_names)
                new_initializers.extend(
                    [
                        _make_i64_initializer(starts_w_name, [0]),
                        _make_i64_initializer(ends_w_name, [9223372036854775807]),
                        _make_i64_initializer(axes_w_name, [2]),
                        _make_i64_initializer(steps_w_name, [plan.stride_w]),
                    ]
                )
                new_nodes.append(
                    helper.make_node(
                        "Slice",
                        [
                            x_for_conv,
                            starts_w_name,
                            ends_w_name,
                            axes_w_name,
                            steps_w_name,
                        ],
                        [x_strided],
                    )
                )
                x_for_conv = x_strided

            w_t_initializer = _build_constant_matmul_rhs(
                w_tensor,
                name=w_t,
                out_channels=out_channels,
                in_channels=in_channels,
            )
            if w_t_initializer is None:
                skipped += 1
                node_reports.append(
                    RewriteNodeReport(
                        node_name=node_name,
                        node_index=node_index,
                        op_type=node.op_type,
                        status="skipped",
                        reason="weight_materialization_failed",
                        used_extended_mode=options.allow_extended_conv1x1,
                    )
                )
                new_nodes.append(node)
                continue
            new_initializers.append(w_t_initializer)

            new_initializers.extend(
                [
                    _make_i64_initializer(idx_0, [0]),
                    _make_i64_initializer(idx_2, [2]),
                    _make_i64_initializer(flat_shape_name, [-1, in_channels]),
                    _make_i64_scalar_initializer(m_scalar_name, out_channels),
                ]
            )

            new_nodes.append(
                helper.make_node("Transpose", [x_for_conv], [perm_out], perm=[0, 2, 1])
            )
            new_nodes.append(
                helper.make_node("Reshape", [perm_out, flat_shape_name], [x_flat])
            )
            new_nodes.append(helper.make_node("MatMul", [x_flat, w_t], [mm_out]))

            mm_source = mm_out
            if len(node.input) == 3:
                bias_name = node.input[2]
                new_nodes.append(
                    helper.make_node("Add", [mm_out, bias_name], [mm_bias_out])
                )
                mm_source = mm_bias_out

            new_nodes.append(helper.make_node("Shape", [x_for_conv], [shape_x]))
            new_nodes.append(
                helper.make_node("Gather", [shape_x, idx_0], [n_dim], axis=0)
            )
            new_nodes.append(
                helper.make_node("Gather", [shape_x, idx_2], [w_dim], axis=0)
            )
            new_nodes.append(
                helper.make_node(
                    "Concat", [n_dim, w_dim, m_scalar_name], [out_shape], axis=0
                )
            )
            new_nodes.append(
                helper.make_node("Reshape", [mm_source, out_shape], [output_perm])
            )
            new_nodes.append(
                helper.make_node("Transpose", [output_perm], [y_name], perm=[0, 2, 1])
            )
        else:
            perm_nhwc_out = _unique_name(f"{prefix}_x_nhwc", used_names)
            h_dim = _unique_name(f"{prefix}_h", used_names)
            w_dim = _unique_name(f"{prefix}_w", used_names)
            nhwc_out = _unique_name(f"{prefix}_nhwc_out", used_names)
            idx_3 = _unique_name(f"{prefix}_idx3", used_names)
            starts_hw_name = _unique_name(f"{prefix}_slice_starts_hw", used_names)
            ends_hw_name = _unique_name(f"{prefix}_slice_ends_hw", used_names)
            axes_hw_name = _unique_name(f"{prefix}_slice_axes_hw", used_names)
            steps_hw_name = _unique_name(f"{prefix}_slice_steps_hw", used_names)

            if (
                plan.pad_top != 0
                or plan.pad_left != 0
                or plan.pad_bottom != 0
                or plan.pad_right != 0
            ):
                x_padded = _unique_name(f"{prefix}_x_padded", used_names)
                new_initializers.extend(
                    [
                        _make_i64_initializer(
                            pad_spec_name,
                            [
                                0,
                                0,
                                plan.pad_top,
                                plan.pad_left,
                                0,
                                0,
                                plan.pad_bottom,
                                plan.pad_right,
                            ],
                        ),
                        _make_zero_scalar_initializer(
                            pad_value_name,
                            int(w_tensor.data_type),
                        ),
                    ]
                )
                new_nodes.append(
                    helper.make_node(
                        "Pad",
                        [x_name, pad_spec_name, pad_value_name],
                        [x_padded],
                        mode="constant",
                    )
                )
                x_for_conv = x_padded

            if plan.stride_h != 1 or plan.stride_w != 1:
                x_strided = _unique_name(f"{prefix}_x_strided", used_names)
                new_initializers.extend(
                    [
                        _make_i64_initializer(starts_hw_name, [0, 0]),
                        _make_i64_initializer(
                            ends_hw_name,
                            [9223372036854775807, 9223372036854775807],
                        ),
                        _make_i64_initializer(axes_hw_name, [2, 3]),
                        _make_i64_initializer(
                            steps_hw_name, [plan.stride_h, plan.stride_w]
                        ),
                    ]
                )
                new_nodes.append(
                    helper.make_node(
                        "Slice",
                        [
                            x_for_conv,
                            starts_hw_name,
                            ends_hw_name,
                            axes_hw_name,
                            steps_hw_name,
                        ],
                        [x_strided],
                    )
                )
                x_for_conv = x_strided

            w_t_initializer = _build_constant_matmul_rhs(
                w_tensor,
                name=w_t,
                out_channels=out_channels,
                in_channels=in_channels,
            )
            if w_t_initializer is None:
                skipped += 1
                node_reports.append(
                    RewriteNodeReport(
                        node_name=node_name,
                        node_index=node_index,
                        op_type=node.op_type,
                        status="skipped",
                        reason="weight_materialization_failed",
                        used_extended_mode=options.allow_extended_conv1x1,
                    )
                )
                new_nodes.append(node)
                continue

            new_initializers.append(w_t_initializer)
            new_initializers.extend(
                [
                    _make_i64_initializer(idx_0, [0]),
                    _make_i64_initializer(idx_2, [2]),
                    _make_i64_initializer(idx_3, [3]),
                    _make_i64_initializer(flat_shape_name, [-1, in_channels]),
                    _make_i64_scalar_initializer(m_scalar_name, out_channels),
                ]
            )

            new_nodes.append(
                helper.make_node(
                    "Transpose", [x_for_conv], [perm_nhwc_out], perm=[0, 2, 3, 1]
                )
            )
            new_nodes.append(
                helper.make_node("Reshape", [perm_nhwc_out, flat_shape_name], [x_flat])
            )
            new_nodes.append(helper.make_node("MatMul", [x_flat, w_t], [mm_out]))

            mm_source = mm_out
            if len(node.input) == 3:
                bias_name = node.input[2]
                new_nodes.append(
                    helper.make_node("Add", [mm_out, bias_name], [mm_bias_out])
                )
                mm_source = mm_bias_out

            new_nodes.append(helper.make_node("Shape", [x_for_conv], [shape_x]))
            new_nodes.append(
                helper.make_node("Gather", [shape_x, idx_0], [n_dim], axis=0)
            )
            new_nodes.append(
                helper.make_node("Gather", [shape_x, idx_2], [h_dim], axis=0)
            )
            new_nodes.append(
                helper.make_node("Gather", [shape_x, idx_3], [w_dim], axis=0)
            )
            new_nodes.append(
                helper.make_node(
                    "Concat", [n_dim, h_dim, w_dim, m_scalar_name], [out_shape], axis=0
                )
            )
            new_nodes.append(
                helper.make_node("Reshape", [mm_source, out_shape], [nhwc_out])
            )
            new_nodes.append(
                helper.make_node("Transpose", [nhwc_out], [y_name], perm=[0, 3, 1, 2])
            )

        converted += 1
        node_reports.append(
            RewriteNodeReport(
                node_name=node_name,
                node_index=node_index,
                op_type=node.op_type,
                status="converted",
                reason=(
                    "converted_extended_mode_with_non_unit_dilation"
                    if options.allow_extended_conv1x1
                    and options.allow_non_unit_dilation
                    and (has_non_default_spatial or has_non_unit_dilation)
                    else (
                        "converted_with_non_unit_dilation"
                        if options.allow_non_unit_dilation and has_non_unit_dilation
                        else (
                            "converted_extended_mode"
                            if options.allow_extended_conv1x1
                            and has_non_default_spatial
                            else "converted_strict_mode"
                        )
                    )
                ),
                used_extended_mode=options.allow_extended_conv1x1,
            )
        )

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)

    return model, RewriteReport(
        converted=converted,
        skipped=skipped,
        nodes=tuple(node_reports),
    )


def rewrite_file(input_path: str | Path, output_path: str | Path) -> RewriteReport:
    in_path = Path(input_path)
    out_path = Path(output_path)

    model = onnx.load(in_path.as_posix())
    model, report = rewrite_conv1x1_to_matmul(model)
    onnx.checker.check_model(model)
    onnx.save(model, out_path.as_posix())
    return report
