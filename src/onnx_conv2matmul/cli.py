import json
from collections import Counter
from pathlib import Path
from typing import Any

import click
import numpy as np
import onnx
from onnx.external_data_helper import uses_external_data
from rich.console import Console
from rich.table import Table

from .rewriter import DEFAULT_VALIDATION_SEED, RewriteOptions, rewrite_conv1x1_to_matmul
from .validation import (
    compare_tensors,
    create_cpu_session,
    ensure_finite,
    parse_lengths,
)


def _render_reason_table(console: Console, report_json: dict) -> None:
    reasons = Counter(node["reason"] for node in report_json["nodes"])
    if not reasons:
        return

    table = Table(title="Rewrite Reasons", show_lines=False)
    table.add_column("Reason", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    for reason, count in reasons.most_common():
        table.add_row(reason, str(count))
    console.print(table)


def _has_external_initializers(model: onnx.ModelProto) -> bool:
    return any(uses_external_data(init) for init in model.graph.initializer)


def _is_float_tensor(type_str: str | None) -> bool:
    if not type_str:
        return False
    return type_str in {
        "tensor(float)",
        "tensor(double)",
        "tensor(float16)",
        "tensor(bfloat16)",
    }


def _is_int_tensor(type_str: str | None) -> bool:
    if not type_str:
        return False
    return type_str in {
        "tensor(int64)",
        "tensor(int32)",
        "tensor(uint64)",
        "tensor(uint32)",
    }


def _rank(shape: Any) -> int:
    if shape is None:
        return -1
    try:
        return len(shape)
    except TypeError:
        return -1


def _maybe_int(v: Any) -> int | None:
    return int(v) if isinstance(v, int) and v > 0 else None


def _resolve_verify_io(
    session: Any,
    *,
    signal_input_name: str | None,
    length_input_name: str | None,
    tensor_output_index: int,
    length_output_index: int | None,
) -> tuple[str, str, int, int | None]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    signal_name = signal_input_name
    if signal_name is None:
        for item in inputs:
            if _is_float_tensor(item.type) and _rank(item.shape) == 3:
                signal_name = item.name
                break

    length_name = length_input_name
    if length_name is None:
        for item in inputs:
            if _is_int_tensor(item.type) and _rank(item.shape) == 1:
                length_name = item.name
                break

    if signal_name is None:
        raise click.ClickException(
            "Unable to infer signal input. Provide --verify-signal-input-name."
        )
    if length_name is None:
        raise click.ClickException(
            "Unable to infer length input. Provide --verify-length-input-name."
        )

    input_names = {item.name for item in inputs}
    if signal_name not in input_names:
        raise click.ClickException(
            f"Signal input '{signal_name}' not found in model inputs."
        )
    if length_name not in input_names:
        raise click.ClickException(
            f"Length input '{length_name}' not found in model inputs."
        )

    if tensor_output_index < 0 or tensor_output_index >= len(outputs):
        raise click.ClickException(
            f"--verify-output-index out of range. Model has {len(outputs)} output(s)."
        )
    if length_output_index is not None and (
        length_output_index < 0 or length_output_index >= len(outputs)
    ):
        raise click.ClickException(
            f"--verify-length-output-index out of range. Model has {len(outputs)} output(s)."
        )

    return signal_name, length_name, tensor_output_index, length_output_index


def _run_verify(
    console: Console,
    *,
    original_model_path: Path,
    rewritten_model_path: Path,
    lengths_csv: str,
    seed: int,
    rtol: float,
    atol: float,
    max_abs_threshold: float,
    mean_abs_threshold: float,
    deterministic_cpu_verify: bool,
    verify_signal_input_name: str | None,
    verify_length_input_name: str | None,
    verify_output_index: int,
    verify_length_output_index: int | None,
    verify_channels: int | None,
) -> None:
    if seed < 0:
        raise click.UsageError("--verify-seed must be >= 0")
    if rtol < 0 or atol < 0:
        raise click.UsageError("--verify-rtol and --verify-atol must be >= 0")
    if max_abs_threshold < 0 or mean_abs_threshold < 0:
        raise click.UsageError("verify thresholds must be >= 0")

    try:
        lengths = parse_lengths(lengths_csv)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    console.log("Loading validation sessions")
    try:
        original_sess = create_cpu_session(
            original_model_path.as_posix(),
            deterministic_cpu_verify=deterministic_cpu_verify,
        )
        rewritten_sess = create_cpu_session(
            rewritten_model_path.as_posix(),
            deterministic_cpu_verify=deterministic_cpu_verify,
        )
    except ImportError as exc:
        raise click.ClickException(
            "Validation requires onnxruntime. Install it with: pip install onnxruntime"
        ) from exc

    signal_name, length_name, output_index, length_output_index = _resolve_verify_io(
        original_sess,
        signal_input_name=verify_signal_input_name,
        length_input_name=verify_length_input_name,
        tensor_output_index=verify_output_index,
        length_output_index=verify_length_output_index,
    )

    signal_meta = next(
        item for item in original_sess.get_inputs() if item.name == signal_name
    )
    inferred_channels = None
    if _rank(signal_meta.shape) == 3:
        inferred_channels = _maybe_int(signal_meta.shape[1])
    channels = verify_channels if verify_channels is not None else inferred_channels
    if channels is None:
        raise click.ClickException(
            "Unable to infer signal channels. Provide --verify-channels explicitly."
        )
    if channels < 1:
        raise click.UsageError("--verify-channels must be >= 1")

    rng = np.random.default_rng(seed)
    ok = True
    agg_max_abs = 0.0
    agg_mean_abs = 0.0
    agg_p99_abs = 0.0
    agg_p999_abs = 0.0

    table = Table(title="Verification Summary")
    table.add_column("Length", justify="right")
    table.add_column("MaxAbs", justify="right")
    table.add_column("MeanAbs", justify="right")
    table.add_column("P99", justify="right")
    table.add_column("P99.9", justify="right")
    table.add_column("SymAllClose", justify="center")
    table.add_column("LenEq", justify="center")

    for t in lengths:
        x = rng.normal(size=(1, channels, t)).astype(np.float32)
        length_value = np.array([t], dtype=np.int64)
        feed = {signal_name: x, length_name: length_value}

        out_a = original_sess.run(None, feed)
        out_b = rewritten_sess.run(None, feed)

        tensor_a = out_a[output_index]
        tensor_b = out_b[output_index]
        if not ensure_finite([tensor_a, tensor_b]):
            raise click.ClickException(
                "Validation produced non-finite outputs (NaN or Inf)."
            )

        metrics = compare_tensors(tensor_a, tensor_b, rtol=rtol, atol=atol)
        len_eq = True
        if length_output_index is not None:
            len_eq = bool(
                np.array_equal(out_a[length_output_index], out_b[length_output_index])
            )

        agg_max_abs = max(agg_max_abs, metrics.max_abs)
        agg_mean_abs = max(agg_mean_abs, metrics.mean_abs)
        agg_p99_abs = max(agg_p99_abs, metrics.p99_abs)
        agg_p999_abs = max(agg_p999_abs, metrics.p999_abs)

        case_ok = (
            metrics.allclose_symmetric
            and len_eq
            and metrics.max_abs <= max_abs_threshold
            and metrics.mean_abs <= mean_abs_threshold
        )
        ok = ok and case_ok

        table.add_row(
            str(t),
            f"{metrics.max_abs:.3e}",
            f"{metrics.mean_abs:.3e}",
            f"{metrics.p99_abs:.3e}",
            f"{metrics.p999_abs:.3e}",
            "yes" if metrics.allclose_symmetric else "no",
            "yes" if len_eq else "no",
        )

    console.print(table)
    click.echo(
        "verify_ok="
        f"{ok} "
        f"max_abs={agg_max_abs:.9e} "
        f"max_mean_abs={agg_mean_abs:.9e} "
        f"max_p99_abs={agg_p99_abs:.9e} "
        f"max_p999_abs={agg_p999_abs:.9e} "
        f"seed={seed} lengths={','.join(str(v) for v in lengths)}"
    )
    if not ok:
        raise click.ClickException(
            "Verification failed: numerical thresholds not satisfied."
        )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "output_path", required=False, type=click.Path(dir_okay=False, path_type=Path)
)
@click.option(
    "--inplace",
    is_flag=True,
    help="Overwrite input file instead of writing to a separate output.",
)
@click.option(
    "--extended-conv1x1",
    is_flag=True,
    help="Enable guarded support for Conv1x1 with explicit pads/strides via Pad+Slice.",
)
@click.option(
    "--allow-non-unit-dilation",
    is_flag=True,
    help="Allow non-unit dilations for Conv1x1 (opt-in guardrail).",
)
@click.option(
    "--max-dilation",
    type=int,
    default=None,
    help="Operational guardrail: maximum allowed dilation component.",
)
@click.option(
    "--report-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write detailed conversion report as JSON to the provided path.",
)
@click.option(
    "--report-json-stdout",
    is_flag=True,
    help="Print detailed conversion report JSON to stdout.",
)
@click.option(
    "--skip-checker",
    is_flag=True,
    help="Skip onnx.checker.check_model. Useful for very large external-data models.",
)
@click.option(
    "--verify",
    is_flag=True,
    help="Run numerical verification between original and rewritten models after rewrite.",
)
@click.option(
    "--verify-lengths",
    default="64,80,97,128,191",
    show_default=True,
    help="Comma-separated sequence lengths used by verification.",
)
@click.option(
    "--verify-seed",
    type=int,
    default=DEFAULT_VALIDATION_SEED,
    show_default=True,
    help="Random seed used for verification input generation.",
)
@click.option(
    "--verify-rtol",
    type=float,
    default=3e-5,
    show_default=True,
    help="Relative tolerance used by allclose in verification.",
)
@click.option(
    "--verify-atol",
    type=float,
    default=3e-5,
    show_default=True,
    help="Absolute tolerance used by allclose in verification.",
)
@click.option(
    "--verify-max-abs-threshold",
    type=float,
    default=3e-5,
    show_default=True,
    help="Per-case maximum absolute error threshold.",
)
@click.option(
    "--verify-mean-abs-threshold",
    type=float,
    default=2e-6,
    show_default=True,
    help="Per-case mean absolute error threshold.",
)
@click.option(
    "--verify-deterministic-cpu",
    is_flag=True,
    help="Use deterministic CPU-only SessionOptions for reproducible verification.",
)
@click.option(
    "--verify-signal-input-name",
    default=None,
    help="Override signal input name (auto-inferred by default).",
)
@click.option(
    "--verify-length-input-name",
    default=None,
    help="Override length input name (auto-inferred by default).",
)
@click.option(
    "--verify-output-index",
    type=int,
    default=0,
    show_default=True,
    help="Output index used for tensor comparison.",
)
@click.option(
    "--verify-length-output-index",
    type=int,
    default=None,
    help="Optional output index to compare as exact lengths.",
)
@click.option(
    "--verify-channels",
    type=int,
    default=None,
    help="Input channel count for generated signal; auto-inferred when possible.",
)
def main(
    input_path: Path,
    output_path: Path | None,
    inplace: bool,
    extended_conv1x1: bool,
    allow_non_unit_dilation: bool,
    max_dilation: int | None,
    report_json: Path | None,
    report_json_stdout: bool,
    skip_checker: bool,
    verify: bool,
    verify_lengths: str,
    verify_seed: int,
    verify_rtol: float,
    verify_atol: float,
    verify_max_abs_threshold: float,
    verify_mean_abs_threshold: float,
    verify_deterministic_cpu: bool,
    verify_signal_input_name: str | None,
    verify_length_input_name: str | None,
    verify_output_index: int,
    verify_length_output_index: int | None,
    verify_channels: int | None,
) -> None:
    console = Console()

    if inplace and output_path is not None:
        raise click.UsageError("Use either --inplace or output path, not both.")
    if not inplace and output_path is None:
        raise click.UsageError("Provide an output path or use --inplace.")
    if max_dilation is not None and max_dilation < 1:
        raise click.UsageError("--max-dilation must be >= 1")

    out_path = input_path if inplace else output_path
    assert out_path is not None

    console.log(f"Loading model: {input_path}")
    model_meta = onnx.load(input_path.as_posix(), load_external_data=False)
    save_as_external_data = _has_external_initializers(model_meta)
    model = onnx.load(input_path.as_posix(), load_external_data=True)

    console.log("Running Conv1x1 -> MatMul rewrite")
    model, report = rewrite_conv1x1_to_matmul(
        model,
        options=RewriteOptions(
            allow_extended_conv1x1=extended_conv1x1,
            allow_non_unit_dilation=allow_non_unit_dilation,
            max_dilation=max_dilation,
        ),
    )

    if skip_checker:
        console.log("Skipping ONNX checker (--skip-checker)")
    else:
        console.log("Running ONNX checker")
        onnx.checker.check_model(model)

    console.log(f"Saving model: {out_path}")
    if save_as_external_data:
        external_data_name = f"{out_path.name}.data"
        onnx.save_model(
            model,
            out_path.as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=0,
        )
    else:
        onnx.save(model, out_path.as_posix())

    report_payload = report.to_json_dict()
    if report_json is not None:
        report_json.write_text(
            json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        console.log(f"Wrote JSON report: {report_json}")
    if report_json_stdout:
        click.echo(json.dumps(report_payload, indent=2, sort_keys=True))

    summary = Table(title="Rewrite Summary")
    summary.add_column("Metric", style="green")
    summary.add_column("Value", justify="right", style="yellow")
    summary.add_row("Converted", str(report.converted))
    summary.add_row("Skipped", str(report.skipped))
    summary.add_row("Observed Conv Nodes", str(len(report.nodes)))
    console.print(summary)
    _render_reason_table(console, report_payload)

    if verify:
        _run_verify(
            console,
            original_model_path=input_path,
            rewritten_model_path=out_path,
            lengths_csv=verify_lengths,
            seed=verify_seed,
            rtol=verify_rtol,
            atol=verify_atol,
            max_abs_threshold=verify_max_abs_threshold,
            mean_abs_threshold=verify_mean_abs_threshold,
            deterministic_cpu_verify=verify_deterministic_cpu,
            verify_signal_input_name=verify_signal_input_name,
            verify_length_input_name=verify_length_input_name,
            verify_output_index=verify_output_index,
            verify_length_output_index=verify_length_output_index,
            verify_channels=verify_channels,
        )

    click.echo(
        f"converted={report.converted} skipped={report.skipped} output={out_path}"
    )


if __name__ == "__main__":
    main()
