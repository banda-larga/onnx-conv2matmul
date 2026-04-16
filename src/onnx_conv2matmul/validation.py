from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiffMetrics:
    allclose_ab: bool
    allclose_ba: bool
    allclose_symmetric: bool
    max_abs: float
    mean_abs: float
    p95_abs: float
    p99_abs: float
    p999_abs: float


def compare_tensors(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> DiffMetrics:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")

    diff = np.abs(a - b)
    allclose_ab = bool(np.allclose(a, b, rtol=rtol, atol=atol))
    allclose_ba = bool(np.allclose(b, a, rtol=rtol, atol=atol))

    return DiffMetrics(
        allclose_ab=allclose_ab,
        allclose_ba=allclose_ba,
        allclose_symmetric=allclose_ab and allclose_ba,
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        p95_abs=float(np.percentile(diff, 95.0)),
        p99_abs=float(np.percentile(diff, 99.0)),
        p999_abs=float(np.percentile(diff, 99.9)),
    )


def create_cpu_session(
    model_path: str,
    *,
    deterministic_cpu_verify: bool,
):
    import onnxruntime as ort

    if not deterministic_cpu_verify:
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.use_deterministic_compute = True
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")

    return ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def parse_lengths(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    lengths = [int(v) for v in values]
    if not lengths:
        raise ValueError("at least one length must be provided")
    if any(v <= 0 for v in lengths):
        raise ValueError("all lengths must be positive")
    return lengths


def ensure_finite(arrays: Iterable[np.ndarray]) -> bool:
    return all(np.isfinite(a).all() for a in arrays)
