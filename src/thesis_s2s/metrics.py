"""Latency and interrupt metrics.

T_first_audio: user end-of-speech or accepted barge-in -> first audible PCM.
T_barge_in: user interrupt onset -> assistant playback actually stops.

Official tables must use a 12–24 GB GPU. This module can also *simulate* a
24 GB cap on a larger card via ``cuda_memory_cap_gb``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import mkstemp

import numpy as np

from thesis_s2s import (
    BARGEIN_ACCURACY_TARGET,
    OFFICIAL_VRAM_GB,
    T_BARGE_IN_P95_MS,
    T_FIRST_AUDIO_P50_MS,
)


@dataclass
class LatencySample:
    t_first_audio_ms: float
    t_barge_in_ms: float | None = None
    gpu_name: str = ""
    vram_gb: float | None = None
    memory_capped: bool = False
    notes: str = ""


@dataclass
class LatencyReport:
    n: int
    t_first_audio_p50_ms: float
    t_first_audio_p95_ms: float
    t_first_audio_max_ms: float
    t_barge_in_p50_ms: float | None
    t_barge_in_p95_ms: float | None
    t_barge_in_max_ms: float | None
    t_first_audio_gate_ok: bool
    t_first_audio_literal_max_gate_ok: bool
    t_barge_in_gate_ok: bool
    official_gpu: bool
    samples: list[LatencySample] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


@dataclass
class BargeinConfusion:
    accuracy: float
    interrupt_f1: float
    far: float
    frr: float
    target_ok: bool
    n: int
    labels: list[str]
    matrix: list[list[int]]


def percentile(values: Sequence[float], q: float) -> float:
    try:
        q_value = float(q)
    except (TypeError, ValueError) as exc:
        raise ValueError("percentile must be finite and between 0 and 100") from exc
    if isinstance(q, bool) or not np.isfinite(q_value) or not 0.0 <= q_value <= 100.0:
        raise ValueError("percentile must be finite and between 0 and 100")
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("percentile values must be a finite one-dimensional sequence")
    return float(np.percentile(array, q_value))


def meets_bargein_accuracy_target(accuracy: object) -> bool:
    """Fail closed unless a finite probability is strictly above the target."""

    if isinstance(accuracy, bool):
        return False
    try:
        value = float(accuracy)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and 0.0 <= value <= 1.0 and value > BARGEIN_ACCURACY_TARGET)


def summarize_latency(
    samples: Iterable[LatencySample],
    *,
    official_gpu: bool | None = None,
) -> LatencyReport:
    rows = list(samples)
    for sample in rows:
        values = [sample.t_first_audio_ms, sample.t_barge_in_ms]
        try:
            invalid_latency = any(
                value is not None
                and (isinstance(value, bool) or not np.isfinite(value) or value < 0)
                for value in values
            )
            invalid_vram = sample.vram_gb is not None and (
                isinstance(sample.vram_gb, bool)
                or not np.isfinite(sample.vram_gb)
                or sample.vram_gb <= 0
            )
        except TypeError as exc:
            raise ValueError("latency and VRAM values must be numeric") from exc
        if invalid_latency:
            raise ValueError("latency samples must be finite, non-negative milliseconds")
        if invalid_vram:
            raise ValueError("VRAM values must be finite, positive gigabytes")
    first = [s.t_first_audio_ms for s in rows]
    barge = [s.t_barge_in_ms for s in rows if s.t_barge_in_ms is not None]
    p50_first = percentile(first, 50)
    p95_first = percentile(first, 95)
    max_first = max(first) if first else float("nan")
    p50_barge = percentile(barge, 50) if barge else None
    p95_barge = percentile(barge, 95) if barge else None
    max_barge = max(barge) if barge else None
    hardware_eligible = bool(rows) and all(
        s.vram_gb is not None
        and OFFICIAL_VRAM_GB[0] <= s.vram_gb <= OFFICIAL_VRAM_GB[1]
        and not s.memory_capped
        for s in rows
    )
    if official_gpu is not None and not isinstance(official_gpu, bool):
        raise ValueError("official_gpu must be a boolean or None")
    official_gpu = hardware_eligible if official_gpu is None else official_gpu and hardware_eligible
    return LatencyReport(
        n=len(rows),
        t_first_audio_p50_ms=p50_first,
        t_first_audio_p95_ms=p95_first,
        t_first_audio_max_ms=max_first,
        t_barge_in_p50_ms=p50_barge,
        t_barge_in_p95_ms=p95_barge,
        t_barge_in_max_ms=max_barge,
        t_first_audio_gate_ok=p50_first <= T_FIRST_AUDIO_P50_MS if first else False,
        t_first_audio_literal_max_gate_ok=max_first <= T_FIRST_AUDIO_P50_MS if first else False,
        t_barge_in_gate_ok=(p95_barge is not None and p95_barge <= T_BARGE_IN_P95_MS),
        official_gpu=bool(official_gpu),
        samples=rows,
    )


def binary_scores(y_true: Sequence[int], y_pred: Sequence[int]) -> BargeinConfusion:
    """interrupt=1 vs other=0."""

    raw_yt = np.asarray(y_true)
    raw_yp = np.asarray(y_pred)
    if raw_yt.ndim != 1 or raw_yp.ndim != 1 or len(raw_yt) != len(raw_yp):
        raise ValueError("y_true and y_pred must be one-dimensional and equally sized")
    if len(raw_yt) == 0:
        raise ValueError("at least one prediction is required")
    if (
        not np.issubdtype(raw_yt.dtype, np.number)
        or not np.issubdtype(raw_yp.dtype, np.number)
        or np.issubdtype(raw_yt.dtype, np.bool_)
        or np.issubdtype(raw_yp.dtype, np.bool_)
    ):
        raise ValueError("binary labels must be numeric integers")
    yt_float = raw_yt.astype(np.float64)
    yp_float = raw_yp.astype(np.float64)
    if (
        not np.isfinite(yt_float).all()
        or not np.isfinite(yp_float).all()
        or not np.equal(yt_float, np.floor(yt_float)).all()
        or not np.equal(yp_float, np.floor(yp_float)).all()
    ):
        raise ValueError("binary labels must be finite integers")
    yt = yt_float.astype(int)
    yp = yp_float.astype(int)
    if not np.isin(yt, [0, 1]).all() or not np.isin(yp, [0, 1]).all():
        raise ValueError("binary labels must contain only 0 or 1")
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    far = fp / max(1, fp + tn)
    frr = fn / max(1, fn + tp)
    return BargeinConfusion(
        accuracy=float(acc),
        interrupt_f1=float(f1),
        far=float(far),
        frr=float(frr),
        target_ok=meets_bargein_accuracy_target(acc),
        n=int(len(yt)),
        labels=["other", "interrupt"],
        matrix=[[tn, fp], [fn, tp]],
    )


def binary_score_confidence_intervals(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> dict[str, list[float] | None]:
    """Return deterministic 95% intervals for the detector's primary scores.

    Accuracy, FAR, and FRR use Wilson score intervals. F1 uses a seeded
    non-parametric bootstrap because it is not a simple binomial proportion.
    """

    binary_scores(y_true, y_pred)  # shared strict validation
    if not isinstance(bootstrap_samples, int) or isinstance(bootstrap_samples, bool):
        raise ValueError("bootstrap_samples must be a positive integer")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    def wilson(successes: int, total: int) -> list[float] | None:
        if total <= 0:
            return None
        z = 1.959963984540054
        proportion = successes / total
        denominator = 1.0 + z * z / total
        centre = (proportion + z * z / (2.0 * total)) / denominator
        margin = (
            z
            * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
            / denominator
        )
        return [
            round(float(max(0.0, centre - margin)), 4),
            round(float(min(1.0, centre + margin)), 4),
        ]

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    rng = np.random.default_rng(seed)
    f1_samples: list[float] = []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(yt), len(yt))
        f1_samples.append(binary_scores(yt[indices].tolist(), yp[indices].tolist()).interrupt_f1)
    f1_interval: np.ndarray = np.asarray(np.percentile(f1_samples, [2.5, 97.5]))
    return {
        "accuracy": wilson(tp + tn, len(yt)),
        "interrupt_f1": [round(float(value), 4) for value in f1_interval],
        "far": wilson(fp, fp + tn),
        "frr": wilson(fn, fn + tp),
    }


def mean_confidence_interval(
    values: Sequence[float],
    *,
    seed: int = 0,
    bootstrap_samples: int = 5000,
) -> list[float] | None:
    """Seeded percentile-bootstrap 95% interval for a sample mean."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("values must be a finite one-dimensional sequence")
    if not isinstance(bootstrap_samples, int) or isinstance(bootstrap_samples, bool):
        raise ValueError("bootstrap_samples must be a positive integer")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if len(array) < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(bootstrap_samples, len(array)), replace=True)
    means = samples.mean(axis=1)
    lower = float(np.percentile(means, 2.5))
    upper = float(np.percentile(means, 97.5))
    return [round(lower, 4), round(upper, 4)]


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(payload, dict):
        raise TypeError("metric payload must be a dictionary")
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary_name = mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def gpu_inventory() -> dict:
    """Physical GPU vs official 12–24 GB thesis table."""

    info: dict[str, object] = {
        "device": None,
        "total_gb": None,
        "official_size": False,
        "path_b_possible": False,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            info["error"] = "cuda_unavailable"
            return info
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        info["device"] = props.name
        info["total_gb"] = round(total_gb, 2)
        info["official_size"] = 12.0 <= total_gb <= 24.0
        info["path_b_possible"] = info["official_size"]
    except Exception as exc:
        info["error"] = str(exc)[:200]
    return info


def cuda_memory_cap_gb(cap_gb: float = 24.0) -> dict:
    """Best-effort 24 GB profile on a larger GPU. Disclose this in the thesis."""

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    info: dict[str, object] = {
        "requested_cap_gb": cap_gb,
        "applied": False,
        "device": None,
        "total_gb": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            info["error"] = "cuda_unavailable"
            return info
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        info["device"] = props.name
        info["total_gb"] = round(total_gb, 2)
        fraction = min(1.0, cap_gb / total_gb)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        info["applied"] = True
        info["fraction"] = fraction
        info["memory_capped"] = total_gb > cap_gb + 0.5
    except Exception as exc:  # pragma: no cover - hardware specific
        info["error"] = str(exc)
    return info
