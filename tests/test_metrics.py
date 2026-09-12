import json
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s.metrics import (
    LatencySample,
    binary_score_confidence_intervals,
    binary_scores,
    mean_confidence_interval,
    percentile,
    summarize_latency,
    write_json,
)


def test_latency_gates():
    samples = [
        LatencySample(t_first_audio_ms=180, t_barge_in_ms=40, vram_gb=24, memory_capped=False),
        LatencySample(t_first_audio_ms=220, t_barge_in_ms=55, vram_gb=24, memory_capped=False),
        LatencySample(t_first_audio_ms=190, t_barge_in_ms=70, vram_gb=24, memory_capped=False),
    ]
    report = summarize_latency(samples)
    assert report.t_first_audio_gate_ok
    assert report.t_barge_in_gate_ok
    assert report.official_gpu
    assert report.t_first_audio_max_ms == 220
    assert report.t_first_audio_literal_max_gate_ok is True


def test_path_a_capped_is_unofficial():
    samples = [
        LatencySample(t_first_audio_ms=180, t_barge_in_ms=40, vram_gb=93, memory_capped=True),
        LatencySample(t_first_audio_ms=220, t_barge_in_ms=55, vram_gb=93, memory_capped=True),
    ]
    report = summarize_latency(samples, official_gpu=False)
    assert report.official_gpu is False
    assert report.t_first_audio_gate_ok


def test_mean_confidence_interval_is_deterministic():
    first = mean_confidence_interval([1, 2, 3, 4, 5], seed=7, bootstrap_samples=200)
    second = mean_confidence_interval([1, 2, 3, 4, 5], seed=7, bootstrap_samples=200)
    assert first == second


def test_latency_cannot_be_forced_official_without_eligible_hardware():
    report = summarize_latency(
        [LatencySample(t_first_audio_ms=100, vram_gb=80, memory_capped=False)],
        official_gpu=True,
    )
    assert report.official_gpu is False
    with pytest.raises(ValueError, match="official_gpu"):
        summarize_latency([], official_gpu="yes")
    with pytest.raises(ValueError, match="VRAM"):
        summarize_latency([LatencySample(t_first_audio_ms=100, vram_gb=float("nan"))])
    with pytest.raises(ValueError, match="latency"):
        summarize_latency([LatencySample(t_first_audio_ms=True)])


def test_binary_scores_reject_labels_that_would_be_silently_coerced():
    for invalid in ([0.5, 1.0], [True, False], ["0", "1"], [0, float("nan")]):
        with pytest.raises(ValueError, match="binary labels"):
            binary_scores(invalid, [0, 1])
    with pytest.raises(ValueError, match="positive integer"):
        binary_score_confidence_intervals([0, 1], [0, 1], bootstrap_samples=0)


def test_percentile_and_bootstrap_configuration_fail_closed():
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], 101)
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], "not-a-percentile")
    with pytest.raises(ValueError, match="finite one-dimensional"):
        percentile([float("inf")], 50)
    with pytest.raises(ValueError, match="positive integer"):
        mean_confidence_interval([1, 2], bootstrap_samples=1.5)
    with pytest.raises(ValueError, match="positive integer"):
        mean_confidence_interval([1], bootstrap_samples=0)


def test_metric_json_is_finite_atomic_and_preserves_old_file_on_failure(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "metrics.json"
    write_json(path, {"value": 1, "label": "فارسی"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1, "label": "فارسی"}
    assert list(tmp_path.glob(".metrics.json.*.tmp")) == []

    old_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="JSON compliant"):
        write_json(path, {"value": np.nan})
    assert path.read_bytes() == old_bytes

    monkeypatch.setattr(
        "thesis_s2s.metrics.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        write_json(path, {"value": 2})
    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob(".metrics.json.*.tmp")) == []
