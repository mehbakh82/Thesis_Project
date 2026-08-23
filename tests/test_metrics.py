from thesis_s2s.metrics import LatencySample, mean_confidence_interval, summarize_latency


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
