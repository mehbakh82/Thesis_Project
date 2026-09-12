import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s import audio
from thesis_s2s.runtime import tts


def test_audio_conversion_rejects_nonfinite_complex_and_channels_first() -> None:
    with pytest.raises(ValueError, match="finite"):
        audio.to_float32_mono(np.asarray([float("nan")], dtype=np.float32))
    with pytest.raises(ValueError, match="complex"):
        audio.to_float32_mono(np.asarray([1 + 2j]))
    with pytest.raises(ValueError, match="sample-major"):
        audio.to_float32_mono(np.zeros((2, 100), dtype=np.float32))
    with pytest.raises(ValueError, match="numeric"):
        audio.to_float32_mono(np.asarray([True]))
    with pytest.raises(ValueError, match="numeric"):
        audio.to_float32_mono(np.asarray(["not audio"]))
    assert audio.to_float32_mono(np.asarray([2.0], dtype=np.float32))[0] == 1.0
    unsigned = audio.to_float32_mono(np.asarray([0, 255], dtype=np.uint8))
    assert unsigned[0] == -1.0 and unsigned[1] > 0.99


def test_sample_rates_and_empty_audio_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source sample rate"):
        audio.resample(np.ones(10, dtype=np.float32), 0)
    with pytest.raises(ValueError, match="sample rate"):
        audio.duration_seconds(np.ones(10, dtype=np.float32), 0)
    with pytest.raises(ValueError, match="empty audio"):
        audio.write_wav(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32))


def test_wav_write_is_atomic_and_round_trips(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "audio.wav"
    original = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    audio.write_wav(path, original)
    decoded, sample_rate = audio.read_wav(path)
    assert sample_rate == 16_000
    assert np.allclose(decoded, original, atol=1 / 32767)
    assert list(tmp_path.glob(".audio.wav.*.tmp")) == []

    old_bytes = path.read_bytes()
    monkeypatch.setattr(audio.wave, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")))
    with pytest.raises(OSError, match="write failed"):
        audio.write_wav(path, original)
    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob(".audio.wav.*.tmp")) == []


def test_ffmpeg_decode_requires_file_timeout_and_nonempty_output(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "audio.bin"
    source.write_bytes(b"input")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=np.asarray([100], dtype=np.int16).tobytes())

    monkeypatch.setenv("AUDIO_DECODE_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    decoded, sample_rate = audio.read_audio_ffmpeg(source)
    assert decoded.size == 1 and sample_rate == 16_000
    assert calls[0][1]["timeout"] == 12.5

    monkeypatch.setattr(
        audio.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout=b""),
    )
    with pytest.raises(ValueError, match="decoded audio is empty"):
        audio.read_audio_ffmpeg(source)
    with pytest.raises(FileNotFoundError):
        audio.read_audio_ffmpeg(tmp_path / "missing")
    for invalid in ("invalid", "0", "nan"):
        monkeypatch.setenv("AUDIO_DECODE_TIMEOUT_SECONDS", invalid)
        with pytest.raises(ValueError, match="positive and finite"):
            audio.read_audio_ffmpeg(source)


def test_resample_and_empty_wav_validation(tmp_path: Path) -> None:
    resampled = audio.resample(np.ones(8, dtype=np.float32), 8_000, 16_000)
    assert resampled.dtype == np.float32 and len(resampled) == 16
    assert audio.resample(np.zeros(0, dtype=np.float32), 8_000).size == 0
    empty = tmp_path / "empty.wav"
    with wave.open(str(empty), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
    with pytest.raises(ValueError, match="audio file is empty"):
        audio.read_wav(empty)


def test_formant_seed_and_output_are_stable_and_inputs_are_validated() -> None:
    assert tts._text_seed("سلام") == tts._text_seed("سلام")
    assert np.array_equal(tts.formant_synthesize("سلام"), tts.formant_synthesize("سلام"))
    with pytest.raises(ValueError, match="non-empty"):
        tts.formant_synthesize("")
    with pytest.raises(ValueError, match="frequency"):
        tts.formant_synthesize("سلام", f0=float("nan"))


def test_piper_model_discovery_never_selects_an_arbitrary_voice(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PIPER_MODEL", raising=False)
    monkeypatch.setattr(tts, "project_piper_dir", lambda: tmp_path)
    (tmp_path / "other-language.onnx").write_bytes(b"model")
    assert tts.os_piper_model() is None
    preferred = tmp_path / "fa_IR-mana-medium.onnx"
    preferred.write_bytes(b"model")
    assert tts.os_piper_model() == preferred
    monkeypatch.setenv("PIPER_MODEL", str(tmp_path / "missing.onnx"))
    assert tts.os_piper_model() is None


def test_piper_subprocess_has_timeout_and_fails_closed(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(tts.shutil, "which", lambda _name: "/usr/bin/piper")
    monkeypatch.setenv("PIPER_TIMEOUT_SECONDS", "4")
    calls = []

    def timeout(cmd, **kwargs):
        calls.append((cmd, kwargs))
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(tts.subprocess, "run", timeout)
    assert tts._piper_subprocess_synthesize("سلام", model, 16_000, deterministic=True) is None
    assert calls[0][1]["timeout"] == 4.0
    monkeypatch.setenv("PIPER_TIMEOUT_SECONDS", "nan")
    assert tts._piper_subprocess_synthesize("سلام", model, 16_000, deterministic=False) is None


def test_piper_subprocess_rejects_failed_decode(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(tts.shutil, "which", lambda _name: "/usr/bin/piper")

    def fake_run(cmd, **_kwargs):
        Path(cmd[cmd.index("--output_file") + 1]).write_bytes(b"invalid wav")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(tts.subprocess, "run", fake_run)
    monkeypatch.setattr(tts, "read_wav", lambda *_args: (_ for _ in ()).throw(ValueError("bad")))
    assert tts._piper_subprocess_synthesize("سلام", model, 16_000, deterministic=False) is None


def test_piper_wav_parser_and_synthesize_reject_invalid_audio(monkeypatch) -> None:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.zeros(20, dtype=np.int16).tobytes())
    assert tts._pcm_from_wav_bytes(buffer.getvalue(), 16_000) is None

    monkeypatch.setattr(
        tts,
        "piper_synthesize",
        lambda _text, _sr: np.asarray([float("nan")], dtype=np.float32),
    )
    rendered, backend = tts.synthesize("سلام")
    assert backend == "formant"
    assert np.isfinite(rendered).all()
    with pytest.raises(ValueError, match="non-empty"):
        tts.synthesize("")


def test_piper_runtime_ready_probes_and_caches_by_model_identity(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "fa_IR-mana-medium.onnx"
    model.write_bytes(b"model")
    calls = []
    monkeypatch.setattr(tts, "os_piper_model", lambda: model)
    monkeypatch.setattr(
        tts,
        "piper_synthesize",
        lambda *_args, **_kwargs: calls.append(True) or np.ones(10, dtype=np.float32),
    )
    monkeypatch.setattr(tts, "_PIPER_READY_KEY", None)
    monkeypatch.setattr(tts, "_PIPER_READY", False)

    assert tts.piper_runtime_ready() is True
    assert tts.piper_runtime_ready() is True
    assert len(calls) == 1


def test_piper_runtime_readiness_and_packets_fail_closed(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "fa_IR-mana-medium.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(tts, "os_piper_model", lambda: model)
    monkeypatch.setattr(tts, "_PIPER_READY_KEY", None)
    monkeypatch.setattr(tts, "piper_synthesize", lambda *_args: (_ for _ in ()).throw(RuntimeError()))
    assert tts.piper_runtime_ready() is False

    with pytest.raises(ValueError, match="sample rate"):
        tts.first_packet(np.ones(10, dtype=np.float32), sr=0)
    with pytest.raises(ValueError, match="duration"):
        tts.first_packet(np.ones(10, dtype=np.float32), seconds=float("nan"))
    with pytest.raises(ValueError, match="one-dimensional"):
        tts.first_packet(np.asarray([float("inf")], dtype=np.float32))
