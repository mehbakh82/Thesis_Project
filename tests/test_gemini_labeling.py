from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s.labeling import gemini
from thesis_s2s.labeling.gemini import GeminiClient, GeminiError, parse_json_object, wav_bytes


def test_parse_json_object_accepts_fenced_reply() -> None:
    assert parse_json_object('```json\n{"transcript": "سلام"}\n```') == {"transcript": "سلام"}


@pytest.mark.parametrize("reply", ["no json here", "[1, 2]", "{not json}"])
def test_parse_json_object_rejects_invalid_replies(reply: str) -> None:
    with pytest.raises(GeminiError):
        parse_json_object(reply)


def test_wav_bytes_round_trips_mono_pcm16() -> None:
    audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    raw = wav_bytes(audio, 16_000)
    with wave.open(io.BytesIO(raw)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16_000
        assert handle.getnframes() == 1600


def test_wav_bytes_rejects_long_or_non_finite_audio() -> None:
    with pytest.raises(ValueError):
        wav_bytes(np.zeros(16_000 * 121, dtype=np.float32), 16_000)
    with pytest.raises(ValueError):
        wav_bytes(np.array([np.nan], dtype=np.float32), 16_000)


def test_client_caches_by_model_prompt_and_audio(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(gemini.KEY_ENV, "test-key")
    calls: list[dict] = []

    def fake_post(self, payload):  # noqa: ANN001
        calls.append(payload)
        return {
            "model": "google/gemini-3.8-flash",
            "choices": [{"message": {"content": '{"transcript": "سلام", "speech_present": true}'}}],
        }

    monkeypatch.setattr(GeminiClient, "_post", fake_post)
    client = GeminiClient(cache_dir=tmp_path, requests_per_minute=6000)
    first = client.complete_json("prompt", audio=b"abc", required_keys=["transcript"])
    second = client.complete_json("prompt", audio=b"abc", required_keys=["transcript"])
    assert first == second
    assert first["_meta"]["response_model"] == "google/gemini-3.8-flash"
    assert len(calls) == 1
    cached = next(tmp_path.rglob("*.json"))
    assert "test-key" not in cached.read_text(encoding="utf-8")
    client.complete_json("prompt", audio=b"abd", required_keys=["transcript"])
    assert len(calls) == 2


def test_client_fails_after_missing_required_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(gemini.KEY_ENV, "test-key")
    monkeypatch.setattr(
        GeminiClient,
        "_post",
        lambda self, payload: {"choices": [{"message": {"content": json.dumps({"x": 1})}}]},
    )
    client = GeminiClient(
        cache_dir=tmp_path, requests_per_minute=6000, max_attempts=2, retry_backoff_sec=0
    )
    with pytest.raises(GeminiError, match="required keys"):
        client.complete_json("prompt", required_keys=["transcript"])


def test_auth_errors_are_not_retried(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(gemini.KEY_ENV, "test-key")
    attempts = []

    def deny(self, payload):  # noqa: ANN001
        attempts.append(1)
        raise gemini.GeminiAuthError("gateway HTTP 401")

    monkeypatch.setattr(GeminiClient, "_post", deny)
    client = GeminiClient(cache_dir=tmp_path, requests_per_minute=6000, retry_backoff_sec=0)
    with pytest.raises(gemini.GeminiAuthError):
        client.complete_json("prompt")
    assert attempts == [1]


def test_key_falls_back_to_shared_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(gemini.KEY_ENV, raising=False)
    config = tmp_path / "config.yaml"
    config.write_text("ground_truth:\n  api:\n    api_key: from-config\n", encoding="utf-8")
    client = GeminiClient(cache_dir=tmp_path / "cache", config_path=config)
    assert client.key == "from-config"
