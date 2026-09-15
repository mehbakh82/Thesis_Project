import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from thesis_s2s.bargein.detector import EnergyVadBaseline
from thesis_s2s.bargein.synthetic import _harmonic
from thesis_s2s.runtime import duplex as duplex_mod
from thesis_s2s.runtime import session_log
from thesis_s2s.runtime.duplex import build_app


class AlwaysInterrupt(EnergyVadBaseline):
    def predict_binary(self, audio, assistant_mask=None):
        return 1


class SpyTalker:
    backend = "test-tts"
    last_transcript = "متن آزمایشی"
    last_reply_text = "پاسخ آزمایشی"
    last_asr_error = None
    responder_backend = "test-responder"
    responder_initialization_error = None
    last_responder_fallback_used = False
    last_responder_error = None
    last_responder_generation_attempts = 1
    last_responder_language_retry_used = False

    def __init__(self, *, fail_once: bool = False):
        self.inputs: list[np.ndarray] = []
        self.fail_once = fail_once

    def reply_audio(self, user_audio: np.ndarray, text: str | None = None) -> np.ndarray:
        self.inputs.append(np.asarray(user_audio, dtype=np.float32).copy())
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("CUDA out of memory")
        return np.full(800, 0.1, dtype=np.float32)


def _patch_runtime(monkeypatch, tmp_path: Path, talker: SpyTalker) -> None:
    monkeypatch.setattr(session_log, "project_root", lambda: tmp_path)
    monkeypatch.setattr(duplex_mod, "project_root", lambda: tmp_path)
    monkeypatch.setattr(duplex_mod, "default_talker", lambda: talker)
    monkeypatch.setattr(
        duplex_mod,
        "gpu_inventory",
        lambda: {"device": "test-cpu", "total_gb": None},
    )


def _pcm(samples: int = 1600) -> bytes:
    audio = _harmonic(samples / 16_000, 190)
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16).tobytes()


def _meta() -> dict:
    return {
        "session_id": "Sduplex",
        "speaker_id": "Pduplex",
        "age_bin": "under_60",
        "prompt_id": "interrupt_story",
        "interrupt_label": "interrupt",
        "consent": True,
    }


def _complete_turn(ws, audio: bytes) -> dict:
    ws.send_json({"event": "begin_utterance", **_meta()})
    ws.send_bytes(audio)
    ws.send_json({"event": "end_of_speech", **_meta()})
    assert len(ws.receive_bytes()) == 1600
    return ws.receive_json()


def test_automatic_bargein_becomes_next_turn_and_records_client_ack(
    tmp_path: Path, monkeypatch
) -> None:
    talker = SpyTalker()
    _patch_runtime(monkeypatch, tmp_path, talker)
    app = build_app(AlwaysInterrupt(), retention="metrics")
    client = TestClient(app)
    chunk = _pcm()

    with client.websocket_connect("/ws") as ws:
        first_ready = _complete_turn(ws, chunk)
        assert first_ready["event"] == "reply_ready"
        assert first_ready["talker"] == "SpyTalker"
        assert first_ready["tts_backend"] == "test-tts"
        assert first_ready["responder_backend"] == "test-responder"
        assert first_ready["responder_fallback_used"] is False
        ws.send_json({"event": "playback_started", "t_first_audio_ms": 41.0, **_meta()})
        ws.send_bytes(chunk)
        ws.send_bytes(chunk)
        stopped = ws.receive_json()
        assert stopped["event"] == "stopped"

        ws.send_json({"event": "playback_stopped_ack", "t_barge_in_ms": 17.0, **_meta()})
        ws.send_bytes(chunk)
        ws.send_json({"event": "end_of_speech", **_meta()})
        assert len(ws.receive_bytes()) == 1600
        second_ready = ws.receive_json()
        assert second_ready["event"] == "reply_ready"
        ws.send_json({"event": "playback_ended", **_meta()})

    assert [len(audio) for audio in talker.inputs] == [1600, 4800]
    folder = tmp_path / "data" / "recordings" / "Sduplex"
    rows = [
        json.loads(line)
        for line in (folder / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["t_first_audio_ms"] == 41.0
    assert rows[0]["t_barge_in_ms"] == 17.0
    assert rows[0]["client_playback_started_ack"] is True
    assert rows[0]["client_playback_stopped_ack"] is True
    assert rows[0]["stopped"] is True
    assert rows[0]["retention"] == "metrics"
    assert rows[1]["duration"] == 0.3
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["measurement_source"] == "live_browser"
    assert not list(folder.glob("*.wav"))


def test_reply_ready_redacts_internal_model_errors(tmp_path: Path, monkeypatch) -> None:
    talker = SpyTalker()
    talker.responder_initialization_error = "OSError: /secret/model/path"
    talker.last_responder_error = "RuntimeError: private diagnostic"
    talker.last_asr_error = "connection failed at https://private.invalid"
    _patch_runtime(monkeypatch, tmp_path, talker)
    app = build_app(EnergyVadBaseline())

    with TestClient(app).websocket_connect("/ws") as ws:
        ready = _complete_turn(ws, _pcm())

    assert ready["responder_initialization_error"] == "model_initialization_failed"
    assert ready["responder_error"] == "model_generation_failed"
    assert ready["asr_error"] == "asr_request_failed"
    assert "/secret/model/path" not in json.dumps(ready)
    assert "private.invalid" not in json.dumps(ready)

    health = TestClient(app).get("/health").json()
    assert health["ok"] is True
    assert health["validated_cascade_ready"] is False


def test_health_requires_the_exact_validated_cascade(tmp_path: Path, monkeypatch) -> None:
    talker = object.__new__(duplex_mod.CascadeTalker)
    talker.backend = "piper"
    talker.responder = SimpleNamespace(
        model=object(),
        tokenizer=object(),
        model_name=duplex_mod.QWEN4B_MODEL,
        model_revision=duplex_mod.QWEN4B_REVISION,
        prompt_profile="qwen4b_v2",
        backend=duplex_mod.QWEN4B_MODEL,
    )
    monkeypatch.setattr(duplex_mod, "default_talker", lambda: talker)
    monkeypatch.setattr(duplex_mod, "project_root", lambda: tmp_path)
    monkeypatch.setattr(duplex_mod, "piper_runtime_ready", lambda: True)
    monkeypatch.setattr(
        duplex_mod,
        "gpu_inventory",
        lambda: {"device": "test-cpu", "total_gb": None},
    )

    health = TestClient(build_app(EnergyVadBaseline())).get("/health").json()

    assert health["validated_cascade_ready"] is True
    assert health["responder_backend"] == duplex_mod.QWEN4B_MODEL
    assert health["responder_revision"] == duplex_mod.QWEN4B_REVISION
    assert health["responder_prompt_profile"] == "qwen4b_v2"


def test_cancel_discards_pending_turn_and_clears_stale_audio(
    tmp_path: Path, monkeypatch
) -> None:
    talker = SpyTalker()
    _patch_runtime(monkeypatch, tmp_path, talker)
    app = build_app(EnergyVadBaseline(), retention="metrics")
    client = TestClient(app)
    chunk = _pcm()

    with client.websocket_connect("/ws") as ws:
        assert _complete_turn(ws, chunk)["event"] == "reply_ready"
        ws.send_json({"event": "cancel", **_meta()})
        assert ws.receive_json() == {"event": "cancelled"}

        ws.send_json({"event": "end_of_speech", **_meta()})
        assert ws.receive_json() == {
            "event": "error",
            "message": "no microphone audio received",
        }

        assert _complete_turn(ws, chunk)["event"] == "reply_ready"
        ws.send_json({"event": "playback_ended", **_meta()})

    assert [len(audio) for audio in talker.inputs] == [1600, 1600]
    turns = (
        tmp_path / "data" / "recordings" / "Sduplex" / "turns.jsonl"
    ).read_text(encoding="utf-8")
    assert len([line for line in turns.splitlines() if line]) == 1


def test_simultaneous_new_turn_and_reconnect_start_with_clean_buffers(
    tmp_path: Path, monkeypatch
) -> None:
    talker = SpyTalker()
    _patch_runtime(monkeypatch, tmp_path, talker)
    app = build_app(EnergyVadBaseline(), retention="metrics")
    client = TestClient(app)
    chunk = _pcm()

    with client.websocket_connect("/ws") as ws:
        assert _complete_turn(ws, chunk)["event"] == "reply_ready"
        # Starting a new utterance while the previous reply is playing stops
        # that playback, closes its pending record, and creates a fresh buffer.
        assert _complete_turn(ws, chunk)["event"] == "reply_ready"
        ws.send_json({"event": "playback_ended", **_meta()})

    with client.websocket_connect("/ws") as ws:
        assert _complete_turn(ws, chunk)["event"] == "reply_ready"
        ws.send_json({"event": "playback_ended", **_meta()})

    assert [len(audio) for audio in talker.inputs] == [1600, 1600, 1600]
    turns_path = tmp_path / "data" / "recordings" / "Sduplex" / "turns.jsonl"
    rows = [json.loads(line) for line in turns_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["stopped"] is True
    assert rows[1]["stopped"] is False
    assert rows[2]["stopped"] is False


def test_socket_recovers_from_malformed_input_silence_and_generation_oom(
    tmp_path: Path, monkeypatch
) -> None:
    talker = SpyTalker(fail_once=True)
    _patch_runtime(monkeypatch, tmp_path, talker)
    app = build_app(EnergyVadBaseline())
    client = TestClient(app)
    chunk = _pcm()

    with client.websocket_connect("/ws") as ws:
        ws.send_text("{")
        assert ws.receive_json() == {"event": "error", "message": "invalid JSON event"}
        ws.send_text("[]")
        assert ws.receive_json() == {
            "event": "error",
            "message": "JSON event must be an object",
        }
        ws.send_json({"event": "unsupported"})
        assert ws.receive_json() == {"event": "error", "message": "unknown event"}
        ws.send_json({"event": "begin_utterance", **_meta()})
        ws.send_bytes(b"\x00")
        assert ws.receive_json() == {
            "event": "error",
            "message": "invalid or empty PCM16 frame",
        }
        ws.send_bytes(b"")
        assert ws.receive_json() == {
            "event": "error",
            "message": "invalid or empty PCM16 frame",
        }
        ws.send_json({"event": "end_of_speech", **_meta()})
        assert ws.receive_json() == {
            "event": "error",
            "message": "no microphone audio received",
        }

        ws.send_json({"event": "begin_utterance", **_meta()})
        ws.send_bytes(chunk)
        ws.send_json({"event": "end_of_speech", **_meta()})
        assert ws.receive_json() == {
            "event": "error",
            "message": "generation_out_of_memory",
        }

        ready = _complete_turn(ws, chunk)
        assert ready["event"] == "reply_ready"
        assert ready["talker"] == "SpyTalker"

    assert [len(audio) for audio in talker.inputs] == [1600, 1600]
