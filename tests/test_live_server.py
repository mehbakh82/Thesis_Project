from __future__ import annotations

import json
import struct
import time
from collections.abc import Iterator

import numpy as np
from fastapi.testclient import TestClient

from thesis_s2s.runtime.bargein_live import BargeInController
from thesis_s2s.runtime.live import FRAME, LiveComponents, LiveConfig, build_live_app
from thesis_s2s.runtime.vad import SAMPLE_RATE, EnergyScorer


class _Responder:
    def stream(self, text: str, history=(), cancel=None) -> Iterator[str]:
        for delta in ["سلام", " دوست", " من", ".", " حالت", " خوبه", "؟"]:
            if cancel is not None and cancel.is_set():
                return
            yield delta


def _frames(audio: np.ndarray, start_seq: int) -> list[bytes]:
    out = []
    for index in range(audio.size // FRAME):
        pcm = (audio[index * FRAME : (index + 1) * FRAME] * 32767).astype("<i2").tobytes()
        out.append(struct.pack("<I", start_seq + index) + pcm)
    return out


def _tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def _quiet(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.001 * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def _app(tmp_path, telemetry=True):
    components = LiveComponents(
        asr=lambda audio: ("سلام چطوری", None),
        responder=_Responder(),  # type: ignore[arg-type]
        tts=lambda text: _tone(0.3, 0.1),
        identity={"asr_backend": "fake"},
        scorer_factory=EnergyScorer,
    )
    path = tmp_path / "telemetry.jsonl" if telemetry else None
    return build_live_app(components, LiveConfig(hangover_ms=256, telemetry_path=path)), path


def _receive_until(ws, predicate, limit: int = 200) -> list:
    seen = []
    for _ in range(limit):
        message = ws.receive()
        if message.get("bytes") is not None:
            seen.append(("audio", message["bytes"]))
        else:
            payload = json.loads(message["text"])
            seen.append(("json", payload))
            if predicate(payload):
                return seen
    raise AssertionError(f"predicate not met; saw {[s[1] for s in seen if s[0] == 'json']}")


def test_hands_free_turn_streams_clauses_and_records_telemetry(tmp_path) -> None:
    app, telemetry = _app(tmp_path)
    client = TestClient(app)
    audio = np.concatenate([_quiet(0.6), _tone(0.8), _quiet(0.6, 1)])
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"event": "hello"}))
        assert json.loads(ws.receive()["text"])["event"] == "ready"
        for frame in _frames(audio, 0):
            ws.send_bytes(frame)
        seen = _receive_until(ws, lambda p: p.get("event") == "reply_done")
        endpoint = next(p for kind, p in seen if kind == "json" and p["event"] == "endpoint")
        assert abs(endpoint["end_sample"] - int(1.4 * SAMPLE_RATE)) <= 1024
        chunks = [struct.unpack("<II", b[:8]) for kind, b in seen if kind == "audio"]
        assert chunks == [(1, 0), (1, 1)]
        done = seen[-1][1]
        assert done["transcript"] == "سلام چطوری" and done["clauses"] == 2
        ws.send_text(json.dumps({"event": "playback_started", "turn": 1, "t_first_audio_ms": 321.0}))
        ws.send_text(json.dumps({"event": "playback_ended", "turn": 1}))
        time.sleep(0.2)
    rows = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert rows[0]["client_t_first_audio_ms"] == 321.0 and rows[0]["outcome"] == "completed"
    assert rows[0]["server_timings_ms"]["first_audio_ready_ms"] is not None


def test_sustained_speech_during_response_ducks_then_stops(tmp_path) -> None:
    app, _ = _app(tmp_path, telemetry=False)
    client = TestClient(app)
    audio = np.concatenate([_quiet(0.6), _tone(0.8), _quiet(0.6, 1)])
    with client.websocket_connect("/ws") as ws:
        frames = _frames(audio, 0)
        for frame in frames:
            ws.send_bytes(frame)
        _receive_until(ws, lambda p: p.get("event") == "endpoint")
        barge = _frames(_tone(1.0, 0.3), len(frames))
        for frame in barge:
            ws.send_bytes(frame)
        seen = _receive_until(ws, lambda p: p.get("event") == "stop")
        kinds = [p["event"] for kind, p in seen if kind == "json"]
        assert "duck" in kinds and kinds.index("duck") < kinds.index("stop")
        stop = seen[-1][1]
        assert abs(stop["onset_sample"] - len(frames) * FRAME) <= 2 * 512


def test_short_backchannel_resumes_instead_of_stopping() -> None:
    controller = BargeInController(scorer=EnergyScorer(), confirm_speech_ms=480)
    audio = np.concatenate([_quiet(0.5), _tone(0.25, 0.3), _quiet(0.6, 2)])
    events = []
    for start in range(0, audio.size, FRAME):
        events.extend(controller.push(audio[start : start + FRAME]))
    assert [e.kind for e in events] == ["duck", "resume"]


def test_classifier_can_confirm_before_duration_rule() -> None:
    controller = BargeInController(
        scorer=EnergyScorer(), classifier=lambda window: 0.95, classifier_min_ms=256
    )
    audio = np.concatenate([_quiet(0.5), _tone(0.6, 0.3)])
    events = []
    for start in range(0, audio.size, FRAME):
        events.extend(controller.push(audio[start : start + FRAME]))
    stop = next(e for e in events if e.kind == "stop")
    assert stop.probability == 0.95
    assert (stop.decided_at_sample - stop.onset_sample) / SAMPLE_RATE < 0.48
    assert controller.captured_since_onset().size > 0


def test_resumed_speech_cancels_speculative_turn_and_merges(tmp_path) -> None:
    app, _ = _app(tmp_path, telemetry=False)
    client = TestClient(app)
    # A 200 ms pause (longer than the 160 ms speculation point, shorter than the
    # 256 ms hangover) must not produce a reply for the first fragment.
    audio = np.concatenate(
        [_quiet(0.5), _tone(0.6), _quiet(0.2, 3), _tone(0.6), _quiet(0.7, 4)]
    )
    with client.websocket_connect("/ws") as ws:
        for frame in _frames(audio, 0):
            ws.send_bytes(frame)
        seen = _receive_until(ws, lambda p: p.get("event") == "reply_done")
        endpoints = [p for kind, p in seen if kind == "json" and p["event"] == "endpoint"]
        assert len(endpoints) == 1 and endpoints[0]["turn"] == 2
        assert abs(endpoints[0]["end_sample"] - int(1.9 * SAMPLE_RATE)) <= 1024
        audio_turns = {struct.unpack("<II", b[:8])[0] for kind, b in seen if kind == "audio"}
        assert audio_turns == {2}


def test_endpointer_emits_pause_and_resume_before_hangover() -> None:
    from thesis_s2s.runtime.vad import Endpointer

    endpointer = Endpointer(scorer=EnergyScorer(), hangover_ms=256, tentative_ms=160)
    audio = np.concatenate([_quiet(0.5), _tone(0.5), _quiet(0.2, 5), _tone(0.5), _quiet(0.5, 6)])
    kinds = []
    for start in range(0, audio.size, FRAME):
        kinds.extend(e.kind for e in endpointer.push(audio[start : start + FRAME]))
    assert kinds == [
        "speech_start",
        "speech_pause",
        "speech_resume",
        "speech_pause",
        "speech_end",
    ]


def test_second_turn_after_playback_keeps_absolute_sample_timeline(tmp_path) -> None:
    app, _ = _app(tmp_path, telemetry=False)
    client = TestClient(app)
    first = np.concatenate([_quiet(0.6), _tone(0.8), _quiet(0.6, 1)])
    second = np.concatenate([_quiet(0.6, 7), _tone(0.8), _quiet(0.6, 8)])
    with client.websocket_connect("/ws") as ws:
        frames = _frames(first, 0)
        for frame in frames:
            ws.send_bytes(frame)
        _receive_until(ws, lambda p: p.get("event") == "reply_done")
        ws.send_text(json.dumps({"event": "playback_ended", "turn": 1}))
        time.sleep(0.1)
        for frame in _frames(second, len(frames)):
            ws.send_bytes(frame)
        seen = _receive_until(ws, lambda p: p.get("event") == "endpoint")
        endpoint = seen[-1][1]
        expected = (len(frames) * FRAME) + int(1.4 * SAMPLE_RATE)
        assert endpoint["turn"] == 2
        assert abs(endpoint["end_sample"] - expected) <= 1024


class _AlwaysSpeech:
    """Scorer that calls every frame speech, like Silero on residual echo."""

    backend = "always_speech"

    def score(self, frame: np.ndarray) -> float:
        return 0.99

    def reset(self) -> None:
        return None


def test_echo_floor_gating_ignores_quiet_residual_echo_but_ducks_on_loud_user() -> None:
    controller = BargeInController(scorer=_AlwaysSpeech(), echo_margin_db=12.0)
    echo = _tone(2.0, 0.01)  # about -43 dBFS: speech-like but quiet
    events = []
    for start in range(0, echo.size, FRAME):
        events.extend(controller.push(echo[start : start + FRAME]))
    assert events == []
    user = _tone(0.8, 0.3) + _tone(0.8, 0.01)
    for start in range(0, user.size, FRAME):
        events.extend(controller.push(user[start : start + FRAME]))
    assert [e.kind for e in events][:1] == ["duck"]


def test_echo_gating_can_be_disabled() -> None:
    controller = BargeInController(scorer=_AlwaysSpeech(), echo_margin_db=None)
    echo = _tone(1.0, 0.01)
    events = []
    for start in range(0, echo.size, FRAME):
        events.extend(controller.push(echo[start : start + FRAME]))
    assert events and events[0].kind == "duck"
