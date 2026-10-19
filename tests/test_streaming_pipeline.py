from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from thesis_s2s.runtime.streaming import (
    ClauseChunker,
    LatinMaskProcessor,
    latin_token_ids,
    run_streamed_turn,
)
from thesis_s2s.runtime.vad import FRAME_SAMPLES, SAMPLE_RATE, Endpointer, EnergyScorer


def _chunks(chunker: ClauseChunker, deltas: Sequence[str]) -> list[str]:
    out: list[str] = []
    for delta in deltas:
        out.extend(chunker.push(delta))
    rest = chunker.flush()
    return out + ([rest] if rest else [])


def test_short_first_clause_merges_forward() -> None:
    deltas = ["بله", "،", " حتماً", " کمکتان", " می‌کنم", ".", " هوا", " خوب", " است", "."]
    assert _chunks(ClauseChunker(), deltas) == ["بله، حتماً کمکتان می‌کنم.", "هوا خوب است."]


def test_unpunctuated_text_is_cut_at_word_limits() -> None:
    words = [f"w{i} " for i in range(1, 14)]
    chunks = _chunks(ClauseChunker(first_min_words=2, first_max_words=4, max_words=5), words)
    assert chunks == ["w1 w2 w3 w4", "w5 w6 w7 w8 w9", "w10 w11 w12 w13"]


def test_incomplete_trailing_word_is_not_cut() -> None:
    chunker = ClauseChunker(first_min_words=1, first_max_words=2, max_words=2)
    assert chunker.push("سلام خوب") == []
    assert chunker.push("ید ") == ["سلام خوبید"]


def test_invalid_chunk_limits_rejected() -> None:
    with pytest.raises(ValueError):
        ClauseChunker(first_min_words=5, first_max_words=3)


class _FakeTokenizer:
    all_special_ids = [3]
    pieces = ["سلام", "hello", " دنیا", "<eos>", "Ab"]

    def __len__(self) -> int:
        return len(self.pieces)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self.pieces[i] for i in ids)


def test_latin_token_ids_skip_special_tokens() -> None:
    assert latin_token_ids(_FakeTokenizer()) == [1, 4]


def test_latin_mask_processor_forbids_banned_ids() -> None:
    torch = pytest.importorskip("torch")
    scores = torch.zeros((1, 5))
    masked = LatinMaskProcessor([1, 4])(None, scores)
    assert torch.isinf(masked[0, 1]) and torch.isinf(masked[0, 4])
    assert masked[0, 0] == 0 and scores[0, 1] == 0


class _FakeResponder:
    def __init__(self, deltas: Sequence[str], fail: bool = False) -> None:
        self.deltas = deltas
        self.fail = fail
        self.seen: list[tuple[str, tuple]] = []

    def stream(self, text: str, history=(), cancel=None) -> Iterator[str]:
        self.seen.append((text, tuple(history)))
        yield from self.deltas
        if self.fail:
            raise RuntimeError("generation failed")


def test_streamed_turn_emits_clauses_in_order_with_timings() -> None:
    emitted: list[tuple[int, int]] = []
    responder = _FakeResponder(["سلام", " دوست", " من", ".", " خوبی", "؟"])
    turn = run_streamed_turn(
        np.zeros(1600, dtype=np.float32),
        asr=lambda audio: ("حالت چطوره", None),
        responder=responder,  # type: ignore[arg-type]
        tts=lambda text: np.ones(100 * len(text.split()), dtype=np.float32),
        emit_audio=lambda audio, index: emitted.append((index, audio.size)),
        history=[("قبلی", "پاسخ")],
    )
    assert turn.clauses == ["سلام دوست من.", "خوبی؟"]
    assert emitted == [(0, 300), (1, 100)]
    assert responder.seen == [("حالت چطوره", (("قبلی", "پاسخ"),))]
    timings = turn.timings.as_ms()
    assert all(timings[key] is not None for key in timings)
    assert timings["first_audio_ready_ms"] <= timings["turn_done_ms"]


def test_empty_transcript_speaks_repeat_request() -> None:
    spoken: list[str] = []
    turn = run_streamed_turn(
        np.zeros(10, dtype=np.float32),
        asr=lambda audio: ("", "asr failed"),
        responder=_FakeResponder([]),  # type: ignore[arg-type]
        tts=lambda text: spoken.append(text) or np.ones(5, dtype=np.float32),
        emit_audio=lambda audio, index: None,
    )
    assert turn.asr_error == "asr failed" and spoken and turn.clauses == spoken


def test_cancel_stops_further_audio() -> None:
    stop = threading.Event()
    emitted: list[int] = []

    def emit(audio: np.ndarray, index: int) -> None:
        emitted.append(index)
        stop.set()

    turn = run_streamed_turn(
        np.zeros(10, dtype=np.float32),
        asr=lambda audio: ("سلام", None),
        responder=_FakeResponder(["یک دو سه.", " چهار پنج شش.", " هفت هشت نه."]),  # type: ignore[arg-type]
        tts=lambda text: np.ones(10, dtype=np.float32),
        emit_audio=emit,
        cancel=stop,
    )
    assert emitted == [0] and turn.cancelled


def test_responder_failure_is_reported_not_raised() -> None:
    turn = run_streamed_turn(
        np.zeros(10, dtype=np.float32),
        asr=lambda audio: ("سلام", None),
        responder=_FakeResponder(["یک دو سه."], fail=True),  # type: ignore[arg-type]
        tts=lambda text: np.ones(10, dtype=np.float32),
        emit_audio=lambda audio, index: None,
    )
    assert turn.responder_error and turn.clauses == ["یک دو سه."]


def _tone(seconds: float, amplitude: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_endpointer_reports_acoustic_end_not_decision_time() -> None:
    rng = np.random.default_rng(0)
    silence = lambda s: (0.001 * rng.standard_normal(int(s * SAMPLE_RATE))).astype(np.float32)  # noqa: E731
    audio = np.concatenate([silence(1.0), _tone(1.0, 0.3), silence(1.0)])
    endpointer = Endpointer(scorer=EnergyScorer(), hangover_ms=256)
    events = []
    for start in range(0, audio.size, 320):
        events.extend(endpointer.push(audio[start : start + 320]))
    kinds = [e.kind for e in events]
    assert kinds == ["speech_start", "speech_end"]
    start_event, end_event = events
    assert abs(start_event.sample - SAMPLE_RATE) <= 2 * FRAME_SAMPLES
    assert abs(end_event.sample - 2 * SAMPLE_RATE) <= 2 * FRAME_SAMPLES
    assert end_event.decided_at_sample - end_event.sample >= 256 * SAMPLE_RATE / 1000 - FRAME_SAMPLES


def test_endpointer_ignores_clicks_shorter_than_min_speech() -> None:
    rng = np.random.default_rng(1)
    quiet = (0.001 * rng.standard_normal(SAMPLE_RATE)).astype(np.float32)
    audio = np.concatenate([quiet, _tone(0.08, 0.4), quiet])
    endpointer = Endpointer(scorer=EnergyScorer(), onset_frames=1, min_speech_ms=160)
    events = []
    for start in range(0, audio.size, 512):
        events.extend(endpointer.push(audio[start : start + 512]))
    assert [e.kind for e in events if e.kind == "speech_end"] == []
