from __future__ import annotations

import numpy as np

from thesis_s2s.bargein.lexicon import content_tokens, echo_vocabulary, listener_tokens_only
from thesis_s2s.runtime.bargein_live import BargeInController
from thesis_s2s.runtime.vad import SAMPLE_RATE, EnergyScorer

FRAME = 320


def _tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def _quiet(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.001 * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def _run(controller: BargeInController, audio: np.ndarray) -> list:
    events = []
    for start in range(0, audio.size - FRAME + 1, FRAME):
        events.extend(controller.push(audio[start : start + FRAME]))
    return events


def test_content_tokens_drop_listener_tokens_and_echo_words() -> None:
    echo = echo_vocabulary(["بهتر است با پزشک مشورت کنید."])
    assert content_tokens("آره، باشه", echo) == []
    assert content_tokens("با پزشک", echo) == []
    assert content_tokens("پزشگ", echo) == []  # one edit from an echo word
    assert content_tokens("صبر کن یه سوال دارم", echo) == ["صبر", "کن", "یه", "سوال", "دارم"]
    assert listener_tokens_only("اوهوم آره") and not listener_tokens_only("")


def test_lexical_confirm_resumes_on_backchannel() -> None:
    calls = []

    def transcribe(audio: np.ndarray) -> str:
        calls.append(audio.size)
        return "آره"

    controller = BargeInController(scorer=EnergyScorer(), echo_margin_db=None, transcriber=transcribe)
    events = _run(controller, np.concatenate([_quiet(0.5), _tone(0.7), _quiet(0.6, 2)]))
    assert [e.kind for e in events] == ["duck", "resume"]
    assert events[-1].transcript == "آره" and calls


def test_lexical_confirm_stops_on_content_words_even_for_short_speech() -> None:
    controller = BargeInController(
        scorer=EnergyScorer(), echo_margin_db=None, transcriber=lambda audio: "صبر کن"
    )
    events = _run(controller, np.concatenate([_quiet(0.5), _tone(0.3), _quiet(0.6, 2)]))
    assert [e.kind for e in events] == ["duck", "stop"]


def test_lexical_confirm_ignores_the_assistants_own_words() -> None:
    reply = echo_vocabulary(["قرص فشار را صبح بخورید"])
    controller = BargeInController(
        scorer=EnergyScorer(),
        echo_margin_db=None,
        transcriber=lambda audio: "قرص فشار صبح",
        echo_words=lambda: reply,
    )
    events = _run(controller, np.concatenate([_quiet(0.5), _tone(1.0), _quiet(0.6, 2)]))
    assert [e.kind for e in events] == ["duck", "resume"]


def test_lexical_confirm_waits_while_transcript_is_empty_then_caps() -> None:
    transcripts = iter(["", "", "", "", "", "بلیط"])
    controller = BargeInController(
        scorer=EnergyScorer(),
        echo_margin_db=None,
        transcriber=lambda audio: next(transcripts, "بلیط"),
        lexical_after_ms=400,
        lexical_interval_ms=200,
    )
    events = _run(controller, np.concatenate([_quiet(0.5), _tone(2.0), _quiet(0.6, 2)]))
    # (the server stops feeding the controller after a stop; here it keeps going)
    assert [e.kind for e in events[:2]] == ["duck", "stop"]
    # decided only after several re-checks, not at the first one
    assert events[1].decided_at_sample - events[1].onset_sample > 0.9 * SAMPLE_RATE


def test_v2_matching_accepts_words_cut_at_the_window_edges_and_drops_function_words() -> None:
    echo = echo_vocabulary(["این کار را کرده بود، یه کانورسیشن ساده"])
    assert content_tokens("کانورسی", echo) == ["کانورسی"]  # two letters cut off
    assert content_tokens("کانورسی", echo, matching="v2") == []
    assert content_tokens("کرد", echo) == ["کرد"]
    assert content_tokens("کرد", echo, matching="v2") == []
    assert content_tokens("از این", frozenset(), drop_function_words=True) == []
    assert content_tokens("از بلیط", frozenset(), drop_function_words=True) == ["بلیط"]


def test_conservative_coupling_prior_avoids_a_duck_when_the_echo_arrives_late() -> None:
    from thesis_s2s.runtime.vad import FRAME_SAMPLES

    class _AlwaysSpeech:
        def score(self, frame: np.ndarray) -> float:
            return 0.99

        def reset(self) -> None:
            return None

    def run(prior: float | None) -> list:
        controller = BargeInController(
            scorer=_AlwaysSpeech(), echo_margin_db=12.0, dtd_margin_db=14.0, initial_coupling_db=prior
        )
        played = _tone(1.5, 0.1)  # reference playback level (about -23 dBFS)
        delay = int(0.06 * SAMPLE_RATE)
        quiet = 0.0003 * np.random.default_rng(0).standard_normal(delay).astype(np.float32)
        mic = np.concatenate([quiet, 0.2 * played[:-delay]])  # echo 14 dB down, 60 ms late
        level = np.full(played.size, float(np.sqrt(np.mean(played**2))), np.float32)
        events = []
        for start in range(0, played.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            events.extend(
                controller.push(
                    mic[start : start + FRAME_SAMPLES], reference_level=level[start : start + FRAME_SAMPLES]
                )
            )
        return events

    assert [e.kind for e in run(None)][:1] == ["duck"]  # legacy: coupling learned before the echo
    assert run(0.0) == []
