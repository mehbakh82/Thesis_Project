import numpy as np

from thesis_s2s.bargein.detector import EnergyVadBaseline
from thesis_s2s.bargein.synthetic import _harmonic
from thesis_s2s.model.llama_omni2 import train_smoke
from thesis_s2s.runtime import cascade
from thesis_s2s.runtime.cascade import CascadeTalker, TextResponder
from thesis_s2s.runtime.duplex import DummyTalker, DuplexSession, default_talker


def test_s2s_smoke_loss_drops():
    report = train_smoke(steps=3, device="cpu")
    assert report["loss_last"] is not None
    assert report["steps"] == 3


def test_duplex_session_first_audio():
    session = DuplexSession(EnergyVadBaseline(), DummyTalker())
    user = _harmonic(0.5, 200)
    reply = session.on_user_end(user)
    assert len(reply) > 0
    assert session.log.t_first_audio_ms is None  # populated only by browser playback acknowledgement
    assert session.log.server_generation_ms is not None
    assert session.log.server_generation_ms < 500


def test_default_talker_is_always_the_implemented_cascade(monkeypatch):
    monkeypatch.setenv("TEXT_LLM_ENABLED", "0")
    assert isinstance(default_talker(), CascadeTalker)


def test_cascade_generates_and_returns_the_complete_reply(monkeypatch):
    monkeypatch.setenv("TEXT_LLM_ENABLED", "0")
    expected = np.linspace(-0.3, 0.3, 12_000, dtype=np.float32)
    generated_text: list[str] = []

    def fake_synthesize(text: str):
        generated_text.append(text)
        return expected, "test-tts"

    monkeypatch.setattr(cascade, "synthesize", fake_synthesize)
    talker = CascadeTalker(TextResponder())
    reply = talker.reply_audio(np.zeros(1600, dtype=np.float32), text="سلام")

    assert np.array_equal(reply, expected)
    assert talker.last_transcript == "سلام"
    assert talker.last_reply_text == "سلام، چطورید؟"
    assert generated_text == [talker.last_reply_text]
    assert talker.backend == "test-tts"


def test_cascade_handles_failed_asr_without_inventing_a_transcript(monkeypatch):
    monkeypatch.setenv("TEXT_LLM_ENABLED", "0")
    generated_text: list[str] = []
    monkeypatch.setattr(cascade, "asr_http", lambda _: {"error": "offline", "persian": None})
    monkeypatch.setattr(
        cascade,
        "synthesize",
        lambda text: (generated_text.append(text) or np.ones(800, dtype=np.float32), "test-tts"),
    )

    talker = CascadeTalker(TextResponder())
    reply = talker.reply_audio(np.zeros(1600, dtype=np.float32))

    assert len(reply) == 800
    assert talker.last_transcript == ""
    assert talker.last_asr_error == "offline"
    assert "دوباره" in generated_text[0]
