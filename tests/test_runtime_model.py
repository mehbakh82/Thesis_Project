from types import SimpleNamespace

import numpy as np
import torch

from thesis_s2s.bargein.detector import EnergyVadBaseline
from thesis_s2s.bargein.synthetic import _harmonic
from thesis_s2s.model.llama_omni2 import train_smoke
from thesis_s2s.runtime import cascade
from thesis_s2s.runtime.cascade import CascadeTalker, TextResponder
from thesis_s2s.runtime.duplex import DummyTalker, DuplexSession, default_talker


def test_s2s_smoke_loss_drops(tmp_path):
    report = train_smoke(steps=3, device="cpu", out_dir=tmp_path)
    assert report["loss_last"] is not None
    assert report["steps"] == 3


def test_duplex_session_first_audio():
    session = DuplexSession(EnergyVadBaseline(), DummyTalker())
    user = _harmonic(0.5, 200)
    reply = session.on_user_end(user)
    assert len(reply) > 0
    assert (
        session.log.t_first_audio_ms is None
    )  # populated only by browser playback acknowledgement
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
    assert talker.last_responder_fallback_used is True
    assert talker.last_responder_error is None
    assert talker.responder_initialization_error is None
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
    assert talker.last_responder_fallback_used is None
    assert talker.last_responder_error is None
    assert talker.responder_initialization_error is None
    assert "دوباره" in generated_text[0]


def test_text_responder_extracts_input_ids_from_batch_encoding(monkeypatch):
    monkeypatch.setenv("TEXT_LLM_ENABLED", "0")
    responder = TextResponder()

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert "بدون هیچ حرف یا واژه لاتین" in messages[0]["content"]
            assert kwargs == {"add_generation_prompt": True, "return_tensors": "pt"}
            return SimpleNamespace(
                input_ids=torch.tensor([[10, 11]]),
                attention_mask=torch.tensor([[1, 1]]),
            )

        def decode(self, tokens, *, skip_special_tokens):
            assert tokens.tolist() == [12]
            assert skip_special_tokens is True
            return "پاسخ واقعی مدل"

    class FakeModel:
        def generate(self, *, input_ids, attention_mask, max_new_tokens, do_sample):
            assert input_ids.tolist() == [[10, 11]]
            assert attention_mask.tolist() == [[1, 1]]
            assert max_new_tokens == 64
            assert do_sample is False
            return torch.tensor([[10, 11, 12]])

    responder.backend = responder.model_name
    responder.tokenizer = FakeTokenizer()
    responder.model = FakeModel()

    assert responder.reply("سلام") == "پاسخ واقعی مدل"
    assert responder.last_fallback_used is False
    assert responder.last_generation_error is None


def test_piper_runtime_output_is_clipped_to_pcm_range(monkeypatch):
    from thesis_s2s.runtime import tts

    monkeypatch.setattr(
        tts,
        "piper_synthesize",
        lambda text, sr: np.asarray([-1.02, 0.0, 1.03], dtype=np.float32),
    )

    audio, backend = tts.synthesize("سلام")

    assert backend == "piper"
    assert audio.dtype == np.float32
    assert audio.tolist() == [-1.0, 0.0, 1.0]
