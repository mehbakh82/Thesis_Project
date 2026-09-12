import sys
from types import SimpleNamespace

import numpy as np
import pytest
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
    assert talker.last_responder_generation_attempts == 0
    assert talker.last_responder_language_retry_used is False
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
    assert talker.last_responder_generation_attempts is None
    assert talker.last_responder_language_retry_used is None
    assert talker.responder_initialization_error is None
    assert "دوباره" in generated_text[0]


def test_text_responder_extracts_input_ids_from_batch_encoding(monkeypatch):
    monkeypatch.setenv("TEXT_LLM_ENABLED", "0")
    monkeypatch.setenv("TEXT_LLM_PROMPT_PROFILE", "legacy")
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
    assert responder.last_generation_attempts == 1
    assert responder.last_language_retry_used is False


def test_text_responder_retries_qwen_once_for_non_persian_output(monkeypatch):
    monkeypatch.setenv("TEXT_LLM_ENABLED", "0")
    monkeypatch.setenv("TEXT_LLM_PROMPT_PROFILE", "legacy")
    responder = TextResponder()
    replies = iter(["visit example dot com", "این یک پاسخ فارسی است"])
    prompts: list[str] = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return SimpleNamespace(
                input_ids=torch.tensor([[10, 11]]),
                attention_mask=torch.tensor([[1, 1]]),
            )

        def decode(self, tokens, *, skip_special_tokens):
            return next(replies)

    class FakeModel:
        def generate(self, **kwargs):
            return torch.tensor([[10, 11, 12]])

    responder.backend = responder.model_name
    responder.tokenizer = FakeTokenizer()
    responder.model = FakeModel()

    assert responder.reply("یک سایت معرفی کن") == "این یک پاسخ فارسی است"
    assert len(prompts) == 2
    assert "هیچ کد" in prompts[1]
    assert responder.last_fallback_used is False
    assert responder.last_generation_error is None
    assert responder.last_generation_attempts == 2
    assert responder.last_language_retry_used is True


def test_default_responder_uses_validated_qwen4b_prompt_v2(monkeypatch):
    monkeypatch.setenv("TEXT_LLM_ENABLED", "0")
    monkeypatch.delenv("TEXT_LLM_MODEL", raising=False)
    monkeypatch.delenv("TEXT_LLM_PROMPT_PROFILE", raising=False)

    responder = TextResponder()

    assert responder.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert responder.model_revision == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert responder.prompt_profile == "qwen4b_v2"
    assert responder.model_source in {
        "Qwen/Qwen3-4B-Instruct-2507",
        str(cascade.QWEN4B_PROJECT_DIR),
    }


def test_default_cached_hub_load_is_bound_to_exact_qwen_revision(tmp_path, monkeypatch):
    calls: list[tuple[str, dict]] = []

    class Loader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return Loader()

        def to(self, _device):
            return self

        def eval(self):
            return self

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        bfloat16="bfloat16",
        float32="float32",
    )
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=Loader,
        AutoTokenizer=Loader,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(cascade, "QWEN4B_PROJECT_DIR", tmp_path / "missing")
    monkeypatch.setenv("TEXT_LLM_ENABLED", "1")
    monkeypatch.delenv("TEXT_LLM_MODEL", raising=False)
    monkeypatch.setenv("TEXT_LLM_REVISION", "unvalidated-override")

    responder = TextResponder()

    assert responder.initialization_error is None
    assert responder.model_revision == cascade.QWEN4B_REVISION
    assert len(calls) == 2
    assert all(source == cascade.QWEN4B_MODEL for source, _ in calls)
    assert all(kwargs["revision"] == cascade.QWEN4B_REVISION for _, kwargs in calls)


def test_asr_http_rejects_unbounded_or_malformed_responses(monkeypatch):
    class Response:
        def __init__(self, payload: bytes):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, limit: int) -> bytes:
            return self.payload[:limit]

    monkeypatch.setattr(cascade, "urlopen", lambda *_args, **_kwargs: Response(b"[]"))
    assert cascade.asr_http(b"wav")["error"].startswith("ValueError")

    oversized = b"x" * (cascade.MAX_ASR_RESPONSE_BYTES + 1)
    monkeypatch.setattr(cascade, "urlopen", lambda *_args, **_kwargs: Response(oversized))
    assert cascade.asr_http(b"wav")["error"].startswith("ValueError")

    assert cascade.asr_http(b"")["error"].startswith("ValueError")
    assert cascade.asr_http(b"wav", "https://user:secret@example.invalid")[
        "error"
    ].startswith("ValueError")

    monkeypatch.setattr(
        cascade,
        "urlopen",
        lambda *_args, **_kwargs: Response('{"persian":" سلام "}'.encode()),
    )
    assert cascade._asr_transcript(cascade.asr_http(b"wav")) == "سلام"


def test_asr_timeout_and_wav_serialization_fail_closed(monkeypatch):
    for value in ("0", "-1", "nan", "inf", "invalid"):
        monkeypatch.setenv("ASR_TIMEOUT_SECONDS", value)
        assert cascade.asr_http(b"wav")["error"].startswith("ValueError")

    for audio in (
        np.zeros(0, dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.asarray([float("nan")], dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="user audio"):
            cascade._wav_bytes(audio)
    with pytest.raises(ValueError, match="sample rate"):
        cascade._wav_bytes(np.zeros(10, dtype=np.float32), 0)


def test_component_diagnostic_never_invents_transcript_or_official_timing(monkeypatch):
    monkeypatch.setattr(cascade, "asr_http", lambda _wav: {"error": "offline"})
    monkeypatch.setattr(
        cascade,
        "synthesize",
        lambda text: (np.ones(800, dtype=np.float32), "test-tts"),
    )

    result = cascade.cascade_first_audio(np.zeros(1600, dtype=np.float32))

    assert result["transcript"] == ""
    assert result["asr_usable"] is False
    assert result["official_e2e_eligible"] is False
    assert result["evidence_class"] == "component_diagnostic"
    assert "دوباره" in result["reply_text"]


def test_qwen_v2_reply_validator_rejects_latin_or_overlong_output() -> None:
    assert cascade._valid_model_reply("این پاسخ فارسی است", strict_v2=True)
    assert not cascade._valid_model_reply("این پاسخ fa است", strict_v2=True)
    assert not cascade._valid_model_reply(" ".join(["واژه"] * 26), strict_v2=True)


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
