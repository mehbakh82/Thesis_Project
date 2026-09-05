"""NeMo streaming cascade baseline: ASR HTTP + small LLM/text + Piper/formant TTS."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.data.verbatim import verbatim_normalize
from thesis_s2s.runtime.tts import FormantTalker, synthesize


def asr_http(wav_bytes: bytes, base: str | None = None) -> dict:
    base = (base or os.environ.get("ASR_API_URL") or "http://127.0.0.1:8090").rstrip("/")
    req = Request(base + "/transcribe/binary?align=true", data=wav_bytes, method="POST")
    req.add_header("content-type", "application/octet-stream")
    try:
        timeout = float(os.environ.get("ASR_TIMEOUT_SECONDS", "3"))
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"error": str(exc), "persian": None}


def _reply_text(user_text: str) -> str:
    t = user_text.strip()
    if "سلام" in t:
        return "سلام، چطورید؟"
    if "خداحافظ" in t or "بای" in t:
        return "خداحافظ، روز خوبی داشته باشید."
    if "ساعت" in t:
        return "ساعت را دقیق نمی‌دانم، ولی گوش می‌دهم."
    if len(t) < 2:
        return "بفرمایید."
    return "متوجه شدم. بفرمایید ادامه دهید."


class TextResponder:
    """Local Persian Qwen responder with a deterministic rule fallback."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get(
            "TEXT_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )
        self.backend = "rules"
        self.model: Any = None
        self.tokenizer: Any = None
        self.device = "cpu"
        self.initialization_error: str | None = None
        self.last_fallback_used = True
        self.last_generation_error: str | None = None
        if os.environ.get("TEXT_LLM_ENABLED", "1").lower() in {"0", "false", "no"}:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            loaded_model: Any = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
            self.model = loaded_model.to(self.device).eval()
            self.backend = self.model_name
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.initialization_error = f"{type(exc).__name__}: {exc}"

    def reply(self, user_text: str) -> str:
        self.last_generation_error = None
        if self.model is None or self.tokenizer is None:
            self.last_fallback_used = True
            return _reply_text(user_text)
        messages = [
            {
                "role": "system",
                "content": "فقط با خط فارسی و بدون هیچ حرف یا واژه لاتین، کوتاه و طبیعی پاسخ بده.",
            },
            {"role": "user", "content": user_text},
        ]
        try:
            encoded: Any = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            tokens: Any = getattr(encoded, "input_ids", None)
            if tokens is None:
                tokens = encoded
            tokens = tokens.to(self.device)
            attention_mask: Any = getattr(encoded, "attention_mask", None)
            generation_arguments: dict[str, Any] = {
                "input_ids": tokens,
                "max_new_tokens": 64,
                "do_sample": False,
            }
            if attention_mask is not None:
                generation_arguments["attention_mask"] = attention_mask.to(self.device)
            output = self.model.generate(**generation_arguments)
            answer = str(
                self.tokenizer.decode(output[0, tokens.shape[-1] :], skip_special_tokens=True)
            ).strip()
            if answer:
                self.last_fallback_used = False
                return answer
            self.last_generation_error = "empty model response"
        except Exception as exc:
            self.last_generation_error = f"{type(exc).__name__}: {exc!r}"
        self.last_fallback_used = True
        return _reply_text(user_text)


class CascadeTalker:
    """Working Persian ASR -> text responder -> full Piper/formant speech baseline."""

    def __init__(self, responder: TextResponder | None = None):
        self.responder = responder or TextResponder()
        self.backend = "uninitialized"
        self.asr_backend = "nemo-soroush-http"
        self.responder_backend = self.responder.backend
        self.responder_initialization_error = self.responder.initialization_error
        self.last_transcript = ""
        self.last_reply_text = ""
        self.last_responder_fallback_used: bool | None = None
        self.last_responder_error: str | None = None
        self.last_asr_error: str | None = None

    def reply_audio(self, user_audio: np.ndarray, text: str | None = None) -> np.ndarray:
        if text is None:
            asr = asr_http(_wav_bytes(user_audio))
            self.last_asr_error = asr.get("error")
            user_text = verbatim_normalize(str(asr.get("persian") or asr.get("text") or ""))
        else:
            self.last_asr_error = None
            user_text = verbatim_normalize(text)
        self.last_transcript = user_text
        if not user_text:
            reply = "متأسفم، صدای شما را درست نشنیدم. لطفاً دوباره بگویید."
            self.last_responder_fallback_used = None
            self.last_responder_error = None
        else:
            reply = self.responder.reply(user_text)
            self.last_responder_fallback_used = self.responder.last_fallback_used
            self.last_responder_error = self.responder.last_generation_error
        audio, backend = synthesize(reply)
        self.backend = backend
        self.responder_backend = self.responder.backend
        self.responder_initialization_error = self.responder.initialization_error
        self.last_reply_text = reply
        return audio

    def first_chunk(self, user_audio: np.ndarray, text: str | None = None) -> np.ndarray:
        return self.reply_audio(user_audio, text)[: int(0.25 * SAMPLE_RATE)]


def _wav_bytes(user_audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    import io
    import wave

    pcm = np.clip(user_audio * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return buf.getvalue()


def cascade_first_audio(user_audio: np.ndarray) -> dict:
    """Cascade T_first_audio: NeMo ASR (if up) + formant/Piper TTS. Sortformer stays off."""

    t0 = time.perf_counter()
    asr = asr_http(_wav_bytes(user_audio))
    text = verbatim_normalize(str(asr.get("persian") or asr.get("text") or "سلام"))
    reply = _reply_text(text)
    audio, backend = synthesize(reply)
    first = audio[: int(0.25 * SAMPLE_RATE)]
    t_first = 1000.0 * (time.perf_counter() - t0)
    return {
        "system": "nemo_cascade",
        "transcript": text,
        "reply_text": reply,
        "tts_backend": backend,
        "t_first_audio_ms": t_first,
        "reply_samples": int(len(first)),
        "asr_error": asr.get("error"),
        "note": "Piper used if PIPER_MODEL is set; else Persian formant TTS. Sortformer off the S2S hot path.",
        "sample_rate": SAMPLE_RATE,
    }


def cascade_talker() -> FormantTalker:
    return FormantTalker()
