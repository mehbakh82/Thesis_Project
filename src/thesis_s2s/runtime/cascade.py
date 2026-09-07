"""NeMo streaming cascade: ASR HTTP + validated local responder + Piper TTS."""

from __future__ import annotations

import json
import os
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.data.verbatim import verbatim_normalize
from thesis_s2s.runtime.tts import FormantTalker, synthesize

QWEN4B_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
QWEN4B_PROJECT_DIR = Path(__file__).resolve().parents[3] / "models/qwen3-4b-instruct-2507"
QWEN4B_PROMPT_V2 = (
    "متن کاربر ممکن است خروجی ناقص گفتاربه‌متن باشد. منظور اصلی قابل‌فهم را تشخیص بده و "
    "دقیقاً یک جمله کامل و روان فارسی با حداکثر بیست‌وپنج واژه بنویس که مستقیماً به همان "
    "منظور پاسخ دهد. متن کاربر را تکرار نکن، جمله را نیمه‌تمام نگذار، حرف لاتین، فهرست یا "
    "توضیح حاشیه‌ای ننویس. اگر هیچ منظوری قابل‌فهم نیست، در یک جمله درخواست تکرار کن."
)
QWEN4B_RETRY_PROMPT_V2 = (
    "فقط یک جمله کامل، کوتاه، روان و مرتبط با منظور اصلی کاربر به خط فارسی بنویس. "
    "هیچ حرف لاتین، کد، فهرست، تکرار ورودی یا جمله نیمه‌تمام ننویس."
)


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


def _persian_letter_fraction(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    persian = [
        character for character in letters if "ARABIC" in unicodedata.name(character, "")
    ]
    return len(persian) / max(1, len(letters))


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
        configured_model = model_name or os.environ.get("TEXT_LLM_MODEL")
        self.model_name = configured_model or QWEN4B_MODEL
        self.model_source = (
            str(QWEN4B_PROJECT_DIR)
            if configured_model is None and QWEN4B_PROJECT_DIR.is_dir()
            else self.model_name
        )
        configured_profile = os.environ.get("TEXT_LLM_PROMPT_PROFILE", "auto")
        self.prompt_profile = (
            "qwen4b_v2"
            if configured_profile == "qwen4b_v2"
            or (configured_profile == "auto" and QWEN4B_MODEL in self.model_name)
            else "legacy"
        )
        self.backend = "rules"
        self.model: Any = None
        self.tokenizer: Any = None
        self.device = "cpu"
        self.initialization_error: str | None = None
        self.last_fallback_used = True
        self.last_generation_error: str | None = None
        self.last_generation_attempts = 0
        self.last_language_retry_used = False
        if os.environ.get("TEXT_LLM_ENABLED", "1").lower() in {"0", "false", "no"}:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_source, local_files_only=True
            )
            loaded_model: Any = AutoModelForCausalLM.from_pretrained(
                self.model_source,
                local_files_only=True,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
            self.model = loaded_model.to(self.device).eval()
            self.backend = self.model_name
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.initialization_error = f"{type(exc).__name__}: {exc}"

    def reply(self, user_text: str) -> str:
        self.last_generation_error = None
        self.last_generation_attempts = 0
        self.last_language_retry_used = False
        if self.model is None or self.tokenizer is None:
            self.last_fallback_used = True
            return _reply_text(user_text)
        system_prompts = (
            [QWEN4B_PROMPT_V2, QWEN4B_RETRY_PROMPT_V2]
            if self.prompt_profile == "qwen4b_v2"
            else [
                "فقط با خط فارسی و بدون هیچ حرف یا واژه لاتین، کوتاه و طبیعی پاسخ بده.",
                (
                    "در یک جمله کوتاه و مرتبط پاسخ بده. پاسخ فقط باید شامل خط فارسی باشد؛ "
                    "هیچ کد، حرف انگلیسی یا واژه لاتین ننویس."
                ),
            ]
        )
        for attempt_index, system_prompt in enumerate(system_prompts):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
            try:
                self.last_generation_attempts += 1
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
                if answer and _persian_letter_fraction(answer) >= 0.8:
                    self.last_generation_error = None
                    self.last_fallback_used = False
                    return answer
                self.last_generation_error = "model response failed the Persian-script constraint"
                if attempt_index == 0:
                    self.last_language_retry_used = True
                    continue
            except Exception as exc:
                self.last_generation_error = f"{type(exc).__name__}: {exc!r}"
                break
        self.last_fallback_used = True
        return _reply_text(user_text)


class CascadeTalker:
    """Persian ASR -> Qwen3-4B prompt-v2 -> full Piper/formant speech system."""

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
        self.last_responder_generation_attempts: int | None = None
        self.last_responder_language_retry_used: bool | None = None
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
            self.last_responder_generation_attempts = None
            self.last_responder_language_retry_used = None
        else:
            reply = self.responder.reply(user_text)
            self.last_responder_fallback_used = self.responder.last_fallback_used
            self.last_responder_error = self.responder.last_generation_error
            self.last_responder_generation_attempts = self.responder.last_generation_attempts
            self.last_responder_language_retry_used = self.responder.last_language_retry_used
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
