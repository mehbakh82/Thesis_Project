"""NeMo streaming cascade: ASR HTTP + validated local responder + Piper TTS."""

from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.data.verbatim import verbatim_normalize
from thesis_s2s.runtime.tts import FormantTalker, synthesize

QWEN4B_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
QWEN4B_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
QWEN4B_PROJECT_DIR = Path(__file__).resolve().parents[3] / "models/qwen3-4b-instruct-2507"
MAX_ASR_RESPONSE_BYTES = 1_048_576
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


def _asr_timeout_seconds() -> float:
    raw = os.environ.get("ASR_TIMEOUT_SECONDS", "3")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("ASR_TIMEOUT_SECONDS must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("ASR_TIMEOUT_SECONDS must be a positive finite number")
    return timeout


def _validated_asr_base(value: str) -> str:
    base = value.strip().rstrip("/")
    parsed = urlsplit(base)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "ASR_API_URL must be an http(s) base URL without credentials, query, or fragment"
        )
    return base


def asr_http(wav_bytes: bytes, base: str | None = None) -> dict[str, Any]:
    try:
        if not isinstance(wav_bytes, bytes) or not wav_bytes:
            raise ValueError("ASR request body must be non-empty bytes")
        resolved_base = _validated_asr_base(
            base or os.environ.get("ASR_API_URL") or "http://127.0.0.1:8090"
        )
        req = Request(
            resolved_base + "/transcribe/binary?align=true", data=wav_bytes, method="POST"
        )
        req.add_header("content-type", "application/octet-stream")
        timeout = _asr_timeout_seconds()
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read(MAX_ASR_RESPONSE_BYTES + 1)
        if len(raw) > MAX_ASR_RESPONSE_BYTES:
            raise ValueError("ASR response exceeded the size limit")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ASR response must be a JSON object")
        return payload
    except (URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: ASR request failed", "persian": None}


def _asr_transcript(payload: dict[str, Any]) -> str:
    for key in ("persian", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return verbatim_normalize(value)
    return ""


def _persian_letter_fraction(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    persian = [
        character for character in letters if "ARABIC" in unicodedata.name(character, "")
    ]
    return len(persian) / max(1, len(letters))


def _valid_model_reply(text: str, *, strict_v2: bool) -> bool:
    if not text or _persian_letter_fraction(text) < 0.8:
        return False
    if re.search(r"[A-Za-z]", text):
        return False
    return not strict_v2 or len(text.split()) <= 25


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
        self.model_revision = (
            QWEN4B_REVISION
            if self.model_name == QWEN4B_MODEL
            else os.environ.get("TEXT_LLM_REVISION", "").strip() or None
        )
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
                self.model_source,
                local_files_only=True,
                **(
                    {"revision": self.model_revision}
                    if self.model_revision and not Path(self.model_source).is_dir()
                    else {}
                ),
            )
            loaded_model: Any = AutoModelForCausalLM.from_pretrained(
                self.model_source,
                local_files_only=True,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                **(
                    {"revision": self.model_revision}
                    if self.model_revision and not Path(self.model_source).is_dir()
                    else {}
                ),
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
                if _valid_model_reply(answer, strict_v2=self.prompt_profile == "qwen4b_v2"):
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
            self.last_asr_error = str(asr["error"]) if asr.get("error") else None
            user_text = _asr_transcript(asr)
            if not user_text and self.last_asr_error is None:
                self.last_asr_error = "ASR response contained no usable transcript"
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

    samples = np.asarray(user_audio, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("user audio must be a non-empty finite one-dimensional array")
    if not isinstance(sr, int) or isinstance(sr, bool) or sr <= 0:
        raise ValueError("sample rate must be a positive integer")
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return buf.getvalue()


def cascade_first_audio(user_audio: np.ndarray) -> dict:
    """Unofficial component diagnostic; browser timing remains the E2E authority."""

    t0 = time.perf_counter()
    asr = asr_http(_wav_bytes(user_audio))
    text = _asr_transcript(asr)
    reply = (
        _reply_text(text)
        if text
        else "متأسفم، صدای شما را درست نشنیدم. لطفاً دوباره بگویید."
    )
    audio, backend = synthesize(reply)
    first = audio[: int(0.25 * SAMPLE_RATE)]
    t_first = 1000.0 * (time.perf_counter() - t0)
    return {
        "system": "nemo_cascade_rule_diagnostic",
        "evidence_class": "component_diagnostic",
        "official_e2e_eligible": False,
        "asr_usable": bool(text),
        "transcript": text,
        "reply_text": reply,
        "tts_backend": backend,
        "t_first_audio_ms": t_first,
        "reply_samples": int(len(first)),
        "asr_error": asr.get("error"),
        "note": (
            "Rule responder diagnostic only; Piper is used if PIPER_MODEL is set, otherwise "
            "formant TTS. This is never browser-measured end-to-end evidence."
        ),
        "sample_rate": SAMPLE_RATE,
    }


def cascade_talker() -> FormantTalker:
    return FormantTalker()
