"""Conservative S2S/TTS text: YouTube CSV captions after fa-verbatim-2, never ASR normalise()."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _try_serving_normalizer():
    roots = [
        Path("/mnt/md0/mehbakh/asr_nemo_soroush/asr_api"),
        Path(__file__).resolve().parents[3] / "asr_nemo_soroush" / "asr_api",
    ]
    for root in roots:
        if not (root / "postprocessing.py").exists():
            continue
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from postprocessing import PersianTextNormalizer  # type: ignore[import-not-found]

            return PersianTextNormalizer(enabled=True)
        except Exception:
            continue
    return None


_SERVING: Any = None


def s2s_text(row: dict) -> str:
    """Spoken S2S/TTS target: CSV caption, never NeMo/Whisper teacher ASR."""

    for key in ("transcript_caption", "text"):
        val = str(row.get(key) or "").strip()
        if val:
            return val
    return ""


def verbatim_normalize(text: str) -> str:
    """fa-verbatim-2 when the serving repo is importable; else a safe subset."""

    global _SERVING
    if _SERVING is False:
        return _fallback(text)
    if _SERVING is None:
        _SERVING = _try_serving_normalizer() or False
    if _SERVING is False:
        return _fallback(text)
    result = _SERVING.normalize(text, requested=True)
    return result.text


def _fallback(text: str) -> str:
    table = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ؤ": "و", "إ": "ا", "أ": "ا"})
    cleaned = unicodedata_nfc(text).translate(table)
    return " ".join(cleaned.split())


def unicodedata_nfc(text: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFC", text)
