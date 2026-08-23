"""SNR, script LID, and music/caption filters for the conversational mix."""

from __future__ import annotations

import re

import numpy as np

from thesis_s2s.bargein.features import FeatureConfig, log_energy

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
MUSIC_RE = re.compile(r"(موسیقی|music|applause|\[.*\])", re.IGNORECASE)


def estimate_snr_db(audio: np.ndarray, cfg: FeatureConfig | None = None) -> float:
    """Frame-energy SNR proxy: p90 speech minus p10 noise floor."""

    cfg = cfg or FeatureConfig()
    audio = np.asarray(audio, dtype=np.float32)
    win, hop = cfg.win, cfg.hop
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    n = 1 + (len(audio) - win) // hop
    index = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    energy = log_energy(audio[index])
    if len(energy) < 4:
        return float("nan")
    return float(np.percentile(energy, 90) - np.percentile(energy, 10))


def arabic_script_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha() or ARABIC_RE.search(ch)]
    if not letters:
        return 0.0
    arabic = sum(1 for ch in letters if ARABIC_RE.search(ch))
    return arabic / len(letters)


def looks_persian(text: str, min_ratio: float = 0.35) -> bool:
    if not text or not text.strip():
        return False
    return arabic_script_ratio(text) >= min_ratio


def looks_music_only(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 4:
        return True
    if MUSIC_RE.fullmatch(stripped):
        return True
    return False


def conversational_ok(
    *,
    duration: float,
    text: str,
    snr_db: float | None,
    language_status: str | None = None,
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    min_snr_db: float = 3.0,
    min_words: int = 3,
) -> tuple[bool, str]:
    if duration < min_duration or duration > max_duration:
        return False, "duration"
    if language_status in {"no_persian_speech", "no_speech", "language_uncertain"}:
        return False, f"lid:{language_status}"
    if looks_music_only(text):
        return False, "music"
    if not looks_persian(text):
        return False, "not_persian"
    if len(text.split()) < min_words:
        return False, "short_text"
    if snr_db is not None and snr_db == snr_db and snr_db < min_snr_db:
        return False, "low_snr"
    return True, "ok"
