"""Persian speech output: Piper if usable, else a formant diagnostic fallback."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import read_wav

# Approximate Persian grapheme → IPA-ish phones (no short-vowel restoration).
G2P = {
    "ا": "a",
    "آ": "a",
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ث": "s",
    "ج": "dZ",
    "چ": "tS",
    "ح": "h",
    "خ": "x",
    "د": "d",
    "ذ": "z",
    "ر": "r",
    "ز": "z",
    "ژ": "Z",
    "س": "s",
    "ش": "S",
    "ص": "s",
    "ض": "z",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "G",
    "ف": "f",
    "ق": "q",
    "ک": "k",
    "ك": "k",
    "گ": "g",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "و": "v",
    "ه": "h",
    "ی": "i",
    "ي": "i",
    "ئ": "i",
    "ء": "",
    " ": " ",
    "،": " ",
    ".": " ",
    "؟": " ",
    "!": " ",
}

# F1, F2, duration_ms for vowels / voiced defaults
FORMANTS = {
    "a": (750, 1200, 90),
    "e": (500, 1800, 70),
    "i": (300, 2300, 70),
    "o": (500, 900, 80),
    "u": (350, 800, 80),
    "b": (200, 800, 50),
    "p": (200, 900, 40),
    "t": (400, 1800, 40),
    "d": (400, 1700, 45),
    "k": (400, 2000, 40),
    "g": (400, 1900, 45),
    "q": (350, 1100, 50),
    "f": (500, 1400, 50),
    "s": (400, 5000, 55),
    "z": (400, 4500, 50),
    "S": (400, 2400, 60),
    "Z": (400, 2200, 55),
    "tS": (400, 2300, 55),
    "dZ": (400, 2100, 55),
    "x": (500, 1600, 55),
    "G": (450, 1200, 55),
    "h": (600, 1400, 40),
    "l": (400, 1400, 50),
    "r": (450, 1500, 40),
    "m": (250, 1000, 50),
    "n": (300, 1400, 50),
    "v": (400, 1200, 50),
    " ": (0, 0, 40),
}


def g2p(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    phones: list[str] = []
    for ch in text:
        phones.append(G2P.get(ch, "a" if "\u0600" <= ch <= "\u06ff" else " "))
    if not phones:
        phones = ["a"]
    return phones


def _buzz(n: int, f0: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / sr
    sig = 0.4 * np.sin(2 * np.pi * f0 * t)
    sig += 0.15 * np.sin(2 * np.pi * 2 * f0 * t)
    sig += 0.04 * rng.normal(size=n)
    return sig.astype(np.float32)


def _resonator(x: np.ndarray, freq: float, sr: int, bw: float = 80.0) -> np.ndarray:
    if freq <= 0:
        return np.zeros_like(x)
    r = np.exp(-np.pi * bw / sr)
    omega = 2 * np.pi * freq / sr
    a1 = 2 * r * np.cos(omega)
    a2 = -(r**2)
    y = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        y[i] = x[i] + a1 * (y[i - 1] if i else 0) + a2 * (y[i - 2] if i > 1 else 0)
    peak = np.max(np.abs(y)) + 1e-9
    return (0.35 * y / peak).astype(np.float32)


def _text_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _validate_synthesis_inputs(text: str, sr: int) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("synthesis text must be a non-empty string")
    if not isinstance(sr, int) or isinstance(sr, bool) or sr <= 0:
        raise ValueError("synthesis sample rate must be a positive integer")


def formant_synthesize(text: str, sr: int = SAMPLE_RATE, f0: float = 140.0) -> np.ndarray:
    _validate_synthesis_inputs(text, sr)
    if not math.isfinite(f0) or f0 <= 0:
        raise ValueError("formant fundamental frequency must be positive and finite")
    rng = np.random.default_rng(_text_seed(text))
    chunks: list[np.ndarray] = []
    for phone in g2p(text):
        f1, f2, dur_ms = FORMANTS.get(phone, (500, 1500, 60))
        n = max(1, int(sr * dur_ms / 1000.0))
        if phone == " " or f1 == 0:
            chunks.append(np.zeros(n, dtype=np.float32))
            continue
        src = _buzz(n, f0 + rng.uniform(-8, 8), sr, rng)
        y = 0.6 * _resonator(src, f1, sr) + 0.4 * _resonator(src, f2, sr, bw=120)
        fade = min(40, n // 4)
        env: np.ndarray = np.ones(n, dtype=np.float32)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        chunks.append((y * env).astype(np.float32))
    audio = np.concatenate(chunks) if chunks else _buzz(sr // 4, f0, sr, rng)
    peak = np.max(np.abs(audio)) + 1e-9
    return (0.85 * audio / peak).astype(np.float32)


def project_piper_dir() -> Path:
    from thesis_s2s.config import project_root

    return project_root() / "models" / "piper"


def os_piper_model() -> Path | None:
    env = os.environ.get("PIPER_MODEL")
    if env:
        configured = Path(env)
        return configured if configured.is_file() else None
    preferred = project_piper_dir() / "fa_IR-mana-medium.onnx"
    if preferred.is_file():
        return preferred
    return None


def piper_available() -> bool:
    return os_piper_model() is not None


def warm_piper() -> str | None:
    """Load the ONNX voice once so duplex T_first_audio is not the cold-start."""

    if not piper_available():
        return None
    audio = piper_synthesize("سلام")
    return "piper" if audio is not None and len(audio) > 0 else None


def _pcm_from_wav_bytes(raw: bytes, sr: int) -> np.ndarray | None:
    import io
    import wave

    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                return None
            frames = wav.readframes(wav.getnframes())
            src_sr = wav.getframerate()
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if src_sr != sr:
            from thesis_s2s.audio import resample

            audio = resample(audio, src_sr, sr)
        return audio
    except Exception:
        return None


_PIPER_VOICE = None
_PIPER_PATH = None
_PIPER_READY_KEY: tuple[str, int, int] | None = None
_PIPER_READY = False


def _piper_voice():
    global _PIPER_VOICE, _PIPER_PATH
    model = os_piper_model()
    if model is None:
        return None
    key = str(model)
    if _PIPER_VOICE is not None and _PIPER_PATH == key:
        return _PIPER_VOICE
    try:
        from piper import PiperVoice

        _PIPER_VOICE = PiperVoice.load(key)
        _PIPER_PATH = key
        return _PIPER_VOICE
    except Exception:
        _PIPER_VOICE = None
        _PIPER_PATH = None
        return None


def _piper_subprocess_synthesize(
    text: str,
    model: Path,
    sr: int,
    *,
    deterministic: bool,
) -> np.ndarray | None:
    """Render in a short-lived process so ONNX arenas return to the OS."""

    exe = shutil.which("piper")
    if not exe:
        return None
    raw_timeout = os.environ.get("PIPER_TIMEOUT_SECONDS", "120")
    try:
        timeout = float(raw_timeout)
    except ValueError:
        return None
    if not math.isfinite(timeout) or timeout <= 0:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "out.wav"
        command = [exe, "--model", str(model), "--output_file", str(wav_path)]
        if deterministic:
            command.extend(
                [
                    "--noise-scale",
                    "0",
                    "--noise-w-scale",
                    "0",
                ]
            )
        try:
            proc = subprocess.run(
                command,
                input=(text + "\n").encode("utf-8"),
                capture_output=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0 or not wav_path.is_file():
            return None
        try:
            audio, _ = read_wav(wav_path, sr)
        except (OSError, ValueError):
            return None
        return audio


def piper_synthesize(
    text: str,
    sr: int = SAMPLE_RATE,
    *,
    deterministic: bool = False,
    isolated: bool = False,
) -> np.ndarray | None:
    _validate_synthesis_inputs(text, sr)
    model = os_piper_model()
    if model is None:
        return None
    if isolated:
        audio = _piper_subprocess_synthesize(text, model, sr, deterministic=deterministic)
        if audio is None or audio.ndim != 1 or len(audio) == 0 or not np.isfinite(audio).all():
            return None
        return audio
    voice = _piper_voice()
    if voice is not None:
        try:
            import io
            import wave

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_handle:
                wav_handle.setnchannels(1)
                wav_handle.setsampwidth(2)
                wav_handle.setframerate(getattr(voice, "sample_rate", 22050))
                syn_config = None
                if deterministic:
                    from piper.config import SynthesisConfig

                    syn_config = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)
                if hasattr(voice, "synthesize_wav"):
                    voice.synthesize_wav(text, wav_handle, syn_config=syn_config)
                else:
                    if deterministic:
                        return None
                    voice.synthesize(text, wav_handle)
            audio = _pcm_from_wav_bytes(buf.getvalue(), sr)
            if (
                audio is not None
                and audio.ndim == 1
                and len(audio) > 0
                and np.isfinite(audio).all()
            ):
                return audio
        except Exception:
            pass
    audio = _piper_subprocess_synthesize(text, model, sr, deterministic=deterministic)
    if audio is None or audio.ndim != 1 or len(audio) == 0 or not np.isfinite(audio).all():
        return None
    return audio


def piper_runtime_ready() -> bool:
    """Probe and cache whether the configured voice can actually render finite audio."""

    global _PIPER_READY, _PIPER_READY_KEY
    model = os_piper_model()
    if model is None:
        return False
    try:
        stat = model.stat()
    except OSError:
        return False
    key = (str(model.resolve()), stat.st_mtime_ns, stat.st_size)
    if key != _PIPER_READY_KEY:
        try:
            audio = piper_synthesize("سلام")
        except Exception:
            audio = None
        _PIPER_READY = bool(
            audio is not None and audio.ndim == 1 and len(audio) > 0 and np.isfinite(audio).all()
        )
        _PIPER_READY_KEY = key
    return _PIPER_READY


def first_packet(audio: np.ndarray, sr: int = SAMPLE_RATE, seconds: float = 0.25) -> np.ndarray:
    if not isinstance(sr, int) or isinstance(sr, bool) or sr <= 0:
        raise ValueError("packet sample rate must be a positive integer")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("packet duration must be positive and finite")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("packet audio must be a finite one-dimensional array")
    n = min(len(audio), int(seconds * sr))
    return audio[: max(n, int(0.12 * sr))]


def synthesize(text: str, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, str]:
    _validate_synthesis_inputs(text, sr)
    piper = piper_synthesize(text, sr)
    if (
        piper is not None
        and piper.ndim == 1
        and len(piper) > 0
        and np.isfinite(piper).all()
    ):
        return np.clip(piper, -1.0, 1.0).astype(np.float32), "piper"
    return formant_synthesize(text, sr), "formant"


class FormantTalker:
    """Streaming-capable first-chunk talker used by duplex and cascade.

    Prefers Piper/ManaTTS only when the configured voice passes a render probe.
    """

    def __init__(self, reply_text: str = "سلام، چطور می‌تونم کمکتون کنم؟"):
        self.reply_text = reply_text
        self.backend = "formant"
        self._cached_first: np.ndarray | None = None
        if piper_runtime_ready():
            warm_piper()
            audio, backend = synthesize("سلام")
            self.backend = backend
            self._cached_first = first_packet(audio)

    def first_chunk(self, user_audio: np.ndarray, text: str | None = None) -> np.ndarray:
        phrase = text or self.reply_text
        if text is None and self._cached_first is not None:
            return self._cached_first
        audio, backend = synthesize(phrase)
        self.backend = backend
        packet = first_packet(audio)
        if text is None:
            self._cached_first = packet
        return packet

    def full_reply(self, text: str | None = None) -> np.ndarray:
        audio, backend = synthesize(text or self.reply_text)
        self.backend = backend
        return audio


class PiperTalker(FormantTalker):
    """Piper/ManaTTS first packet when a Persian ONNX voice is on disk."""
