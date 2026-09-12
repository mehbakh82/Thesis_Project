"""Audio I/O and resampling at 16 kHz mono."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from thesis_s2s import SAMPLE_RATE


def _validated_sample_rate(sample_rate: int, *, name: str = "sample rate") -> int:
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return sample_rate


def to_float32_mono(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio)
    if data.ndim not in {1, 2}:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {data.shape}")
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_):
        raise ValueError("audio samples must be numeric")
    if np.issubdtype(data.dtype, np.complexfloating):
        raise ValueError("complex audio samples are unsupported")
    source_dtype = data.dtype
    if data.ndim == 2:
        if data.shape[1] == 0 or data.shape[1] > 32:
            raise ValueError("2-D audio must use sample-major shape (samples, channels<=32)")
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    if not np.isfinite(data).all():
        raise ValueError("audio samples must be finite")
    if np.issubdtype(source_dtype, np.integer):
        limits = np.iinfo(source_dtype)
        if np.issubdtype(source_dtype, np.unsignedinteger):
            midpoint = (limits.max + 1) / 2.0
            data = (data - midpoint) / midpoint
        else:
            data = data / float(max(abs(limits.min), limits.max))
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int = SAMPLE_RATE) -> np.ndarray:
    _validated_sample_rate(src_sr, name="source sample rate")
    _validated_sample_rate(dst_sr, name="destination sample rate")
    mono = to_float32_mono(audio)
    if mono.size == 0:
        return mono
    if src_sr == dst_sr:
        return mono
    from math import gcd

    g = gcd(src_sr, dst_sr)
    return resample_poly(mono, dst_sr // g, src_sr // g).astype(np.float32)


def read_audio_ffmpeg(path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-readable file to 16 kHz mono float32."""

    path = Path(path)
    _validated_sample_rate(target_sr, name="target sample rate")
    if not path.is_file():
        raise FileNotFoundError(f"audio file does not exist: {path}")
    raw_timeout = os.environ.get("AUDIO_DECODE_TIMEOUT_SECONDS", "300")
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError("AUDIO_DECODE_TIMEOUT_SECONDS must be positive and finite") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("AUDIO_DECODE_TIMEOUT_SECONDS must be positive and finite")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "s16le",
        "-",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if data.size == 0:
        raise ValueError(f"decoded audio is empty: {path}")
    return data, target_sr


def read_wav(path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    path = Path(path)
    _validated_sample_rate(target_sr, name="target sample rate")
    try:
        handle = wave.open(str(path), "rb")
    except Exception:
        return read_audio_ffmpeg(path, target_sr)
    with handle:
        n_channels = handle.getnchannels()
        sampwidth = handle.getsampwidth()
        src_sr = handle.getframerate()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)
    if n_frames <= 0:
        raise ValueError(f"audio file is empty: {path}")
    try:
        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported sample width {sampwidth} in {path}")
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)
        data = resample(data, src_sr, target_sr)
        return data, target_sr
    except Exception:
        return read_audio_ffmpeg(path, target_sr)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path = Path(path)
    _validated_sample_rate(sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = to_float32_mono(audio)
    if mono.size == 0:
        raise ValueError("cannot write empty audio")
    pcm = np.rint(np.clip(mono * 32767.0, -32768, 32767)).astype(np.int16)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with wave.open(str(temporary_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def duration_seconds(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    _validated_sample_rate(sample_rate)
    return float(len(to_float32_mono(audio)) / sample_rate)
