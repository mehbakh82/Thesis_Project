"""Audio I/O and resampling at 16 kHz mono."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from thesis_s2s import SAMPLE_RATE


def to_float32_mono(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio)
    if data.ndim not in {1, 2}:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {data.shape}")
    source_dtype = data.dtype
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    if np.issubdtype(source_dtype, np.integer):
        limits = np.iinfo(source_dtype)
        if np.issubdtype(source_dtype, np.unsignedinteger):
            midpoint = (limits.max + 1) / 2.0
            data = (data - midpoint) / midpoint
        else:
            data = data / float(max(abs(limits.min), limits.max))
    elif data.size and np.max(np.abs(data)) > 1.5:
        # Some callers provide PCM values already promoted to floating point.
        data = data / 32768.0
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int = SAMPLE_RATE) -> np.ndarray:
    mono = to_float32_mono(audio)
    if src_sr == dst_sr:
        return mono
    from math import gcd

    g = gcd(src_sr, dst_sr)
    return resample_poly(mono, dst_sr // g, src_sr // g).astype(np.float32)


def read_audio_ffmpeg(path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-readable file to 16 kHz mono float32."""

    import subprocess

    path = Path(path)
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
    result = subprocess.run(cmd, check=True, capture_output=True)
    data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return data, target_sr


def read_wav(path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    path = Path(path)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(to_float32_mono(audio) * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def duration_seconds(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return float(len(to_float32_mono(audio)) / sample_rate)
