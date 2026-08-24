"""Frame-level energy, ZCR, F0 (YIN), and MFCC features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, rfft

from thesis_s2s import SAMPLE_RATE


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = SAMPLE_RATE
    frame_ms: float = 20.0
    context_ms: float = 400.0
    n_mfcc: int = 13
    n_mels: int = 26
    f0_min_hz: float = 60.0
    f0_max_hz: float = 400.0
    preemphasis: float = 0.97
    yin_threshold: float = 0.15

    @property
    def hop(self) -> int:
        return max(1, int(self.sample_rate * self.frame_ms / 1000.0))

    @property
    def win(self) -> int:
        return max(self.hop, int(self.sample_rate * 0.04))

    @property
    def context_frames(self) -> int:
        return max(1, int(round(self.context_ms / self.frame_ms)))


def _frames(audio: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    n = 1 + (len(audio) - win) // hop
    index = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return audio[index]


def log_energy(frames: np.ndarray) -> np.ndarray:
    power = np.mean(frames**2, axis=1) + 1e-12
    return 10.0 * np.log10(power)


def zcr(frames: np.ndarray) -> np.ndarray:
    signs = np.sign(frames)
    signs[signs == 0] = 1
    return np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)


def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    bank: np.ndarray = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        right = min(right, bank.shape[1] - 1)
        center = min(max(center, left + 1), right - 1) if right - left > 2 else center
        if center > left:
            bank[i - 1, left:center] = (np.arange(left, center) - left) / max(1, center - left)
        if right > center:
            bank[i - 1, center:right] = (right - np.arange(center, right)) / max(1, right - center)
    return bank


def mfcc(frames: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    window = np.hamming(frames.shape[1])
    emphasized = np.empty_like(frames)
    emphasized[:, 0] = frames[:, 0]
    emphasized[:, 1:] = frames[:, 1:] - cfg.preemphasis * frames[:, :-1]
    spec = np.abs(rfft(emphasized * window, n=512, axis=1)) ** 2
    bank = _mel_filterbank(cfg.n_mels, 512, cfg.sample_rate)
    mel = np.log(spec @ bank.T + 1e-10)
    coef = dct(mel, type=2, axis=1, norm="ortho")[:, : cfg.n_mfcc]
    return coef.astype(np.float32)


def yin_f0(frames: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """Batched YIN F0 in Hz and a voiced flag."""

    sr = cfg.sample_rate
    tau_min = max(2, int(sr / cfg.f0_max_hz))
    tau_max = min(frames.shape[1] // 2 - 1, int(sr / cfg.f0_min_hz))
    w = frames - frames.mean(axis=1, keepdims=True)
    n, win = w.shape
    diff = np.empty((n, tau_max + 1), dtype=np.float64)
    diff[:, 0] = 1.0
    for tau in range(1, tau_max + 1):
        d = w[:, :-tau] - w[:, tau:]
        diff[:, tau] = np.sum(d * d, axis=1)
    cum = np.cumsum(diff[:, 1:], axis=1)
    cmd = np.empty_like(diff)
    cmd[:, 0] = 1.0
    cmd[:, 1:] = diff[:, 1:] * np.arange(1, tau_max + 1)[None, :] / np.maximum(cum, 1e-12)
    region = cmd[:, tau_min:tau_max]
    tau_rel = np.argmin(region, axis=1)
    tau = tau_rel + tau_min
    min_cmd = region[np.arange(n), tau_rel]
    below = region < cfg.yin_threshold
    has_candidate = below.any(axis=1)
    first = np.argmax(below, axis=1)
    tau = np.where(has_candidate, first + tau_min, tau)
    voiced = ((min_cmd <= 0.45) | has_candidate) & (np.max(np.abs(w), axis=1) > 1e-4)
    f0 = np.zeros(n, dtype=np.float32)
    valid = voiced & (tau > 1) & (tau < tau_max)
    idx = np.where(valid)[0]
    t = tau[idx]
    s0 = cmd[idx, t - 1]
    s1 = cmd[idx, t]
    s2 = cmd[idx, t + 1]
    denom = 2 * s1 - s0 - s2
    shift = np.where(np.abs(denom) > 1e-12, (s0 - s2) / (2 * denom), 0.0)
    f0[idx] = sr / (t + shift)
    return f0, voiced.astype(np.float32)


def frame_feature_matrix(audio: np.ndarray, cfg: FeatureConfig | None = None) -> np.ndarray:
    cfg = cfg or FeatureConfig()
    frames = _frames(np.asarray(audio, dtype=np.float32), cfg.win, cfg.hop)
    energy = log_energy(frames)
    rate = zcr(frames)
    f0, voiced = yin_f0(frames, cfg)
    coef = mfcc(frames, cfg)
    delta = np.diff(coef, axis=0, prepend=coef[:1])
    feats = np.column_stack([energy, rate, f0, voiced, coef, delta])
    return feats.astype(np.float32)


def context_vector(feats: np.ndarray, end_index: int, context_frames: int) -> np.ndarray:
    start = max(0, end_index - context_frames + 1)
    window = feats[start : end_index + 1]
    if len(window) == 0:
        return np.zeros(feats.shape[1] * 3, dtype=np.float32)
    stats = np.concatenate([window.mean(axis=0), window.std(axis=0), window.max(axis=0)])
    return stats.astype(np.float32)


def streaming_context_vectors(feats: np.ndarray, context_frames: int) -> np.ndarray:
    return np.stack([context_vector(feats, i, context_frames) for i in range(len(feats))])
