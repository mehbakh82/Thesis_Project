"""Synthetic duplex mixtures with interrupt / backchannel / noise labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.bargein.detector import LABELS
from thesis_s2s.bargein.features import FeatureConfig


def _harmonic(
    duration_s: float, f0: float, sr: int = SAMPLE_RATE, rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    t = np.arange(int(duration_s * sr), dtype=np.float64) / sr
    signal = 0.35 * np.sin(2 * np.pi * f0 * t)
    signal += 0.18 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += 0.08 * np.sin(2 * np.pi * 3 * f0 * t)
    vibrato = 1 + 0.003 * np.sin(2 * np.pi * 5 * t)
    signal *= vibrato
    env = np.ones_like(signal)
    fade = int(0.02 * sr)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    noise = 0.01 * rng.normal(size=signal.shape)
    return (signal * env + noise).astype(np.float32)


@dataclass
class DuplexClip:
    audio: np.ndarray
    frame_labels: np.ndarray  # 0 none, 1 interrupt, 2 backchannel, 3 noise
    binary_interrupt: int
    assistant_mask: np.ndarray
    kind: str
    group_id: str | None = None


def make_clip(
    kind: str,
    *,
    sr: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
    feat_cfg: FeatureConfig | None = None,
) -> DuplexClip:
    rng = rng or np.random.default_rng()
    feat_cfg = feat_cfg or FeatureConfig()
    duration = float(rng.uniform(1.2, 2.2))
    assistant = _harmonic(duration, f0=float(rng.uniform(90, 140)), sr=sr, rng=rng)
    mix = assistant.copy()
    labels: np.ndarray = np.zeros(1 + (len(mix) - feat_cfg.win) // feat_cfg.hop, dtype=np.int32)
    assistant_mask: np.ndarray = np.ones(len(mix), dtype=np.float32)

    def mark(start_s: float, end_s: float, label: int) -> None:
        for i in range(len(labels)):
            t = (i * feat_cfg.hop) / sr
            if start_s <= t < end_s:
                labels[i] = label

    if kind == "interrupt":
        start = float(rng.uniform(0.45, 0.9))
        user = _harmonic(duration - start, f0=float(rng.uniform(180, 260)), sr=sr, rng=rng)
        end = min(len(mix), int(start * sr) + len(user))
        mix[int(start * sr) : end] += 0.9 * user[: end - int(start * sr)]
        mark(start, duration, LABELS.index("interrupt"))
    elif kind == "backchannel":
        start = float(rng.uniform(0.5, 1.0))
        length = int(0.07 * sr)
        burst = 0.25 * rng.normal(size=length).astype(np.float32)
        s = int(start * sr)
        mix[s : s + length] += burst
        mark(start, start + 0.08, LABELS.index("backchannel"))
    elif kind == "noise":
        mix += 0.08 * rng.normal(size=mix.shape).astype(np.float32)
        labels[:] = LABELS.index("noise")
    elif kind == "none":
        pass
    else:
        raise ValueError(kind)

    peak = np.max(np.abs(mix)) + 1e-6
    mix = (0.9 * mix / peak).astype(np.float32)
    return DuplexClip(
        audio=mix,
        frame_labels=labels,
        binary_interrupt=int(kind == "interrupt"),
        assistant_mask=assistant_mask,
        kind=kind,
    )


def make_dataset(
    n_per_class: int = 40,
    seed: int = 0,
) -> list[DuplexClip]:
    rng = np.random.default_rng(seed)
    clips: list[DuplexClip] = []
    for kind in ("none", "interrupt", "backchannel", "noise"):
        for _ in range(n_per_class):
            clips.append(make_clip(kind, rng=rng))
    rng.shuffle(clips)
    return clips
