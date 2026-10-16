"""Streaming voice-activity detection and end-of-turn endpointing.

The endpointer consumes 16 kHz mono float PCM in arbitrary chunk sizes and
emits ``speech_start`` / ``speech_end`` decisions with exact sample indices, so
latency can be measured from the acoustic end of speech (the last voiced
frame), not from when the decision was made. Silero VAD (ONNX, MIT license)
is used when its model file is available; otherwise an adaptive energy
detector is used and the backend name says so.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # Silero's fixed 32 ms frame at 16 kHz
SILERO_CONTEXT = 64
DEFAULT_SILERO_PATH = Path(__file__).resolve().parents[3] / "models" / "vad" / "silero_vad.onnx"


class FrameScorer(Protocol):
    backend: str

    def score(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


class SileroScorer:
    """Silero VAD v5/v6 ONNX scorer for 512-sample frames."""

    backend = "silero_vad_onnx"

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state: np.ndarray = np.zeros((2, 1, 128), dtype=np.float32)
        self._context: np.ndarray = np.zeros((1, SILERO_CONTEXT), dtype=np.float32)

    def score(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -SILERO_CONTEXT:]
        return float(out[0][0])


class EnergyScorer:
    """Adaptive energy scorer mapped to a pseudo-probability in [0, 1]."""

    backend = "adaptive_energy"

    def __init__(self, margin_db: float = 12.0) -> None:
        self.margin_db = margin_db
        self.reset()

    def reset(self) -> None:
        self._floor_db = -60.0

    def score(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
        level = 20.0 * np.log10(rms + 1e-12)
        if level < self._floor_db + 3.0:
            self._floor_db = 0.95 * self._floor_db + 0.05 * level
        else:
            self._floor_db += 0.001 * (level - self._floor_db)
        return float(1.0 / (1.0 + np.exp(-(level - self._floor_db - self.margin_db) / 2.0)))


def default_scorer() -> FrameScorer:
    path = Path(os.environ.get("VAD_MODEL", str(DEFAULT_SILERO_PATH)))
    if path.is_file():
        try:
            return SileroScorer(path)
        except Exception:  # noqa: BLE001 - fall back to a disclosed energy backend
            pass
    return EnergyScorer()


@dataclass
class EndpointEvent:
    kind: str  # "speech_start" or "speech_end"
    sample: int  # acoustic start, or end of the last voiced frame
    decided_at_sample: int


@dataclass
class Endpointer:
    """Hysteresis endpointer on per-frame speech probabilities.

    ``hangover_ms`` is the silence that must follow speech before the turn is
    declared over; it is the dominant, tunable part of end-of-turn latency.
    """

    scorer: FrameScorer = field(default_factory=default_scorer)
    onset_threshold: float = 0.5
    offset_threshold: float = 0.35
    onset_frames: int = 2
    hangover_ms: float = 256.0
    min_speech_ms: float = 160.0
    tentative_ms: float | None = None
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _consumed: int = 0
    _in_speech: bool = False
    _voiced_run: int = 0
    _silence_samples: int = 0
    _speech_start: int = 0
    _last_voiced_end: int = 0
    _paused: bool = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self.scorer.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._consumed = 0
        self._in_speech = False
        self._voiced_run = 0
        self._silence_samples = 0
        self._paused = False

    def push(self, pcm: np.ndarray) -> list[EndpointEvent]:
        """Feed PCM and return any endpoint decisions it completes."""

        events: list[EndpointEvent] = []
        self._pending = np.concatenate([self._pending, np.asarray(pcm, dtype=np.float32)])
        hangover = int(self.hangover_ms * SAMPLE_RATE / 1000)
        min_speech = int(self.min_speech_ms * SAMPLE_RATE / 1000)
        tentative = (
            int(self.tentative_ms * SAMPLE_RATE / 1000) if self.tentative_ms is not None else None
        )
        while self._pending.size >= FRAME_SAMPLES:
            frame, self._pending = self._pending[:FRAME_SAMPLES], self._pending[FRAME_SAMPLES:]
            start = self._consumed
            self._consumed += FRAME_SAMPLES
            prob = self.scorer.score(frame)
            if not self._in_speech:
                self._voiced_run = self._voiced_run + 1 if prob >= self.onset_threshold else 0
                if self._voiced_run >= self.onset_frames:
                    self._in_speech = True
                    self._speech_start = start - (self.onset_frames - 1) * FRAME_SAMPLES
                    self._last_voiced_end = self._consumed
                    self._silence_samples = 0
                    events.append(EndpointEvent("speech_start", self._speech_start, self._consumed))
            elif prob >= self.offset_threshold:
                self._last_voiced_end = self._consumed
                self._silence_samples = 0
                if self._paused:
                    self._paused = False
                    events.append(EndpointEvent("speech_resume", self._consumed, self._consumed))
            else:
                self._silence_samples += FRAME_SAMPLES
                long_enough = self._last_voiced_end - self._speech_start >= min_speech
                if (
                    tentative is not None
                    and not self._paused
                    and long_enough
                    and tentative <= self._silence_samples < hangover
                ):
                    self._paused = True
                    events.append(
                        EndpointEvent("speech_pause", self._last_voiced_end, self._consumed)
                    )
                if self._silence_samples >= hangover:
                    self._in_speech = False
                    self._voiced_run = 0
                    self._paused = False
                    if long_enough:
                        events.append(
                            EndpointEvent("speech_end", self._last_voiced_end, self._consumed)
                        )
        return events
