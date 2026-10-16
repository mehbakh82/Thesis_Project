"""Two-stage barge-in control for streamed assistant playback.

Stage 1 (*duck*): as soon as voice activity is sustained for ``duck_frames``
frames during playback, the assistant's audio is lowered. This is the fast
"detect and adapt the ongoing response" reaction the proposal asks to happen
within 150 ms.

Stage 2 (*confirm*): the controller then decides between a real interruption
(stop playback, cancel generation, and keep the captured speech as the next
user turn) and a backchannel or noise (restore the volume and continue). The
decision uses an optional acoustic classifier on energy/F0/MFCC features
(probability of ``interrupt``) together with how long the user keeps talking:
backchannels such as «آره» or «اوهوم» are short.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from thesis_s2s.runtime.vad import FRAME_SAMPLES, SAMPLE_RATE, FrameScorer, default_scorer

InterruptClassifier = Callable[[np.ndarray], float]


@dataclass
class BargeInEvent:
    kind: str  # "duck", "stop", or "resume"
    onset_sample: int
    decided_at_sample: int
    probability: float | None = None


@dataclass
class BargeInController:
    scorer: FrameScorer = field(default_factory=default_scorer)
    classifier: InterruptClassifier | None = None
    voice_threshold: float = 0.5
    duck_frames: int = 2
    confirm_speech_ms: float = 480.0
    classifier_min_ms: float = 256.0
    classifier_threshold: float = 0.7
    resume_silence_ms: float = 320.0
    context_ms: float = 400.0
    echo_margin_db: float | None = 12.0
    floor_alpha: float = 0.05
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _history: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _consumed: int = 0
    _voiced_run: int = 0
    _ducked: bool = False
    _onset: int = 0
    _speech_samples: int = 0
    _silence_samples: int = 0
    _floor_db: float | None = None
    _frames_seen: int = 0

    @property
    def ducked(self) -> bool:
        return self._ducked

    def reset(self) -> None:
        self.scorer.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._history = np.zeros(0, dtype=np.float32)
        self._voiced_run = 0
        self._ducked = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._floor_db = None
        self._frames_seen = 0

    def _user_voice(self, frame: np.ndarray) -> bool:
        """Speech-like and clearly louder than the running playback/echo floor."""

        speech = self.scorer.score(frame) >= self.voice_threshold
        if self.echo_margin_db is None:
            return speech
        level = 20.0 * float(np.log10(np.sqrt(np.mean(frame**2)) + 1e-9))
        if self._floor_db is None:
            self._floor_db = level
        above = level >= self._floor_db + self.echo_margin_db
        if not self._ducked and not (speech and above):
            # Track the echo + noise level while the user is not talking; adapt
            # faster for the first half second of playback.
            alpha = 0.2 if self._frames_seen < 16 else self.floor_alpha
            self._floor_db += alpha * (level - self._floor_db)
        self._frames_seen += 1
        return speech and above

    def captured_since_onset(self) -> np.ndarray:
        """Microphone audio from the detected onset (pre-roll for the next turn)."""

        start = max(0, self._history.size - (self._consumed - self._onset))
        return self._history[start:].copy()

    def push(self, pcm: np.ndarray, sample_offset: int | None = None) -> list[BargeInEvent]:
        """Feed microphone PCM captured while the assistant is speaking."""

        if sample_offset is not None and not self._pending.size:
            self._consumed = sample_offset
        events: list[BargeInEvent] = []
        self._pending = np.concatenate([self._pending, np.asarray(pcm, dtype=np.float32)])
        keep = int(4 * SAMPLE_RATE)
        while self._pending.size >= FRAME_SAMPLES:
            frame, self._pending = self._pending[:FRAME_SAMPLES], self._pending[FRAME_SAMPLES:]
            frame_start = self._consumed
            self._consumed += FRAME_SAMPLES
            self._history = np.concatenate([self._history, frame])[-keep:]
            voiced = self._user_voice(frame)
            if not self._ducked:
                self._voiced_run = self._voiced_run + 1 if voiced else 0
                if self._voiced_run >= self.duck_frames:
                    self._ducked = True
                    self._onset = frame_start - (self.duck_frames - 1) * FRAME_SAMPLES
                    self._speech_samples = self.duck_frames * FRAME_SAMPLES
                    self._silence_samples = 0
                    events.append(BargeInEvent("duck", self._onset, self._consumed))
                continue
            if voiced:
                self._speech_samples += FRAME_SAMPLES
                self._silence_samples = 0
            else:
                self._silence_samples += FRAME_SAMPLES
            speech_ms = 1000.0 * self._speech_samples / SAMPLE_RATE
            probability: float | None = None
            if self.classifier is not None and speech_ms >= self.classifier_min_ms:
                window = self._history[-int(self.context_ms * SAMPLE_RATE / 1000) :]
                probability = float(self.classifier(window))
                if probability >= self.classifier_threshold:
                    events.append(BargeInEvent("stop", self._onset, self._consumed, probability))
                    self._ducked = False
                    self._voiced_run = 0
                    continue
            if speech_ms >= self.confirm_speech_ms:
                events.append(BargeInEvent("stop", self._onset, self._consumed, probability))
                self._ducked = False
                self._voiced_run = 0
            elif 1000.0 * self._silence_samples / SAMPLE_RATE >= self.resume_silence_ms:
                events.append(BargeInEvent("resume", self._onset, self._consumed, probability))
                self._ducked = False
                self._voiced_run = 0
        return events
