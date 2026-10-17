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

With a ``transcriber`` (the deployed ASR), stage 2 is *lexical*: the captured
speech since the onset is transcribed once ``lexical_after_ms`` of it has been
heard (and again every ``lexical_interval_ms`` while undecided, or when the
user falls silent). Words of the assistant's own reply (residual echo, via
``echo_words``) and listener tokens are discarded; any remaining word means a
real interruption, while an empty or listener-token-only transcript means a
backchannel or noise. The ASR time is added to the decision time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from thesis_s2s.bargein.lexicon import content_tokens
from thesis_s2s.runtime.vad import FRAME_SAMPLES, SAMPLE_RATE, FrameScorer, default_scorer

InterruptClassifier = Callable[[np.ndarray], float]
Transcriber = Callable[[np.ndarray], str]


@dataclass
class BargeInEvent:
    kind: str  # "duck", "stop", or "resume"
    onset_sample: int
    decided_at_sample: int
    probability: float | None = None
    transcript: str | None = None


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
    dtd_margin_db: float = 10.0
    coupling_alpha: float = 0.1
    reference_active_db: float = -55.0
    transcriber: Transcriber | None = None
    echo_words: Callable[[], set[str]] | None = None
    lexical_matching: str = "v1"
    drop_function_words: bool = False
    initial_coupling_db: float | None = None
    lexical_after_ms: float = 400.0
    lexical_interval_ms: float = 200.0
    confirm_cap_ms: float = 1600.0
    _next_check_ms: float = 0.0
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _pending_ref: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _recent_ref_db: list[float] = field(default_factory=list)
    _coupling_db: float | None = None
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
        self._pending_ref = np.zeros(0, dtype=np.float32)
        self._recent_ref_db = []
        self._coupling_db = None

    def _user_voice(self, frame: np.ndarray, reference_rms: float | None = None) -> bool:
        """Speech-like and louder than the expected echo (double-talk detection).

        With a playback reference level, the expected echo is the loudest recent
        reference frame (covering echo delay) plus an adaptively estimated
        echo coupling; the microphone must exceed it by ``dtd_margin_db`` (a
        Geigel-style detector). Without a reference, a running noise/echo floor
        plus ``echo_margin_db`` is used.
        """

        speech = self.scorer.score(frame) >= self.voice_threshold
        if self.echo_margin_db is None:
            return speech
        level = 20.0 * float(np.log10(np.sqrt(np.mean(frame**2)) + 1e-9))
        if self._floor_db is None:
            self._floor_db = level
        above = level >= self._floor_db + self.echo_margin_db
        reference_active = False
        if reference_rms is not None:
            ref_db = 20.0 * float(np.log10(reference_rms + 1e-9))
            self._recent_ref_db = (self._recent_ref_db + [ref_db])[-8:]
            ref_max = max(self._recent_ref_db)
            reference_active = ref_max > self.reference_active_db
            if reference_active:
                if self._coupling_db is None:
                    # Legacy: the first active frame (often before the delayed
                    # echo arrives, so far too low). A conservative prior instead
                    # assumes strong coupling and adapts down while the user is quiet.
                    self._coupling_db = (
                        level - ref_max if self.initial_coupling_db is None else self.initial_coupling_db
                    )
                above = level >= ref_max + self._coupling_db + self.dtd_margin_db
                if not self._ducked and not (speech and above):
                    self._coupling_db += self.coupling_alpha * (level - ref_max - self._coupling_db)
        if not reference_active and not self._ducked and not (speech and above):
            # Track the noise level while the user is not talking; adapt faster
            # for the first half second.
            alpha = 0.2 if self._frames_seen < 16 else self.floor_alpha
            self._floor_db += alpha * (level - self._floor_db)
        self._frames_seen += 1
        return speech and above

    def _lexical_step(self, speech_ms: float) -> BargeInEvent | None:
        """Transcribe when due; return a stop/resume decision or None to keep waiting."""

        ended = 1000.0 * self._silence_samples / SAMPLE_RATE >= self.resume_silence_ms
        capped = speech_ms >= self.confirm_cap_ms
        if not (ended or capped or speech_ms >= self._next_check_ms):
            return None
        transcriber = self.transcriber
        if transcriber is None:
            return None
        started = time.perf_counter()
        transcript = transcriber(self.captured_since_onset())
        latency = int(round((time.perf_counter() - started) * SAMPLE_RATE))
        echo = self.echo_words() if self.echo_words is not None else set()
        content = content_tokens(
            transcript, echo, matching=self.lexical_matching, drop_function_words=self.drop_function_words
        )
        if content:
            kind = "stop"
        elif ended or capped:
            kind = "resume"
        else:
            self._next_check_ms = speech_ms + self.lexical_interval_ms
            return None
        return BargeInEvent(kind, self._onset, self._consumed + latency, None, transcript)

    def captured_since_onset(self) -> np.ndarray:
        """Microphone audio from the detected onset (pre-roll for the next turn)."""

        start = max(0, self._history.size - (self._consumed - self._onset))
        return self._history[start:].copy()

    def push(
        self,
        pcm: np.ndarray,
        sample_offset: int | None = None,
        reference_level: np.ndarray | None = None,
    ) -> list[BargeInEvent]:
        """Feed microphone PCM captured while the assistant is speaking.

        ``reference_level`` optionally gives, per microphone sample, the RMS of
        the assistant audio being played at that moment (as reported by the
        client); it enables double-talk detection against residual echo.
        """

        if sample_offset is not None and not self._pending.size:
            self._consumed = sample_offset
        events: list[BargeInEvent] = []
        pcm = np.asarray(pcm, dtype=np.float32)
        self._pending = np.concatenate([self._pending, pcm])
        if reference_level is not None:
            self._pending_ref = np.concatenate(
                [self._pending_ref, np.asarray(reference_level, dtype=np.float32)[: pcm.size]]
            )
        keep = int(4 * SAMPLE_RATE)
        while self._pending.size >= FRAME_SAMPLES:
            frame, self._pending = self._pending[:FRAME_SAMPLES], self._pending[FRAME_SAMPLES:]
            reference_rms: float | None = None
            if self._pending_ref.size >= FRAME_SAMPLES:
                reference_rms = float(np.mean(self._pending_ref[:FRAME_SAMPLES]))
                self._pending_ref = self._pending_ref[FRAME_SAMPLES:]
            frame_start = self._consumed
            self._consumed += FRAME_SAMPLES
            self._history = np.concatenate([self._history, frame])[-keep:]
            voiced = self._user_voice(frame, reference_rms)
            if not self._ducked:
                self._voiced_run = self._voiced_run + 1 if voiced else 0
                if self._voiced_run >= self.duck_frames:
                    self._ducked = True
                    self._onset = frame_start - (self.duck_frames - 1) * FRAME_SAMPLES
                    self._speech_samples = self.duck_frames * FRAME_SAMPLES
                    self._silence_samples = 0
                    self._next_check_ms = self.lexical_after_ms
                    events.append(BargeInEvent("duck", self._onset, self._consumed))
                continue
            if voiced:
                self._speech_samples += FRAME_SAMPLES
                self._silence_samples = 0
            else:
                self._silence_samples += FRAME_SAMPLES
            speech_ms = 1000.0 * self._speech_samples / SAMPLE_RATE
            if self.transcriber is not None:
                decision = self._lexical_step(speech_ms)
                if decision is not None:
                    events.append(decision)
                    self._ducked = False
                    self._voiced_run = 0
                continue
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
