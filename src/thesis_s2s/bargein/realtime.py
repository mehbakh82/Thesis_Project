"""Realtime playback controller owned by the barge-in detector."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from thesis_s2s.bargein.detector import BargeinDetector, EnergyVadBaseline


@dataclass
class PlaybackController:
    """Stop assistant playback when the classical detector fires.

    ``stop_playback`` is the duplex control point: drop remaining tokens,
    flush the speaker buffer, and start a new user turn.
    """

    detector: BargeinDetector | EnergyVadBaseline
    sample_rate: int = 16_000
    tail_seconds: float = 0.45
    min_consecutive: int = 2
    _playing: bool = False
    _stopped_at: float | None = None
    _interrupt_onset: float | None = None
    _tail: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _positive_hops: int = 0
    events: list[dict] = field(default_factory=list)

    def start_playback(self) -> None:
        self._playing = True
        self._stopped_at = None
        self._interrupt_onset = None
        self._tail = np.zeros(0, dtype=np.float32)
        self._positive_hops = 0

    def stop_playback(self, reason: str = "bargein") -> None:
        if not self._playing:
            return
        self._playing = False
        self._stopped_at = time.perf_counter()
        self.events.append({"event": "stop_playback", "reason": reason, "t": self._stopped_at})

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def microphone_tail(self) -> np.ndarray:
        """Return a copy of the current rolling microphone window.

        The copy is intentional: study persistence may reduce the window to
        aggregate features while the controller immediately starts another
        turn and clears its internal buffer.
        """

        return self._tail.copy()

    def t_barge_in_ms(self) -> float | None:
        if self._stopped_at is None or self._interrupt_onset is None:
            return None
        return 1000.0 * (self._stopped_at - self._interrupt_onset)

    def mark_interrupt_onset(self) -> None:
        if self._playing and self._interrupt_onset is None:
            self._interrupt_onset = time.perf_counter()

    def on_mic_chunk(self, chunk: np.ndarray, *, interrupt_onset: bool = False) -> bool:
        """Feed a microphone chunk while the assistant is speaking.

        Returns True if playback was stopped on this chunk.
        """

        if not self._playing:
            return False
        if interrupt_onset:
            self._interrupt_onset = time.perf_counter()
        current = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if current.size == 0:
            return False
        need = int(self.tail_seconds * self.sample_rate)
        self._tail = np.concatenate((self._tail, current))[-need:]
        tail = self._tail
        if isinstance(self.detector, BargeinDetector):
            fired = self.detector.interrupt_now(tail)
        else:
            fired = bool(self.detector.predict_binary(tail, np.ones(len(tail))))
        self._positive_hops = self._positive_hops + 1 if fired else 0
        if self._positive_hops >= max(1, self.min_consecutive):
            if self._interrupt_onset is None:
                self._interrupt_onset = time.perf_counter() - len(current) / self.sample_rate
            self.stop_playback("bargein")
            return True
        return False
