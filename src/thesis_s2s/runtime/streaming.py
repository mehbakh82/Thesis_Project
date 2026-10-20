"""Low-latency streaming turn pipeline: ASR -> streamed LLM -> clause-level TTS.

The legacy cascade generated a whole reply and synthesized it in one block.
Here the responder streams tokens, a Persian clause chunker releases speakable
fragments as soon as they are complete, and each fragment is synthesized and
emitted while the language model keeps generating. Latin-script tokens are
masked during decoding instead of being rejected after generation, so no
second generation pass is needed. Every stage is timestamped with
``time.perf_counter`` so first-audio latency can be decomposed.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CLAUSE_END = re.compile(r"([.!?؟،؛:\n])")
LATIN = re.compile(r"[A-Za-z]")


class ClauseChunker:
    """Release speakable Persian fragments from a stream of text deltas.

    A fragment ends at sentence/clause punctuation. If no punctuation arrives,
    the first fragment is cut after ``first_max_words`` words and later ones
    after ``max_words`` words, so the first audio is never held hostage by a
    long unpunctuated sentence.
    """

    def __init__(self, first_min_words: int = 3, first_max_words: int = 7, max_words: int = 14):
        if not 1 <= first_min_words <= first_max_words <= max_words:
            raise ValueError("invalid chunk word limits")
        self.first_min_words = first_min_words
        self.first_max_words = first_max_words
        self.max_words = max_words
        self._buffer = ""
        self._emitted = 0

    def _limits(self) -> tuple[int, int]:
        if self._emitted == 0:
            return self.first_min_words, self.first_max_words
        return 1, self.max_words

    def push(self, delta: str) -> list[str]:
        self._buffer += delta
        out: list[str] = []
        while True:
            fragment_end = self._next_boundary()
            if fragment_end is None:
                return out
            fragment = self._buffer[:fragment_end].strip()
            self._buffer = self._buffer[fragment_end:]
            if fragment:
                out.append(fragment)
                self._emitted += 1

    def _next_boundary(self) -> int | None:
        """Return the end index of the next releasable fragment, if any."""

        minimum, maximum = self._limits()
        for match in CLAUSE_END.finditer(self._buffer):
            if len(self._buffer[: match.end()].split()) >= minimum:
                return match.end()
        # No acceptable punctuation: cut after ``maximum`` complete words.
        boundaries = [m.start() for m in re.finditer(r"\s+", self._buffer)]
        words_seen = 0
        previous = 0
        for position in boundaries:
            if self._buffer[previous:position].strip():
                words_seen += 1
                if words_seen >= maximum:
                    return position
            previous = position
        return None

    def flush(self) -> str | None:
        rest = self._buffer.strip()
        self._buffer = ""
        if rest:
            self._emitted += 1
            return rest
        return None


def latin_token_ids(tokenizer: Any) -> list[int]:
    """Return vocabulary ids whose decoded text contains a Latin letter."""

    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    banned: list[int] = []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        if LATIN.search(piece):
            banned.append(token_id)
    return banned


class LatinMaskProcessor:
    """``transformers`` logits processor that forbids Latin-letter tokens."""

    def __init__(self, banned_ids: Sequence[int]) -> None:
        self._banned = list(banned_ids)
        self._index: Any = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        if self._index is None or self._index.device != scores.device:
            self._index = torch.tensor(self._banned, dtype=torch.long, device=scores.device)
        scores = scores.clone()
        scores[:, self._index] = float("-inf")
        return scores


@dataclass
class TurnTimings:
    """``perf_counter`` stamps (seconds) for one streamed turn."""

    start: float = 0.0
    asr_done: float | None = None
    first_token: float | None = None
    first_clause: float | None = None
    first_audio_ready: float | None = None
    done: float | None = None

    def as_ms(self) -> dict[str, float | None]:
        def rel(value: float | None) -> float | None:
            return None if value is None else round(1000.0 * (value - self.start), 2)

        return {
            "asr_ms": rel(self.asr_done),
            "first_token_ms": rel(self.first_token),
            "first_clause_ms": rel(self.first_clause),
            "first_audio_ready_ms": rel(self.first_audio_ready),
            "turn_done_ms": rel(self.done),
        }


@dataclass
class StreamedTurn:
    transcript: str
    reply_text: str
    clauses: list[str]
    timings: TurnTimings
    asr_error: str | None = None
    responder_error: str | None = None
    cancelled: bool = False
    audio_samples: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class StreamingResponder:
    """Stream a Persian reply from a local Hugging Face chat model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        system_prompt: str,
        device: str = "cuda",
        max_new_tokens: int = 64,
        history_turns: int = 2,
        mask_latin: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.history_turns = history_turns
        self._processors = [LatinMaskProcessor(latin_token_ids(tokenizer))] if mask_latin else []

    def messages(self, user_text: str, history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        for user, assistant in list(history)[-self.history_turns :] if self.history_turns else []:
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": assistant})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def stream(
        self,
        user_text: str,
        history: Sequence[tuple[str, str]] = (),
        cancel: threading.Event | None = None,
    ) -> Iterator[str]:
        from transformers import (
            LogitsProcessorList,
            StoppingCriteria,
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

        encoded: Any = self.tokenizer.apply_chat_template(
            self.messages(user_text, history),
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0
        )
        stop = cancel or threading.Event()

        class _Cancelled(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return stop.is_set()

        arguments = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "streamer": streamer,
            "logits_processor": LogitsProcessorList(self._processors),
            "stopping_criteria": StoppingCriteriaList([_Cancelled()]),
        }
        errors: list[BaseException] = []

        def run() -> None:
            try:
                self.model.generate(**arguments)
            except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer
                errors.append(exc)
                streamer.end()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        for delta in streamer:
            if stop.is_set():
                break
            yield delta
        worker.join()
        if errors:
            raise RuntimeError(f"generation failed: {errors[0]!r}")


PERSIAN_ONLY_REGEX = r"[\u0600-\u06FF\u200c\u200f ،؛؟!.:0-9۰-۹]{1,400}"


class OpenAIStreamingResponder:
    """Stream from an OpenAI-compatible server (e.g. vLLM) with a Persian-only constraint.

    The constraint is a regular-expression structured output, so Latin-script
    text cannot be produced at all; it plays the role of the local logits mask.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        system_prompt: str,
        max_new_tokens: int = 64,
        history_turns: int = 2,
        persian_only: bool = True,
        timeout: float = 30.0,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.chat_template_kwargs = chat_template_kwargs
        self.model = model
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.history_turns = history_turns
        self.persian_only = persian_only
        self.timeout = timeout

    def messages(self, user_text: str, history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        for user, assistant in list(history)[-self.history_turns :] if self.history_turns else []:
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": assistant})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def stream(
        self,
        user_text: str,
        history: Sequence[tuple[str, str]] = (),
        cancel: threading.Event | None = None,
    ) -> Iterator[str]:
        import json
        import urllib.request

        body: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages(user_text, history),
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
            "stream": True,
        }
        if self.persian_only:
            body["structured_outputs"] = {"regex": PERSIAN_ONLY_REGEX}
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        request = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                for raw in response:
                    if cancel is not None and cancel.is_set():
                        return
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    choices = json.loads(line[5:]).get("choices") or []
                    delta = (choices[0].get("delta") or {}).get("content") if choices else None
                    if delta:
                        yield delta
        except OSError as exc:
            raise RuntimeError(f"generation failed: {exc!r}") from exc


def run_streamed_turn(
    user_audio: np.ndarray,
    *,
    asr: Callable[[np.ndarray], tuple[str, str | None]],
    responder: StreamingResponder | OpenAIStreamingResponder,
    tts: Callable[[str], np.ndarray],
    emit_audio: Callable[[np.ndarray, int], None],
    history: Sequence[tuple[str, str]] = (),
    cancel: threading.Event | None = None,
    chunker: ClauseChunker | None = None,
    empty_reply: str = "متأسفم، صدای شما را درست نشنیدم. لطفاً دوباره بگویید.",
    on_transcript: Callable[[str], None] | None = None,
) -> StreamedTurn:
    """Run one turn, emitting each synthesized clause as soon as it is ready.

    ``on_transcript`` is called with the ASR transcript before generation starts
    (the end-of-turn gate reuses the speculative transcript).
    """

    stop = cancel or threading.Event()
    timings = TurnTimings(start=time.perf_counter())
    transcript, asr_error = asr(user_audio)
    timings.asr_done = time.perf_counter()
    if on_transcript is not None:
        on_transcript(transcript)
    chunker = chunker or ClauseChunker()
    clauses: list[str] = []
    samples = 0
    responder_error: str | None = None

    def speak(text: str) -> None:
        nonlocal samples
        if stop.is_set():
            return
        if timings.first_clause is None:
            timings.first_clause = time.perf_counter()
        audio = tts(text)
        if stop.is_set() or audio.size == 0:
            return
        if timings.first_audio_ready is None:
            timings.first_audio_ready = time.perf_counter()
        emit_audio(audio, len(clauses))
        clauses.append(text)
        samples += int(audio.size)

    if not transcript.strip():
        speak(empty_reply)
    else:
        try:
            for delta in responder.stream(transcript, history, stop):
                if timings.first_token is None and delta.strip():
                    timings.first_token = time.perf_counter()
                for clause in chunker.push(delta):
                    speak(clause)
                if stop.is_set():
                    break
            rest = chunker.flush()
            if rest and not stop.is_set():
                speak(rest)
        except RuntimeError as exc:
            responder_error = str(exc)
    timings.done = time.perf_counter()
    return StreamedTurn(
        transcript=transcript,
        reply_text=" ".join(clauses),
        clauses=clauses,
        timings=timings,
        asr_error=asr_error,
        responder_error=responder_error,
        cancelled=stop.is_set(),
        audio_samples=samples,
    )
