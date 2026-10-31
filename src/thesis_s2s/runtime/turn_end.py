"""Semantic end-of-turn gates (``docs/END_OF_TURN_PROTOCOL.md``).

At the speculative start the live server asks a gate for P(complete) of the
utterance so far; below the threshold the current pause is held longer before
the end of the turn is confirmed. The prompt and the word list below are part
of the frozen protocol.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SAMPLE_RATE = 16_000
DEFAULT_SMART_TURN_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "turn" / "smart-turn-v3.2-cpu.onnx"
)
# Stage 2: Smart Turn v3.2 fine-tuned on Persian (scripts/smart_turn_persian.py)
DEFAULT_SMART_TURN_FA_PATH = DEFAULT_SMART_TURN_PATH.with_name("smart-turn-v3.2-fa.onnx")
# Stage 2b (docs/END_OF_TURN_V2_PROTOCOL.md): fine-tuned on the restored diarized windows
DEFAULT_SMART_TURN_FA_V2_PATH = DEFAULT_SMART_TURN_PATH.with_name("smart-turn-v3.2-fa-v2.onnx")
SMART_TURN_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
SMART_TURN_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"

COMPLETENESS_PROMPT = (
    "You judge whether a Persian speaker has finished their turn. The user message is an "
    "automatic transcript of what they have said so far, without punctuation, and it may "
    "contain recognition errors. Answer 1 if it is a finished request, question or statement "
    "that a listener could reply to now. Answer 0 if the speaker is clearly in the middle of a "
    "sentence, for example if it ends in a conjunction, a preposition or «را», or the verb of "
    "the sentence is still missing. Answer with a single digit."
)

# Words that cannot end a Persian utterance: conjunctions, prepositions, the
# object marker, determiners and discourse fillers (formal and colloquial).
CONTINUATION_WORDS = frozenset(
    "و که با از به در برای یا اما ولی تا اگر اگه چون را رو این اون آن یه یک هم یعنی "
    "مثلاً مثلا بعد اینکه توی بر اِ اه اوم".split()
)
_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه"})
_STRIP = re.compile(r"[ً-ْٰ«»،؛؟!.,:\"'()\[\]-]")


class TurnEndGate(Protocol):
    name: str
    needs_transcript: bool

    def p_complete(self, audio: np.ndarray, transcript: str | None) -> float: ...


def last_word(transcript: str) -> str:
    words = _STRIP.sub(" ", transcript.translate(_ARABIC_TO_PERSIAN)).split()
    return words[-1] if words else ""


def smart_turn_features(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Whisper log-mel of the last 8 s (zero-padded at the start), normalized input.

    Equal to ``WhisperFeatureExtractor(chunk_length=8)`` with ``do_normalize``,
    as in the reference Smart Turn inference (tested to 1e-5), but vectorized so
    the gate fits its time budget.
    """

    samples = 8 * SAMPLE_RATE
    x = np.asarray(audio, dtype=np.float64)[-samples:]
    if x.size < samples:
        x = np.pad(x, (samples - x.size, 0))
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, 200, mode="reflect"), 400)
    power = np.abs(np.fft.rfft(frames[::160] * np.hanning(401)[:-1], axis=1)) ** 2
    log_mel = np.log10(np.maximum(power @ mel_filters, 1e-10)).T[:, :-1]
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    return ((log_mel + 4.0) / 4.0).astype(np.float32)[None]


class LexicalGate:
    """Candidate L: hold if the transcript ends in a word that cannot end an utterance."""

    name = "lexical"
    needs_transcript = True

    def p_complete(self, audio: np.ndarray, transcript: str | None) -> float:
        word = last_word(transcript or "")
        return 0.0 if not word or word in CONTINUATION_WORDS else 1.0


class SmartTurnGate:
    """Candidate A: Smart Turn v3.2 (audio only), with the reference preprocessing."""

    name = "smart_turn_v3_2"
    needs_transcript = False

    def __init__(
        self, model_path: Path | None = None, threads: int = 2, name: str = "smart_turn_v3_2"
    ) -> None:
        import onnxruntime as ort

        self.name = name
        from transformers import WhisperFeatureExtractor

        path = Path(model_path or os.environ.get("SMART_TURN_MODEL") or DEFAULT_SMART_TURN_PATH)
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.mel_filters = WhisperFeatureExtractor(chunk_length=8).mel_filters
        self.model_path = path

    def p_complete(self, audio: np.ndarray, transcript: str | None = None) -> float:
        features = smart_turn_features(audio, self.mel_filters)
        output = self.session.run(None, {"input_features": features})[0]
        return float(output.reshape(-1)[0])


class LLMCompletenessGate:
    """Candidate B: the deployed LLM judges the speculative transcript."""

    name = "llm_completeness"
    needs_transcript = True

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout: float = 2.0,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.chat_template_kwargs = chat_template_kwargs

    def p_complete(self, audio: np.ndarray, transcript: str | None) -> float:
        if not (transcript or "").strip():
            return 0.0
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": COMPLETENESS_PROMPT},
                {"role": "user", "content": transcript},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
        }
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        request = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0].get("logprobs", {}).get("content") or []
        top = content[0].get("top_logprobs", []) if content else []
        probs = {"0": 0.0, "1": 0.0}
        for entry in top:
            token = entry.get("token", "").strip()
            if token in probs:
                probs[token] += float(np.exp(entry["logprob"]))
        total = probs["0"] + probs["1"]
        return probs["1"] / total if total > 0 else 1.0


def make_gate(
    name: str,
    *,
    endpoint: str = "",
    model: str = "",
    chat_template_kwargs: dict[str, Any] | None = None,
) -> TurnEndGate:
    if name == "lexical":
        return LexicalGate()
    if name == "smart_turn_v3_2":
        return SmartTurnGate()
    if name == "smart_turn_v3_2_fa":
        path = Path(os.environ.get("SMART_TURN_FA_MODEL") or DEFAULT_SMART_TURN_FA_PATH)
        return SmartTurnGate(model_path=path, name=name)
    if name == "smart_turn_v3_2_fa_v2":
        path = Path(os.environ.get("SMART_TURN_FA_V2_MODEL") or DEFAULT_SMART_TURN_FA_V2_PATH)
        return SmartTurnGate(model_path=path, name=name)
    if name == "llm_completeness":
        if not endpoint:
            raise ValueError("the LLM completeness gate needs TEXT_LLM_ENDPOINT")
        return LLMCompletenessGate(
            endpoint=endpoint, model=model, chat_template_kwargs=chat_template_kwargs
        )
    raise ValueError(f"unknown end-of-turn gate: {name}")
