#!/usr/bin/env python3
"""Reference-blind, per-clip-timed ASR predictions for a frozen panel.

Run with the interpreter that owns the chosen backend. Most backends are the
user's pinned, previously audited inference wrappers from the noisy-speech
study (pass ``--external-backends`` pointing at that directory); ``gemma4``
is implemented here. Each clip is transcribed alone (batch size 1) after a
warm-up, so ``latency_ms`` is the per-utterance processing time. The output
never contains reference text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

GEMMA_PROMPT = (
    "Transcribe this Persian (Farsi) speech verbatim in Persian script. "
    "Output only the transcription, with no translation or commentary."
)


class GemmaAudioBackend:
    """Gemma 4 E2B/E4B/12B native audio input used as a zero-shot ASR."""

    def __init__(self, checkpoint: Path, max_new_tokens: int = 256) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(checkpoint))
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(checkpoint), dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        self.max_new_tokens = max_new_tokens

    def predict(self, paths: list[Path]) -> list[str]:
        import wave

        import numpy as np

        outputs = []
        for path in paths:
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
                channels = handle.getnchannels()
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            if rate != 16_000:
                raise ValueError("gemma4 backend expects 16 kHz panels")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": np.asarray(audio, dtype=np.float32)},
                        {"type": "text", "text": GEMMA_PROMPT},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            text = self.processor.decode(
                generated[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            outputs.append(text.strip())
        return outputs


def load_backend(args: argparse.Namespace) -> Any:
    if args.backend == "gemma4":
        return GemmaAudioBackend(args.checkpoint)
    if not args.external_backends:
        raise SystemExit("--external-backends is required for this backend")
    sys.path.insert(0, str(args.external_backends))
    os.chdir(args.external_backends)
    import infer  # type: ignore[import-not-found]

    try:
        import infer_additional  # type: ignore[import-not-found]  # noqa: F401 - registers extras
    except Exception as exc:  # noqa: BLE001
        print(f"note: optional backends unavailable: {exc}", file=sys.stderr)
    return infer.BACKENDS[args.backend](args.checkpoint, 1, args.beam_size, None, 0.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--external-backends", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.panel = args.panel.resolve()
    args.out = args.out.resolve()
    args.checkpoint = args.checkpoint.resolve()

    rows = [json.loads(line) for line in args.panel.read_text(encoding="utf-8").splitlines() if line]
    done = set()
    if args.out.is_file():
        done = {json.loads(line)["id"] for line in args.out.read_text().splitlines() if line}
    load_started = time.perf_counter()
    backend = load_backend(args)
    load_seconds = time.perf_counter() - load_started
    backend.predict([Path(rows[0]["audio_filepath"])])  # warm-up, not recorded
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with args.out.open("a", encoding="utf-8") as sink:
        for row in rows:
            if row["id"] in done:
                continue
            path = Path(row["audio_filepath"])
            started = time.perf_counter()
            error = None
            try:
                hypothesis = backend.predict([path])[0]
            except Exception as exc:  # noqa: BLE001 - failure is scored as full deletion
                hypothesis, error = "", f"{type(exc).__name__}: {exc}"
            latency = 1000.0 * (time.perf_counter() - started)
            sink.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "arm": args.arm,
                        "hypothesis": hypothesis,
                        "latency_ms": round(latency, 2),
                        "error": error,
                        "audio_seconds": row.get("duration"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    meta = {
        "arm": args.arm,
        "backend": args.backend,
        "checkpoint": str(args.checkpoint),
        "beam_size": args.beam_size,
        "load_seconds": round(load_seconds, 2),
        "peak_torch_gib": round(peak, 3) if peak is not None else None,
        "panel_sha256": sha256_file(args.panel),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
