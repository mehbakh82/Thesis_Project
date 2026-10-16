#!/usr/bin/env python3
"""Minimal loopback ASR service for a NeMo hybrid checkpoint (stdlib HTTP).

Run with the NeMo environment's interpreter. It implements the runtime's
``asr_http`` contract: ``POST /transcribe/binary`` with a 16 kHz mono PCM16 WAV
body returns ``{"text": ..., "model_sha256": ..., "decoder": ...}``;
``GET /health`` reports readiness. One request is served at a time so the
reported latency reflects a single utterance.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NemoTranscriber:
    def __init__(self, checkpoint: Path, decoder: str) -> None:
        import torch
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        from omegaconf import OmegaConf

        self.torch = torch
        self.model = EncDecHybridRNNTCTCBPEModel.restore_from(str(checkpoint), map_location="cpu")
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.change_decoding_strategy(
            OmegaConf.create({"strategy": "greedy_batch"}), decoder_type=decoder
        )
        self.decoder = decoder
        self.lock = threading.Lock()

    def transcribe(self, audio: np.ndarray) -> str:
        with self.lock, self.torch.inference_mode():
            result = self.model.transcribe([audio], batch_size=1, verbose=False)
        first = result[0] if isinstance(result, list) else result
        if isinstance(first, list):
            first = first[0]
        return str(getattr(first, "text", first)).strip()


def decode_wav(body: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(body), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("expected mono PCM16 WAV")
        if handle.getframerate() != 16_000:
            raise ValueError("expected 16 kHz WAV")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder", choices=("ctc", "rnnt"), default="ctc")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("the ASR service binds to loopback only")

    model_sha = sha256_file(args.checkpoint)
    transcriber = NemoTranscriber(args.checkpoint, args.decoder)
    transcriber.transcribe(np.zeros(16_000, dtype=np.float32))  # warm-up

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "model_sha256": model_sha, "decoder": args.decoder})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if not self.path.startswith("/transcribe/binary"):
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 50 * 1024 * 1024:
                self._send(413, {"error": "invalid body size"})
                return
            try:
                audio = decode_wav(self.rfile.read(length))
                started = time.perf_counter()
                text = transcriber.transcribe(audio)
                elapsed = 1000.0 * (time.perf_counter() - started)
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._send(
                200,
                {
                    "text": text,
                    "model_sha256": model_sha,
                    "decoder": args.decoder,
                    "inference_ms": round(elapsed, 2),
                },
            )

        def log_message(self, format: str, *log_args: object) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"ready": True, "port": args.port, "model_sha256": model_sha}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
