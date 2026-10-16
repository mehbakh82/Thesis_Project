#!/usr/bin/env python3
"""Stream panel clips to the live server at real-time pace and time each turn.

This is a protocol-level probe (no browser audio stack): the first-audio time
is measured from the moment the frame holding the server-reported acoustic end
of speech was sent to the arrival of the first synthesized clause. The browser
loopback harness (`scripts/loopback_latency.py`) adds client scheduling and
output latency on top of this.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.audio import read_wav  # noqa: E402

FRAME = 320


async def one_turn(url: str, audio: np.ndarray, timeout: float) -> dict:
    import websockets

    padded = np.concatenate([np.zeros(8000, np.float32), audio, np.zeros(48000, np.float32)])
    sent: dict[int, float] = {}
    result: dict = {}
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"event": "hello"}))
        await ws.recv()

        async def sender() -> None:
            start = time.perf_counter()
            for index in range(padded.size // FRAME):
                frame = (padded[index * FRAME : (index + 1) * FRAME] * 32767).astype("<i2")
                target = start + index * FRAME / 16000
                await asyncio.sleep(max(0.0, target - time.perf_counter()))
                sent[index] = time.perf_counter()
                await ws.send(struct.pack("<I", index) + frame.tobytes())

        task = asyncio.create_task(sender())
        end_sample = None
        deadline = time.perf_counter() + timeout
        try:
            while time.perf_counter() < deadline:
                message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                if isinstance(message, bytes):
                    if "first_audio_ms" not in result and end_sample is not None:
                        frame_index = end_sample // FRAME
                        acoustic_end = sent[frame_index] + (end_sample % FRAME) / 16000
                        result["first_audio_ms"] = 1000 * (time.perf_counter() - acoustic_end)
                    continue
                payload = json.loads(message)
                if payload["event"] == "endpoint" and end_sample is None:
                    end_sample = payload["end_sample"]
                elif payload["event"] == "reply_done":
                    result.update(
                        transcript=payload["transcript"],
                        reply=payload["reply_text"],
                        server=payload["server_timings_ms"],
                    )
                    break
        finally:
            task.cancel()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--url", default="ws://127.0.0.1:8766/ws")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.panel.read_text(encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if args.min_seconds <= float(r["duration"]) <= args.max_seconds]
    results = []
    for row in rows[: args.limit]:
        audio, _ = read_wav(row["audio_filepath"])
        outcome = asyncio.run(one_turn(args.url, audio, timeout=30.0))
        outcome["id"] = row["id"]
        results.append(outcome)
        print(json.dumps({k: outcome.get(k) for k in ("id", "first_audio_ms", "server")}, ensure_ascii=False))
    values = [r["first_audio_ms"] for r in results if "first_audio_ms" in r]
    summary = {
        "turns": len(results),
        "measured": len(values),
        "first_audio_ms_median": statistics.median(values) if values else None,
        "first_audio_ms_max": max(values) if values else None,
    }
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(
            json.dumps({"summary": summary, "turns": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
