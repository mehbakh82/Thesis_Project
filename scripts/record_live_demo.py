#!/usr/bin/env python3
"""Record a real interaction with the live server as a single listenable WAV.

The user clip is streamed at real-time pace; every synthesized clause is placed
on the same timeline at the moment it arrived (as the browser would schedule
it, 10 ms later). With ``--interrupt-clip`` a second user utterance starts
``--interrupt-after`` seconds after the first reply audio arrives. As in the
browser client, the assistant audio is lowered to 0.2 gain from a ``duck``
event until ``resume``, and the stopped turn is cut at the ``stop`` event. A
backchannel clip (``--expect-replies 1``) should not create a second reply.
The output mixes both sides (user slightly quieter) and a JSON sidecar records
the measured timings. This is a recording of actual system behaviour, not a
re-synthesis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.audio import read_wav  # noqa: E402

RATE = 16_000
FRAME = 320


async def record(
    url: str, first: np.ndarray, interrupt: np.ndarray | None, after: float, needed: int
) -> dict:
    import websockets

    timeline_user = np.concatenate([np.zeros(int(0.5 * RATE), np.float32), first])
    chunks: list[tuple[float, int, np.ndarray]] = []
    events: list[tuple[float, dict]] = []
    t0 = time.perf_counter()
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"event": "hello"}))
        await ws.recv()
        state: dict = {"first_audio_t": None, "stop_t": None, "stop_turn": None, "done": False}
        user_stream = [timeline_user]

        async def sender() -> None:
            seq = 0
            injected = False
            cursor = 0
            buffer = user_stream[0]
            while not state["done"]:
                if interrupt is not None and not injected and state["first_audio_t"] is not None:
                    if time.perf_counter() - t0 >= state["first_audio_t"] + after:
                        start_sample = seq * FRAME
                        buffer = np.concatenate([buffer[:start_sample], np.zeros(max(0, start_sample - buffer.size), np.float32), interrupt])
                        user_stream[0] = buffer
                        injected = True
                frame = buffer[cursor : cursor + FRAME]
                if frame.size < FRAME:
                    frame = np.concatenate([frame, np.zeros(FRAME - frame.size, np.float32)])
                target = t0 + seq * FRAME / RATE
                await asyncio.sleep(max(0.0, target - time.perf_counter()))
                await ws.send(struct.pack("<I", seq) + (frame * 32767).astype("<i2").tobytes())
                seq += 1
                cursor += FRAME

        task = asyncio.create_task(sender())
        deadline = time.perf_counter() + 60
        replies_done = 0
        play_end: dict[int, float] = {}  # virtual playback end per turn, as the browser queue would play it
        ended: set[int] = set()
        stopped: set[int] = set()
        try:
            while time.perf_counter() < deadline:
                now = time.perf_counter() - t0
                # emulate the browser: report the end of playback so the server returns to listening
                for turn, end in list(play_end.items()):
                    if turn not in ended and turn not in stopped and now > end:
                        ended.add(turn)
                        await ws.send(json.dumps({"event": "playback_ended", "turn": turn}))
                if replies_done >= needed and all(t in ended or t in stopped for t in play_end) and now > max(
                    play_end.values(), default=0.0
                ) + 1.5:
                    break
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                now = time.perf_counter() - t0
                if isinstance(message, bytes):
                    turn, index = struct.unpack("<II", message[:8])
                    pcm = np.frombuffer(message[8:], dtype="<i2").astype(np.float32) / 32768.0
                    if state["first_audio_t"] is None:
                        state["first_audio_t"] = now
                    start = max(now + 0.01, play_end.get(turn, 0.0))
                    play_end[turn] = start + pcm.size / RATE
                    chunks.append((now + 0.01, turn, pcm))
                    continue
                payload = json.loads(message)
                events.append((now, payload))
                if payload.get("event") == "stop":
                    stopped.add(int(payload.get("turn") or 0))
                    if state["stop_t"] is None:
                        state["stop_t"] = now
                        state["stop_turn"] = payload.get("turn")
                if payload.get("event") == "reply_done":
                    replies_done += 1
        finally:
            state["done"] = True
            task.cancel()
    return {
        "chunks": chunks,
        "events": events,
        "user": user_stream[0],
        "stop_t": state["stop_t"],
        "stop_turn": state["stop_turn"],
    }


def mix(result: dict) -> np.ndarray:
    end = max([t + c.size / RATE for t, _turn, c in result["chunks"]] + [result["user"].size / RATE]) + 0.5
    out = np.zeros(int(end * RATE), np.float32)
    user = result["user"]
    out[: user.size] += 0.8 * user
    gain = np.ones(out.size, np.float32)
    ducked_at: float | None = None
    for t, event in result["events"]:
        kind = event.get("event")
        if kind == "duck":
            ducked_at = t
        elif kind in {"resume", "stop"} and ducked_at is not None:
            gain[int(ducked_at * RATE) : int(t * RATE)] = 0.2
            ducked_at = None
    if ducked_at is not None:
        gain[int(ducked_at * RATE) :] = 0.2
    assistant = np.zeros(out.size, np.float32)
    play_cursor: dict[int, float] = {}
    stop_t, stop_turn = result["stop_t"], result["stop_turn"]
    for t, turn, pcm in result["chunks"]:
        start = max(t, play_cursor.get(turn, 0.0))
        play_cursor[turn] = start + pcm.size / RATE
        a = int(start * RATE)
        segment = pcm
        if stop_t is not None and turn == stop_turn:
            if start >= stop_t:
                continue
            segment = pcm[: max(0, int((stop_t - start) * RATE))]
        assistant[a : a + segment.size] += segment[: max(0, out.size - a)]
    out += assistant * gain
    return np.clip(out, -1, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--interrupt-clip", type=Path)
    parser.add_argument("--interrupt-after", type=float, default=1.2)
    parser.add_argument("--expect-replies", type=int, help="default: 2 with an interrupt clip, else 1")
    parser.add_argument("--url", default="ws://127.0.0.1:8766/ws")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    first, _ = read_wav(args.clip)
    interrupt = read_wav(args.interrupt_clip)[0] if args.interrupt_clip else None
    needed = args.expect_replies or (2 if interrupt is not None else 1)
    result = asyncio.run(record(args.url, first, interrupt, args.interrupt_after, needed))
    audio = mix(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((audio * 32767).astype("<i2").tobytes())
    sidecar = {
        "clip": args.clip.name,
        "interrupt_clip": args.interrupt_clip.name if args.interrupt_clip else None,
        "events": [{"t": round(t, 3), **{k: v for k, v in e.items() if k in ("event", "turn", "transcript", "reply_text", "server_timings_ms")}} for t, e in result["events"]],
        "first_audio_arrival_s": round(result["chunks"][0][0], 3) if result["chunks"] else None,
        "stop_t": result["stop_t"],
        "stop_turn": result["stop_turn"],
        "bargein_events": [e.get("event") for _t, e in result["events"] if e.get("event") in {"duck", "resume", "stop"}],
    }
    args.out.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(sidecar, ensure_ascii=False)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
