#!/usr/bin/env python3
"""Automated browser loopback latency test (proposal section 6.4).

Pre-recorded Persian utterances (1-5 s) are concatenated into one fake
microphone WAV with long pauses between turns. Headless Chromium opens the
real live client page with that file as its microphone, so audio travels the
same path as a real user: Web Audio capture -> 20 ms PCM frames -> WebSocket ->
endpointing/ASR/LLM/TTS -> streamed clauses -> Web Audio playback scheduling.
First-audio latency is measured by the client from the captured frame holding
the acoustic end of speech to the scheduled start of the first clause.

Optional scripted interruptions place a second utterance ``--barge-offset``
seconds after a turn ends (during the reply); optional backchannels place a
short listener token at the same offset. A panel row may instead carry its own
``barge_audio_filepath`` (and ``barge_offset_s``), for scripted dialogues where
the interruption belongs to the request; ``--keep-order`` keeps the panel order. This is automated loopback evidence on
the machine where it runs; it is never a human-study or target-GPU result by
itself.

Run with an interpreter that has Playwright (and its Chromium) installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RATE = 16_000


def read_mono16(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        if handle.getframerate() != RATE or handle.getsampwidth() != 2:
            raise ValueError(f"expected 16 kHz PCM16: {path}")
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        if handle.getnchannels() > 1:
            frames = frames.reshape(-1, handle.getnchannels()).mean(axis=1).astype("<i2")
    return frames.astype(np.float32) / 32768.0


def write_mono16(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def build_timeline(
    utterances: list[dict],
    *,
    gap_seconds: float,
    barge: list[np.ndarray | None],
    barge_offset: float,
    barge_offsets: list[float | None] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    parts = [np.zeros(int(1.5 * RATE), np.float32)]
    cursor = parts[0].size
    plan = []
    offsets = barge_offsets or [None] * len(utterances)
    for row, extra, own_offset in zip(utterances, barge, offsets, strict=True):
        audio = read_mono16(row["audio_filepath"])
        start, end = cursor, cursor + audio.size
        segment = np.zeros(audio.size + int(gap_seconds * RATE), np.float32)
        segment[: audio.size] = audio
        entry = {"id": row["id"], "start_s": start / RATE, "end_s": end / RATE}
        if extra is not None:
            offset = audio.size + int((barge_offset if own_offset is None else own_offset) * RATE)
            segment[offset : offset + extra.size] += extra
            entry["barge_onset_s"] = (cursor + offset) / RATE
            entry["barge_seconds"] = extra.size / RATE
        parts.append(segment)
        cursor += segment.size
        plan.append(entry)
    return np.concatenate(parts), plan


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True, help="JSONL with id/audio_filepath/duration")
    parser.add_argument("--telemetry", type=Path, required=True, help="server telemetry JSONL")
    parser.add_argument("--url", default="http://127.0.0.1:8766/?autostart=1")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--gap-seconds", type=float, default=7.0)
    parser.add_argument("--barge-panel", type=Path, help="JSONL of clips to overlay during replies")
    parser.add_argument("--barge-offset", type=float, default=0.9)
    parser.add_argument("--condition", default="plain")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--keep-order", action="store_true", help="do not shuffle the panel")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    rows = [json.loads(line) for line in args.panel.read_text(encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if args.min_seconds <= float(r["duration"]) <= args.max_seconds]
    rng = random.Random(args.seed)
    if not args.keep_order:
        rng.shuffle(rows)
    rows = rows[: args.turns]
    barge: list[np.ndarray | None] = [None] * len(rows)
    if args.barge_panel:
        clips = [json.loads(line) for line in args.barge_panel.read_text().splitlines() if line]
        rng.shuffle(clips)
        barge = [read_mono16(clips[i % len(clips)]["audio_filepath"]) for i in range(len(rows))]
    for i, row in enumerate(rows):  # a row's own interruption takes precedence
        if row.get("barge_audio_filepath"):
            barge[i] = read_mono16(row["barge_audio_filepath"])
    audio, plan = build_timeline(
        rows,
        gap_seconds=args.gap_seconds,
        barge=barge,
        barge_offset=args.barge_offset,
        barge_offsets=[row.get("barge_offset_s") for row in rows],
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    mic = args.work_dir / f"fake_mic_{args.condition}.wav"
    write_mono16(mic, audio)
    offset = args.telemetry.stat().st_size if args.telemetry.exists() else 0
    duration = audio.size / RATE

    started = time.time()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={mic}%noloop",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        page = browser.new_page()
        console: list[str] = []
        page.on("console", lambda msg: console.append(msg.text))
        page.goto(args.url)
        page.wait_for_timeout(int((duration + 8.0) * 1000))
        events = page.evaluate("window.__events || []")
        browser.close()

    lines = args.telemetry.read_bytes()[offset:].decode("utf-8").splitlines()
    turns = [json.loads(line) for line in lines if line.strip()]
    first = [float(t["client_t_first_audio_ms"]) for t in turns if t.get("client_t_first_audio_ms") is not None]
    audible = [float(t["client_t_first_audible_ms"]) for t in turns if t.get("client_t_first_audible_ms") is not None]
    duck = [float(t["client_t_duck_ms"]) for t in turns if t.get("client_t_duck_ms") is not None]
    stop = [float(t["client_t_stop_ms"]) for t in turns if t.get("client_t_stop_ms") is not None]
    outcomes: dict[str, int] = {}
    for t in turns:
        outcomes[str(t.get("outcome"))] = outcomes.get(str(t.get("outcome")), 0) + 1
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "automated_browser_loopback",
        "official_e2e_eligible": False,
        "condition": args.condition,
        "browser": "playwright-chromium-headless (fake microphone file)",
        "planned_user_turns": len(plan),
        "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        "fake_mic_sha256": hashlib.sha256(mic.read_bytes()).hexdigest(),
        "fake_mic_seconds": round(duration, 2),
        "wall_seconds": round(time.time() - started, 1),
        "server_turn_records": len(turns),
        "outcomes": outcomes,
        "t_first_audio_ms": summarize(first),
        "t_first_audible_ms": summarize(audible),
        "t_duck_ms": summarize(duck),
        "t_stop_ms": summarize(stop),
        "plan": plan,
        "turns": [
            {k: t.get(k) for k in (
                "turn", "end_sample", "utterance_seconds", "client_t_first_audio_ms", "client_t_first_audible_ms",
                "client_t_duck_ms", "client_t_stop_ms", "server_timings_ms", "outcome",
                "speculative_start", "speculation_confirmed", "bargein_events", "asr_error",
                "responder_error", "clauses", "speculation_cancelled", "client_played_ms", "heard_words",
            )}
            for t in turns
        ],
        "client_events": len(events),
        "console_errors": [c for c in console if "error" in c.lower()][:20],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in (
        "condition", "planned_user_turns", "server_turn_records", "outcomes",
        "t_first_audio_ms", "t_duck_ms", "t_stop_ms")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
