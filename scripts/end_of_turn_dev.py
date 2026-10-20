#!/usr/bin/env python3
"""Development benchmark and selection for the semantic end-of-turn gate.

Protocol: ``docs/END_OF_TURN_PROTOCOL.md``.

``build`` selects the development clips (barge-in speech pool, sessions disjoint
from every loopback test panel, 1-5 s) and writes their simulated-elderly copies
(ffmpeg ``atempo=0.85,volume=-3dB``, as for the test panel).

``score`` replays every trial through the deployed ``Endpointer`` with a gate in
the loop. The gate is asked at each speculative start, exactly as the live
server does, and its answer takes effect only after its measured compute time
(for transcript gates: the speculative ASR call plus the gate; for the LLM gate
while the speculative reply streams on the same server). Gate outputs are
cached per utterance span, so every grid configuration replays the same
answers.

``select`` applies the pre-registered rule and writes the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.runtime.vad import SAMPLE_RATE, Endpointer, default_scorer  # noqa: E402

FRAME = 320  # the live client's 20 ms frame
PRE_ROLL = int(0.2 * SAMPLE_RATE)
HANGOVER_MS = 256.0
SPECULATIVE_MS = 96.0
LEAD, TAIL = int(0.3 * SAMPLE_RATE), int(1.5 * SAMPLE_RATE)
CANDIDATES = ("lexical", "smart_turn_v3_2", "llm_completeness")
THRESHOLDS = (0.3, 0.5, 0.7)
HOLDS_MS = (640.0, 1024.0)
COST_ORDER = {"none": -1, "lexical": 0, "smart_turn_v3_2": 1, "llm_completeness": 2}
TEST_PANELS = ("plain", "elderly", "interruptions", "backchannels", "backchannels_seamless")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getsampwidth() != 2:
            raise ValueError(f"expected 16 kHz PCM16: {path}")
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        if handle.getnchannels() > 1:
            frames = frames.reshape(-1, handle.getnchannels()).mean(axis=1)
    return frames.astype(np.float32) / 32768.0


def build(args: argparse.Namespace) -> int:
    panels = args.panels_dir
    pool = [json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines() if line]
    by_file = {Path(r["audio_filepath"]).name: r for r in pool}
    test_sessions: set[str] = set()
    for name in TEST_PANELS:
        for line in (panels / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("session_id"):
                test_sessions.add(row["session_id"])
            source = by_file.get(Path(row["audio_filepath"]).name.removeprefix("corpus-"))
            if source is not None:
                test_sessions.add(source["session_id"])
    clips = [
        r for r in pool
        if r["kind"] == "speech" and r["session_id"] not in test_sessions and 1.0 <= r["duration"] <= 5.0
    ]
    elderly_dir = args.work / "elderly"
    elderly_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    for row in sorted(clips, key=lambda r: r["audio_filepath"]):
        clip_id = Path(row["audio_filepath"]).stem
        trials.append({"trial": f"plain/{clip_id}", "condition": "plain", "audio": row["audio_filepath"],
                       "session_id": row["session_id"]})
        out = elderly_dir / f"{clip_id}.wav"
        if not out.exists():
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", row["audio_filepath"],
                 "-af", "atempo=0.85,volume=-3dB", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(out)],
                check=True,
            )
        trials.append({"trial": f"elderly/{clip_id}", "condition": "simulated_elderly", "audio": str(out),
                       "session_id": row["session_id"]})
    path = args.work / "trials.jsonl"
    path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trials), encoding="utf-8")
    print(json.dumps({"clips": len(clips), "trials": len(trials), "sessions": len({r["session_id"] for r in clips}),
                      "excluded_test_sessions": len(test_sessions)}))
    return 0


class GateCache:
    """Gate answers and their measured readiness, per (trial, utterance start, pause)."""

    def __init__(self, path: Path, candidate: str) -> None:
        self.path = path
        self.candidate = candidate
        self.rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.rows[row["key"]] = row
        self._gate: Any = None
        self._asr: Any = None
        self._responder: Any = None
        self._transcripts: dict[str, tuple[str, float]] = {}

    def _setup(self) -> None:
        if self._gate is not None:
            return
        from thesis_s2s.runtime.turn_end import make_gate

        endpoint = os.environ.get("TEXT_LLM_ENDPOINT", "")
        served = os.environ.get("TEXT_LLM_SERVED_NAME", "qwen3-4b-instruct-2507")
        self._gate = make_gate(self.candidate, endpoint=endpoint, model=served)
        if self._gate.needs_transcript:
            from thesis_s2s.runtime.cascade import _asr_transcript, _wav_bytes, asr_http

            def asr(audio: np.ndarray) -> str:
                return _asr_transcript(asr_http(_wav_bytes(audio)))

            self._asr = asr
        if self.candidate == "llm_completeness":
            from thesis_s2s.runtime.cascade import QWEN4B_PROMPT_V2
            from thesis_s2s.runtime.streaming import OpenAIStreamingResponder

            self._responder = OpenAIStreamingResponder(
                endpoint=endpoint, model=served, system_prompt=QWEN4B_PROMPT_V2
            )

    def answer(self, key: str, utterance: np.ndarray) -> dict[str, Any]:
        if key in self.rows:
            return self.rows[key]
        self._setup()
        transcript: str | None = None
        asr_ms = 0.0
        if self._gate.needs_transcript:
            started = time.perf_counter()
            transcript = self._asr(utterance)
            asr_ms = (time.perf_counter() - started) * 1000
        background: threading.Thread | None = None
        if self._responder is not None and transcript:
            # the speculative reply streams on the same server while the gate asks
            background = threading.Thread(
                target=lambda: list(self._responder.stream(transcript)), daemon=True
            )
            background.start()
        started = time.perf_counter()
        p_complete = float(self._gate.p_complete(utterance, transcript))
        gate_ms = (time.perf_counter() - started) * 1000
        if background is not None:
            background.join()
        row = {
            "key": key,
            "p_complete": p_complete,
            "asr_ms": round(asr_ms, 1),
            "gate_ms": round(gate_ms, 1),
            "ready_ms": round(asr_ms + gate_ms, 1),
            "transcript_words": len(transcript.split()) if transcript else None,
        }
        self.rows[key] = row
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return row


def true_end(audio: np.ndarray) -> int:
    """End of the clip's last voiced segment (the last end the deployed endpointer reports)."""

    endpointer = Endpointer(scorer=default_scorer(), hangover_ms=HANGOVER_MS, tentative_ms=None)
    end = 0
    for start in range(0, audio.size, FRAME):
        for event in endpointer.push(audio[start : start + FRAME]):
            if event.kind == "speech_end":
                end = event.sample
    return end


def replay(
    audio: np.ndarray,
    trial: str,
    *,
    cache: GateCache | None,
    threshold: float,
    hold_ms: float,
) -> dict[str, Any]:
    """Run one trial up to the first confirmed end of a turn, as the live server does."""

    endpointer = Endpointer(scorer=default_scorer(), hangover_ms=HANGOVER_MS, tentative_ms=SPECULATIVE_MS)
    utterance_start: int | None = None
    pending: tuple[int, bool] | None = None  # (sample at which the answer is ready, hold?)
    held = False
    budget = int((HANGOVER_MS - SPECULATIVE_MS) * SAMPLE_RATE / 1000)
    late = 0
    for start in range(0, audio.size, FRAME):
        position = start + FRAME
        for event in endpointer.push(audio[start:position], sample_offset=start):
            if event.kind == "speech_start":
                utterance_start = max(0, event.sample - PRE_ROLL)
                held = False
            elif event.kind == "speech_pause" and utterance_start is not None and cache is not None:
                key = f"{trial}|{utterance_start}|{event.sample}"
                answer = cache.answer(key, audio[utterance_start : event.sample])
                ready = event.decided_at_sample + int(answer["ready_ms"] * SAMPLE_RATE / 1000)
                wants_hold = answer["p_complete"] < threshold
                if wants_hold and ready - event.decided_at_sample >= budget:
                    late += 1
                pending = (ready, wants_hold)
            elif event.kind == "speech_resume":
                pending, held = None, False
            elif event.kind == "speech_end":
                return {"end_sample": event.sample, "decided_sample": event.decided_at_sample,
                        "held": held, "late": late}
        if pending is not None and position >= pending[0]:
            if pending[1] and endpointer.hold(hold_ms):
                held = True
            pending = None
    return {"end_sample": None, "decided_sample": None, "held": held, "late": late}


def score(args: argparse.Namespace) -> int:
    trials = [json.loads(line) for line in (args.work / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        trials = trials[: args.limit]
    configs: list[tuple[str, float, float]] = [("none", 0.0, HANGOVER_MS)]
    if args.candidate != "none":
        thresholds = (1.0,) if args.candidate == "lexical" else THRESHOLDS
        configs = [(args.candidate, t, h) for t in thresholds for h in HOLDS_MS]
    cache = None if args.candidate == "none" else GateCache(args.work / f"gate_{args.candidate}.jsonl", args.candidate)
    results: dict[str, list[dict[str, Any]]] = {}
    for index, trial in enumerate(trials):
        clip = read_wav(Path(trial["audio"]))
        audio = np.concatenate([np.zeros(LEAD, np.float32), clip, np.zeros(TAIL, np.float32)])
        last_voiced = true_end(audio)
        for candidate, threshold, hold_ms in configs:
            name = f"{candidate}|{threshold}|{int(hold_ms)}"
            out = replay(audio, trial["trial"], cache=cache, threshold=threshold, hold_ms=hold_ms)
            premature = out["end_sample"] is not None and out["end_sample"] < last_voiced - FRAME
            silence_ms = (
                (out["decided_sample"] - out["end_sample"]) * 1000 / SAMPLE_RATE
                if out["end_sample"] is not None else None
            )
            results.setdefault(name, []).append({
                "trial": trial["trial"],
                "condition": trial["condition"],
                "premature": bool(premature),
                "final_held": bool(out["held"] and not premature),
                "added_delay_ms": max(0.0, silence_ms - HANGOVER_MS) if silence_ms is not None and not premature else 0.0,
                "late_holds": out["late"],
                "unended": out["end_sample"] is None,
            })
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(trials)}", flush=True)
    out_path = args.work / f"replay_{args.candidate}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    if cache is not None:
        ready = [r["ready_ms"] for r in cache.rows.values()]
        print(json.dumps({"gate_calls": len(ready), "ready_ms_median": statistics.median(ready) if ready else None}))
    print(json.dumps(summarize(results), ensure_ascii=False, indent=1))
    return 0


def summarize(results: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for name, rows in results.items():
        candidate, threshold, hold = name.split("|")
        n = len(rows)
        delays = [r["added_delay_ms"] for r in rows]
        per_condition = {}
        for condition in sorted({r["condition"] for r in rows}):
            sub = [r for r in rows if r["condition"] == condition]
            per_condition[condition] = {
                "trials": len(sub),
                "premature": sum(r["premature"] for r in sub),
                "false_holds": sum(r["final_held"] for r in sub),
            }
        summary[name] = {
            "candidate": candidate,
            "threshold": float(threshold),
            "hold_ms": int(hold),
            "trials": n,
            "premature_rate": sum(r["premature"] for r in rows) / n,
            "false_hold_rate": sum(r["final_held"] for r in rows) / n,
            "late_hold_attempts": sum(r["late_holds"] for r in rows),
            "unended": sum(r["unended"] for r in rows),
            "mean_added_delay_ms": statistics.fmean(delays),
            "p90_added_delay_ms": float(np.percentile(delays, 90)),
            "by_condition": per_condition,
        }
    return summary


def select(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite the write-once receipt {args.out}")
    summary: dict[str, dict[str, Any]] = {}
    for candidate in ("none",) + CANDIDATES:
        path = args.work / f"replay_{candidate}.json"
        if path.exists():
            summary.update(summarize(json.loads(path.read_text(encoding="utf-8"))))
    baseline = next(v for v in summary.values() if v["candidate"] == "none")
    eligible = [v for v in summary.values() if v["candidate"] != "none" and v["false_hold_rate"] <= 0.05]
    eligible.sort(key=lambda v: (v["premature_rate"], v["false_hold_rate"], v["hold_ms"], COST_ORDER[v["candidate"]]))
    chosen = None
    if eligible:
        best = eligible[0]["premature_rate"]
        tied = [v for v in eligible if v["premature_rate"] <= best + 0.01]
        tied.sort(key=lambda v: (v["false_hold_rate"], v["hold_ms"], COST_ORDER[v["candidate"]]))
        chosen = tied[0]
    reduction = (
        1 - chosen["premature_rate"] / baseline["premature_rate"]
        if chosen and baseline["premature_rate"] > 0 else None
    )
    adopted = bool(chosen and reduction is not None and reduction >= 0.25)
    trials_path = args.work / "trials.jsonl"
    receipt = {
        "schema_version": 1,
        "experiment": "end_of_turn_gate_development",
        "protocol": "docs/END_OF_TURN_PROTOCOL.md",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trials_sha256": hashlib.sha256(trials_path.read_bytes()).hexdigest(),
        "endpointer": {"hangover_ms": HANGOVER_MS, "speculative_ms": SPECULATIVE_MS, "frame": FRAME},
        "rule": "lowest pooled premature-end rate among configurations with false-hold rate <= 5%; ties within "
                "1 point: lower false-hold rate, smaller hold, cheaper candidate; adopted only if the "
                "premature-end rate falls by at least 25% relative to no gate",
        "baseline": baseline,
        "configurations": summary,
        "chosen": chosen,
        "relative_reduction": reduction,
        "adopted": adopted,
        "next": "loopback v3 with the chosen gate" if adopted else "stage 2 (fine-tune Smart Turn on Persian)",
        "text_retained": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"chosen": chosen and {k: chosen[k] for k in ("candidate", "threshold", "hold_ms",
                      "premature_rate", "false_hold_rate")}, "baseline": baseline["premature_rate"],
                      "reduction": reduction, "adopted": adopted}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--pool", type=Path, required=True)
    p_build.add_argument("--panels-dir", type=Path, required=True)
    p_build.add_argument("--work", type=Path, required=True)
    p_score = sub.add_parser("score")
    p_score.add_argument("--work", type=Path, required=True)
    p_score.add_argument("--candidate", choices=("none",) + CANDIDATES, required=True)
    p_score.add_argument("--limit", type=int, default=0)
    p_select = sub.add_parser("select")
    p_select.add_argument("--work", type=Path, required=True)
    p_select.add_argument("--out", type=Path, default=ROOT / "results/eval/end_of_turn_dev.json")
    args = parser.parse_args()
    return {"build": build, "score": score, "select": select}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
