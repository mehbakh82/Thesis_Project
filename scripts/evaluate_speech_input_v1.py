#!/usr/bin/env python3
"""Speech-input LLM versus cascade on spoken Persian requests (W3).

Panels: the 60 natural user turns (real speech from unseen sessions) and the
60 elderly-assistant requests rendered as simulated elderly speech (slow male
voices with room noise). Arms:

* ``cascade_nemo_qwen3_promptv2`` – production: NeMo ASR -> Qwen3-4B prompt-v2;
* ``cascade_nemo_gemma_elderly``  – NeMo ASR -> Gemma 4 E4B text, elderly prompt;
* ``speechin_gemma_elderly``      – Gemma 4 E4B hears the audio directly (no ASR),
  elderly prompt (optionally a LoRA adapter served under another model name).

Replies are judged blind against the reference request text by the local
Qwen3.8 judge (and later Gemini) with the responder-v2 rubric. Latency per turn
is recorded (ASR, time to first token, total).
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_responder_candidates_v2 import (  # noqa: E402
    ELDERLY_PROMPT_V1,
    SCORE_KEYS,
    judge_gemini,
    judge_qwen38,
)

from thesis_s2s.audio import read_wav  # noqa: E402
from thesis_s2s.runtime.cascade import (  # noqa: E402
    QWEN4B_PROMPT_V2,
    _asr_transcript,
    _wav_bytes,
    asr_http,
)
from thesis_s2s.runtime.streaming import PERSIAN_ONLY_REGEX  # noqa: E402

RESPOND_PROMPT = "به گفتهٔ صوتی کاربر پاسخ بده."
INCUMBENT = "cascade_nemo_qwen3_promptv2"
PANELS = ("natural_turns", "elderly_requests", "all")


def stream_chat(endpoint: str, model: str, messages: list[dict[str, Any]]) -> tuple[str, float, float]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 96,
        "temperature": 0,
        "stream": True,
        "structured_outputs": {"regex": PERSIAN_ONLY_REGEX},
    }
    request = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first: float | None = None
    text = ""
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            choices = json.loads(line[5:]).get("choices") or []
            delta = (choices[0].get("delta") or {}).get("content") if choices else None
            if delta:
                if first is None:
                    first = time.perf_counter()
                text += delta
    end = time.perf_counter()
    return text.strip(), 1000 * ((first or end) - started), 1000 * (end - started)


def run_arm(arm: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        record: dict[str, Any] = {"row_id": row["row_id"], "panel": row["panel"], "arm": arm}
        if arm.startswith("cascade"):
            audio, _ = read_wav(row["audio_filepath"])
            started = time.perf_counter()
            transcript = _asr_transcript(asr_http(_wav_bytes(audio)))
            record["asr_ms"] = 1000 * (time.perf_counter() - started)
            record["asr_transcript"] = transcript
            if arm == "cascade_nemo_qwen3_promptv2":
                endpoint, model, prompt = args.qwen_endpoint, "qwen3-4b-instruct-2507", QWEN4B_PROMPT_V2
            else:
                endpoint, model, prompt = args.gemma_endpoint, args.gemma_model, ELDERLY_PROMPT_V1
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": transcript or "..."}]
        else:
            data = base64.b64encode(Path(row["audio_filepath"]).read_bytes()).decode("ascii")
            endpoint = args.gemma_endpoint
            model = args.speech_model or args.gemma_model
            record["asr_ms"] = 0.0
            messages = [
                {"role": "system", "content": ELDERLY_PROMPT_V1},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
                        {"type": "text", "text": RESPOND_PROMPT},
                    ],
                },
            ]
        reply, ttft, total = stream_chat(endpoint, model, messages)
        record.update(reply=reply, ttft_ms=ttft, llm_total_ms=total, first_token_after_audio_ms=record["asr_ms"] + ttft)
        out.append(record)
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def judge_with_retry(
    reference: str, reply: str, endpoint: str, wait_s: float = 1500.0, judge: str = "qwen38", cache: Path | None = None
) -> dict[str, Any]:
    """The shared judge restarts on a day/night schedule; wait it out instead of failing."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            if judge == "gemini":
                assert cache is not None  # noqa: S101 - script-level invariant
                return judge_gemini(reference, reply, cache)
            return judge_qwen38(reference, reply, endpoint)
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if time.monotonic() > deadline:
                raise
            time.sleep(15)


def paired_bootstrap(a: list[float], b: list[float], draws: int = 10_000) -> list[float]:
    rng = random.Random(20261016)
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    means = sorted(statistics.fmean(rng.choice(diffs) for _ in diffs) for _ in range(draws))
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def summarize(replies: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for panel_name in PANELS:
        subset = [r for r in replies if panel_name == "all" or r["panel"] == panel_name]
        summary[panel_name] = {k: statistics.fmean(r[k] for r in subset) for k in SCORE_KEYS}
        firsts = sorted(r["first_token_after_audio_ms"] for r in subset)
        summary[panel_name]["first_token_after_audio_ms_median"] = statistics.median(firsts)
        summary[panel_name]["first_token_after_audio_ms_p90"] = firsts[int(0.9 * (len(firsts) - 1))]
        summary[panel_name]["empty_replies"] = sum(1 for r in subset if not r["reply"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--qwen-endpoint", default="http://127.0.0.1:8020")
    parser.add_argument("--gemma-endpoint", default="http://127.0.0.1:8021")
    parser.add_argument("--gemma-model", default="gemma-4-e4b-it")
    parser.add_argument("--speech-model", help="served name for a speech-input LoRA adapter")
    parser.add_argument("--judge-endpoint", default="http://127.0.0.1:8003")
    parser.add_argument("--judge", choices=("qwen38", "gemini"), default="qwen38")
    parser.add_argument("--gemini-cache", type=Path, help="cache directory for the Gemini judge")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    panel = {r["row_id"]: r for r in load_jsonl(args.work / "panel.jsonl")}
    rows = []
    for name, file in (("natural_turns", "natural_audio.jsonl"), ("elderly_requests", "spoken_elderly.jsonl")):
        for item in load_jsonl(args.work / file):
            rows.append({**item, "panel": name, "reference": panel[item["row_id"]]["text"]})
    refs = {r["row_id"]: r["reference"] for r in rows}
    results: dict[str, Any] = {"schema_version": 1, "panel_rows": len(rows), "arms": {}}
    judged: dict[str, list[dict[str, Any]]] = {}
    for arm in args.arms:
        suffix = "" if args.judge == "qwen38" else f".{args.judge}"
        judged_path = args.work / f"speech_input_{arm}{suffix}.jsonl"
        replies_path = args.work / f"speech_input_{arm}.replies.jsonl"
        first_judge_path = args.work / f"speech_input_{arm}.jsonl"
        if judged_path.exists():
            judged[arm] = load_jsonl(judged_path)
        else:
            if replies_path.exists():
                replies = load_jsonl(replies_path)
            elif args.judge != "qwen38" and first_judge_path.exists():
                # same replies as judged by the first judge, scores removed
                replies = [{k: v for k, v in r.items() if k not in SCORE_KEYS} for r in load_jsonl(first_judge_path)]
            else:
                run_arm(arm, rows[:1], args)  # warm-up of this path
                replies = run_arm(arm, rows, args)
                write_jsonl(replies_path, replies)
            order = list(range(len(replies)))
            random.Random(f"speech-input-v1:{arm}").shuffle(order)
            for index in order:
                record = replies[index]
                raw = judge_with_retry(
                    refs[record["row_id"]], record["reply"], args.judge_endpoint, judge=args.judge, cache=args.gemini_cache
                )
                record.update({k: int(raw[k]) for k in SCORE_KEYS})
            write_jsonl(judged_path, replies)
            judged[arm] = replies
        results["arms"][arm] = summarize(judged[arm])
        print(arm, json.dumps(results["arms"][arm]["all"], ensure_ascii=False), flush=True)
    if INCUMBENT in judged:
        base = {r["row_id"]: r for r in judged[INCUMBENT]}
        for arm, replies in judged.items():
            if arm == INCUMBENT:
                continue
            delta: dict[str, Any] = {}
            for panel_name in PANELS:
                subset = [r for r in replies if panel_name == "all" or r["panel"] == panel_name]
                delta[panel_name] = {
                    f"{k}_minus_incumbent_ci95": paired_bootstrap(
                        [float(r[k]) for r in subset], [float(base[r["row_id"]][k]) for r in subset]
                    )
                    for k in SCORE_KEYS
                }
            results["arms"][arm]["paired_vs_incumbent"] = delta
    results["incumbent"] = INCUMBENT
    results["judge"] = (
        "qwen3.8-27b (local, temperature 0)" if args.judge == "qwen38" else "gemini-3.8-flash (independent automatic judge)"
    )
    results["text_retained"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
