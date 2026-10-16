#!/usr/bin/env python3
"""Prepare Persian speech-input training data for the Gemma 4 LoRA (W3).

From the corrected Moshi v2 training split only (validation/test sessions are
excluded), user turns of 1-30 s are cut from the stereo exports' user channel.
Each yields two examples:

* ``asr``: audio -> the caption transcript (fa-verbatim-2 text);
* ``respond``: audio -> the production responder's reply (Qwen3-4B, elderly
  prompt, greedy, Persian-only) to that caption transcript, i.e. DiVA-style
  distillation of the text assistant into the speech-input model.

Teacher replies come from an OpenAI-compatible endpoint (``--endpoint``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_responder_candidates_v2 import ELDERLY_PROMPT_V1  # noqa: E402

from thesis_s2s.runtime.streaming import OpenAIStreamingResponder  # noqa: E402

RATE = 16_000


def user_segment(row: dict, wav: Path) -> np.ndarray:
    with wave.open(str(wav), "rb") as handle:
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    stereo = frames.reshape(-1, 2).astype(np.float32) / 32768.0
    start = max(0.0, float(row["source_user_interval"][0]) - float(row["source_span_start"]))
    a = int(round(start * RATE))
    b = a + int(round(float(row["user_duration"]) * RATE))
    return stereo[a:b, 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "data/processed/moshi_finetune_v2/train.jsonl")
    parser.add_argument("--conversations", type=Path, default=ROOT / "data/processed/manifests/conversations.jsonl")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8020")
    parser.add_argument("--served-name", default="qwen3-4b-instruct-2507")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if "test" in args.split_manifest.name or "val" in args.split_manifest.name:
        raise SystemExit("speech LoRA data must come from the training split only")

    conversations = {}
    for line in args.conversations.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            conversations[row["utt_id"]] = row
    exports = [json.loads(line) for line in args.split_manifest.read_text().splitlines() if line]
    teacher = OpenAIStreamingResponder(
        endpoint=args.endpoint,
        model=args.served_name,
        system_prompt=ELDERLY_PROMPT_V1,
        max_new_tokens=96,
        history_turns=0,
    )
    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for export in exports:
        wav = Path(export["path"])
        sidecar = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        row = conversations.get(sidecar["source_pair_id"])
        if row is None:
            continue
        duration = float(row["user_duration"])
        text = str(row.get("text") or "").strip()
        if not (1.0 <= duration <= 30.0) or len(text.split()) < 2:
            continue
        segment = user_segment(row, wav)
        if segment.size < RATE:
            continue
        out = audio_dir / f"{row['utt_id']}.wav"
        with wave.open(str(out), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(RATE)
            handle.writeframes((np.clip(segment, -1, 1) * 32767).astype("<i2").tobytes())
        items.append({"utt_id": row["utt_id"], "session_id": row["session_id"], "audio_filepath": str(out), "transcript": text, "duration": round(segment.size / RATE, 3)})

    def reply(item: dict) -> dict:
        try:
            item["teacher_reply"] = "".join(teacher.stream(item["transcript"])).strip()
        except RuntimeError as exc:
            item["teacher_reply"] = ""
            item["teacher_error"] = str(exc)
        return item

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        items = list(pool.map(reply, items))
    examples = []
    for item in items:
        examples.append({**item, "task": "asr", "target": item["transcript"]})
        if item.get("teacher_reply"):
            examples.append({**item, "task": "respond", "target": item["teacher_reply"]})
    manifest = args.out_dir / "train.jsonl"
    manifest.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8")
    summary = {
        "turns": len(items),
        "examples": len(examples),
        "hours": round(sum(i["duration"] for i in items) / 3600, 3),
        "sessions": len({i["session_id"] for i in items}),
        "teacher_failures": sum(1 for i in items if not i.get("teacher_reply")),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
