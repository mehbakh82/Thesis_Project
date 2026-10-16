#!/usr/bin/env python3
"""Mine isolated real backchannels and single-speaker speech segments.

From locally available diarized windows, an *isolated backchannel* is a short
exclusive turn (0.15-0.9 s) of one speaker sandwiched between turns of another
speaker, whose ASR transcript consists only of listener tokens (lexicon below).
*Speech segments* are exclusive single-speaker turns of 1.2-4.0 s. Both are
written as 16 kHz mono clips with their session so the simulation can split by
session. The ASR service follows the runtime contract (``ASR_API_URL``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.eval.asr_normalize import normalize_for_scoring  # noqa: E402
from thesis_s2s.runtime.cascade import _asr_transcript, _wav_bytes, asr_http  # noqa: E402

RATE = 16_000
LEXICON = {
    "آره", "اره", "آرہ", "بله", "بلی", "اوهوم", "اوهم", "هوم", "آها", "اها", "آهان", "اهان",
    "باشه", "درسته", "دقیقا", "دقیقاً", "عالیه", "خب", "خوبه", "آفرین", "جدی", "واقعا", "نه",
    "حتما", "حتماً", "صحیح", "اوکی", "اهوم", "ممنون", "مرسی", "هه", "ههه", "آ", "ا", "اوه",
}


def read_window(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        channels = handle.getnchannels()
    audio = data.reshape(-1, channels).mean(axis=1) if channels > 1 else data
    return audio.astype(np.float32) / 32768.0


def write_clip(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-speech-per-window", type=int, default=6)
    args = parser.parse_args()

    records = []
    stats = {"windows": 0, "bc_candidates": 0, "bc_kept": 0, "speech_kept": 0}
    for manifest in args.windows:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            window = json.loads(line)
            path = ROOT / window["audio_filepath"]
            if not path.is_file():
                continue
            stats["windows"] += 1
            audio = read_window(path)
            turns = sorted(
                (float(t["start"]), float(t["end"]), str(t["speaker"]))
                for t in window.get("exclusive_speaker_turns") or []
            )
            speech_taken = 0
            for index, (start, end, speaker) in enumerate(turns):
                duration = end - start
                key = hashlib.sha256(f"{window['window_id']}|{start:.3f}".encode()).hexdigest()[:16]
                base = {
                    "session_id": window["session_id"],
                    "channel": window.get("channel"),
                    "window_id": window["window_id"],
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
                if 0.15 <= duration <= 0.9 and 0 < index < len(turns) - 1:
                    before, after = turns[index - 1], turns[index + 1]
                    flanked = before[2] != speaker and after[2] != speaker
                    gaps_ok = start - before[1] < 1.0 and after[0] - end < 1.5
                    if not (flanked and gaps_ok):
                        continue
                    stats["bc_candidates"] += 1
                    clip = audio[max(0, int((start - 0.05) * RATE)) : int((end + 0.05) * RATE)]
                    result = asr_http(_wav_bytes(clip))
                    text = normalize_for_scoring(_asr_transcript(result))
                    tokens = text.split()
                    if tokens and len(tokens) <= 3 and all(t in LEXICON for t in tokens):
                        out = args.out_dir / "backchannel" / f"bc-{key}.wav"
                        write_clip(out, clip)
                        records.append({**base, "kind": "backchannel", "audio_filepath": str(out), "asr_text": text, "duration": round(clip.size / RATE, 3)})
                        stats["bc_kept"] += 1
                elif 1.2 <= duration <= 4.0 and speech_taken < args.max_speech_per_window:
                    clip = audio[int(start * RATE) : int(end * RATE)]
                    out = args.out_dir / "speech" / f"sp-{key}.wav"
                    write_clip(out, clip)
                    records.append({**base, "kind": "speech", "audio_filepath": str(out), "duration": round(clip.size / RATE, 3)})
                    speech_taken += 1
                    stats["speech_kept"] += 1
    (args.out_dir / "pool.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    stats["sessions"] = len({r["session_id"] for r in records})
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
