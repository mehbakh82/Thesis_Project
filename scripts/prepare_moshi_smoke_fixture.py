#!/usr/bin/env python3
"""Create a project-generated Moshi wiring fixture with no human/source media."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
DURATION_SECONDS = 5.0


def prepare(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = round(DURATION_SECONDS * SAMPLE_RATE)
    stereo = np.zeros((samples, 2), dtype=np.float32)
    user_start, user_end = round(0.4 * SAMPLE_RATE), round(2.2 * SAMPLE_RATE)
    assistant_start, assistant_end = round(2.5 * SAMPLE_RATE), round(4.6 * SAMPLE_RATE)
    user_t = np.arange(user_end - user_start, dtype=np.float32) / SAMPLE_RATE
    assistant_t = np.arange(assistant_end - assistant_start, dtype=np.float32) / SAMPLE_RATE
    stereo[user_start:user_end, 1] = 0.12 * np.sin(2 * np.pi * 181.0 * user_t)
    stereo[assistant_start:assistant_end, 0] = 0.12 * np.sin(2 * np.pi * 233.0 * assistant_t)
    wav_path = out_dir / "synthetic_wiring_only.wav"
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    sidecar = {
        "alignments": [
            [
                "این فقط آزمایش اتصال آموزشگر است",
                [assistant_start / SAMPLE_RATE, assistant_end / SAMPLE_RATE],
                "SPEAKER_MAIN",
            ]
        ],
        "source": "project_generated_harmonic_fixture",
        "scientific_evidence": False,
    }
    wav_path.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = out_dir / "train.jsonl"
    manifest.write_text(
        json.dumps({"path": str(wav_path.resolve()), "duration": DURATION_SECONDS}) + "\n",
        encoding="utf-8",
    )
    report = {
        "out_dir": str(out_dir),
        "manifest": str(manifest),
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "source": "project_generated_harmonic_fixture",
        "scientific_evidence": False,
        "purpose": "one-step trainer/model/loss wiring validation only",
    }
    (out_dir / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/moshi_smoke"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.out_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
