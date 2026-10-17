#!/usr/bin/env python3
"""Moshi v7a predeclared secondary measure: Persian ASR intelligibility.

The v7a evaluator retained the generated assistant speech of the nine panel
rows for the last evaluated checkpoint (step 200; earlier candidates were
overwritten, see docs/MOSHI_V7_RESULT.md). Each clip is resampled to 16 kHz,
transcribed by the selected project ASR (NeMo v4 CTC, ``ASR_API_URL``), and
scored, as the protocol states, by character error rate against the
equal-length prefix of the row's Persian target (the Persian-script assistant
text of the v6 overfit sidecar), with the same punctuation-free normalization
as the ASR comparisons. Reported, not gated. No transcript text is retained.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.audio import resample  # noqa: E402
from thesis_s2s.eval.asr_normalize import NORMALIZER_VERSION, normalize_for_scoring  # noqa: E402
from thesis_s2s.eval.wer import edit_distance  # noqa: E402
from thesis_s2s.runtime.cascade import _asr_transcript, _wav_bytes, asr_http  # noqa: E402


def read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    audio = data.reshape(-1, channels)[:, 0].astype(np.float32) / 32768.0
    return resample(audio, rate, 16_000) if rate != 16_000 else audio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "checkpoints/moshi_v7a_overfit/runtime_audio")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/moshi_finetune_v6_overfit/train.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/moshi_v7a_step200_intelligibility.json")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite write-once receipt {args.out}")
    targets = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if line:
            wav = Path(json.loads(line)["path"])
            sidecar = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
            alignments = sidecar["alignments"]
            if isinstance(alignments, str):
                alignments = ast.literal_eval(alignments)
            targets[wav.stem] = " ".join(str(a[0]) for a in alignments)
    rows = []
    model_sha = None
    for clip in sorted(args.audio_dir.glob("*.wav")):
        result = asr_http(_wav_bytes(read(clip)))
        model_sha = result.get("model_sha256")
        hyp = normalize_for_scoring(_asr_transcript(result)).replace(" ", "")
        ref = normalize_for_scoring(targets[clip.stem]).replace(" ", "")
        prefix = ref[: len(hyp)]
        cer = edit_distance(list(hyp), list(prefix)) / max(len(hyp), len(prefix), 1) if hyp else 1.0
        rows.append({"pair_id": clip.stem, "audio_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(), "hypothesis_chars": len(hyp), "prefix_cer": round(cer, 4)})
    cers = [r["prefix_cer"] for r in rows]
    receipt = {
        "schema_version": 1,
        "experiment": "moshi_v7a_secondary_intelligibility",
        "checkpoint": "step 200 (only retained candidate)",
        "asr_model_sha256": model_sha,
        "normalizer": NORMALIZER_VERSION,
        "metric": "character error rate of the ASR transcript against the equal-length prefix of the Persian target",
        "rows": rows,
        "mean_prefix_cer": sum(cers) / len(cers),
        "rows_cer_le_0_75": sum(c <= 0.75 for c in cers),
        "empty_transcripts": sum(r["hypothesis_chars"] == 0 for r in rows),
        "gated": False,
        "text_retained": False,
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("mean_prefix_cer", "rows_cer_le_0_75", "empty_transcripts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
