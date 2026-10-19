#!/usr/bin/env python3
"""Build Moshi v7 sidecars: romanized, word-level-aligned inner-monologue text.

v7 changes exactly two things relative to the frozen v6 training targets:

1. the assistant text is the deterministic ASCII romanization of the
   espeak-ng phonemes that the pinned Mana Piper voice actually spoke; and
2. every word carries its exact start/end time, taken from Piper's own
   phoneme-to-sample alignment of a re-synthesis that must reproduce the
   stored assistant channel (lengths equal, correlation >= 0.999).

Audio is never rewritten: each stereo WAV is hard-linked into the v7 tree (or
symlinked across filesystems) and only the sidecar JSON differs. Run with an
interpreter that has ``piper-tts`` and ``onnx`` (alignment patch), e.g.::

    PYTHONPATH=src <python with piper-tts and onnx> \\
        scripts/prepare_moshi_v7.py --source data/processed/moshi_finetune_v6_overfit/train.jsonl \\
        --out-dir data/processed/moshi_v7a_overfit --split train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.data.romanize import (  # noqa: E402
    ROMANIZATION_VERSION,
    _map_word,
    alignment_letters,
)

PIPER_MODEL = ROOT / "models/piper/fa_IR-mana-medium.onnx"
PIPER_SHA256 = "e390c0e74ba71fd97c49ba662ee0c6e1724b462ba2d4561698af4f564840f126"
EXPORT_SAMPLE_RATE = 16_000
MIN_REPRODUCTION_CORRELATION = 0.999
PUNCTUATION = {",": ",", ".": ".", "?": "?", "!": "!", ";": ",", ":": ""}
NON_PHONEME_MARKERS = {"^", "$", "_"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def assistant_channel(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 2 or handle.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit stereo export: {path}")
        if handle.getframerate() != EXPORT_SAMPLE_RATE:
            raise ValueError(f"unexpected export sample rate: {path}")
        frames = handle.readframes(handle.getnframes())
    stereo = np.frombuffer(frames, dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768.0
    return stereo[:, 0]


def word_timings(chunks: list[Any]) -> tuple[list[tuple[str, float, float]], np.ndarray, int]:
    """Return (romanized word, start, end) in seconds from Piper alignments."""

    words: list[tuple[str, float, float]] = []
    audio_parts: list[np.ndarray] = []
    offset_samples = 0
    sample_rate = int(chunks[0].sample_rate)
    for chunk in chunks:
        if chunk.phoneme_alignments is None:
            raise RuntimeError("Piper returned no phoneme alignments")
        cursor = offset_samples
        phones: list[str] = []
        start: int | None = None
        end = cursor

        def flush(end: int) -> None:
            nonlocal phones, start
            joined = "".join(phones)
            trail = ""
            while joined and joined[-1] in PUNCTUATION:
                trail = PUNCTUATION[joined[-1]] + trail
                joined = joined[:-1]
            body = _map_word(joined) if joined else ""
            if body and start is not None:
                words.append((body + trail, start / sample_rate, end / sample_rate))
            elif trail and words:
                last = words[-1]
                words[-1] = (last[0] + trail, last[1], last[2])
            phones, start = [], None

        for alignment in chunk.phoneme_alignments:
            samples = int(alignment.num_samples)
            symbol = str(alignment.phoneme)
            if symbol == " ":
                flush(end)
            elif symbol not in NON_PHONEME_MARKERS:
                if symbol not in PUNCTUATION:
                    if start is None:
                        start = cursor
                    end = cursor + samples
                phones.append(symbol)
            cursor += samples
        flush(end)
        audio = np.asarray(chunk.audio_float_array, dtype=np.float32)
        audio_parts.append(audio)
        offset_samples += audio.size
    return words, np.concatenate(audio_parts), sample_rate


def link_audio(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source)
        return "symlink"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="source Moshi manifest")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    from piper import PiperVoice
    from piper.config import SynthesisConfig

    from thesis_s2s.audio import resample

    if sha256_file(PIPER_MODEL) != PIPER_SHA256:
        raise SystemExit("Piper voice hash does not match the pinned training target")
    if "test" in args.split.lower() or "test" in args.source.name.lower():
        raise SystemExit("v7 preparation must not read the sealed test split")
    voice = PiperVoice.load(str(PIPER_MODEL), include_alignments=True)
    config = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)

    rows = read_jsonl(args.source)
    out_rows: list[dict[str, Any]] = []
    correlations: list[float] = []
    tokens_note: list[int] = []
    link_kinds: dict[str, int] = {}
    failures: list[dict[str, str]] = []
    audio_dir = args.out_dir / "audio" / args.split
    for row in rows:
        wav = Path(str(row["path"]))
        metadata = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        alignments = metadata["alignments"]
        if len(alignments) != 1:
            failures.append({"pair": wav.stem, "reason": "expected one whole-turn alignment"})
            continue
        text, (span_start, span_end), speaker = alignments[0]
        chunks = list(voice.synthesize(text, syn_config=config, include_alignments=True))
        words, piper_audio, piper_rate = word_timings(chunks)
        rendered = resample(piper_audio, piper_rate, EXPORT_SAMPLE_RATE)
        stored = assistant_channel(wav)
        start_sample = int(round(float(span_start) * EXPORT_SAMPLE_RATE))
        segment = stored[start_sample : start_sample + rendered.size]
        length_ok = segment.size == rendered.size
        correlation = (
            float(np.corrcoef(segment, rendered)[0, 1])
            if length_ok and rendered.size > 1 and np.std(rendered) > 0
            else 0.0
        )
        if not length_ok or correlation < MIN_REPRODUCTION_CORRELATION or not words:
            failures.append(
                {"pair": wav.stem, "reason": f"reproduction failed (corr={correlation:.4f})"}
            )
            continue
        correlations.append(correlation)
        offset = float(span_start)
        v7_alignments = [
            [word, [round(offset + start, 4), round(offset + end, 4)], speaker]
            for word, start, end in words
        ]
        for word, _start, _end in words:
            alignment_letters(word)
        tokens_note.append(len(words))
        target = audio_dir / wav.name
        kind = link_audio(wav, target)
        link_kinds[kind] = link_kinds.get(kind, 0) + 1
        sidecar = dict(metadata)
        sidecar["alignments"] = v7_alignments
        sidecar["v7_text_representation"] = ROMANIZATION_VERSION
        sidecar["v7_source_persian_text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        sidecar["v7_alignment_source"] = "piper_phoneme_alignment_resynthesis"
        target.with_suffix(".json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        new_row = dict(row)
        new_row["path"] = str(target.resolve().parent / target.name) if kind == "hardlink" else str(target)
        new_row["metadata_sha256"] = sha256_file(target.with_suffix(".json"))
        out_rows.append(new_row)

    manifest = args.out_dir / f"{args.split}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8"
    )
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Moshi v7 romanized word-aligned inner-monologue targets",
        "final_test_accessed": False,
        "audio_rewritten": False,
        "inputs": {
            "source_manifest": str(args.source),
            "source_manifest_sha256": sha256_file(args.source),
            "piper_model_sha256": PIPER_SHA256,
            "romanization_version": ROMANIZATION_VERSION,
            "romanize_source_sha256": sha256_file(ROOT / "src/thesis_s2s/data/romanize.py"),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "output": {
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "rows": len(out_rows),
            "audio_links": link_kinds,
        },
        "reproduction": {
            "minimum_correlation_required": MIN_REPRODUCTION_CORRELATION,
            "minimum_correlation_observed": min(correlations) if correlations else None,
            "median_correlation": statistics.median(correlations) if correlations else None,
        },
        "words_per_row_median": statistics.median(tokens_note) if tokens_note else None,
        "failures": failures,
        "passes": not failures and len(out_rows) == len(rows),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("output", "reproduction", "passes")}, indent=2))
    return 0 if receipt["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
