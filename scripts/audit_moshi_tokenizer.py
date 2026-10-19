#!/usr/bin/env python3
"""Measure how Moshi's text tokenizer handles the Persian response targets.

The receipt compares the original Persian-script inner-monologue text with the
deterministic romanization of the phonemes Piper spoke, and reports how many
text tokens each assistant turn needs relative to its 12.5 Hz frame budget.
It reads only the exported sidecar JSON files; no audio or model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.data.romanize import ROMANIZATION_VERSION, romanize  # noqa: E402

TOKENIZER = ROOT / "hf_cache/pinned/moshika-pytorch-bf16-a49141e/tokenizer_spm_32k_3.model"
SIDECARS = ROOT / "data/processed/moshi_finetune/audio"
PIPER_MODEL = ROOT / "models/piper/fa_IR-mana-medium.onnx"
FRAME_RATE_HZ = 12.5
PERSIAN_LETTERS = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیآ"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/moshi_tokenizer_audit.json")
    args = parser.parse_args()

    import sentencepiece as spm
    from piper import PiperVoice

    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    voice = PiperVoice.load(str(PIPER_MODEL))
    byte_ids = {i for i in range(tokenizer.get_piece_size()) if tokenizer.is_byte(i)}
    single_letter = [c for c in PERSIAN_LETTERS if tokenizer.piece_to_id(c) != tokenizer.unk_id()]

    totals = {"words": 0, "fa_tokens": 0, "fa_byte_tokens": 0, "roman_words": 0, "roman_tokens": 0}
    fa_per_frame: list[float] = []
    roman_per_frame: list[float] = []
    over_budget_fa = over_budget_roman = turns = 0
    for sidecar in sorted(SIDECARS.glob("*/*.json")):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        for text, (start, end), _speaker in payload["alignments"]:
            frames = max(1.0, (float(end) - float(start)) * FRAME_RATE_HZ)
            fa_ids = tokenizer.encode(text)
            roman_words = romanize(text, voice)
            roman_ids = tokenizer.encode(" ".join(roman_words))
            totals["words"] += len(text.split())
            totals["fa_tokens"] += len(fa_ids)
            totals["fa_byte_tokens"] += sum(1 for i in fa_ids if i in byte_ids)
            totals["roman_words"] += len(roman_words)
            totals["roman_tokens"] += len(roman_ids)
            fa_per_frame.append(len(fa_ids) / frames)
            roman_per_frame.append(len(roman_ids) / frames)
            over_budget_fa += len(fa_ids) > frames
            over_budget_roman += len(roman_ids) > frames
            turns += 1

    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "evidence_class": "tokenizer_capacity_diagnostic",
        "inputs": {
            "tokenizer": str(TOKENIZER.relative_to(ROOT)),
            "tokenizer_sha256": _sha256(TOKENIZER),
            "piper_model_sha256": _sha256(PIPER_MODEL),
            "sidecar_count": len(list(SIDECARS.glob("*/*.json"))),
            "romanization_version": ROMANIZATION_VERSION,
            "romanize_source_sha256": _sha256(ROOT / "src/thesis_s2s/data/romanize.py"),
            "script_sha256": _sha256(Path(__file__)),
        },
        "assistant_turns": turns,
        "frame_rate_hz": FRAME_RATE_HZ,
        "persian_script": {
            "tokens_per_word": totals["fa_tokens"] / totals["words"],
            "byte_fallback_token_share": totals["fa_byte_tokens"] / totals["fa_tokens"],
            "persian_letters_with_own_piece": len(single_letter),
            "persian_letters_total": len(PERSIAN_LETTERS),
            "tokens_per_frame": _summary(fa_per_frame),
            "turns_over_frame_budget_fraction": over_budget_fa / turns,
            "aggregate_tokens_per_frame": sum(fa_per_frame) / turns,
        },
        "romanized_phonemes": {
            "tokens_per_word": totals["roman_tokens"] / totals["roman_words"],
            "tokens_per_frame": _summary(roman_per_frame),
            "turns_over_frame_budget_fraction": over_budget_roman / turns,
            "aggregate_tokens_per_frame": sum(roman_per_frame) / turns,
        },
        "interpretation": (
            "Moshi emits at most one text token per frame. A tokens-per-frame ratio "
            "above 1 means the inner-monologue text cannot keep pace with the "
            "assistant speech it should precede. This is a capacity diagnostic, "
            "not evidence that romanized training will succeed."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("persian_script", "romanized_phonemes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
