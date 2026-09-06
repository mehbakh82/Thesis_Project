#!/usr/bin/env python3
"""Measure privacy-safe NeMo round-trip CER/WER for the frozen cascade panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_cascade_real_service_v4 import (  # noqa: E402
    atomic_write,
    percentile,
    qwen_revision,
    sha256_file,
    sha256_text,
)
from scripts.evaluate_cascade_validation import (  # noqa: E402
    PANEL_INDICES,
    read_user_channel,
)
from thesis_s2s.data.verbatim import verbatim_normalize  # noqa: E402
from thesis_s2s.eval.wer import (  # noqa: E402
    character_tokens,
    edit_distance,
    tokenize,
)
from thesis_s2s.metrics import gpu_inventory  # noqa: E402
from thesis_s2s.runtime.cascade import (  # noqa: E402
    CascadeTalker,
    TextResponder,
    _wav_bytes,
    asr_http,
)

EXPECTED_ROWS = 131
EXPECTED_PANEL_ROWS = 9
USER_CHANNEL_INDEX = 1
MANIFEST_SHA256 = "a2762495830c82ce12d3dc4cc8469e2110ae21a77ea91f9b70498a6812e187a7"
PARENT_PANEL_SHA256 = "ae515381db2b72e9ed0b8dcb491c0765990732ed8bb8c46383499b9d60f89ced"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PIPER_MODEL_SHA256 = "e390c0e74ba71fd97c49ba662ee0c6e1724b462ba2d4561698af4f564840f126"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def _rate(distance: int, reference_units: int) -> float:
    if reference_units == 0:
        return 0.0 if distance == 0 else 1.0
    return distance / reference_units


def text_error_counts(reference: str, hypothesis: str) -> dict[str, int | float]:
    reference_words = tokenize(reference)
    hypothesis_words = tokenize(hypothesis)
    reference_characters = character_tokens(reference)
    hypothesis_characters = character_tokens(hypothesis)
    word_edits = edit_distance(reference_words, hypothesis_words)
    character_edits = edit_distance(reference_characters, hypothesis_characters)
    return {
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "word_edits": word_edits,
        "wer": round(_rate(word_edits, len(reference_words)), 6),
        "reference_characters": len(reference_characters),
        "hypothesis_characters": len(hypothesis_characters),
        "character_edits": character_edits,
        "cer": round(_rate(character_edits, len(reference_characters)), 6),
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    wers = [float(sample["wer"]) for sample in samples]
    cers = [float(sample["cer"]) for sample in samples]
    reference_words = sum(int(sample["reference_words"]) for sample in samples)
    reference_characters = sum(int(sample["reference_characters"]) for sample in samples)
    word_edits = sum(int(sample["word_edits"]) for sample in samples)
    character_edits = sum(int(sample["character_edits"]) for sample in samples)
    return {
        "rows": len(samples),
        "asr_failures": sum(sample.get("roundtrip_asr_error") is not None for sample in samples),
        "exact_word_match_rows": sum(int(sample["word_edits"]) == 0 for sample in samples),
        "exact_character_match_rows": sum(
            int(sample["character_edits"]) == 0 for sample in samples
        ),
        "word_error_rate": {
            "micro": round(_rate(word_edits, reference_words), 6),
            "macro_mean": round(float(np.mean(wers)), 6),
            "p50": percentile(wers, 50),
            "p95": percentile(wers, 95),
            "max": round(max(wers), 6),
            "reference_words": reference_words,
            "edits": word_edits,
        },
        "character_error_rate": {
            "micro": round(_rate(character_edits, reference_characters), 6),
            "macro_mean": round(float(np.mean(cers)), 6),
            "p50": percentile(cers, 50),
            "p95": percentile(cers, 95),
            "max": round(max(cers), 6),
            "reference_characters": reference_characters,
            "edits": character_edits,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/moshi_finetune/val.jsonl")
    )
    parser.add_argument(
        "--parent-panel",
        type=Path,
        default=Path("results/eval/cascade_real_service_validation_panel.json"),
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("docs/CASCADE_INTELLIGIBILITY_PROTOCOL.md")
    )
    parser.add_argument(
        "--piper-model", type=Path, default=Path("models/piper/fa_IR-mana-medium.onnx")
    )
    parser.add_argument("--asr-url", default="http://127.0.0.1:8090")
    parser.add_argument("--asr-timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--out", type=Path, default=Path("results/eval/cascade_intelligibility_proxy.json")
    )
    args = parser.parse_args()

    manifest_path = (ROOT / args.manifest).resolve()
    parent_path = (ROOT / args.parent_panel).resolve()
    protocol_path = (ROOT / args.protocol).resolve()
    piper_path = (ROOT / args.piper_model).resolve()
    piper_config_path = piper_path.with_suffix(piper_path.suffix + ".json")
    output_path = (ROOT / args.out).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    if "test" in manifest_path.name.lower() or "test" in manifest_path.parent.name.lower():
        raise ValueError("sealed test manifests are forbidden")
    for required in (manifest_path, parent_path, protocol_path, piper_path, piper_config_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("validation manifest hash drift")
    if sha256_file(parent_path) != PARENT_PANEL_SHA256:
        raise RuntimeError("parent validation-panel hash drift")
    if sha256_file(piper_path) != PIPER_MODEL_SHA256:
        raise RuntimeError("Piper model hash drift")

    rows = load_jsonl(manifest_path)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_samples = list(parent.get("samples") or [])
    if len(rows) != EXPECTED_ROWS or len(parent_samples) != EXPECTED_PANEL_ROWS:
        raise RuntimeError("frozen row count drift")
    if tuple(int(sample["manifest_row_index"]) for sample in parent_samples) != PANEL_INDICES:
        raise RuntimeError("parent panel-index drift")

    os.environ["ASR_API_URL"] = args.asr_url
    os.environ["ASR_TIMEOUT_SECONDS"] = str(args.asr_timeout_seconds)
    os.environ["TEXT_LLM_ENABLED"] = "1"
    os.environ["TEXT_LLM_MODEL"] = QWEN_MODEL
    os.environ["PIPER_MODEL"] = str(piper_path)

    responder = TextResponder(QWEN_MODEL)
    talker = CascadeTalker(responder)
    samples: list[dict[str, Any]] = []
    for panel_position, row_index in enumerate(PANEL_INDICES):
        row = rows[row_index]
        parent_sample = parent_samples[panel_position]
        audio_path = Path(str(row["path"])).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        user_audio, audio_format = read_user_channel(audio_path)
        reply_audio = np.asarray(talker.reply_audio(user_audio), dtype=np.float32)
        roundtrip = asr_http(_wav_bytes(reply_audio))
        raw_reference = talker.last_reply_text
        reference = verbatim_normalize(raw_reference)
        hypothesis = verbatim_normalize(
            str(roundtrip.get("persian") or roundtrip.get("text") or "")
        )
        counts = text_error_counts(reference, hypothesis)
        samples.append(
            {
                "panel_position": panel_position,
                "manifest_row_index": row_index,
                "source_audio_sha256": sha256_file(audio_path),
                "input_transcript_sha256": sha256_text(talker.last_transcript),
                "reference_reply_sha256": sha256_text(raw_reference),
                "normalized_reference_reply_sha256": sha256_text(reference),
                "roundtrip_transcript_sha256": sha256_text(hypothesis),
                **counts,
                "input_asr_error": talker.last_asr_error,
                "roundtrip_asr_error": roundtrip.get("error"),
                "selected_user_channel": audio_format["selected_user_channel"],
                "responder_backend": talker.responder_backend,
                "responder_fallback_used": talker.last_responder_fallback_used,
                "tts_backend": talker.backend,
                "matches_parent_input_transcript": (
                    sha256_text(talker.last_transcript) == parent_sample["transcript_sha256"]
                ),
                "matches_parent_reply": (
                    sha256_text(raw_reference) == parent_sample["reply_text_sha256"]
                ),
            }
        )

    measured_revision = qwen_revision(QWEN_MODEL)
    validity = {
        "parent_panel_passed": (
            parent.get("status") == "passed"
            and (parent.get("claim") or {}).get("positive_working_user_turn_result") is True
        ),
        "parent_final_test_not_accessed": (
            (parent.get("panel") or {}).get("final_test_accessed") is False
        ),
        "manifest_hash_matches": sha256_file(manifest_path) == MANIFEST_SHA256,
        "parent_panel_hash_matches": sha256_file(parent_path) == PARENT_PANEL_SHA256,
        "fixed_panel_exact": tuple(sample["manifest_row_index"] for sample in samples)
        == PANEL_INDICES,
        "all_source_hashes_match_parent": all(
            sample["source_audio_sha256"] == parent_samples[index]["source_audio_sha256"]
            for index, sample in enumerate(samples)
        ),
        "all_input_transcripts_reproduced": all(
            sample["matches_parent_input_transcript"] for sample in samples
        ),
        "all_reply_text_reproduced": all(sample["matches_parent_reply"] for sample in samples),
        "all_input_asr_calls_succeeded": all(
            sample["input_asr_error"] is None for sample in samples
        ),
        "all_roundtrip_asr_calls_succeeded": all(
            sample["roundtrip_asr_error"] is None for sample in samples
        ),
        "qwen_backend_exact": responder.backend == QWEN_MODEL,
        "qwen_revision_exact": measured_revision == QWEN_REVISION,
        "no_rule_fallback": all(sample["responder_fallback_used"] is False for sample in samples),
        "piper_backend_exact": all(sample["tts_backend"] == "piper" for sample in samples),
        "audited_user_channel_used": all(
            sample["selected_user_channel"] == USER_CHANNEL_INDEX for sample in samples
        ),
        "exactly_nine_rows_measured": len(samples) == EXPECTED_PANEL_ROWS,
        "final_test_not_accessed": True,
    }
    valid = all(validity.values())
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "valid" if valid else "invalid",
        "evidence_class": "automatic_asr_roundtrip_intelligibility_proxy",
        "claim": {
            "measurement_valid": valid,
            "automatic_intelligibility_proxy_measured": valid,
            "human_intelligibility_result": False,
            "pronunciation_or_naturalness_result": False,
            "semantic_relevance_result": False,
            "population_generalization_result": False,
        },
        "panel": {
            "manifest_rows": len(rows),
            "indices": list(PANEL_INDICES),
            "sample_count": len(samples),
            "split_unit": "source_session_id",
            "final_test_accessed": False,
        },
        "components": {
            "asr": {"backend": "nemo-soroush-http", "base_url": args.asr_url},
            "responder": {
                "backend": responder.backend,
                "revision": measured_revision,
                "initialization_error": responder.initialization_error,
            },
            "tts": {"backend": "piper", "model_sha256": sha256_file(piper_path)},
            "hardware": gpu_inventory(),
        },
        "normalization": {
            "text": "thesis_s2s.data.verbatim.verbatim_normalize",
            "words": "whitespace tokens with ZWNJ treated as a boundary",
            "characters": "Unicode code points excluding whitespace and ZWNJ",
        },
        "threshold": None,
        "aggregate": summarize(samples),
        "samples": samples,
        "validity_requirements": validity,
        "artifacts": {
            "manifest_sha256": sha256_file(manifest_path),
            "parent_panel_sha256": sha256_file(parent_path),
            "protocol_sha256": sha256_file(protocol_path),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "runtime_cascade_sha256": sha256_file(
                ROOT / "src/thesis_s2s/runtime/cascade.py"
            ),
            "error_metric_sha256": sha256_file(ROOT / "src/thesis_s2s/eval/wer.py"),
        },
        "limitations": [
            "This is automatic ASR round-trip evidence, not a human listening result.",
            "The same ASR family participates in the system and in this proxy.",
            "Nine outputs from one synthesized voice do not establish population intelligibility.",
            "CER/WER do not measure pronunciation quality, naturalness, semantics, or usefulness.",
            "No post-hoc quality threshold or confidence interval is asserted.",
        ],
        "privacy": {
            "plaintext_input_transcripts_stored": False,
            "plaintext_reply_text_stored": False,
            "plaintext_roundtrip_transcripts_stored": False,
            "audio_stored": False,
        },
    }
    atomic_write(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": report["aggregate"]["rows"],
                "micro_wer": report["aggregate"]["word_error_rate"]["micro"],
                "micro_cer": report["aggregate"]["character_error_rate"]["micro"],
                "output": output_path.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
