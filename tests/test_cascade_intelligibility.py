from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.evaluate_cascade_intelligibility import (
    MANIFEST_SHA256,
    PANEL_INDICES,
    PARENT_PANEL_SHA256,
    summarize,
    text_error_counts,
)
from thesis_s2s.eval.wer import (
    cer,
    character_tokens,
    edit_distance,
    tokenize,
    wer,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_word_and_character_metrics_have_declared_persian_tokenization() -> None:
    assert tokenize("می\u200cروم خانه") == ["می", "روم", "خانه"]
    assert character_tokens("می\u200c روم") == list("میروم")
    assert edit_distance(list("سلام"), list("سالم")) == 2
    assert wer("سلام دنیا", "سلام") == 0.5
    assert cer("سلام", "سلا") == 0.25
    assert text_error_counts("سلام دنیا", "سلام") == {
        "reference_words": 2,
        "hypothesis_words": 1,
        "word_edits": 1,
        "wer": 0.5,
        "reference_characters": 8,
        "hypothesis_characters": 4,
        "character_edits": 4,
        "cer": 0.5,
    }


def test_summary_reports_micro_macro_extremes_and_failures() -> None:
    rows = [
        {
            **text_error_counts("یک دو", "یک دو"),
            "roundtrip_asr_error": None,
        },
        {
            **text_error_counts("یک دو", "یک"),
            "roundtrip_asr_error": "timeout",
        },
    ]
    report = summarize(rows)
    assert report["rows"] == 2
    assert report["asr_failures"] == 1
    assert report["exact_word_match_rows"] == 1
    assert report["word_error_rate"]["micro"] == 0.25
    assert report["word_error_rate"]["macro_mean"] == 0.25
    assert report["word_error_rate"]["max"] == 0.5


def test_protocol_constants_bind_existing_validation_without_final_test() -> None:
    manifest = ROOT / "data/processed/moshi_finetune/val.jsonl"
    parent = ROOT / "results/eval/cascade_real_service_validation_panel.json"
    assert _sha256(manifest) == MANIFEST_SHA256
    assert _sha256(parent) == PARENT_PANEL_SHA256
    assert PANEL_INDICES == (0, 16, 32, 48, 65, 81, 97, 113, 130)
    protocol = (ROOT / "docs/CASCADE_INTELLIGIBILITY_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    assert "sealed Moshi final-test manifest is forbidden" in protocol
    assert "No quality threshold" in protocol


def test_canonical_result_is_valid_privacy_safe_and_hash_bound() -> None:
    result_path = ROOT / "results/eval/cascade_intelligibility_proxy.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "valid"
    assert payload["panel"]["final_test_accessed"] is False
    assert payload["aggregate"]["rows"] == 9
    assert payload["aggregate"]["asr_failures"] == 0
    assert payload["aggregate"]["word_error_rate"]["micro"] == 0.304094
    assert payload["aggregate"]["character_error_rate"]["micro"] == 0.071429
    assert payload["artifacts"]["parent_panel_sha256"] == PARENT_PANEL_SHA256
    assert all(payload["validity_requirements"].values())
    assert payload["privacy"] == {
        "plaintext_input_transcripts_stored": False,
        "plaintext_reply_text_stored": False,
        "plaintext_roundtrip_transcripts_stored": False,
        "audio_stored": False,
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for item in value.values() for key in keys(item)}
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    assert keys(payload).isdisjoint(
        {"transcript", "reply_text", "roundtrip_transcript", "audio_filepath", "path"}
    )
