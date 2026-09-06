import json
from pathlib import Path

from scripts.analyze_cascade_validation import analyze


def test_analysis_is_descriptive_fail_closed_and_privacy_safe(tmp_path: Path) -> None:
    report = tmp_path / "panel.json"
    samples = []
    for index, (transcript_chars, reply_chars, seconds, elapsed) in enumerate(
        ((10, 5, 1.0, 500.0), (20, 10, 2.0, 1000.0), (30, 15, 3.0, 2000.0))
    ):
        samples.append(
            {
                "manifest_row_index": index * 10,
                "passes": index != 1,
                "transcript_sha256": f"transcript-{index}",
                "reply_text_sha256": "duplicate" if index < 2 else "unique",
                "transcript_statistics": {"characters": transcript_chars},
                "reply_statistics": {"characters": reply_chars},
                "reply_audio_duration_seconds": seconds,
                "reply_audio_rms": 0.1 + index / 10,
                "full_turn_generation_ms": elapsed,
                "responder_fallback_used": False,
                "responder_language_retry_used": False,
                "asr_error": None,
                "requirements": {"mechanics": index != 1},
            }
        )
    report.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_class": "test-panel",
                "requirements": {"panel_exact": True},
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )

    result = analyze(report)

    assert result["panel_rows"] == 3
    assert result["panel_passed_rows"] == 2
    assert result["sample_requirement_failure_counts"] == {"mechanics": 1}
    assert result["execution_outcomes"]["duplicate_reply_rows"] == 1
    assert result["distributions"]["transcript_characters"]["p50"] == 20.0
    assert result["row_extremes"]["slowest_full_turn_ms"] == {
        "manifest_row_index": 20,
        "value": 2000.0,
    }
    assert result["claim_boundary"]["plaintext_transcripts_or_replies_read_or_emitted"] is False
    assert result["claim_boundary"]["semantic_quality_claim_allowed"] is False
    assert result["claim_boundary"]["final_test_accessed"] is False
    assert result["analysis_integrity"]["all_required_measurements_and_hashes_present"] is True
    assert result["status"] == "source_report_incomplete_or_has_failures"


def test_analysis_fails_closed_when_a_required_measurement_is_missing(tmp_path: Path) -> None:
    report = tmp_path / "panel.json"
    report.write_text(
        json.dumps(
            {
                "status": "passed",
                "requirements": {"panel_exact": True},
                "samples": [
                    {
                        "manifest_row_index": 0,
                        "passes": True,
                        "transcript_sha256": "transcript",
                        "reply_text_sha256": "reply",
                        "transcript_statistics": {"characters": 10},
                        "reply_statistics": {"characters": 5},
                        "reply_audio_duration_seconds": 1.0,
                        "reply_audio_rms": 0.1,
                        "requirements": {"mechanics": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = analyze(report)

    assert result["analysis_integrity"]["measurement_missing_rows"] == {
        "transcript_characters": 0,
        "reply_characters": 0,
        "reply_audio_seconds": 0,
        "reply_audio_rms": 0,
        "full_turn_generation_ms": 1,
    }
    assert result["status"] == "source_report_incomplete_or_has_failures"
