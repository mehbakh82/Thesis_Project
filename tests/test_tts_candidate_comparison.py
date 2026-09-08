import json

from scripts.evaluate_tts_candidates_v1 import (
    ALL_ARMS,
    CHALLENGER_ARM,
    CHANNELS,
    CURRENT_ARM,
    EXPECTED_ROWS,
    arm_summary,
    panel_receipt,
    responder_panel_sessions,
    select_panels,
    select_winner,
)


def _source_rows():
    rows = []
    for channel in CHANNELS:
        for session_index in range(8):
            session = f"{channel}/episode-{session_index}"
            for row_index in range(5):
                rows.append(
                    {
                        "utt_id": f"{session}/row-{row_index}",
                        "session_id": session,
                        "split": "train",
                        "text": "این متن پرسش فارسی معتبر است",
                        "response_text": "این پاسخ فارسی برای آزمون گفتار است",
                    }
                )
    return rows


def _metric_samples(character_edits: int, word_edits: int, rtf: float):
    samples = []
    for index in range(EXPECTED_ROWS):
        samples.append(
            {
                "row_hash": f"row-{index}",
                "session_hash": f"session-{index // 5}",
                "channel": CHANNELS[index // 10],
                "reference_words": 2,
                "hypothesis_words": 2 - int(word_edits > 0),
                "word_edits": word_edits,
                "wer": word_edits / 2,
                "reference_characters": 10,
                "hypothesis_characters": 10 - character_edits,
                "character_edits": character_edits,
                "cer": character_edits / 10,
                "empty_hypothesis": False,
                "synthesis_seconds": 0.1,
                "real_time_factor": rtf,
                "audio_duration_seconds": 1.0,
            }
        )
    return samples


def test_tts_panels_are_new_and_group_disjoint():
    rows = _source_rows()
    excluded = responder_panel_sessions(rows)
    panels = select_panels(rows)
    development_sessions = {row["session_id"] for row in panels["development"]}
    final_sessions = {row["session_id"] for row in panels["final"]}

    assert len(panels["development"]) == EXPECTED_ROWS
    assert len(panels["final"]) == EXPECTED_ROWS
    assert not (development_sessions & final_sessions)
    assert not ((development_sessions | final_sessions) & excluded)
    assert all(
        sum(row["session_id"].startswith(channel + "/") for row in panel) == 10
        for panel in panels.values()
        for channel in CHANNELS
    )


def test_panel_receipt_does_not_retain_plaintext():
    panel = select_panels(_source_rows())["development"]
    receipt = panel_receipt(panel)
    serialized = json.dumps(receipt, ensure_ascii=False)

    assert receipt["rows"] == EXPECTED_ROWS
    assert "این پاسخ" not in serialized
    assert "episode-" not in serialized


def test_challenger_advances_only_on_material_paired_improvement():
    samples = {
        CURRENT_ARM: _metric_samples(character_edits=3, word_edits=1, rtf=0.1),
        CHALLENGER_ARM: _metric_samples(character_edits=1, word_edits=0, rtf=0.2),
    }
    summaries = {arm: arm_summary(samples[arm]) for arm in ALL_ARMS}
    selection = select_winner(samples, summaries)

    assert selection["status"] == "challenger_advanced"
    assert selection["final_panel_unlocked"] is True
    assert selection["production_promotion_allowed"] is False


def test_failure_remains_in_denominator_and_blocks_promotion():
    incumbent = _metric_samples(character_edits=3, word_edits=1, rtf=0.1)
    challenger = _metric_samples(character_edits=1, word_edits=0, rtf=0.2)
    challenger[0]["failure_type"] = "RuntimeError"
    challenger[0]["failure_stage"] = "synthesis"
    samples = {CURRENT_ARM: incumbent, CHALLENGER_ARM: challenger}
    summaries = {arm: arm_summary(samples[arm]) for arm in ALL_ARMS}
    selection = select_winner(samples, summaries)

    assert summaries[CHALLENGER_ARM]["rows"] == EXPECTED_ROWS
    assert summaries[CHALLENGER_ARM]["failures"] == 1
    assert summaries[CHALLENGER_ARM]["mechanically_valid"] is False
    assert selection["status"] == "incumbent_retained"
    assert selection["final_panel_unlocked"] is False
