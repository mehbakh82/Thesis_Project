import json

import pytest

from thesis_s2s.data.conversation_balance import audit_conversation_balance


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_conversation_balance_reports_dominance_and_remediation(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "session_id": "A/one",
                "duration": 60,
                "user_duration": 30,
                "assistant_duration": 29,
                "human_verified": False,
            },
            {
                "session_id": "A/two",
                "duration": 30,
                "user_duration": 14,
                "assistant_duration": 15,
                "human_verified": False,
            },
            {
                "session_id": "B/three",
                "duration": 60,
                "user_duration": 29,
                "assistant_duration": 30,
                "human_verified": True,
                "automatic_interaction_candidate": "interrupt",
            },
        ],
    )

    report = audit_conversation_balance(
        manifest,
        tmp_path / "report.json",
        max_channel_hour_share=0.55,
        min_channels=2,
        root=tmp_path,
    )

    assert report["totals"]["pairs"] == 3
    assert report["totals"]["human_verified_rows"] == 1
    assert report["dominance"]["channel"] == "A"
    assert report["dominance"]["hour_share"] == pytest.approx(0.6)
    assert report["dominance"]["balance_gate_passes"] is False
    assert report["dominance"]["hours_to_replace_at_fixed_total"] == pytest.approx(
        7.5 / 3600, abs=1e-6
    )
    assert report["channels"]["B"]["interaction_candidates"] == 1
    assert report["strict_representative_balance_passes"] is False


def test_conversation_balance_rejects_invalid_threshold(tmp_path):
    with pytest.raises(ValueError, match="max_channel_hour_share"):
        audit_conversation_balance(
            tmp_path / "missing.jsonl",
            tmp_path / "report.json",
            max_channel_hour_share=0.0,
            root=tmp_path,
        )
