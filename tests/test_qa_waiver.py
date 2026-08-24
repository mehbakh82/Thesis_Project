import json
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import write_wav
from thesis_s2s.data.conversation import (
    audit_conversation_manifest,
    build_conversation_manifest,
)
from thesis_s2s.data.moshi import export_moshi_finetune_dataset
from thesis_s2s.data.qa_policy import load_qa_waiver


def _write_waiver(tmp_path: Path, *, supervisor_claimed: bool = False) -> Path:
    path = tmp_path / "conversation_qa_waiver.yaml"
    path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "status: acknowledged",
                "policy: automatic_only_documented_waiver",
                'decision_date: "2026-08-24"',
                "decision_authority: student_project_owner",
                f"supervisor_approval_claimed: {str(supervisor_claimed).lower()}",
                "reason: insufficient_time_and_no_available_delegate",
                "scope:",
                "  window_manual_qa: waived",
                "  interaction_manual_qa: waived",
                "  internal_training: allowed_under_waiver",
                "claims:",
                "  human_verified_data: false",
                "  human_verified_interruptions: false",
                "  strict_thesis_data_coverage: false",
                "preservation:",
                "  qa_sheets: true",
                "  reviewer_guides: true",
                "  playback_helper: true",
                "  future_review_supported: true",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _rights() -> dict:
    basis = "supervisor-approved-internal-research"
    return {
        "license": "pending-youtube-rights-review",
        "license_verified": False,
        "internal_research_authorized": True,
        "authorization_basis": basis,
        "redistribution_allowed": False,
        "rights_review": {
            "status": "approved",
            "authorization_basis": basis,
            "decision_complete": True,
            "permissions": {
                "internal_training": True,
                "thesis_reporting": True,
                "derived_artifacts": True,
                "redistribution": False,
            },
            "evidence_reference": "supervisor-decision",
            "approved_by": "thesis-supervisor",
            "approval_date": "2026-08-23",
        },
    }


def test_waiver_rejects_fabricated_supervisor_approval(tmp_path: Path):
    waiver_path = _write_waiver(tmp_path, supervisor_claimed=True)

    with pytest.raises(ValueError, match="supervisor_approval_not_claimed"):
        load_qa_waiver(waiver_path)


def test_builder_preserves_automatic_candidate_without_human_claim(tmp_path: Path):
    waiver_path = _write_waiver(tmp_path)
    source_audio = tmp_path / "source.wav"
    write_wav(source_audio, np.zeros(2 * SAMPLE_RATE, dtype=np.float32))
    source_manifest = tmp_path / "source.jsonl"
    row = {
        **_rights(),
        "window_id": "window-1",
        "session_id": "session-1",
        "audio_filepath": str(source_audio),
        "duration": 2.0,
        "automatic_multi_speaker_verified": True,
        "reference_alignment_status": "complete",
        "manual_qa_reviewed": True,
        "human_verified": True,
        "segments": [
            {"start": 0.0, "end": 0.9, "speaker": "A", "text": "پرسش آزمایشی"},
            {"start": 0.9, "end": 1.8, "speaker": "B", "text": "پاسخ آزمایشی"},
        ],
        "speaker_turns": [
            {"start": 0.0, "end": 1.2, "speaker": "A"},
            {"start": 0.7, "end": 1.8, "speaker": "B"},
        ],
        "noise_condition": "podcast-room",
    }
    source_manifest.write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "conversations.jsonl"

    build = build_conversation_manifest(
        source_manifest,
        output,
        tmp_path / "clips",
        qa_waiver_path=waiver_path,
    )
    pair = json.loads(output.read_text(encoding="utf-8").strip())

    assert build["qa_policy"]["valid"] is True
    assert pair["automatic_interaction_candidate"]["human_verified"] is False
    assert pair["manual_qa_waived"] is True
    assert pair["human_verified"] is False
    assert pair["human_verified_interaction_label"] is False
    assert pair["qa_waiver_sha256"] == load_qa_waiver(waiver_path)["sha256"]


def _waived_pair(
    tmp_path: Path,
    waiver: dict,
    *,
    index: int,
    split: str,
) -> dict:
    user_path = tmp_path / f"user-{index}.wav"
    response_path = tmp_path / f"response-{index}.wav"
    write_wav(user_path, np.zeros(SAMPLE_RATE, dtype=np.float32))
    write_wav(response_path, np.zeros(SAMPLE_RATE, dtype=np.float32))
    return {
        **_rights(),
        "utt_id": f"pair-{index}",
        "session_id": f"session-{index}",
        "speaker_id": f"session-{index}:A",
        "assistant_speaker_id": f"session-{index}:B",
        "split": split,
        "audio_filepath": str(user_path),
        "response_audio_filepath": str(response_path),
        "source_audio_filepath": str(tmp_path / f"source-{index}.wav"),
        "source_span_start": 0.0,
        "source_span_end": 180000.0,
        "source_user_interval": [0.0, 1.0],
        "source_response_interval": [0.8, 1.8],
        "duration": 180000.0,
        "assistant_text": "پاسخ آزمایشی",
        "response_text": "پاسخ آزمایشی",
        "interrupt_label": "overlap_unattributed",
        "overlap_intervals": [[0.8, 1.0]],
        "automatic_interaction_candidate": {
            "candidate_id": f"candidate-{index}",
            "automatic_label": "interrupt",
            "human_verified": False,
        },
        "noise_condition": "podcast-room",
        "manual_qa_waived": True,
        "qa_policy": waiver["policy"],
        "qa_waiver_sha256": waiver["sha256"],
        "human_verified": False,
        "human_verified_interaction_label": False,
    }


def test_audit_opens_only_waiver_training_gate(tmp_path: Path):
    waiver_path = _write_waiver(tmp_path)
    waiver = load_qa_waiver(waiver_path)
    rows = [
        _waived_pair(tmp_path, waiver, index=0, split="train"),
        _waived_pair(tmp_path, waiver, index=1, split="val"),
    ]
    manifest = tmp_path / "conversations.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    audit = audit_conversation_manifest(manifest, qa_waiver_path=waiver_path)

    assert audit["training_ready_under_qa_waiver"] is True
    assert audit["thesis_coverage_ok"] is False
    assert audit["requirements"]["manual_verification_sample_present"] is False
    assert audit["requirements"]["interruptions_present"] is False
    assert audit["claims"] == {
        "human_verified_data": False,
        "human_verified_interruptions": False,
        "strict_thesis_data_coverage": False,
    }


def test_moshi_export_requires_exact_waiver_stamp(tmp_path: Path):
    waiver_path = _write_waiver(tmp_path)
    waiver = load_qa_waiver(waiver_path)
    row = _waived_pair(tmp_path, waiver, index=0, split="train")
    manifest = tmp_path / "conversations.jsonl"
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    report = export_moshi_finetune_dataset(
        manifest,
        tmp_path / "moshi",
        assistant_audio_mode="source",
        qa_waiver_path=waiver_path,
    )

    assert report["qa_policy"]["sha256"] == waiver["sha256"]
    assert report["final_training_ready"] is False
    assert report["claims"]["human_verified_data"] is False
    assert report["waiver_requirements"]["all_pairs_declare_same_waiver"] is True

    row["human_verified"] = True
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact documented waiver"):
        export_moshi_finetune_dataset(
            manifest,
            tmp_path / "rejected",
            assistant_audio_mode="source",
            qa_waiver_path=waiver_path,
        )
