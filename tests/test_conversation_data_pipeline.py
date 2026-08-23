import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s.audio import write_wav
from thesis_s2s.data.channels import TABAGHE16, YOUTUBE_CHANNELS, remote_csv
from thesis_s2s.data.conversation import SpeakerSegment, _interaction_label
from thesis_s2s.data.conversation_selection import (
    audit_episode_selection,
    select_conversation_episodes,
)
from thesis_s2s.data.diarize import (
    align_reference_segments,
    audit_diarized_windows,
    diarize_episode_manifest,
    merge_conversation_window_manifests,
)
from thesis_s2s.data.episode_prepare import (
    audit_prepared_episode_windows,
    preparation_stats_path,
    rclone_filter_literal,
    reconstruct_episode_windows,
)
from thesis_s2s.data.ingest import _conversation_candidate
from thesis_s2s.data.manual_qa import apply_manual_qa, sample_manual_qa
from thesis_s2s.data.rights import (
    apply_conversation_rights_review,
    create_conversation_rights_review,
    normalize_pending_rights_metadata,
    rights_record_verified,
    training_use_authorized,
)
from thesis_s2s.data.s3_inventory import rclone_process_env


def test_all_sources_use_2tb_and_s3_credentials_stay_out_of_argv(monkeypatch):
    assert len(YOUTUBE_CHANNELS) == 5
    assert TABAGHE16 in YOUTUBE_CHANNELS
    assert TABAGHE16.bucket == "asr"
    assert "STT/YT_PodCast_Chunks/CSVs/طبقه 16" in remote_csv(TABAGHE16)

    monkeypatch.setenv("S3_ACCESS_KEY_ID", "access-test")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "secret-test")
    monkeypatch.setenv("S3_ENDPOINT", "https://s3.example.invalid")
    monkeypatch.delenv("RCLONE_CONFIG_S3_ACCESS_KEY_ID", raising=False)
    env = rclone_process_env()
    assert env["RCLONE_S3_ACCESS_KEY_ID"] == "access-test"
    assert env["RCLONE_S3_SECRET_ACCESS_KEY"] == "secret-test"
    assert "RCLONE_CONFIG_S3_ACCESS_KEY_ID" not in env


def test_monologue_source_is_fail_closed_except_interview_titles():
    kooshiar = next(spec for spec in YOUTUBE_CHANNELS if spec.name == "Kooshiar")
    assert _conversation_candidate(kooshiar, "آموزش سرمایه گذاری") == (
        False,
        "likely_monologue_excluded_by_default",
    )
    assert _conversation_candidate(kooshiar, "گفتگو با یک مهمان") == (
        True,
        "interview_title_needs_diarization",
    )


def test_weighted_selection_is_balanced_whole_episode_and_not_final_evidence(tmp_path: Path):
    csv_path = tmp_path / "reference.csv"
    csv_path.write_text("start_time,end_time,text\n", encoding="utf-8")
    policies = [
        ("Tabaghe16", 0.50, "interview_podcast", 1),
        ("Mehran Rowshan Persian", 0.28, "interview_podcast", 1),
        ("Digiato", 0.11, "technology_mixed", 2),
        ("Zoomit", 0.11, "technology_mixed", 2),
    ]
    episodes = []
    for channel, weight, source_kind, priority in policies:
        for index in range(8):
            episodes.append(
                {
                    "episode_id": f"{channel}/episode-{index}",
                    "channel": channel,
                    "stem": f"episode-{index}",
                    "csv_path": str(csv_path),
                    "csv_hours": 10.0,
                    "selection_weight": weight,
                    "selection_eligible": True,
                    "conversation_priority": priority,
                    "source_kind": source_kind,
                    "conversation_candidate_reason": "podcast_prior_needs_diarization",
                    "reference_transcript_source": "provided_youtube_csv",
                    "split": "train",
                }
            )
    selected = select_conversation_episodes(
        episodes,
        target_hours=100,
        min_hours=100,
        max_hours=120,
        max_channel_share=0.55,
    )
    report = audit_episode_selection(
        selected,
        min_hours=100,
        max_hours=120,
        target_hours=100,
        max_channel_share=0.55,
    )
    assert report["selected_candidate_hours"] == 100
    assert report["planning_gate_passes"] is True
    assert report["thesis_evidence_gate_passes"] is False
    spoofed_report = audit_episode_selection(
        [{**row, "license_verified": True} for row in selected],
        min_hours=100,
        max_hours=120,
        target_hours=100,
        max_channel_share=0.55,
    )
    assert spoofed_report["evidence_requirements"]["training_authorized_for_selected"] is False
    assert max(item["share"] for item in report["channels"].values()) <= 0.55
    assert all(row["selected_for_conversation_prep"] for row in selected)
    selected_ids = {str(row["episode_id"]) for row in selected}
    repeated = select_conversation_episodes(
        episodes,
        target_hours=100,
        min_hours=100,
        max_hours=120,
        max_channel_share=0.55,
    )
    assert [row["episode_id"] for row in repeated] == [row["episode_id"] for row in selected]
    reserve = select_conversation_episodes(
        episodes,
        target_hours=100,
        min_hours=100,
        max_hours=120,
        max_channel_share=0.55,
        exclude_episode_ids=selected_ids,
    )
    assert sum(float(row["csv_hours"]) for row in reserve) == 100
    assert selected_ids.isdisjoint(str(row["episode_id"]) for row in reserve)


def test_primary_and_reserve_window_merge_is_atomic_and_rejects_duplicates(tmp_path: Path):
    primary = tmp_path / "primary.jsonl"
    reserve = tmp_path / "reserve.jsonl"
    out = tmp_path / "combined.jsonl"
    primary.write_text(
        json.dumps(
            {
                "window_id": "primary-1",
                "episode_id": "primary",
                "channel": "Tabaghe16",
                "duration": 3600.0,
                "automatic_multi_speaker_verified": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reserve.write_text(
        json.dumps({"window_id": "reserve-1", "episode_id": "reserve", "channel": "Zoomit"}) + "\n",
        encoding="utf-8",
    )
    report = merge_conversation_window_manifests([primary, reserve], out)
    assert report["windows"] == 2
    assert report["automatic_multi_speaker_hours"] == 1.0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(ValueError, match="duplicate window_id"):
        merge_conversation_window_manifests([primary, primary], out)


def test_reference_alignment_is_temporal_and_fail_closed():
    references = [
        {"start": 0.0, "end": 2.0, "text": "سلام دوست من"},
        {"start": 2.0, "end": 4.0, "text": "پاسخ مهمان"},
        {"start": 5.0, "end": 6.0, "text": "بدون گفتار"},
    ]
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "A"},
        {"start": 2.0, "end": 4.0, "speaker": "B"},
    ]
    aligned, usable, stats = align_reference_segments(references, turns)
    assert [item["speaker"] for item in usable] == ["A", "B"]
    assert aligned[-1]["alignment_status"] == "low_confidence"
    assert stats["usable_coverage"] == 0.6667


def test_episode_window_reconstruction_tracks_chunk_limitations(tmp_path: Path):
    stem = "episode [one]"
    csv_path = tmp_path / f"{stem}.csv"
    csv_path.write_text(
        "start_time,end_time,text\n"
        "00:00:00,00:00:01,سلام دوست من\n"
        "00:00:01,00:00:02,پاسخ مهمان من\n"
        "00:00:02,00:00:03,ادامه گفتگو است\n",
        encoding="utf-8",
    )
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    audio = np.zeros(16000, dtype=np.float32)
    for index in range(1, 4):
        write_wav(chunks / f"{stem}_chunk_{index:04d}.wav", audio)
    episode = {
        "episode_id": f"Tabaghe16/{stem}",
        "stem": stem,
        "channel": "Tabaghe16",
        "csv_path": str(csv_path),
        "split": "train",
        "source_kind": "interview_podcast",
        "expected_multi_speaker": True,
        "conversation_candidate_reason": "podcast_prior_needs_diarization",
    }
    windows, report = reconstruct_episode_windows(episode, chunks, tmp_path / "out")
    assert report["status"] == "complete"
    assert report["chunk_coverage"] == 1.0
    assert len(windows) == 1
    assert Path(windows[0]["audio_filepath"]).is_file()
    assert len(windows[0]["reference_segments"]) == 3
    assert windows[0]["cross_chunk_overlap_recoverable"] is False
    assert windows[0]["diarization_status"] == "not_run"
    assert windows[0]["license_verified"] is False
    assert windows[0]["internal_research_authorized"] is False
    assert windows[0]["authorization_basis"] == "pending"
    assert windows[0]["redistribution_allowed"] is False
    assert preparation_stats_path(Path("conversation_episode_windows.jsonl")).name == (
        "conversation_episode_prepare_stats.json"
    )
    assert preparation_stats_path(Path("reserve.jsonl")).name == "reserve_prepare_stats.json"
    assert rclone_filter_literal(stem) == r"episode \[one\]"
    second = dict(windows[0])
    second.update(
        {
            "window_id": "Zoomit/episode-two#window-0000",
            "episode_id": "Zoomit/episode-two",
            "session_id": "Zoomit/episode-two",
            "recording_id": "Zoomit/episode-two",
            "channel": "Zoomit",
        }
    )
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in (
                {"episode_id": episode["episode_id"], "split": "train"},
                {"episode_id": second["episode_id"], "split": "train"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "prepared.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in (windows[0], second)) + "\n",
        encoding="utf-8",
    )
    stats = tmp_path / "conversation_episode_prepare_stats.json"
    stats.write_text(
        '{"finished_at":"now","in_progress":false,"episodes_failed":0,'
        '"episodes_prepared":2,"episodes_resumed":0,"windows":2}\n',
        encoding="utf-8",
    )
    audit = audit_prepared_episode_windows(
        selection, manifest, stats_path=stats, min_hours=0, max_hours=1
    )
    assert audit["reconstruction_gate_passes"] is True
    assert audit["legacy_or_implicit_rights_rows"] == 0
    assert audit["requirements"]["rights_metadata_explicit"] is True
    assert (
        audit["artifact_sha256"]["window_manifest"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    legacy_rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    legacy_rows[0]["license"] = "unknown"
    legacy_rows[0].pop("license_verified")
    legacy_rows[1]["license"] = "pending-youtube-rights-review"
    legacy_rows[1]["license_verified"] = True
    legacy_rows[1]["internal_research_authorized"] = True
    legacy_rows[1]["authorization_basis"] = "supervisor-approved-internal-research"
    legacy_rows[1]["redistribution_allowed"] = True
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in legacy_rows) + "\n",
        encoding="utf-8",
    )
    legacy_audit = audit_prepared_episode_windows(
        selection, manifest, stats_path=stats, min_hours=0, max_hours=1
    )
    assert legacy_audit["legacy_or_implicit_rights_rows"] == 2
    assert legacy_audit["requirements"]["rights_metadata_explicit"] is False
    assert legacy_audit["reconstruction_gate_passes"] is False
    normalized = normalize_pending_rights_metadata(manifest)
    assert normalized["changed_rows"] == 2
    assert normalized["normalized_legacy_rows"] == 1
    assert normalized["normalized_invalid_flags"] == 1
    assert normalized["normalized_authorization_flags"] == 1
    assert normalized["license_approval_inferred"] is False
    assert normalized["training_authorization_inferred"] is False
    normalized_rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(row["license"] == "pending-youtube-rights-review" for row in normalized_rows)
    assert all(row["license_verified"] is False for row in normalized_rows)
    assert all(row["internal_research_authorized"] is False for row in normalized_rows)
    assert all(row["authorization_basis"] == "pending" for row in normalized_rows)
    assert all(row["redistribution_allowed"] is False for row in normalized_rows)
    normalized_audit = audit_prepared_episode_windows(
        selection, manifest, stats_path=stats, min_hours=0, max_hours=1
    )
    assert normalized_audit["reconstruction_gate_passes"] is True
    assert (
        normalized_audit["artifact_sha256"]["window_manifest"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert normalize_pending_rights_metadata(manifest)["changed_rows"] == 0
    stats.write_text(
        '{"finished_at":"old","episodes_failed":0,"episodes_prepared":1,'
        '"episodes_resumed":0,"windows":1}\n',
        encoding="utf-8",
    )
    stale = audit_prepared_episode_windows(
        selection, manifest, stats_path=stats, min_hours=0, max_hours=1
    )
    assert stale["requirements"]["preparation_report_current_and_finished"] is False
    assert stale["reconstruction_gate_passes"] is False


def test_episode_diarization_aligns_and_failed_rows_are_retried(tmp_path: Path, monkeypatch):
    audio_path = tmp_path / "window.wav"
    write_wav(audio_path, np.zeros(4 * 16000, dtype=np.float32))
    source = tmp_path / "windows.jsonl"
    row = {
        "window_id": "episode-1#window-0000",
        "episode_id": "episode-1",
        "channel": "Tabaghe16",
        "audio_filepath": str(audio_path),
        "duration": 4.0,
        "split": "train",
        "cross_chunk_overlap_recoverable": False,
        "reference_segments": [
            {"start": 0.0, "end": 2.0, "text": "پرسش میزبان"},
            {"start": 2.0, "end": 4.0, "text": "پاسخ مهمان"},
        ],
    }
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "diarized.jsonl"
    failed = dict(row)
    failed["diarization_status"] = "failed"
    output.write_text(json.dumps(failed, ensure_ascii=False) + "\n", encoding="utf-8")

    def fake_diarize(path, base=None):
        assert path == audio_path
        return {
            "available": True,
            "speaker_turns": [
                {"start": 0.0, "end": 2.0, "speaker": "A"},
                {"start": 2.0, "end": 4.0, "speaker": "B"},
            ],
            "exclusive_speaker_turns": [
                {"start": 0.0, "end": 2.0, "speaker": "A"},
                {"start": 2.0, "end": 4.0, "speaker": "B"},
            ],
            "overlap_intervals": [],
        }

    monkeypatch.setattr("thesis_s2s.data.diarize.diarize_file", fake_diarize)
    report = diarize_episode_manifest(
        source,
        output,
        min_secondary_speaker_seconds=1.0,
        min_secondary_speaker_share=0.1,
    )
    assert report["attempted"] == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["diarization_status"] == "complete"
    assert rows[0]["license"] == "pending-youtube-rights-review"
    assert rows[0]["license_verified"] is False
    assert rows[0]["automatic_multi_speaker_verified"] is True
    assert [item["speaker"] for item in rows[0]["segments"]] == ["A", "B"]
    audit = audit_diarized_windows(output, min_hours=0.0, max_hours=1.0)
    assert audit["automatic_multi_speaker_hours"] > 0
    assert audit["requirements"]["manual_qa_sample_present"] is False

    assert audit["requirements"]["training_use_authorized"] is False
    rows[0]["license"] = "youtube-internal"
    rows[0].pop("license_verified")
    output.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")

    def unexpected_diarization(*args, **kwargs):
        raise AssertionError("completed windows must be resumed without another GPU request")

    monkeypatch.setattr("thesis_s2s.data.diarize.diarize_file", unexpected_diarization)
    resumed = diarize_episode_manifest(source, output)
    assert resumed["attempted"] == 0
    assert resumed["complete"] == 1
    assert resumed["completed_this_run"] == 0
    assert resumed["automatic_multi_speaker_hours"] > 0
    normalized = json.loads(output.read_text(encoding="utf-8"))
    assert normalized["license"] == "pending-youtube-rights-review"
    assert normalized["license_verified"] is False
    progress = json.loads(
        (tmp_path / "conversation_episode_diarization_stats.json").read_text(encoding="utf-8")
    )
    assert progress["in_progress"] is False
    assert progress["output_windows"] == 1
    assert progress["source_manifest_complete"] is True


def test_known_diarization_overlap_can_label_an_interruption():
    user = SpeakerSegment(0.0, 0.7, "A", "پرسش")
    assistant = SpeakerSegment(1.0, 2.0, "B", "پاسخ")
    label, overlap = _interaction_label(user, assistant, [[0.8, 0.9]])
    assert label == "interrupt"
    assert overlap == [[0.8, 0.9]]


def test_manual_qa_handoff_is_stratified_and_fail_closed(tmp_path: Path):
    source = tmp_path / "diarized.jsonl"
    source_rows = []
    for channel in ("Tabaghe16", "Zoomit"):
        for verified in (False, True):
            source_rows.append(
                {
                    "window_id": f"{channel}-{verified}",
                    "episode_id": f"{channel}/episode",
                    "channel": channel,
                    "audio_filepath": f"/internal/{channel}-{verified}.wav",
                    "duration": 60.0,
                    "diarization_status": "complete",
                    "automatic_multi_speaker_verified": verified,
                    "reference_alignment": {"usable_coverage": 0.9},
                    "speaker_evidence": {
                        "n_speakers": 2 if verified else 1,
                        "secondary_speaker_share": 0.2 if verified else 0.0,
                    },
                    "overlap_intervals": [],
                    "license": "pending-youtube-rights-review",
                    "license_verified": False,
                }
            )
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in source_rows) + "\n",
        encoding="utf-8",
    )
    qa_csv = tmp_path / "qa.csv"
    sampled = sample_manual_qa(source, qa_csv, per_stratum=1)
    assert sampled["sampled_windows"] == 4

    with qa_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        decisions = list(csv.DictReader(handle))
        fieldnames = list(decisions[0])
    for decision in decisions:
        decision.update(
            {
                "review_status": "pass",
                "speaker_count_correct": "yes",
                "speaker_assignment_correct": "yes",
                "caption_acceptable": "yes",
                "overlap_annotation_correct": "yes",
                "reviewer_id": "reviewer-1",
            }
        )
    decisions[0]["caption_acceptable"] = "no"
    decisions[0]["automatic_status"] = "tampered"
    with qa_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)

    reviewed = tmp_path / "reviewed.jsonl"
    report = apply_manual_qa(
        source,
        qa_csv,
        reviewed,
        report_path=tmp_path / "qa-report.json",
    )
    assert report["counts"]["approved"] == 3
    assert report["counts"]["rejected"] == 1
    reviewed_rows = [json.loads(line) for line in reviewed.read_text(encoding="utf-8").splitlines()]
    assert sum(bool(row["human_verified"]) for row in reviewed_rows) == 3
    assert all(row["license_verified"] is False for row in reviewed_rows)
    audit = audit_diarized_windows(reviewed, min_hours=0.0, max_hours=1.0)
    assert audit["manual_qa_reviewed_windows"] == 4
    assert audit["human_verified_windows"] == 3
    assert audit["requirements"]["manual_qa_sample_present"] is True
    assert len(audit["manual_qa_reviewed_strata"]) == 4


def test_source_authorization_distinguishes_supervisor_approval_from_license(
    tmp_path: Path,
):
    source = tmp_path / "reviewed.jsonl"
    source_rows = [
        {
            "window_id": "tabaghe-1",
            "episode_id": "Tabaghe16/one",
            "channel": "Tabaghe16",
            "duration": 3600.0,
            "license": "pending-youtube-rights-review",
            "license_verified": False,
            "internal_research_authorized": False,
            "authorization_basis": "pending",
            "redistribution_allowed": False,
        },
        {
            "window_id": "zoomit-1",
            "episode_id": "Zoomit/one",
            "channel": "Zoomit",
            "duration": 1800.0,
            "license": "pending-youtube-rights-review",
            "license_verified": False,
            "internal_research_authorized": False,
            "authorization_basis": "pending",
            "redistribution_allowed": False,
        },
    ]
    source.write_text(
        "\n".join(json.dumps(row) for row in source_rows) + "\n",
        encoding="utf-8",
    )
    review_csv = tmp_path / "rights.csv"
    created = create_conversation_rights_review(source, review_csv)
    assert created["review_scopes"] == 2
    with pytest.raises(FileExistsError):
        create_conversation_rights_review(source, review_csv)

    with review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        decisions = list(csv.DictReader(handle))
        fieldnames = list(decisions[0])
    assert "authorization_basis" in fieldnames
    assert "license_name" in fieldnames
    for decision in decisions:
        decision.update(
            {
                "approval_status": "approved",
                "authorization_basis": "supervisor-approved-internal-research",
                "license_name": "",
                "internal_training_allowed": "yes",
                "thesis_reporting_allowed": "yes",
                "derived_artifacts_allowed": "yes",
                "redistribution_allowed": "no",
                "evidence_reference": "supervisor-approval-1",
                "approved_by": "thesis-supervisor",
                "approval_date": "2026-08-23",
            }
        )
    decisions[1]["evidence_reference"] = ""
    with review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)

    approved = tmp_path / "approved.jsonl"
    partial = apply_conversation_rights_review(source, review_csv, approved)
    assert partial["training_authorization_gate_passes"] is False
    partial_rows = [json.loads(line) for line in approved.read_text().splitlines()]
    assert sum(row["internal_research_authorized"] is True for row in partial_rows) == 1
    assert all(row["license_verified"] is False for row in partial_rows)
    assert all(row["redistribution_allowed"] is False for row in partial_rows)

    decisions[1]["evidence_reference"] = "supervisor-approval-2"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)
    complete = apply_conversation_rights_review(
        source,
        review_csv,
        approved,
        report_path=tmp_path / "rights-report.json",
    )
    assert complete["training_authorization_gate_passes"] is True
    assert complete["rights_gate_passes"] is True
    assert complete["license_gate_passes"] is False
    assert complete["authorized_hours"] == 1.5
    assert complete["license_verified_hours"] == 0.0
    approved_rows = [json.loads(line) for line in approved.read_text().splitlines()]
    assert all(row["license"] == "pending-youtube-rights-review" for row in approved_rows)
    assert all(row["license_verified"] is False for row in approved_rows)
    assert all(row["internal_research_authorized"] is True for row in approved_rows)
    assert all(
        row["authorization_basis"] == "supervisor-approved-internal-research"
        for row in approved_rows
    )
    assert all(training_use_authorized(row) for row in approved_rows)
    assert not any(rights_record_verified(row) for row in approved_rows)

    approved_rows[0]["rights_review"]["evidence_reference"] = ""
    assert training_use_authorized(approved_rows[0]) is False
    spoofed = dict(approved_rows[1])
    spoofed["redistribution_allowed"] = True
    assert training_use_authorized(spoofed) is False
    malformed = dict(approved_rows[1])
    malformed["rights_review"] = "approved"
    assert training_use_authorized(malformed) is False

    for decision in decisions:
        decision["authorization_basis"] = "explicit-source-license"
        decision["license_name"] = "CC-BY-4.0"
        decision["redistribution_allowed"] = "yes"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)
    explicitly_licensed = tmp_path / "explicitly-licensed.jsonl"
    licensed_report = apply_conversation_rights_review(
        source,
        review_csv,
        explicitly_licensed,
    )
    licensed_rows = [json.loads(line) for line in explicitly_licensed.read_text().splitlines()]
    assert licensed_report["license_gate_passes"] is True
    assert all(row["license"] == "CC-BY-4.0" for row in licensed_rows)
    assert all(row["license_verified"] is True for row in licensed_rows)
    assert all(row["internal_research_authorized"] is False for row in licensed_rows)
    assert all(row["redistribution_allowed"] is True for row in licensed_rows)
    assert all(rights_record_verified(row) for row in licensed_rows)
    assert all(training_use_authorized(row) for row in licensed_rows)
