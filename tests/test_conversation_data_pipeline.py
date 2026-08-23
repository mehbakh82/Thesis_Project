import csv
import json
from pathlib import Path

import numpy as np

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
)
from thesis_s2s.data.episode_prepare import (
    audit_prepared_episode_windows,
    rclone_filter_literal,
    reconstruct_episode_windows,
)
from thesis_s2s.data.ingest import _conversation_candidate
from thesis_s2s.data.manual_qa import apply_manual_qa, sample_manual_qa
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

    assert audit["requirements"]["licenses_verified"] is False


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
