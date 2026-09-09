import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s.audio import write_wav
from thesis_s2s.data.channels import TABAGHE16, YOUTUBE_CHANNELS, remote_csv
from thesis_s2s.data.conversation import (
    SpeakerSegment,
    _automatic_interaction_candidate,
    _interaction_label,
    audit_conversation_manifest,
    build_conversation_manifest,
    estimate_conversation_pair_yield,
)
from thesis_s2s.data.conversation_balance import (
    merge_conversation_pair_manifests,
    plan_balanced_supplement,
    select_balanced_conversation_pairs,
)
from thesis_s2s.data.conversation_selection import (
    audit_episode_selection,
    load_jsonl,
    select_conversation_episodes,
    write_jsonl,
)
from thesis_s2s.data.diarize import (
    align_reference_segments,
    audit_diarized_windows,
    diarization_stats_path,
    diarize_episode_manifest,
    merge_conversation_window_manifests,
)
from thesis_s2s.data.episode_prepare import (
    audit_prepared_episode_windows,
    preparation_stats_path,
    prepare_selected_episodes,
    rclone_filter_literal,
    reconstruct_episode_windows,
)
from thesis_s2s.data.ingest import _conversation_candidate
from thesis_s2s.data.interaction_qa import apply_interaction_qa, sample_interaction_qa
from thesis_s2s.data.manual_qa import apply_manual_qa, sample_manual_qa
from thesis_s2s.data.rights import (
    apply_conversation_rights_review,
    create_conversation_rights_review,
    normalize_pending_rights_metadata,
    rights_record_verified,
    training_use_authorized,
)
from thesis_s2s.data.s3_inventory import (
    rclone_path,
    rclone_prefix,
    rclone_process_env,
)


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


def test_named_rclone_remote_needs_no_exported_credentials(monkeypatch):
    monkeypatch.setenv("THESIS_RCLONE_REMOTE", "s3")
    for key in ("S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)

    assert rclone_prefix() == ["rclone"]
    assert rclone_path(":s3:asr/path") == "s3:asr/path"
    env = rclone_process_env()
    assert "RCLONE_S3_SECRET_ACCESS_KEY" not in env


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


def test_balanced_supplement_plan_is_disjoint_hash_bound_and_conservative(
    tmp_path: Path,
) -> None:
    inventory_path = tmp_path / "inventory.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    windows_path = tmp_path / "windows.jsonl"
    selection_path = tmp_path / "supplement.jsonl"
    report_path = tmp_path / "plan.json"

    inventory: list[dict] = []
    used_windows: list[dict] = []
    channel_policies = {
        "Tabaghe16": (0.50, 100.0),
        "Digiato": (0.11, 20.0),
        "Mehran Rowshan Persian": (0.28, 20.0),
        "Zoomit": (0.11, 20.0),
    }
    for channel, (weight, used_candidate_hours) in channel_policies.items():
        used_id = f"{channel}/used"
        inventory.append(
            {
                "episode_id": used_id,
                "channel": channel,
                "csv_hours": used_candidate_hours,
                "selection_weight": weight,
                "selection_eligible": True,
                "conversation_priority": 1,
            }
        )
        used_windows.append({"episode_id": used_id, "session_id": used_id})
        if channel != "Tabaghe16":
            for index in range(5):
                inventory.append(
                    {
                        "episode_id": f"{channel}/new-{index}",
                        "channel": channel,
                        "csv_hours": 10.0,
                        "selection_weight": weight,
                        "selection_eligible": True,
                        "conversation_priority": 1,
                    }
                )

    pairs = [
        {"session_id": "Tabaghe16/used", "duration": 75.0 * 3600.0},
        {"session_id": "Digiato/used", "duration": 9.0 * 3600.0},
        {"session_id": "Mehran Rowshan Persian/used", "duration": 8.0 * 3600.0},
        {"session_id": "Zoomit/used", "duration": 8.0 * 3600.0},
    ]

    def write_rows(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_rows(inventory_path, inventory)
    write_rows(pairs_path, pairs)
    write_rows(windows_path, used_windows)
    report = plan_balanced_supplement(
        inventory_path,
        pairs_path,
        windows_path,
        selection_path,
        report_path,
        target_candidate_hours=50.0,
        max_candidate_hours=60.0,
        expected_yield_safety_factor=1.0,
        root=tmp_path,
    )

    selected = [json.loads(line) for line in selection_path.read_text().splitlines()]
    assert report["frozen_v1_mutated"] is False
    assert report["selection"]["candidate_hours"] == 50.0
    assert report["selection"]["disjoint_from_used_episodes"] is True
    assert report["projection"]["minimum_new_non_dominant_pair_hours"] == 20.0
    assert report["projection"]["expected_new_pair_hours"] == 21.0
    assert report["planning_gate_passes"] is True
    assert report["training_ready"] is False
    assert all(row["episode_id"].endswith(tuple(f"new-{i}" for i in range(5))) for row in selected)
    assert all(row["channel"] != "Tabaghe16" for row in selected)
    assert report_path.is_file()


def test_balanced_supplement_plan_can_require_bound_local_audio(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    windows = tmp_path / "windows.jsonl"
    local_manifest = tmp_path / "youtube.jsonl"
    yield_report = tmp_path / "yield.json"
    selection = tmp_path / "selection.jsonl"
    report_path = tmp_path / "report.json"
    audio = tmp_path / "chunk.wav"
    write_wav(audio, np.zeros(16000, dtype=np.float32))
    inventory_rows = []
    local_rows = []
    for channel in ("Digiato", "Mehran Rowshan Persian", "Zoomit"):
        inventory_rows.append(
            {
                "episode_id": f"{channel}/used",
                "channel": channel,
                "csv_hours": 10.0,
                "selection_weight": 0.2,
                "selection_eligible": True,
            }
        )
        for index in range(2):
            source_csv = str(tmp_path / f"{channel}-{index}.csv")
            Path(source_csv).write_text(
                "start_time,end_time,text\n00:00:00,00:00:01,سلام دوست من\n",
                encoding="utf-8",
            )
            inventory_rows.append(
                {
                    "episode_id": f"{channel}/new-{index}",
                    "channel": channel,
                    "csv_path": source_csv,
                    "csv_hours": 10.0,
                    "selection_weight": 0.2,
                    "selection_eligible": True,
                }
            )
            if channel != "Mehran Rowshan Persian":
                local_rows.append({"source_csv": source_csv, "audio_filepath": str(audio)})
    inventory.write_text(
        "".join(json.dumps(row) + "\n" for row in inventory_rows), encoding="utf-8"
    )
    pairs.write_text(
        "".join(
            json.dumps({"session_id": f"{channel}/used", "duration": hours * 3600}) + "\n"
            for channel, hours in (
                ("Tabaghe16", 60.0),
                ("Digiato", 12.0),
                ("Mehran Rowshan Persian", 12.0),
                ("Zoomit", 6.0),
            )
        ),
        encoding="utf-8",
    )
    windows.write_text(
        "".join(
            json.dumps({"episode_id": f"{channel}/used"}) + "\n"
            for channel in ("Digiato", "Mehran Rowshan Persian", "Zoomit")
        ),
        encoding="utf-8",
    )
    local_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in local_rows), encoding="utf-8"
    )
    yield_report.write_text(
        json.dumps(
            {
                "source_manifest_sha256": hashlib.sha256(windows.read_bytes()).hexdigest(),
                "channel_hours": {
                    "Tabaghe16": 60.0,
                    "Digiato": 12.0,
                    "Mehran Rowshan Persian": 12.0,
                    "Zoomit": 6.0,
                },
            }
        ),
        encoding="utf-8",
    )

    report = plan_balanced_supplement(
        inventory,
        pairs,
        windows,
        selection,
        report_path,
        target_candidate_hours=20.0,
        max_candidate_hours=30.0,
        max_supplement_channel_share=0.6,
        expected_yield_safety_factor=1.0,
        current_yield_report_path=yield_report,
        local_chunk_manifest_path=local_manifest,
        root=tmp_path,
    )

    selected = [json.loads(line) for line in selection.read_text().splitlines()]
    assert {row["channel"] for row in selected} == {"Digiato", "Zoomit"}
    assert report["selection"]["all_episodes_have_local_chunks"] is True
    assert report["inputs"]["current_pair_hours_source"] == ("bound_non_mutating_yield_report")


def test_final_balance_selection_retains_minority_and_is_deterministic(tmp_path: Path):
    source = tmp_path / "pairs.jsonl"
    rows = []
    channel_counts = {
        "Tabaghe16": 70,
        "Digiato": 20,
        "Mehran Rowshan Persian": 20,
        "Zoomit": 10,
    }
    for channel, count in channel_counts.items():
        for index in range(count):
            rows.append(
                {
                    "utt_id": f"{channel}-{index}",
                    "session_id": f"{channel}/session-{index % 2}",
                    "channel": channel,
                    "duration": 3600.0,
                    "split": "train" if index % 2 == 0 else "val",
                }
            )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first = tmp_path / "balanced-1.jsonl"
    second = tmp_path / "balanced-2.jsonl"
    report = select_balanced_conversation_pairs(
        source, first, tmp_path / "report-1.json", root=tmp_path
    )
    repeated = select_balanced_conversation_pairs(
        source, second, tmp_path / "report-2.json", root=tmp_path
    )

    selected = [json.loads(line) for line in first.read_text().splitlines()]
    assert report["selection_gate_passes"] is True
    assert report["retained_all_minority_pairs"] is True
    assert report["selected_hours"] == 108.0
    assert report["largest_channel_share"] < 0.54
    assert report["invalid_split_rows"] == 0
    assert report["session_group_split_leaks"] == 0
    assert set(report["split_summary"]) == {"train", "val"}
    assert sum(row["channel"] != "Tabaghe16" for row in selected) == 50
    assert first.read_bytes() == second.read_bytes()
    assert report["out_manifest_sha256"] == repeated["out_manifest_sha256"]


def test_pair_manifest_merge_is_hash_bound_and_rejects_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    out = tmp_path / "merged.jsonl"
    report_path = tmp_path / "merge.json"
    base = {
        "duration": 10.0,
        "channel": "A",
        "split": "train",
    }
    write_jsonl(first, [{**base, "utt_id": "a", "session_id": "A/one"}])
    write_jsonl(
        second,
        [{**base, "utt_id": "b", "session_id": "A/two", "channel": "B"}],
    )

    report = merge_conversation_pair_manifests(
        [first, second], out, report_path, root=tmp_path
    )

    assert report["pairs"] == 2
    assert report["merge_gate_passes"] is True
    assert len(report["inputs"]) == 2
    assert len(load_jsonl(out)) == 2

    write_jsonl(second, [{**base, "utt_id": "a", "session_id": "A/two"}])
    with pytest.raises(ValueError, match="invalid row"):
        merge_conversation_pair_manifests(
            [first, second], out, report_path, root=tmp_path
        )


def test_jsonl_writer_is_atomic_on_serialization_failure(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    target.write_text('{"preserved": true}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        write_jsonl(target, [{"valid": True}, {"invalid": object()}])

    assert target.read_text(encoding="utf-8") == '{"preserved": true}\n'
    assert not (tmp_path / ".rows.jsonl.partial").exists()


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
    assert audit["requirements"]["minimum_channels_present"] is True
    assert audit["audit_contract"]["min_channels"] == 2
    strict_channels = audit_prepared_episode_windows(
        selection,
        manifest,
        stats_path=stats,
        min_hours=0,
        max_hours=1,
        min_channels=3,
    )
    assert strict_channels["requirements"]["minimum_channels_present"] is False
    assert strict_channels["reconstruction_gate_passes"] is False
    with pytest.raises(ValueError, match="min_channels"):
        audit_prepared_episode_windows(
            selection, manifest, stats_path=stats, min_hours=0, max_hours=1, min_channels=0
        )
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


def test_episode_preparation_prefers_bound_local_chunk_manifest(tmp_path: Path, monkeypatch):
    stem = "local episode"
    csv_path = tmp_path / f"{stem}.csv"
    csv_path.write_text(
        "start_time,end_time,text\n00:00:00,00:00:01,سلام دوست من\n",
        encoding="utf-8",
    )
    chunk = tmp_path / f"{stem}_chunk_0001.wav"
    write_wav(chunk, np.zeros(16000, dtype=np.float32))
    local_manifest = tmp_path / "youtube.jsonl"
    local_manifest.write_text(
        json.dumps(
            {"source_csv": str(csv_path), "audio_filepath": str(chunk)},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "episode_id": f"Digiato/{stem}",
                "stem": stem,
                "channel": "Digiato",
                "csv_path": str(csv_path),
                "csv_hours": 1 / 3600,
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def unexpected_remote_copy(*args, **kwargs):
        raise AssertionError("remote copy must not run when local chunks are complete")

    monkeypatch.setattr("thesis_s2s.data.episode_prepare.rclone_copy", unexpected_remote_copy)
    output = tmp_path / "windows.jsonl"
    report = prepare_selected_episodes(
        selection,
        tmp_path / "windows",
        output,
        local_chunk_manifest=local_manifest,
        resume=False,
    )

    assert report["episodes_prepared"] == 1
    assert report["episodes_prepared_from_local"] == 1
    assert report["episodes_prepared_from_remote"] == 0
    assert (
        report["local_chunk_manifest_sha256"]
        == hashlib.sha256(local_manifest.read_bytes()).hexdigest()
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["chunk_source"] == "local_manifest"


def test_v2_pair_selection_recovers_shifted_adjacent_turn_without_reuse(tmp_path: Path):
    source = tmp_path / "diarized.jsonl"
    source.write_text(
        json.dumps(
            {
                "window_id": "window-1",
                "channel": "Digiato",
                "automatic_multi_speaker_verified": True,
                "reference_alignment_status": "complete",
                "segments": [
                    {"start": 0.0, "end": 1.0, "speaker": "A", "text": "یک"},
                    {"start": 1.6, "end": 2.6, "speaker": "A", "text": "دو"},
                    {"start": 2.6, "end": 3.6, "speaker": "B", "text": "سه"},
                    {"start": 4.2, "end": 5.2, "speaker": "B", "text": "چهار"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    v1 = estimate_conversation_pair_yield(source)
    v2 = estimate_conversation_pair_yield(source, pair_selection_method="max_duration_interval_v2")

    assert v1["estimated_pairs"] == 0
    assert v2["estimated_pairs"] == 1
    assert v2["estimated_pair_hours"] > 0
    assert v2["candidate_rule"]["pair_selection_method"] == "max_duration_interval_v2"


def test_alternate_diarization_reports_do_not_overwrite_primary(tmp_path: Path):
    primary = tmp_path / "conversation_episode_windows_diarized.jsonl"
    reserve = tmp_path / "conversation_reserve_windows_diarized.jsonl"

    assert diarization_stats_path(primary).name == "conversation_episode_diarization_stats.json"
    assert (
        diarization_stats_path(reserve).name == "conversation_reserve_windows_diarized_stats.json"
    )
    assert diarization_stats_path(primary) != diarization_stats_path(reserve)


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
    assert audit["audit_contract"]["is_thesis_standard_100_to_200"] is False
    assert audit["thesis_evidence_gate_passes"] is False
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
    progress = json.loads(diarization_stats_path(output).read_text(encoding="utf-8"))
    assert progress["in_progress"] is False
    assert progress["output_windows"] == 1
    assert progress["source_manifest_complete"] is True


def test_known_diarization_overlap_is_not_mislabeled_as_an_interruption():
    user = SpeakerSegment(0.0, 0.7, "A", "پرسش")
    assistant = SpeakerSegment(1.0, 2.0, "B", "پاسخ")
    label, overlap = _interaction_label(user, assistant, [[0.8, 0.9]])
    assert label == "overlap_unattributed"
    assert overlap == [[0.8, 0.9]]


def test_response_turn_overlap_can_label_an_interruption():
    user = SpeakerSegment(0.0, 1.5, "A", "پرسش")
    assistant = SpeakerSegment(1.0, 2.5, "B", "پاسخ بلند")
    label, overlap = _interaction_label(user, assistant)
    assert label == "interrupt"
    assert overlap == [[1.0, 1.5]]


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
    with pytest.raises(FileExistsError, match="reviewer data"):
        sample_manual_qa(source, qa_csv, per_stratum=1)

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


def _interaction_window(window_id: str, channel: str, audio_path: str) -> dict:
    return {
        "window_id": window_id,
        "episode_id": f"{channel}/episode",
        "session_id": f"{channel}/episode",
        "channel": channel,
        "audio_filepath": audio_path,
        "duration": 3.0,
        "diarization_status": "complete",
        "automatic_multi_speaker_verified": True,
        "reference_alignment_status": "complete",
        "segments": [
            {"start": 0.0, "end": 1.0, "speaker": "A", "text": "پرسش گوینده"},
            {"start": 1.0, "end": 2.1, "speaker": "B", "text": "پاسخ مهمان"},
        ],
        "speaker_turns": [
            {"start": 0.0, "end": 1.4, "speaker": "A"},
            {"start": 0.8, "end": 2.2, "speaker": "B"},
        ],
        "overlap_intervals": [[0.8, 1.4]],
        "license": "CC-BY-4.0",
        "license_verified": True,
        "internal_research_authorized": False,
        "authorization_basis": "explicit-source-license",
        "redistribution_allowed": True,
    }


def test_raw_boundary_candidates_are_separate_from_verified_labels():
    row = _interaction_window("window-1", "Tabaghe16", "/internal/window.wav")
    user = SpeakerSegment(0.0, 1.0, "A", "پرسش گوینده")
    response = SpeakerSegment(1.0, 2.1, "B", "پاسخ مهمان")

    candidate = _automatic_interaction_candidate(row, user, response)

    assert candidate is not None
    assert candidate["automatic_label"] == "interrupt"
    assert candidate["overlap_seconds"] == 0.6
    assert candidate["human_verified"] is False
    assert candidate["raw_user_interval"] == [0.0, 1.4]
    assert candidate["raw_response_interval"] == [0.8, 2.2]


def test_interaction_qa_is_balanced_tamper_resistant_and_fail_closed(tmp_path: Path):
    source = tmp_path / "windows.jsonl"
    rows = [
        _interaction_window("tabaghe-1", "Tabaghe16", "/internal/tabaghe.wav"),
        _interaction_window("zoomit-1", "Zoomit", "/internal/zoomit.wav"),
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    qa_csv = tmp_path / "interaction-qa.csv"
    sampled = sample_interaction_qa(source, qa_csv, per_channel=1)
    assert sampled["automatic_candidates"] == 2
    assert sampled["sampled_per_channel"] == {"Tabaghe16": 1, "Zoomit": 1}
    assert sampled["new_recordings_required"] is False

    with qa_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        decisions = list(csv.DictReader(handle))
        fieldnames = list(decisions[0])
    for decision in decisions:
        decision.update(
            {
                "review_status": "pass",
                "speakers_distinct_correct": "yes",
                "user_turn_boundary_correct": "yes",
                "response_turn_boundary_correct": "yes",
                "audible_overlap_correct": "yes",
                "corrected_label": "interrupt",
                "reviewer_id": "reviewer-1",
                "overlap_seconds": "999",
            }
        )
    decisions[0]["audible_overlap_correct"] = "no"
    with qa_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)
    with pytest.raises(FileExistsError, match="reviewer data"):
        sample_interaction_qa(source, qa_csv, per_channel=1)

    reviewed = tmp_path / "reviewed.jsonl"
    report = apply_interaction_qa(source, qa_csv, reviewed)
    assert report["counts"]["approved"] == 1
    assert report["counts"]["rejected"] == 1
    reviewed_rows = [json.loads(line) for line in reviewed.read_text().splitlines()]
    reviews = [review for row in reviewed_rows for review in row["interaction_reviews"]]
    assert sum(review["human_verified"] is True for review in reviews) == 1
    assert all(review["automatic_label"] == "interrupt" for review in reviews)


def test_reviewed_interaction_retimes_export_and_audit(tmp_path: Path):
    audio_path = tmp_path / "conversation.wav"
    write_wav(audio_path, np.zeros(3 * 16000, dtype=np.float32))
    row = _interaction_window("window-1", "Tabaghe16", str(audio_path))
    user = SpeakerSegment(0.0, 1.0, "A", "پرسش گوینده")
    response = SpeakerSegment(1.0, 2.1, "B", "پاسخ مهمان")
    candidate = _automatic_interaction_candidate(row, user, response)
    assert candidate is not None
    row["interaction_reviews"] = [
        {
            "candidate_id": candidate["candidate_id"],
            "label": "interrupt",
            "human_verified": True,
            "reviewer_id": "reviewer-1",
        }
    ]
    source = tmp_path / "reviewed-windows.jsonl"
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = tmp_path / "pairs.jsonl"

    report = build_conversation_manifest(source, manifest, tmp_path / "clips")

    assert report["human_verified_interaction_counts"] == {"interrupt": 1}
    pair = json.loads(manifest.read_text(encoding="utf-8"))
    assert pair["source_user_interval"] == [0.0, 1.4]
    assert pair["source_response_interval"] == [0.8, 2.2]
    assert pair["human_verified_interaction_label"] is True
    audit = audit_conversation_manifest(manifest, check_files=True)
    assert audit["requirements"]["interruptions_present"] is True
    assert audit["human_verified_interaction_label_counts"] == {"interrupt": 1}


def test_unverified_interrupt_label_does_not_pass_interruption_audit(tmp_path: Path):
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "session_id": "s1",
                "speaker_id": "s1:A",
                "assistant_speaker_id": "s1:B",
                "duration": 360000.0,
                "interrupt_label": "interrupt",
                "human_verified_interaction_label": False,
                "response_audio_filepath": "response.wav",
                "assistant_text": "پاسخ",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_conversation_manifest(manifest, check_files=False)

    assert report["requirements"]["interruptions_present"] is False
    assert report["label_counts"] == {"interrupt": 1}
    assert report["human_verified_interaction_label_counts"] == {}
