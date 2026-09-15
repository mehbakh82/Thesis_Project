from __future__ import annotations

import json
from pathlib import Path

import pytest

from thesis_s2s.eval.evidence import (
    build_evidence_status,
    render_evidence_summary,
    write_evidence_summary,
)
from thesis_s2s.repro import sha256_file


def _write(root: Path, relative: str, payload: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(root: Path, relative: str, payload: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_evidence_summary_write_is_atomic(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "SUMMARY.md"
    payload = {"generated_at": "now", "gates": {}, "required_to_complete": []}
    write_evidence_summary(path, payload)
    expected = render_evidence_summary(payload)
    assert path.read_text(encoding="utf-8") == expected
    assert list(tmp_path.glob(".SUMMARY.md.*.tmp")) == []

    old_bytes = path.read_bytes()
    monkeypatch.setattr(
        "thesis_s2s.eval.evidence.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        write_evidence_summary(path, payload)
    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob(".SUMMARY.md.*.tmp")) == []


def test_evidence_status_aggregates_current_artifacts_fail_closed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "results/conversation_audit.json",
        {"pairs": 6754, "hours": 123.796},
    )
    _write(
        tmp_path,
        "results/conversation_balanced_v2_audit.json",
        {
            "pairs": 6551,
            "hours": 110.374,
            "sessions": 186,
            "speakers": 480,
            "training_ready_under_qa_waiver": True,
            "missing_files": 0,
            "reused_source_spans": 0,
            "session_group_split_leaks": 0,
            "human_verified_rows": 0,
        },
    )
    _write(
        tmp_path,
        "results/conversation_balance_v2_audit.json",
        {
            "totals": {"channels": 4},
            "dominance": {"hour_share": 0.54},
            "strict_representative_balance_passes": True,
        },
    )
    _write(
        tmp_path,
        "results/moshi_export_report.json",
        {
            "training_ready_under_qa_waiver": True,
            "final_training_ready": False,
            "source_rows_permitting_redistribution": 0,
            "requirements": {"training_hours_100_to_200": True},
            "claims": {"human_verified_data": False},
        },
    )
    _write(
        tmp_path,
        "results/moshi_export_audit.json",
        {
            "verified_export_pairs": 6754,
            "verified_export_hours": 108.584,
            "split_counts": {"train": 6419, "val": 131, "test": 204},
            "group_split_leaks": 0,
            "audit_passes": True,
            "training_ready_under_qa_waiver": True,
            "strict_thesis_data_coverage": False,
            "sample": {"human_review_complete": False},
            "requirements": {"session_group_splits_isolated": True},
        },
    )
    _write(
        tmp_path,
        "results/hardware/moshi_h100_training.json",
        {"training_run_passes": True, "checkpoint_count": 8, "selected": {"step": 8000}},
    )
    for version, pass_counts in (
        ("v2", [0, 0]),
        ("v3", [0]),
        ("v4", [1, 0]),
        ("v5", [0, 1]),
    ):
        _write(
            tmp_path,
            f"results/moshi_{version}_checkpoint_selection.json",
            {
                "training_complete": True,
                "candidate_count": len(pass_counts),
                "candidates": [
                    {
                        "eval_loss": 2.0 - index / 10,
                        "runtime_panel_pass_count": count,
                        "runtime_panel_count": 9,
                    }
                    for index, count in enumerate(pass_counts)
                ],
                "eligible_steps": [],
                "selected": None,
            },
        )
    _write(
        tmp_path,
        "results/hardware/moshi_v2_negative_selection_finalization.json",
        {"scientific_negative_outcome_finalized": True, "test_access_started": False},
    )
    _write(
        tmp_path,
        "results/hardware/moshi_h100_v3_training.json",
        {"training_run_passes": True, "test_access_started": False},
    )
    _write(
        tmp_path,
        "results/hardware/moshi_h100_v4_training.json",
        {"training_run_passes": True, "test_access_started": False},
    )
    _write(
        tmp_path,
        "results/hardware/moshi_h100_v5_training.json",
        {"training_run_passes": True, "test_access_started": False},
    )
    _write(
        tmp_path,
        "configs/moshi_h100_v6_text_dropout.yaml",
        {
            "duration_sec": 12,
            "max_steps": 200,
            "seed": 20260901,
            "lora": {"rank": 64, "ft_embed": False},
        },
    )
    _write(
        tmp_path,
        "results/hardware/moshi_v6_text_dropout_training.json",
        {
            "status": "passed",
            "training_passes": True,
            "final_test_accessed": False,
            "training": {"peak_allocated_gb": 16.5769},
            "checkpoints": [
                {"full_parameters": ["text_emb.weight", "depformer_text_emb.weight"]}
            ],
        },
    )
    _write(
        tmp_path,
        "results/moshi_v6_text_dropout_reevaluation.json",
        {
            "status": "passed",
            "reevaluation_passes": True,
            "final_test_opened": False,
            "candidates": [
                {
                    "eval_loss": total,
                    "text_eval_loss": text,
                    "audio_eval_loss": total - text,
                    "sample_count": 32,
                }
                for total, text in (
                    (2.6364, 0.8124),
                    (2.2453, 0.6390),
                    (2.0715, 0.5508),
                    (2.0603, 0.5512),
                )
            ],
        },
    )
    _write(
        tmp_path,
        "results/moshi_v6_text_dropout_runtime.json",
        {
            "status": "passed",
            "diagnostic_outcome": "negative",
            "selection": None,
            "final_test_accessed": False,
            "requirements": {"all_candidates_evaluated": True},
            "candidates": [
                {"panel_count": 9, "panel_pass_count": 0} for _ in range(4)
            ],
        },
    )
    _write(
        tmp_path,
        "results/eval/cascade_real_service_validation_panel.json",
        {
            "status": "passed",
            "evidence_class": "real_service_group_disjoint_validation_mechanics",
            "claim": {
                "positive_working_user_turn_result": True,
                "group_disjoint_validation_execution": True,
                "scientific_generalization_result": False,
                "official_end_to_end_latency_result": False,
            },
            "requirements": {"all_nine_samples_pass": True},
            "panel": {
                "selection": "fixed validation panel",
                "split_unit": "source_session_id",
                "selected_user_channel": 1,
                "group_split_leaks": 0,
            },
            "samples": [
                {
                    "passes": True,
                    "responder_fallback_used": False,
                    "responder_language_retry_used": index == 8,
                    "reply_audio_rms": 0.1 + index / 100,
                    "reply_statistics": {"persian_letter_fraction": 1.0},
                }
                for index in range(9)
            ],
        },
    )
    cascade_sha256 = sha256_file(
        tmp_path / "results/eval/cascade_real_service_validation_panel.json"
    )
    _write(
        tmp_path,
        "results/eval/cascade_validation_descriptive_analysis.json",
        {
            "status": "complete",
            "analysis_scope": "post_hoc_privacy_safe_descriptive_error_analysis",
            "source_status": "passed",
            "source_report_sha256": cascade_sha256,
            "panel_rows": 9,
            "panel_passed_rows": 9,
            "panel_failed_rows": 0,
            "report_gate_failures": [],
            "sample_requirement_failure_counts": {},
            "analysis_integrity": {
                "all_required_measurements_and_hashes_present": True
            },
            "execution_outcomes": {
                "asr_error_rows": 0,
                "responder_fallback_rows": 0,
                "language_retry_rows": 1,
                "unique_transcript_hashes": 9,
                "unique_reply_hashes": 9,
                "duplicate_transcript_rows": 0,
                "duplicate_reply_rows": 0,
            },
            "distributions": {
                "full_turn_generation_ms": {"p50": 5500.0, "p95": 13000.0}
            },
            "descriptive_risk_counts": {
                "transcript_over_1000_characters": 3,
                "reply_audio_over_8_seconds": 3,
                "full_turn_over_10_seconds": 2,
            },
            "row_extremes": {},
            "transcript_length_full_turn_pearson_r": 0.885282,
            "interpretation": {"semantic_relevance": "not_measured"},
            "claim_boundary": {
                "plaintext_transcripts_or_replies_read_or_emitted": False,
                "final_test_accessed": False,
                "scientific_generalization_claim_allowed": False,
                "semantic_quality_claim_allowed": False,
                "human_quality_claim_allowed": False,
                "official_latency_claim_allowed": False,
            },
        },
    )
    _write_text(
        tmp_path,
        "docs/CASCADE_INTELLIGIBILITY_PROTOCOL.md",
        "frozen automatic round-trip intelligibility protocol\n",
    )
    intelligibility_protocol_sha256 = sha256_file(
        tmp_path / "docs/CASCADE_INTELLIGIBILITY_PROTOCOL.md"
    )
    intelligibility_samples = [
        {
            "input_asr_error": None,
            "roundtrip_asr_error": None,
            "matches_parent_input_transcript": True,
            "matches_parent_reply": True,
            "responder_backend": "Qwen/Qwen2.5-0.5B-Instruct",
            "responder_fallback_used": False,
            "tts_backend": "piper",
        }
        for _ in range(9)
    ]
    _write(
        tmp_path,
        "results/eval/cascade_intelligibility_proxy.json",
        {
            "status": "valid",
            "evidence_class": "automatic_asr_roundtrip_intelligibility_proxy",
            "claim": {
                "measurement_valid": True,
                "automatic_intelligibility_proxy_measured": True,
                "human_intelligibility_result": False,
                "pronunciation_or_naturalness_result": False,
                "semantic_relevance_result": False,
                "population_generalization_result": False,
            },
            "panel": {"sample_count": 9, "final_test_accessed": False},
            "aggregate": {
                "rows": 9,
                "asr_failures": 0,
                "exact_word_match_rows": 0,
                "exact_character_match_rows": 0,
                "word_error_rate": {
                    "micro": 0.304094,
                    "reference_words": 171,
                    "edits": 52,
                },
                "character_error_rate": {
                    "micro": 0.071429,
                    "reference_characters": 658,
                    "edits": 47,
                },
            },
            "threshold": None,
            "samples": intelligibility_samples,
            "validity_requirements": {
                "parent_panel_passed": True,
                "all_reply_text_reproduced": True,
                "all_roundtrip_asr_calls_succeeded": True,
                "final_test_not_accessed": True,
            },
            "artifacts": {
                "parent_panel_sha256": cascade_sha256,
                "protocol_sha256": intelligibility_protocol_sha256,
            },
            "privacy": {
                "plaintext_input_transcripts_stored": False,
                "plaintext_reply_text_stored": False,
                "plaintext_roundtrip_transcripts_stored": False,
                "audio_stored": False,
            },
            "limitations": ["automatic proxy only"],
        },
    )
    _write(
        tmp_path,
        "results/eval/interrupt_bench.json",
        {
            "evidence_class": "synthetic_proxy",
            "proposed": {"accuracy": 1.0, "n": 32},
            "recorded_eval": False,
            "official_detector_eligible": False,
            "n_recorded": 0,
        },
    )
    _write(
        tmp_path,
        "results/eval/interrupt_recorded_proxy.json",
        {
            "evidence_class": "recorded_youtube_diarization_proxy",
            "recorded_audio": True,
            "label_source": "automatic_diarization_and_reference_alignment_proxy",
            "human_verified_labels": 0,
            "official_detector_eligible": False,
            "heldout_test": {
                "n": 132,
                "sessions": 22,
                "proposed": {
                    "accuracy": 0.8106,
                    "interrupt_f1": 0.7899,
                    "far": 0.0909,
                    "frr": 0.2879,
                },
                "proposed_event_ci95": {"accuracy": [0.7353, 0.8683]},
                "proposed_session_block_bootstrap_ci95": {"accuracy": [0.7444, 0.8718]},
                "recorded_proxy_accuracy_above_80_percent": True,
                "official_target_satisfied": False,
            },
        },
    )
    _write(
        tmp_path,
        "results/eval/human_study.json",
        {
            "status": "not_collected_or_incomplete",
            "participants": 1,
            "elderly_participants": 1,
            "turns": 0,
            "ratings": 0,
            "complete_ratings": 0,
            "invalid_rating_rows": 1,
            "requirements": {
                "participants_5_to_10": False,
                "elderly_participants_at_least_2": False,
                "complete_ratings_cover_participants": False,
                "turn_rows_valid": False,
                "rating_rows_valid": False,
                "eligible_client_first_audio_present": False,
                "eligible_client_barge_in_present": False,
                "physical_gpu_12_to_24_gb": False,
                "real_heldout_detector_report": False,
            },
            "official_ready": False,
        },
    )
    _write(
        tmp_path,
        "results/hardware/current_preflight.json",
        {
            "is_rtx_4090": False,
            "evaluation_hardware_ready": False,
            "gpu": {"device": "H100 NVL", "total_gb": 93.09, "official_size": False},
        },
    )
    _write(
        tmp_path,
        "results/eval/latency_bench.json",
        {
            "official_e2e_eligible": True,
            "evidence_class": "server_component",
            "n": 99,
        },
    )
    _write(
        tmp_path,
        "results/eval/latency_bench_path_b.json",
        {
            "official_e2e_eligible": True,
            "evidence_class": "official_e2e",
            "consented": True,
            "live_browser": True,
            "physical_gpu_12_to_24_gb": True,
            "client_playback_acknowledgements": False,
            "n": 99,
        },
    )

    _write(
        tmp_path,
        "results/hardware/storage_cleanup_20260831.json",
        {
            "status": "passed",
            "space": {
                "observed_phase_reclaimed_bytes": 40_718_569_472,
                "project_du_after": "119G",
            },
            "removed": [{"category": "non_promoted_negative_checkpoint_intermediates"}],
            "retained_representative_adapters": {
                "v1": {"steps": [500, 1000, 2000, 4000, 8000]},
                "v2": {"steps": [400, 2000]},
                "v3": {"steps": [400, 500]},
                "v4": {"steps": [400, 500]},
                "v5": {"steps": [400, 500]},
                "count": 13,
                "all_sha256_match_committed_evidence": True,
            },
            "protected_artifacts_untouched": ["all tracked results and certificates"],
            "postconditions": {"all_retained_adapter_hashes_verified": True},
        },
    )
    _write(
        tmp_path,
        "results/release/final_audit.json",
        {
            "status": "passed",
            "duplex_transport_regression": {
                "tests": 4,
                "automatic_bargein_preroll_samples": 3200,
                "continued_capture_samples": 1600,
                "next_turn_input_samples": 4800,
                "client_acknowledgement_ingestion": "passed",
                "identity_and_fallback_telemetry": "passed",
                "explicit_cancellation": "passed",
                "simultaneous_turn_isolation": "passed",
                "reconnect": "passed",
                "malformed_and_silent_input_recovery": "passed",
                "simulated_generation_oom_recovery": "passed",
                "metrics_retention_writes_no_wav": "passed",
                "evidence_class": "automated_websocket_transport_regression",
            },
            "claim_boundary": {
                "does_not_prove": ["Actual browser AudioBufferSourceNode stop behavior."]
            },
        },
    )
    release_commit = "a" * 40
    _write(
        tmp_path,
        "results/release/submission_tag_attestation.json",
        {
            "schema_version": 1,
            "tag": "submission-test",
            "tag_object_type": "annotated_tag",
            "tag_object": "b" * 40,
            "commit": release_commit,
            "remote": "https://github.com/mehbakh82/Thesis_Project.git",
            "branch_ci": {
                "run_id": 1,
                "head_sha": release_commit,
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "url": "https://github.com/mehbakh82/Thesis_Project/actions/runs/1",
                "updated_at": "2026-09-10T08:00:00Z",
            },
            "tag_ci": {
                "run_id": 2,
                "head_sha": release_commit,
                "head_branch": "submission-test",
                "status": "completed",
                "conclusion": "success",
                "url": "https://github.com/mehbakh82/Thesis_Project/actions/runs/2",
                "updated_at": "2026-09-10T08:00:01Z",
            },
            "verified_at": "2026-09-10T08:01:49Z",
        },
    )

    out = tmp_path / "results/eval/EVIDENCE_STATUS.json"
    report = build_evidence_status(
        out,
        root=tmp_path,
        generated_at="2026-08-30T00:00:00+00:00",
    )

    assert report["authoritative"] is True
    assert report["schema_version"] == 12
    assert report["thesis_ready"] is False
    assert report["generation_policy"]["reads_frozen_final_test_rows"] is False
    assert report["gates"]["audited_export_100_to_200_hours"] is True
    assert report["gates"]["working_persian_s2s_prototype"] is True
    assert report["gates"]["automatic_semantic_final_test_passed"] is False
    assert report["working_system"]["passed_rows"] == 9
    assert report["working_system"]["fallback_rows"] == 0
    assert report["working_system"]["input_channel"] == 1
    assert report["cascade_descriptive_analysis"]["verified"] is True
    assert report["cascade_descriptive_analysis"]["descriptive_risk_counts"] == {
        "transcript_over_1000_characters": 3,
        "reply_audio_over_8_seconds": 3,
        "full_turn_over_10_seconds": 2,
    }
    assert report["cascade_intelligibility_proxy"]["verified"] is True
    assert report["cascade_intelligibility_proxy"]["word_error_rate"]["micro"] == 0.304094
    assert (
        report["cascade_intelligibility_proxy"]["character_error_rate"]["micro"]
        == 0.071429
    )
    assert report["gates"]["data_policy_resolved_under_documented_qa_waiver"] is True
    assert report["data"]["strict_human_qa_complete"] is False
    assert report["data"]["exported_hours"] == 108.584
    assert report["data"]["balanced_v2"] == {
        "evidence_scope": "post_training_recommended_corpus",
        "used_by_existing_moshi_runs": False,
        "pairs": 6551,
        "source_pair_hours": 110.374,
        "sessions": 186,
        "speakers": 480,
        "channels": 4,
        "largest_channel_share": 0.54,
        "balance_gate_passed": True,
        "training_ready_under_documented_qa_waiver": True,
        "missing_files": 0,
        "reused_source_spans": 0,
        "session_group_split_leaks": 0,
        "human_review_complete": False,
    }
    assert report["direct_moshi"]["deployment_eligible"] is False
    assert report["direct_moshi"]["trials"][3]["runtime_panel_pass_counts"] == [1, 0]
    assert report["direct_moshi"]["trials"][4]["runtime_panel_pass_counts"] == [0, 1]
    assert report["direct_moshi"]["trials"][5]["runtime_panel_pass_counts"] == [0, 0, 0, 0]
    assert report["direct_moshi"]["trials"][5]["positive_learning_result"] is True
    assert report["direct_moshi"]["trials"][5]["deployment_eligible"] is False
    assert report["direct_moshi"]["v2_to_v5_final_test_access_started"] is False
    assert report["local_artifact_retention"]["cleanup_passed"] is True
    assert report["local_artifact_retention"]["representative_adapter_count"] == 13
    assert report["local_artifact_retention"]["full_candidate_tensor_sets_retained"] is False
    assert (
        report["local_artifact_retention"]["scientific_results_configs_certificates_retained"]
        is True
    )
    assert report["latency_and_hardware"]["official_e2e_rows"] == 0
    assert report["detector"]["synthetic_accuracy_ci95_wilson"] == [0.8928, 1.0]
    assert report["detector"]["synthetic_failures"] == 0
    assert report["detector"]["recorded_proxy"]["n"] == 132
    assert report["detector"]["recorded_proxy"]["sessions"] == 22
    assert report["detector"]["recorded_proxy"]["accuracy_above_80_percent"] is True
    assert report["detector"]["recorded_proxy"]["official_detector_eligible"] is False
    assert report["gates"]["real_group_heldout_detector_above_80_percent"] is False
    assert report["human_study"]["turns"] == 0
    assert report["human_study"]["ratings"] == 0
    assert report["human_study"]["invalid_rating_rows"] == 1
    assert report["human_study"]["requirements"]["rating_rows_valid"] is False
    assert report["human_study"]["official_ready"] is False
    assert report["reporting_contract"]["failure_denominators_included"] is True
    assert report["duplex_transport"]["automated_websocket_regression_passed"] is True
    assert report["duplex_transport"]["next_turn_input_samples"] == 4800
    assert report["duplex_transport"]["physical_browser_verified"] is False
    assert report["submission_strategy"]["production_candidate"] == "cascade"
    assert report["submission_strategy"]["new_direct_training_before_deadline_recommended"] is False
    assert report["release"]["source_code_license_selected"] is False
    assert report["release"]["immutable_submission_tag_created"] is True
    assert report["release"]["submission_tag_attestation"]["commit"] == release_commit
    assert report["remaining_work_classification"]["waived_not_completed"]
    summary = render_evidence_summary(report)
    assert "## Local artifact retention" in summary
    assert "13 representative" in summary
    assert "passed 9/9" in summary
    assert "32.20%" in summary
    assert "recorded proxy n=132 / 22 sessions" in summary
    assert "turns=0, valid ratings=0, complete ratings=0, invalid rows=1" in summary
    assert "| Immutable release | submission-test | pass |" in summary
    assert "## Duplex transport" in summary
    assert "## Privacy-safe descriptive error analysis" in summary
    assert "descriptive Pearson r=0.885282" in summary
    assert "## Automatic synthesized-speech intelligibility proxy" in summary
    assert "Micro WER=0.304094" in summary
    assert "micro CER=0.071429" in summary
    assert "3,200 pre-roll samples" in summary
    assert json.loads(out.read_text(encoding="utf-8")) == report


def test_evidence_status_rejects_unsubstantiated_human_study_ready(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "results/eval/human_study.json",
        {
            "status": "complete",
            "participants": 5,
            "elderly_participants": 2,
            "complete_ratings": 5,
            "official_ready": True,
        },
    )

    report = build_evidence_status(root=tmp_path)

    assert report["human_study"]["official_ready"] is False
    assert report["gates"]["human_study_complete"] is False
    assert not any(report["human_study"]["requirements"].values())


def _valid_branch_protection_receipt() -> dict:
    return {
        "schema_version": 2,
        "repository": "mehbakh82/Thesis_Project",
        "repository_visibility": "public",
        "write_capable_collaborators": ["mehbakh82"],
        "branch": "main",
        "source": "authenticated_github_rest_api",
        "observed_at": "2026-09-15T05:28:42Z",
        "api_status": 200,
        "enabled": True,
        "pull_request_required": True,
        "required_status_checks": {"strict": True, "contexts": ["test"]},
        "required_pull_request_reviews": {
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": False,
            "require_last_push_approval": False,
            "required_approving_review_count": 0,
        },
        "enforce_admins": True,
        "required_linear_history": True,
        "required_conversation_resolution": True,
        "allow_force_pushes": False,
        "allow_deletions": False,
    }


def test_evidence_status_accepts_verified_main_branch_protection(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "results/release/branch_protection.json",
        _valid_branch_protection_receipt(),
    )

    report = build_evidence_status(root=tmp_path)

    assert report["release"]["main_branch_protection"]["verified"] is True
    assert report["remaining_work_classification"][
        "platform_limited_not_a_scientific_gate"
    ] == []
    assert "| Protected main | mandatory PR + strict CI" in render_evidence_summary(report)


def test_evidence_status_rejects_weakened_main_branch_protection(tmp_path: Path) -> None:
    receipt = _valid_branch_protection_receipt()
    receipt["required_status_checks"]["strict"] = False
    _write(tmp_path, "results/release/branch_protection.json", receipt)

    report = build_evidence_status(root=tmp_path)

    assert report["release"]["main_branch_protection"]["verified"] is False
    assert report["remaining_work_classification"][
        "platform_limited_not_a_scientific_gate"
    ] == ["Main-branch protection has not been verified from the GitHub API."]


def test_evidence_status_rejects_solo_governance_policy_drift(
    tmp_path: Path,
) -> None:
    receipt = _valid_branch_protection_receipt()
    receipt["required_pull_request_reviews"]["required_approving_review_count"] = 1
    _write(tmp_path, "results/release/branch_protection.json", receipt)

    report = build_evidence_status(root=tmp_path)

    assert report["release"]["main_branch_protection"]["verified"] is False


def test_evidence_status_rejects_mismatched_release_attestation(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "results/release/submission_tag_attestation.json",
        {
            "schema_version": 1,
            "tag": "submission-test",
            "tag_object_type": "annotated_tag",
            "tag_object": "b" * 40,
            "commit": "a" * 40,
            "branch_ci": {
                "head_sha": "a" * 40,
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
            },
            "tag_ci": {
                "head_sha": "c" * 40,
                "head_branch": "submission-test",
                "status": "completed",
                "conclusion": "success",
            },
            "verified_at": "2026-09-10T08:01:49Z",
        },
    )

    report = build_evidence_status(root=tmp_path)

    assert report["release"]["immutable_submission_tag_created"] is False
    assert report["release"]["submission_tag_attestation"]["verified"] is False


def test_evidence_status_rejects_malformed_release_ci_receipt(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "results/release/submission_tag_attestation.json",
        {
            "schema_version": 1,
            "tag": "submission-test",
            "tag_object_type": "annotated_tag",
            "tag_object": "b" * 40,
            "commit": "a" * 40,
            "remote": "https://github.com/mehbakh82/Thesis_Project.git",
            "branch_ci": ["not", "an", "object"],
            "tag_ci": "not an object",
            "verified_at": "2026-09-10T08:01:49Z",
        },
    )

    report = build_evidence_status(root=tmp_path)

    assert report["release"]["immutable_submission_tag_created"] is False
    assert report["release"]["submission_tag_attestation"]["branch_ci"] == {}
    assert report["release"]["submission_tag_attestation"]["tag_ci"] == {}


def test_committed_qwen4b_final_test_is_verified_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    report = build_evidence_status(root=root, generated_at="2026-09-06T00:00:00+00:00")
    final = report["qwen4b_v2_automatic_final_test"]

    assert report["gates"]["automatic_semantic_final_test_passed"] is True
    assert report["submission_strategy"]["production_candidate"] == "qwen4b_v2_cascade"
    assert final["verified"] is True
    assert final["rows"] == 40
    assert final["judge_calls_valid"] == 240
    assert final["judge_calls_failed"] == 0
    assert final["base_relevance_mean"] == 1.425
    assert final["candidate_relevance_mean"] == 3.15
    assert final["candidate_coherence_mean"] == 3.325
    assert final["relevance_gain"] == 1.725
    assert final["coherence_gain"] == 1.8
    assert final["relevance_win_rate"] == 0.675
    assert final["relevance_at_least_two_rate"] == 0.875
    assert final["human_semantic_result"] is False


def test_persian_thesis_reporting_tracks_authoritative_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (root / "results/eval/EVIDENCE_STATUS.json").read_text(encoding="utf-8")
    )
    manuscript = (root / "docs/THESIS_REPORTING_FA.md").read_text(encoding="utf-8")
    to_persian = str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫")

    exported_hours = f"{report['data']['exported_hours']:.3f}".translate(to_persian)
    cer_percent = (
        f"{100 * report['cascade_intelligibility_proxy']['character_error_rate']['micro']:.2f}"
        .translate(to_persian)
    )
    wer_percent = (
        f"{100 * report['cascade_intelligibility_proxy']['word_error_rate']['micro']:.2f}"
        .translate(to_persian)
    )
    recorded_accuracy = (
        f"{100 * report['detector']['recorded_proxy']['accuracy']:.2f}".translate(
            to_persian
        )
    )

    assert exported_hours in manuscript
    assert f"CER تجمیعی بازبازشناسی | {cer_percent}٪" in manuscript
    assert f"WER تجمیعی بازبازشناسی | {wer_percent}٪" in manuscript
    assert f"دقت برابر {recorded_accuracy}٪" in manuscript
    assert report["gates"]["working_persian_s2s_prototype"] is True
    assert report["gates"]["audited_export_100_to_200_hours"] is True
    assert report["gates"]["deployment_eligible_persian_direct_model"] is False
    assert report["gates"]["real_group_heldout_detector_above_80_percent"] is False
    assert report["gates"]["physical_12_to_24_gb_fit_and_live_latency"] is False
    assert report["gates"]["human_study_complete"] is False
    assert report["gates"]["source_code_license_selected"] is True
    assert "Moshika با موفقیت برای فارسی تنظیم شد" in manuscript
    assert "تأخیر سامانه کمتر از ۵۰۰ میلی‌ثانیه است" in manuscript
    assert "مطالعه انسانی و ارزیابی اختصاصی سالمندان انجام نشد" in manuscript
