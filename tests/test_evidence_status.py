from __future__ import annotations

import json
from pathlib import Path

from thesis_s2s.eval.evidence import build_evidence_status, render_evidence_summary


def _write(root: Path, relative: str, payload: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evidence_status_aggregates_current_artifacts_fail_closed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "results/conversation_audit.json",
        {"pairs": 6754, "hours": 123.796},
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
            "complete_ratings": 0,
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

    out = tmp_path / "results/eval/EVIDENCE_STATUS.json"
    report = build_evidence_status(
        out,
        root=tmp_path,
        generated_at="2026-08-30T00:00:00+00:00",
    )

    assert report["authoritative"] is True
    assert report["schema_version"] == 6
    assert report["thesis_ready"] is False
    assert report["generation_policy"]["reads_frozen_final_test_rows"] is False
    assert report["gates"]["audited_export_100_to_200_hours"] is True
    assert report["gates"]["data_policy_resolved_under_documented_qa_waiver"] is True
    assert report["data"]["strict_human_qa_complete"] is False
    assert report["data"]["exported_hours"] == 108.584
    assert report["direct_moshi"]["deployment_eligible"] is False
    assert report["direct_moshi"]["trials"][3]["runtime_panel_pass_counts"] == [1, 0]
    assert report["direct_moshi"]["trials"][4]["runtime_panel_pass_counts"] == [0, 1]
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
    assert report["reporting_contract"]["failure_denominators_included"] is True
    assert report["release"]["source_code_license_selected"] is False
    assert report["remaining_work_classification"]["waived_not_completed"]
    summary = render_evidence_summary(report)
    assert "## Local artifact retention" in summary
    assert "13 representative" in summary
    assert "recorded proxy n=132 / 22 sessions" in summary
    assert json.loads(out.read_text(encoding="utf-8")) == report
