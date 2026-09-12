"""Fail-closed aggregation of the project's thesis evidence.

This module reads only committed aggregate artifacts. It never discovers or
opens private manifests. Frozen final-test reports are accepted only when their
predeclared prerequisites, hashes, model identities, counts, privacy boundary,
and automatic gates all verify exactly.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis_s2s.config import load_yaml, project_root
from thesis_s2s.metrics import meets_bargein_accuracy_target, write_json
from thesis_s2s.repro import sha256_file

ARTIFACTS: dict[str, str] = {
    "conversation_audit": "results/conversation_audit.json",
    "conversation_balanced_v2_audit": "results/conversation_balanced_v2_audit.json",
    "conversation_balance_v2_audit": "results/conversation_balance_v2_audit.json",
    "moshi_export_report": "results/moshi_export_report.json",
    "moshi_export_audit": "results/moshi_export_audit.json",
    "moshi_v1_training": "results/hardware/moshi_h100_training.json",
    "moshi_v1_heldout_loss": "results/moshi_heldout_model_eval.json",
    "moshi_v1_protocol": "docs/MOSHI_SELECTION_PROTOCOL.md",
    "moshi_v2_split": "results/moshi_v2_split.json",
    "moshi_v2_config": "configs/moshi_h100_v2.yaml",
    "moshi_v2_selection": "results/moshi_v2_checkpoint_selection.json",
    "moshi_v2_finalization": "results/hardware/moshi_v2_negative_selection_finalization.json",
    "moshi_v3_training": "results/hardware/moshi_h100_v3_training.json",
    "moshi_v3_selection": "results/moshi_v3_checkpoint_selection.json",
    "moshi_v4_training": "results/hardware/moshi_h100_v4_training.json",
    "moshi_v4_selection": "results/moshi_v4_checkpoint_selection.json",
    "moshi_v4_result": "docs/MOSHI_V4_RESULT.md",
    "moshi_v5_training": "results/hardware/moshi_h100_v5_training.json",
    "moshi_v5_selection": "results/moshi_v5_checkpoint_selection.json",
    "moshi_v5_result": "docs/MOSHI_V5_RESULT.md",
    "moshi_v6_2_config": "configs/moshi_h100_v6_text_dropout.yaml",
    "moshi_v6_2_training": "results/hardware/moshi_v6_text_dropout_training.json",
    "moshi_v6_2_reevaluation": "results/moshi_v6_text_dropout_reevaluation.json",
    "moshi_v6_2_runtime": "results/moshi_v6_text_dropout_runtime.json",
    "moshi_v6_2_protocol": "docs/MOSHI_V6_TEXT_DROPOUT_PROTOCOL.md",
    "cascade_working_panel": "results/eval/cascade_real_service_train_panel_v4.json",
    "cascade_working_protocol": "docs/CASCADE_REAL_SERVICE_PROTOCOL_V4.md",
    "cascade_validation": "results/eval/cascade_real_service_validation_panel.json",
    "cascade_descriptive_analysis": (
        "results/eval/cascade_validation_descriptive_analysis.json"
    ),
    "cascade_intelligibility_proxy": "results/eval/cascade_intelligibility_proxy.json",
    "cascade_intelligibility_protocol": "docs/CASCADE_INTELLIGIBILITY_PROTOCOL.md",
    "cascade_validation_protocol": "docs/CASCADE_VALIDATION_PROTOCOL.md",
    "responder_lora_training": "results/training/responder_lora_v1.json",
    "responder_lora_semantic": "results/eval/responder_lora_semantic_proxy.json",
    "qwen4b_v1_semantic": "results/eval/qwen4b_responder_semantic_proxy.json",
    "qwen4b_v2_protocol": "docs/QWEN4B_CASCADE_V2_PROTOCOL.md",
    "qwen4b_v2_evaluator": "scripts/evaluate_qwen4b_responder_v2.py",
    "qwen4b_v2_development": "results/eval/qwen4b_responder_v2_development_proxy.json",
    "qwen4b_v2_final": "results/eval/qwen4b_responder_v2_final_test_proxy.json",
    "interrupt_bench": "results/eval/interrupt_bench.json",
    "interrupt_recorded_proxy": "results/eval/interrupt_recorded_proxy.json",
    "human_study": "results/eval/human_study.json",
    "hardware_preflight": "results/hardware/current_preflight.json",
    "final_hardware_preflight": "results/hardware/final_preflight.json",
    "latency_h100_capped": "results/eval/latency_bench.json",
    "latency_h100_uncapped": "results/eval/latency_bench_path_b.json",
    "metrics_definition": "docs/METRICS.md",
    "storage_cleanup": "results/hardware/storage_cleanup_20260831.json",
    "final_audit": "results/release/final_audit.json",
    "submission_tag_attestation": "results/release/submission_tag_attestation.json",
}


def _read_json(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {relative}")
    return payload


def _release_attestation_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the externally observed Git tag and its two successful CI runs."""

    commit = payload.get("commit")
    tag_object = payload.get("tag_object")
    tag = payload.get("tag")
    branch_ci_payload = payload.get("branch_ci")
    tag_ci_payload = payload.get("tag_ci")
    branch_ci = branch_ci_payload if isinstance(branch_ci_payload, dict) else {}
    tag_ci = tag_ci_payload if isinstance(tag_ci_payload, dict) else {}

    def is_sha1(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 40
            and all(character in "0123456789abcdef" for character in value)
        )

    def ci_receipt_valid(receipt: dict[str, Any], expected_branch: object) -> bool:
        run_id = receipt.get("run_id")
        return bool(
            type(run_id) is int
            and run_id > 0
            and receipt.get("head_sha") == commit
            and receipt.get("head_branch") == expected_branch
            and receipt.get("status") == "completed"
            and receipt.get("conclusion") == "success"
            and receipt.get("url")
            == f"https://github.com/mehbakh82/Thesis_Project/actions/runs/{run_id}"
            and isinstance(receipt.get("updated_at"), str)
            and bool(receipt.get("updated_at"))
        )

    verified = bool(
        payload.get("schema_version") == 1
        and isinstance(tag, str)
        and tag.startswith("submission-")
        and payload.get("tag_object_type") == "annotated_tag"
        and payload.get("remote") == "https://github.com/mehbakh82/Thesis_Project.git"
        and is_sha1(tag_object)
        and is_sha1(commit)
        and ci_receipt_valid(branch_ci, "main")
        and ci_receipt_valid(tag_ci, tag)
        and isinstance(payload.get("verified_at"), str)
        and bool(payload.get("verified_at"))
    )
    return {
        "verified": verified,
        "tag": tag if isinstance(tag, str) else None,
        "tag_object": tag_object if is_sha1(tag_object) else None,
        "commit": commit if is_sha1(commit) else None,
        "branch_ci": branch_ci,
        "tag_ci": tag_ci,
        "verified_at": payload.get("verified_at") if verified else None,
    }


def _artifact_provenance(root: Path) -> dict[str, dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    for name, relative in ARTIFACTS.items():
        path = root / relative
        present = path.is_file()
        provenance[name] = {
            "path": relative,
            "present": present,
            "bytes": path.stat().st_size if present else None,
            "sha256": sha256_file(path) if present else None,
        }
    return provenance


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * (proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2)) ** 0.5
        / denominator
    )
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def _selection_trial(
    version: str,
    selection: dict[str, Any],
    *,
    training: dict[str, Any] | None = None,
    finalization: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = selection.get("candidates")
    candidate_rows = candidates if isinstance(candidates, list) else []
    selected = selection.get("selected")
    selected_step = selected.get("step") if isinstance(selected, dict) else None
    eligible_steps = selection.get("eligible_steps")
    eligible = eligible_steps if isinstance(eligible_steps, list) else []
    training_payload = training or {}
    finalization_payload = finalization or {}
    profile = training_payload.get("profile") or config or {}
    lora = profile.get("lora") or {}
    checkpoints = training_payload.get("checkpoints") or []
    selective_embeddings = (checkpoints[0].get("embedding_parameters") if checkpoints else []) or []
    test_access_started = bool(
        training_payload.get("test_access_started", finalization_payload.get("test_access_started"))
    )
    return {
        "version": version,
        "evidence_scope": "validation_only_checkpoint_selection",
        "scientific_run_complete": bool(
            training_payload.get("training_run_passes")
            or selection.get("training_complete")
            or finalization_payload.get("scientific_negative_outcome_finalized")
        ),
        "candidate_count": int(selection.get("candidate_count") or len(candidate_rows)),
        "eligible_candidate_count": len(eligible),
        "eligible_steps": eligible,
        "selected_step": selected_step,
        "deployment_eligible": bool(
            selection.get("selection_passes") and selected_step in eligible
        ),
        "fixed_validation_losses": [row.get("eval_loss") for row in candidate_rows],
        "fixed_validation_samples_per_candidate": (
            int(candidate_rows[0].get("validation_sample_count") or 0) if candidate_rows else 0
        ),
        "runtime_panel_pass_counts": [
            int(row.get("runtime_panel_pass_count") or 0) for row in candidate_rows
        ],
        "runtime_panel_failure_counts": [
            int(row.get("runtime_panel_count") or 0) - int(row.get("runtime_panel_pass_count") or 0)
            for row in candidate_rows
        ],
        "runtime_panel_size": (
            int(candidate_rows[0].get("runtime_panel_count") or 0) if candidate_rows else 0
        ),
        "selection_failed_closed": not bool(selected) and not bool(eligible),
        "uncertainty": {
            "interval": None,
            "reason": "deterministic complete-scope loss and pass/fail eligibility rule",
        },
        "test_access_started": test_access_started,
        "configuration": {
            "context_seconds": profile.get("duration_sec"),
            "max_steps": profile.get("max_steps"),
            "seed": profile.get("seed"),
            "lora_rank": lora.get("rank"),
            "upstream_ft_embed": lora.get("ft_embed"),
            "selective_embeddings": selective_embeddings,
            "peak_allocated_gb": (training_payload.get("training_metrics") or {}).get(
                "maximum_peak_allocated_gb"
            ),
        },
    }


def _official_e2e_reports(root: Path) -> tuple[int, list[dict[str, Any]]]:
    """Count only explicitly consented live-browser reports on physical target hardware."""

    rows = 0
    reports: list[dict[str, Any]] = []
    for relative in (
        "results/eval/latency_bench.json",
        "results/eval/latency_bench_path_b.json",
        "results/eval/path_ab_and_study.json",
    ):
        payload = _read_json(root, relative)
        evidence_class = payload.get("evidence_class") or payload.get("measurement_scope")
        eligible = bool(
            payload.get("official_e2e_eligible")
            and evidence_class == "official_e2e"
            and payload.get("consented") is True
            and payload.get("live_browser") is True
            and payload.get("physical_gpu_12_to_24_gb") is True
            and payload.get("client_playback_acknowledgements") is True
        )
        report_rows = int(payload.get("n") or payload.get("official_e2e_rows") or 0)
        if eligible:
            rows += report_rows
        reports.append(
            {
                "path": relative,
                "present": bool(payload),
                "declared_evidence_class": evidence_class,
                "qualifies": eligible,
                "qualifying_rows": report_rows if eligible else 0,
            }
        )
    return rows, reports


def _v6_2_capacity_trial(
    training: dict[str, Any],
    reevaluation: dict[str, Any],
    runtime: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Summarize v6.2 without promoting its train-only diagnostic to validation."""

    loss_rows = reevaluation.get("candidates") or []
    runtime_rows = runtime.get("candidates") or []
    text_losses = [
        float(row["text_eval_loss"])
        for row in loss_rows
        if row.get("text_eval_loss") is not None
    ]
    reduction = None
    if len(text_losses) >= 2 and text_losses[0] > 0:
        reduction = (text_losses[0] - min(text_losses[1:])) / text_losses[0]
    checkpoints = training.get("checkpoints") or []
    lora = config.get("lora") or {}
    requirements = runtime.get("requirements") or {}
    run_complete = bool(
        training.get("status") == "passed"
        and training.get("training_passes")
        and reevaluation.get("status") == "passed"
        and reevaluation.get("reevaluation_passes")
        and runtime.get("status") == "passed"
        and len(loss_rows) == 4
        and len(runtime_rows) == 4
        and all(requirements.values())
    )
    return {
        "version": "v6.2",
        "evidence_scope": "train_only_in_sample_capacity_diagnostic",
        "scientific_run_complete": run_complete,
        "scientific_validation_evidence": False,
        "positive_learning_result": bool(reduction is not None and reduction >= 0.30),
        "text_loss_reduction_fraction": round(reduction, 6) if reduction is not None else None,
        "candidate_count": len(runtime_rows),
        "eligible_candidate_count": 0,
        "eligible_steps": [],
        "selected_step": None,
        "deployment_eligible": False,
        "fixed_validation_losses": [row.get("eval_loss") for row in loss_rows],
        "fixed_text_losses": [row.get("text_eval_loss") for row in loss_rows],
        "fixed_audio_losses": [row.get("audio_eval_loss") for row in loss_rows],
        "fixed_validation_samples_per_candidate": (
            int(loss_rows[0].get("sample_count") or 0) if loss_rows else 0
        ),
        "runtime_panel_pass_counts": [
            int(row.get("panel_pass_count") or 0) for row in runtime_rows
        ],
        "runtime_panel_failure_counts": [
            int(row.get("panel_count") or 0) - int(row.get("panel_pass_count") or 0)
            for row in runtime_rows
        ],
        "runtime_panel_size": (
            int(runtime_rows[0].get("panel_count") or 0) if runtime_rows else 0
        ),
        "selection_failed_closed": runtime.get("selection") is None,
        "diagnostic_outcome": runtime.get("diagnostic_outcome"),
        "test_access_started": bool(
            training.get("final_test_accessed")
            or reevaluation.get("final_test_opened")
            or runtime.get("final_test_accessed")
        ),
        "uncertainty": {
            "interval": None,
            "reason": "deterministic in-sample diagnostic; no generalization inference allowed",
        },
        "configuration": {
            "context_seconds": config.get("duration_sec"),
            "max_steps": config.get("max_steps"),
            "seed": config.get("seed"),
            "lora_rank": lora.get("rank"),
            "upstream_ft_embed": lora.get("ft_embed"),
            "selective_embeddings": (
                checkpoints[0].get("full_parameters") if checkpoints else []
            )
            or [],
            "peak_allocated_gb": (training.get("training") or {}).get(
                "peak_allocated_gb"
            ),
        },
    }


def _cascade_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept only the exact, explicit real-service working-demo evidence class."""

    samples = payload.get("samples") or []
    claim = payload.get("claim") or {}
    requirements = payload.get("requirements") or {}
    working = bool(
        payload.get("status") == "passed"
        and payload.get("evidence_class")
        == "real_service_group_disjoint_validation_mechanics"
        and claim.get("positive_working_user_turn_result") is True
        and claim.get("group_disjoint_validation_execution") is True
        and claim.get("scientific_generalization_result") is False
        and claim.get("official_end_to_end_latency_result") is False
        and len(samples) == 9
        and all(sample.get("passes") is True for sample in samples)
        and requirements
        and all(requirements.values())
    )
    reply_fractions = [
        float((sample.get("reply_statistics") or {}).get("persian_letter_fraction") or 0.0)
        for sample in samples
    ]
    audio_rms = [float(sample.get("reply_audio_rms") or 0.0) for sample in samples]
    return {
        "working_prototype": working,
        "architecture": "NeMo Persian ASR -> local Qwen2.5-0.5B -> Piper Persian TTS",
        "evidence_class": payload.get("evidence_class") or "missing",
        "panel_scope": (payload.get("panel") or {}).get("selection"),
        "split_unit": (payload.get("panel") or {}).get("split_unit"),
        "input_channel": (payload.get("panel") or {}).get("selected_user_channel"),
        "group_split_leaks": (payload.get("panel") or {}).get("group_split_leaks"),
        "panel_rows": len(samples),
        "passed_rows": sum(sample.get("passes") is True for sample in samples),
        "fallback_rows": sum(
            sample.get("responder_fallback_used") is True for sample in samples
        ),
        "language_retry_rows": sum(
            sample.get("responder_language_retry_used") is True for sample in samples
        ),
        "minimum_reply_persian_letter_fraction": min(reply_fractions)
        if reply_fractions
        else None,
        "minimum_reply_audio_rms": min(audio_rms) if audio_rms else None,
        "timing_descriptive_only": payload.get("timing_descriptive_only") or {},
        "scientific_generalization_result": False,
        "official_end_to_end_latency_result": False,
        "human_quality_result": False,
        "limitations": [
            "No independent semantic-relevance or human naturalness labels were collected.",
            "A nine-row mechanics panel is not a population-level generalization estimate.",
            "Full-turn generation time is not the thesis T_first_audio browser metric.",
            "The H100 is outside the official 12-24 GB evaluation class.",
        ],
    }


def _cascade_descriptive_result(
    payload: dict[str, Any], *, expected_source_sha256: str | None
) -> dict[str, Any]:
    """Accept only a complete, hash-bound, non-claiming post-hoc analysis."""

    execution = payload.get("execution_outcomes") or {}
    integrity = payload.get("analysis_integrity") or {}
    boundary = payload.get("claim_boundary") or {}
    verified = bool(
        payload.get("status") == "complete"
        and payload.get("analysis_scope")
        == "post_hoc_privacy_safe_descriptive_error_analysis"
        and expected_source_sha256
        and payload.get("source_report_sha256") == expected_source_sha256
        and payload.get("source_status") == "passed"
        and int(payload.get("panel_rows") or 0) == 9
        and int(payload.get("panel_passed_rows") or 0) == 9
        and int(payload.get("panel_failed_rows") or 0) == 0
        and payload.get("report_gate_failures") == []
        and payload.get("sample_requirement_failure_counts") == {}
        and integrity.get("all_required_measurements_and_hashes_present") is True
        and int(execution.get("asr_error_rows") or 0) == 0
        and int(execution.get("responder_fallback_rows") or 0) == 0
        and int(execution.get("unique_transcript_hashes") or 0) == 9
        and int(execution.get("unique_reply_hashes") or 0) == 9
        and int(execution.get("duplicate_transcript_rows") or 0) == 0
        and int(execution.get("duplicate_reply_rows") or 0) == 0
        and boundary.get("plaintext_transcripts_or_replies_read_or_emitted") is False
        and boundary.get("final_test_accessed") is False
        and boundary.get("scientific_generalization_claim_allowed") is False
        and boundary.get("semantic_quality_claim_allowed") is False
        and boundary.get("human_quality_claim_allowed") is False
        and boundary.get("official_latency_claim_allowed") is False
    )
    return {
        "verified": verified,
        "evidence_scope": payload.get("analysis_scope") or "missing",
        "source_report_sha256": payload.get("source_report_sha256"),
        "panel_rows": int(payload.get("panel_rows") or 0),
        "passed_rows": int(payload.get("panel_passed_rows") or 0),
        "failed_rows": int(payload.get("panel_failed_rows") or 0),
        "execution_outcomes": execution,
        "distributions": payload.get("distributions") or {},
        "descriptive_risk_counts": payload.get("descriptive_risk_counts") or {},
        "row_extremes": payload.get("row_extremes") or {},
        "transcript_length_full_turn_pearson_r": payload.get(
            "transcript_length_full_turn_pearson_r"
        ),
        "claim_boundary": boundary,
        "interpretation": payload.get("interpretation") or {},
    }


def _cascade_intelligibility_result(
    payload: dict[str, Any],
    *,
    expected_parent_sha256: str | None,
    expected_protocol_sha256: str | None,
) -> dict[str, Any]:
    """Accept only the predeclared, hash-bound, privacy-safe round-trip proxy."""

    claim = payload.get("claim") or {}
    panel = payload.get("panel") or {}
    aggregate = payload.get("aggregate") or {}
    requirements = payload.get("validity_requirements") or {}
    artifacts = payload.get("artifacts") or {}
    privacy = payload.get("privacy") or {}
    samples = payload.get("samples") or []
    word = aggregate.get("word_error_rate") or {}
    character = aggregate.get("character_error_rate") or {}
    verified = bool(
        payload.get("status") == "valid"
        and payload.get("evidence_class")
        == "automatic_asr_roundtrip_intelligibility_proxy"
        and claim.get("measurement_valid") is True
        and claim.get("automatic_intelligibility_proxy_measured") is True
        and claim.get("human_intelligibility_result") is False
        and claim.get("pronunciation_or_naturalness_result") is False
        and claim.get("semantic_relevance_result") is False
        and claim.get("population_generalization_result") is False
        and expected_parent_sha256
        and artifacts.get("parent_panel_sha256") == expected_parent_sha256
        and expected_protocol_sha256
        and artifacts.get("protocol_sha256") == expected_protocol_sha256
        and int(panel.get("sample_count") or 0) == 9
        and panel.get("final_test_accessed") is False
        and int(aggregate.get("rows") or 0) == 9
        and int(aggregate.get("asr_failures") or 0) == 0
        and int(word.get("reference_words") or 0) > 0
        and int(character.get("reference_characters") or 0) > 0
        and payload.get("threshold") is None
        and len(samples) == 9
        and all(
            sample.get("input_asr_error") is None
            and sample.get("roundtrip_asr_error") is None
            and sample.get("matches_parent_input_transcript") is True
            and sample.get("matches_parent_reply") is True
            and sample.get("responder_backend") == "Qwen/Qwen2.5-0.5B-Instruct"
            and sample.get("responder_fallback_used") is False
            and sample.get("tts_backend") == "piper"
            for sample in samples
        )
        and requirements
        and all(requirements.values())
        and privacy.get("plaintext_input_transcripts_stored") is False
        and privacy.get("plaintext_reply_text_stored") is False
        and privacy.get("plaintext_roundtrip_transcripts_stored") is False
        and privacy.get("audio_stored") is False
    )
    return {
        "verified": verified,
        "evidence_class": payload.get("evidence_class") or "missing",
        "panel_rows": int(aggregate.get("rows") or 0),
        "asr_failures": int(aggregate.get("asr_failures") or 0),
        "exact_word_match_rows": int(aggregate.get("exact_word_match_rows") or 0),
        "exact_character_match_rows": int(
            aggregate.get("exact_character_match_rows") or 0
        ),
        "word_error_rate": word,
        "character_error_rate": character,
        "threshold": payload.get("threshold"),
        "claim_boundary": claim,
        "limitations": payload.get("limitations") or [],
    }


def _qwen4b_v2_final_result(
    payload: dict[str, Any],
    *,
    expected_protocol_sha256: str | None,
    expected_evaluator_sha256: str | None,
    expected_development_sha256: str | None,
) -> dict[str, Any]:
    """Verify the frozen prompt-v2 automatic semantic final-test certificate."""

    claim = payload.get("claim") or {}
    panel = payload.get("panel") or {}
    aggregate = payload.get("aggregate") or {}
    candidate = aggregate.get("qwen4b_v2") or {}
    paired = aggregate.get("paired") or {}
    requirements = payload.get("validity_requirements") or {}
    gates = payload.get("automatic_engineering_gate") or {}
    responder = (payload.get("responders") or {}).get("qwen4b_v2") or {}
    artifacts = payload.get("artifacts") or {}
    privacy = payload.get("privacy") or {}
    samples = payload.get("samples") or []
    verified = bool(
        payload.get("status") == "passed"
        and payload.get("stage") == "final_test"
        and payload.get("evidence_class")
        == "automatic_same_family_llm_as_judge_qwen4b_prompt_v2_final_test"
        and claim.get("measurement_valid") is True
        and claim.get("predeclared_automatic_engineering_gate_passed") is True
        and claim.get("final_test_unlocked") is True
        and claim.get("automatic_final_test_result") is True
        and claim.get("automatic_final_test_gate_passed") is True
        and claim.get("human_semantic_result") is False
        and claim.get("independent_dialogue_benchmark_result") is False
        and claim.get("factuality_or_safety_result") is False
        and claim.get("physical_4090_result") is False
        and panel.get("split") == "test"
        and panel.get("split_unit") == "source_session_id"
        and int(panel.get("manifest_rows") or 0) == 204
        and int(panel.get("sample_count") or 0) == 40
        and len(panel.get("indices") or []) == 40
        and int(aggregate.get("rows") or 0) == 40
        and int(aggregate.get("judge_calls_expected") or 0) == 240
        and int(aggregate.get("judge_calls_valid") or 0) == 240
        and int(aggregate.get("judge_calls_failed") or 0) == 0
        and len(samples) == 40
        and requirements
        and all(value is True for value in requirements.values())
        and gates
        and all(value is True for value in gates.values())
        and responder.get("backend")
        == "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554:prompt-v2"
        and responder.get("revision") == "cdbee75f17c01a7cc42f958dc650907174af0554"
        and responder.get("tree_sha256")
        == "cde447f1326f10c4126061914c57c3664551649286ad6411bffe3d1aa3e3b978"
        and responder.get("license") == "Apache-2.0"
        and artifacts.get("stage_export_sha256")
        == "44d5912201ed359dabe3c026b6ae605b3bf946538e83116f57514448ed0794fe"
        and expected_protocol_sha256
        and artifacts.get("protocol_sha256") == expected_protocol_sha256
        and expected_evaluator_sha256
        and artifacts.get("evaluator_sha256") == expected_evaluator_sha256
        and expected_development_sha256
        and artifacts.get("development_report_sha256") == expected_development_sha256
        and privacy
        and all(value is False for value in privacy.values())
    )
    return {
        "verified": verified,
        "status": payload.get("status") or "missing",
        "evidence_class": payload.get("evidence_class") or "missing",
        "architecture": (
            "NeMo Persian ASR -> Qwen3-4B-Instruct-2507 prompt-v2 -> Piper Persian TTS"
        ),
        "split": panel.get("split"),
        "split_unit": panel.get("split_unit"),
        "rows": int(aggregate.get("rows") or 0),
        "judge_calls_valid": int(aggregate.get("judge_calls_valid") or 0),
        "judge_calls_failed": int(aggregate.get("judge_calls_failed") or 0),
        "base_relevance_mean": (
            ((aggregate.get("base") or {}).get("relevance") or {}).get("mean")
        ),
        "candidate_relevance_mean": (candidate.get("relevance") or {}).get("mean"),
        "candidate_coherence_mean": (candidate.get("coherence") or {}).get("mean"),
        "relevance_gain": paired.get("candidate_minus_base_relevance_mean"),
        "coherence_gain": paired.get("candidate_minus_base_coherence_mean"),
        "relevance_win_rate": paired.get("candidate_relevance_win_rate"),
        "relevance_at_least_two_rate": paired.get(
            "candidate_relevance_at_least_two_rate"
        ),
        "automatic_gate_passed": bool(
            claim.get("automatic_final_test_gate_passed")
            and gates
            and all(gates.values())
        ),
        "human_semantic_result": False,
        "independent_benchmark_result": False,
        "factuality_or_safety_result": False,
        "physical_4090_result": False,
        "limitations": payload.get("limitations") or [],
    }


def build_evidence_status(
    out: str | Path | None = None,
    *,
    root: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build and optionally write the authoritative aggregate evidence status."""

    project = Path(root or project_root()).resolve()
    conversation = _read_json(project, ARTIFACTS["conversation_audit"])
    balanced_v2 = _read_json(project, ARTIFACTS["conversation_balanced_v2_audit"])
    balance_v2 = _read_json(project, ARTIFACTS["conversation_balance_v2_audit"])
    export = _read_json(project, ARTIFACTS["moshi_export_report"])
    export_audit = _read_json(project, ARTIFACTS["moshi_export_audit"])
    v1_training = _read_json(project, ARTIFACTS["moshi_v1_training"])
    v1_heldout = _read_json(project, ARTIFACTS["moshi_v1_heldout_loss"])
    v2_split = _read_json(project, ARTIFACTS["moshi_v2_split"])
    v2_config_path = project / ARTIFACTS["moshi_v2_config"]
    v2_config = load_yaml(v2_config_path) if v2_config_path.is_file() else {}
    v2_selection = _read_json(project, ARTIFACTS["moshi_v2_selection"])
    v2_finalization = _read_json(project, ARTIFACTS["moshi_v2_finalization"])
    v3_training = _read_json(project, ARTIFACTS["moshi_v3_training"])
    v3_selection = _read_json(project, ARTIFACTS["moshi_v3_selection"])
    v4_training = _read_json(project, ARTIFACTS["moshi_v4_training"])
    v4_selection = _read_json(project, ARTIFACTS["moshi_v4_selection"])
    v5_training = _read_json(project, ARTIFACTS["moshi_v5_training"])
    v5_selection = _read_json(project, ARTIFACTS["moshi_v5_selection"])
    v6_2_config_path = project / ARTIFACTS["moshi_v6_2_config"]
    v6_2_config = load_yaml(v6_2_config_path) if v6_2_config_path.is_file() else {}
    v6_2_training = _read_json(project, ARTIFACTS["moshi_v6_2_training"])
    v6_2_reevaluation = _read_json(project, ARTIFACTS["moshi_v6_2_reevaluation"])
    v6_2_runtime = _read_json(project, ARTIFACTS["moshi_v6_2_runtime"])
    cascade_payload = _read_json(project, ARTIFACTS["cascade_validation"])
    cascade_analysis_payload = _read_json(
        project, ARTIFACTS["cascade_descriptive_analysis"]
    )
    cascade_intelligibility_payload = _read_json(
        project, ARTIFACTS["cascade_intelligibility_proxy"]
    )
    qwen4b_v2_final_payload = _read_json(project, ARTIFACTS["qwen4b_v2_final"])
    interrupt = _read_json(project, ARTIFACTS["interrupt_bench"])
    recorded_proxy = _read_json(project, ARTIFACTS["interrupt_recorded_proxy"])
    study = _read_json(project, ARTIFACTS["human_study"])
    cleanup = _read_json(project, ARTIFACTS["storage_cleanup"])
    final_audit = _read_json(project, ARTIFACTS["final_audit"])
    release_attestation = _release_attestation_result(
        _read_json(project, ARTIFACTS["submission_tag_attestation"])
    )
    final_hardware_path = project / "results/hardware/final_preflight.json"
    hardware = (
        _read_json(project, "results/hardware/final_preflight.json")
        if final_hardware_path.is_file()
        else _read_json(project, ARTIFACTS["hardware_preflight"])
    )

    export_requirements = export.get("requirements") or {}
    audit_requirements = export_audit.get("requirements") or {}
    audited_export_ready = bool(
        export.get("training_ready_under_qa_waiver")
        and export_audit.get("audit_passes")
        and export_audit.get("training_ready_under_qa_waiver")
        and export_requirements.get("training_hours_100_to_200")
        and audit_requirements.get("session_group_splits_isolated")
    )
    strict_human_qa = bool(
        export.get("final_training_ready")
        and (export_audit.get("sample") or {}).get("human_review_complete")
        and export_audit.get("strict_thesis_data_coverage")
    )

    v1_selected = v1_training.get("selected")
    trials = [
        {
            "version": "v1",
            "evidence_scope": "heldout_loss_then_postselection_runtime",
            "scientific_run_complete": bool(v1_training.get("training_run_passes")),
            "candidate_count": int(v1_training.get("checkpoint_count") or 0),
            "eligible_candidate_count": 0,
            "eligible_steps": [],
            "selected_step": v1_selected.get("step") if isinstance(v1_selected, dict) else None,
            "deployment_eligible": False,
            "selection_failed_closed": False,
            "test_access_started": bool(
                (v1_training.get("requirements") or {}).get("one_time_heldout_evaluation_passed")
            ),
            "verdict": "runtime_ineligible_after_postselection_validation",
            "reason": "selected adapter decoded near-silent audio and no text",
            "configuration": {
                "context_seconds": (v1_training.get("profile") or {}).get("duration_sec"),
                "max_steps": (v1_training.get("profile") or {}).get("max_steps"),
                "seed": (v1_training.get("profile") or {}).get("seed"),
                "lora_rank": ((v1_training.get("profile") or {}).get("lora") or {}).get("rank"),
                "upstream_ft_embed": ((v1_training.get("profile") or {}).get("lora") or {}).get(
                    "ft_embed"
                ),
                "selective_embeddings": [],
                "peak_allocated_gb": (v1_training.get("training_metrics") or {}).get(
                    "maximum_peak_allocated_gb"
                ),
            },
        },
        _selection_trial("v2", v2_selection, finalization=v2_finalization, config=v2_config),
        _selection_trial("v3", v3_selection, training=v3_training),
        _selection_trial("v4", v4_selection, training=v4_training),
        _selection_trial("v5", v5_selection, training=v5_training),
        _v6_2_capacity_trial(v6_2_training, v6_2_reevaluation, v6_2_runtime, v6_2_config),
    ]
    direct_model_eligible = any(bool(trial["deployment_eligible"]) for trial in trials)
    cascade = _cascade_result(cascade_payload)
    cascade_report_path = project / ARTIFACTS["cascade_validation"]
    cascade_analysis = _cascade_descriptive_result(
        cascade_analysis_payload,
        expected_source_sha256=(
            sha256_file(cascade_report_path) if cascade_report_path.is_file() else None
        ),
    )
    cascade_protocol_path = project / ARTIFACTS["cascade_intelligibility_protocol"]
    cascade_intelligibility = _cascade_intelligibility_result(
        cascade_intelligibility_payload,
        expected_parent_sha256=(
            sha256_file(cascade_report_path) if cascade_report_path.is_file() else None
        ),
        expected_protocol_sha256=(
            sha256_file(cascade_protocol_path) if cascade_protocol_path.is_file() else None
        ),
    )
    qwen4b_protocol_path = project / ARTIFACTS["qwen4b_v2_protocol"]
    qwen4b_evaluator_path = project / ARTIFACTS["qwen4b_v2_evaluator"]
    qwen4b_development_path = project / ARTIFACTS["qwen4b_v2_development"]
    qwen4b_final = _qwen4b_v2_final_result(
        qwen4b_v2_final_payload,
        expected_protocol_sha256=(
            sha256_file(qwen4b_protocol_path) if qwen4b_protocol_path.is_file() else None
        ),
        expected_evaluator_sha256=(
            sha256_file(qwen4b_evaluator_path) if qwen4b_evaluator_path.is_file() else None
        ),
        expected_development_sha256=(
            sha256_file(qwen4b_development_path)
            if qwen4b_development_path.is_file()
            else None
        ),
    )

    proposed = interrupt.get("proposed") or {}
    proposed_matrix = proposed.get("matrix") or []
    synthetic_correct = 0
    if (
        isinstance(proposed_matrix, list)
        and len(proposed_matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in proposed_matrix)
    ):
        synthetic_correct = int(proposed_matrix[0][0]) + int(proposed_matrix[1][1])
    synthetic_n = int(proposed.get("n") or 0)
    if synthetic_correct == 0 and synthetic_n and proposed.get("accuracy") is not None:
        synthetic_correct = round(float(proposed["accuracy"]) * synthetic_n)
    recorded_proxy_test = recorded_proxy.get("heldout_test") or {}
    recorded_proxy_proposed = recorded_proxy_test.get("proposed") or {}
    detector_eligible = bool(
        (
            interrupt.get("recorded_eval")
            and interrupt.get("official_detector_eligible")
            and meets_bargein_accuracy_target(proposed.get("accuracy"))
        )
        or (
            recorded_proxy.get("official_detector_eligible")
            and int(recorded_proxy.get("human_verified_labels") or 0) > 0
            and meets_bargein_accuracy_target(recorded_proxy_proposed.get("accuracy"))
        )
    )
    physical_target_hardware = bool(
        hardware.get("is_rtx_4090")
        and hardware.get("evaluation_hardware_ready")
        and (hardware.get("gpu") or {}).get("official_size")
    )
    study_requirements = study.get("requirements") or {}
    required_study_gates = (
        "participants_5_to_10",
        "elderly_participants_at_least_2",
        "complete_ratings_cover_participants",
        "turn_rows_valid",
        "rating_rows_valid",
        "eligible_client_first_audio_present",
        "eligible_client_barge_in_present",
        "physical_gpu_12_to_24_gb",
        "real_heldout_detector_report",
    )
    human_study_complete = bool(
        study.get("official_ready")
        and all(study_requirements.get(gate) is True for gate in required_study_gates)
    )
    license_files = [
        relative
        for relative in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
        if (project / relative).is_file()
    ]
    source_license_selected = bool(license_files)
    official_rows, official_reports = _official_e2e_reports(project)
    retained_adapters = cleanup.get("retained_representative_adapters") or {}
    removed_categories = {
        str(item.get("category")) for item in cleanup.get("removed") or [] if isinstance(item, dict)
    }
    negative_intermediates_removed = (
        "non_promoted_negative_checkpoint_intermediates" in removed_categories
    )
    cleanup_postconditions = cleanup.get("postconditions") or {}
    cleanup_space = cleanup.get("space") or {}
    transport = final_audit.get("duplex_transport_regression") or {}
    transport_checks = (
        "client_acknowledgement_ingestion",
        "identity_and_fallback_telemetry",
        "explicit_cancellation",
        "simultaneous_turn_isolation",
        "reconnect",
        "malformed_and_silent_input_recovery",
        "simulated_generation_oom_recovery",
        "metrics_retention_writes_no_wav",
    )
    transport_regression_passed = bool(
        final_audit.get("status") == "passed"
        and transport.get("evidence_class") == "automated_websocket_transport_regression"
        and int(transport.get("tests") or 0) >= 4
        and all(transport.get(name) == "passed" for name in transport_checks)
        and int(transport.get("automatic_bargein_preroll_samples") or 0)
        + int(transport.get("continued_capture_samples") or 0)
        == int(transport.get("next_turn_input_samples") or -1)
    )

    gates = {
        "working_persian_s2s_prototype": bool(cascade["working_prototype"]),
        "automatic_semantic_final_test_passed": bool(qwen4b_final["verified"]),
        "audited_export_100_to_200_hours": audited_export_ready,
        "data_policy_resolved_under_documented_qa_waiver": bool(
            audited_export_ready and not strict_human_qa
        ),
        "deployment_eligible_persian_direct_model": direct_model_eligible,
        "real_group_heldout_detector_above_80_percent": detector_eligible,
        "physical_12_to_24_gb_fit_and_live_latency": physical_target_hardware and official_rows > 0,
        "human_study_complete": human_study_complete,
        "source_code_license_selected": source_license_selected,
    }
    thesis_ready = all(gates.values())
    required = {
        "working_persian_s2s_prototype": (
            "Produce a repeatable real-service Persian speech-to-speech result."
        ),
        "automatic_semantic_final_test_passed": (
            "Pass the frozen automatic semantic final test after predeclared development "
            "eligibility without changing its thresholds."
        ),
        "deployment_eligible_persian_direct_model": (
            "Produce a validation-eligible Persian direct-model checkpoint before opening "
            "that experiment's frozen final test."
        ),
        "real_group_heldout_detector_above_80_percent": (
            "Obtain independently human-reviewed recorded labels and establish >80% on a "
            "speaker/session-group-held-out test; the automatic YouTube proxy is not "
            "ground truth."
        ),
        "physical_12_to_24_gb_fit_and_live_latency": (
            "Run the final system and client-acknowledged latency protocol on a physical "
            "12–24 GB GPU such as the planned RTX 4090."
        ),
        "human_study_complete": (
            "Collect consented complete results from 5–10 Persian speakers, including at "
            "least two aged 60+."
        ),
        "source_code_license_selected": "Select and add the project source-code license.",
    }

    payload: dict[str, Any] = {
        "schema_version": 12,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "authoritative": True,
        "thesis_ready": thesis_ready,
        "verdict": "thesis_ready" if thesis_ready else "not_thesis_ready_evidence_gates_pending",
        "generation_policy": {
            "fail_closed": True,
            "reads_frozen_final_test_rows": False,
            "official_e2e_requires": [
                "consented",
                "live_browser",
                "client_playback_acknowledgements",
                "physical_12_to_24_gb_gpu",
            ],
        },
        "gates": gates,
        "data": {
            "conversation_pairs": int(conversation.get("pairs") or 0),
            "conversation_source_hours": float(conversation.get("hours") or 0.0),
            "exported_pairs": int(export_audit.get("verified_export_pairs") or 0),
            "exported_hours": float(export_audit.get("verified_export_hours") or 0.0),
            "split_counts": export_audit.get("split_counts") or {},
            "group_split_leaks": int(export_audit.get("group_split_leaks") or 0),
            "audit_failure_counts": export_audit.get("failure_counts") or {},
            "split_policy": {
                "unit": "source_session_id",
                "group_isolated": bool(audit_requirements.get("session_group_splits_isolated")),
                "v2_to_v5_rule": v2_split.get("rule") or {},
                "v2_to_v5_sessions": v2_split.get("sessions") or {},
            },
            "automatic_integrity_audit_passed": bool(export_audit.get("audit_passes")),
            "training_ready_under_documented_qa_waiver": bool(
                export_audit.get("training_ready_under_qa_waiver")
            ),
            "human_review_complete": bool(
                (export_audit.get("sample") or {}).get("human_review_complete")
            ),
            "strict_human_qa_complete": strict_human_qa,
            "qa_waiver_is_a_documented_limitation_not_an_open_action": bool(
                audited_export_ready and not strict_human_qa
            ),
            "human_verified_claim_allowed": bool(
                (export.get("claims") or {}).get("human_verified_data")
            ),
            "raw_redistribution_permitted_rows": int(
                export.get("source_rows_permitting_redistribution") or 0
            ),
            "balanced_v2": {
                "evidence_scope": "post_training_recommended_corpus",
                "used_by_existing_moshi_runs": False,
                "pairs": int(balanced_v2.get("pairs") or 0),
                "source_pair_hours": float(balanced_v2.get("hours") or 0.0),
                "sessions": int(balanced_v2.get("sessions") or 0),
                "speakers": int(balanced_v2.get("speakers") or 0),
                "channels": int((balance_v2.get("totals") or {}).get("channels") or 0),
                "largest_channel_share": float(
                    (balance_v2.get("dominance") or {}).get("hour_share") or 0.0
                ),
                "balance_gate_passed": bool(
                    balance_v2.get("strict_representative_balance_passes")
                ),
                "training_ready_under_documented_qa_waiver": bool(
                    balanced_v2.get("training_ready_under_qa_waiver")
                ),
                "missing_files": int(balanced_v2.get("missing_files") or 0),
                "reused_source_spans": int(balanced_v2.get("reused_source_spans") or 0),
                "session_group_split_leaks": int(
                    balanced_v2.get("session_group_split_leaks") or 0
                ),
                "human_review_complete": bool(balanced_v2.get("human_verified_rows")),
            },
        },
        "working_system": cascade,
        "qwen4b_v2_automatic_final_test": qwen4b_final,
        "cascade_descriptive_analysis": cascade_analysis,
        "cascade_intelligibility_proxy": cascade_intelligibility,
        "direct_moshi": {
            "deployment_eligible": direct_model_eligible,
            "promoted_adapter": None,
            "trials": trials,
            "v2_to_v5_final_test_access_started": any(
                bool(trial["test_access_started"]) for trial in trials[1:5]
            ),
            "v2_to_v6_2_final_test_access_started": any(
                bool(trial["test_access_started"]) for trial in trials[1:]
            ),
            "v1_automatic_heldout_loss_evidence": {
                "evidence_class": "automatic_model_loss_only",
                "rows": int((v1_heldout.get("test_protocol") or {}).get("rows") or 0),
                "chunks": int((v1_heldout.get("test_protocol") or {}).get("chunks") or 0),
                "selected_total_loss": (
                    ((v1_heldout.get("modes") or {}).get("selected_adapter") or {}).get(
                        "total_loss"
                    )
                    or {}
                ),
                "human_perceptual_claim_allowed": bool(
                    v1_heldout.get("human_perceptual_claim_allowed")
                ),
                "deployment_claim_allowed": False,
            },
            "conclusion": (
                "V1 through v5 are held-out/validation-governed negative trials. V6.2 is "
                "positive in-sample learning evidence but negative direct-generation "
                "evidence. No direct adapter is deployment eligible."
            ),
        },
        "local_artifact_retention": {
            "evidence_class": "post_finalization_storage_receipt",
            "cleanup_passed": bool(cleanup.get("status") == "passed"),
            "observed_reclaimed_bytes": int(
                cleanup_space.get("cumulative_verified_reclaimed_bytes")
                or cleanup_space.get("observed_phase_reclaimed_bytes")
                or 0
            ),
            "project_size_after": cleanup_space.get("project_du_after"),
            "representative_adapter_count": int(retained_adapters.get("count") or 0),
            "current_v6_2_checkpoint_count": len(v6_2_training.get("checkpoints") or []),
            "representative_adapter_steps": {
                version: list((retained_adapters.get(version) or {}).get("steps") or [])
                for version in ("v1", "v2", "v3", "v4", "v5")
            },
            "all_retained_adapter_hashes_verified": bool(
                cleanup_postconditions.get("all_retained_adapter_hashes_verified")
                and retained_adapters.get("all_sha256_match_committed_evidence")
            ),
            "full_candidate_tensor_sets_retained": bool(
                cleanup and not negative_intermediates_removed
            ),
            "scientific_results_configs_certificates_retained": bool(
                "all tracked results and certificates"
                in (cleanup.get("protected_artifacts_untouched") or [])
            ),
            "deleted_candidate_tensor_recreation_requires_retraining": bool(
                negative_intermediates_removed
            ),
            "interpretation": (
                "Candidate counts, hashes, losses, and runtime verdicts describe the "
                "certified historical experiments. Only representative adapter tensors "
                "remain locally after verified storage cleanup."
            ),
        },
        "detector": {
            "evidence_class": interrupt.get("evidence_class") or "missing",
            "synthetic_accuracy": proposed.get("accuracy"),
            "synthetic_accuracy_ci95_wilson": _wilson_interval(synthetic_correct, synthetic_n),
            "synthetic_train_n": int(interrupt.get("n_train") or 0),
            "synthetic_n": synthetic_n,
            "synthetic_failures": synthetic_n - synthetic_correct,
            "recorded_n": int(interrupt.get("n_recorded") or 0),
            "recorded_uncertainty": {
                "interval": None,
                "reason": "no independently human-labeled held-out events",
            },
            "recorded_proxy": {
                "present": bool(recorded_proxy),
                "evidence_class": recorded_proxy.get("evidence_class"),
                "recorded_audio": bool(recorded_proxy.get("recorded_audio")),
                "label_source": recorded_proxy.get("label_source"),
                "human_verified_labels": int(recorded_proxy.get("human_verified_labels") or 0),
                "n": int(recorded_proxy_test.get("n") or 0),
                "sessions": int(recorded_proxy_test.get("sessions") or 0),
                "accuracy": recorded_proxy_proposed.get("accuracy"),
                "interrupt_f1": recorded_proxy_proposed.get("interrupt_f1"),
                "far": recorded_proxy_proposed.get("far"),
                "frr": recorded_proxy_proposed.get("frr"),
                "event_ci95": recorded_proxy_test.get("proposed_event_ci95"),
                "session_block_bootstrap_ci95": recorded_proxy_test.get(
                    "proposed_session_block_bootstrap_ci95"
                ),
                "accuracy_above_80_percent": bool(
                    recorded_proxy_test.get("recorded_proxy_accuracy_above_80_percent")
                ),
                "official_detector_eligible": bool(
                    recorded_proxy.get("official_detector_eligible")
                ),
                "official_target_satisfied": False,
            },
            "evaluation_unit": "event",
            "split_policy": "speaker/session-group-held-out required for official evidence",
            "recorded_group_heldout_gate_passed": detector_eligible,
        },
        "latency_and_hardware": {
            "current_gpu": (hardware.get("gpu") or {}).get("device"),
            "current_gpu_total_gb": (hardware.get("gpu") or {}).get("total_gb"),
            "physical_target_hardware_ready": physical_target_hardware,
            "official_e2e_rows": official_rows,
            "official_failures_or_timeouts": None,
            "official_uncertainty": {
                "interval": None,
                "reason": "no qualifying official end-to-end rows",
            },
            "reports": official_reports,
            "component_proxy_claim_allowed": False,
        },
        "human_study": {
            "status": study.get("status") or "missing",
            "participants": int(study.get("participants") or 0),
            "elderly_participants": int(study.get("elderly_participants") or 0),
            "turns": int(study.get("turns") or 0),
            "ratings": int(study.get("ratings") or 0),
            "complete_ratings": int(study.get("complete_ratings") or 0),
            "invalid_rating_rows": int(study.get("invalid_rating_rows") or 0),
            "rating_denominator": int(study.get("complete_ratings") or 0),
            "requirements": {
                gate: study_requirements.get(gate) is True for gate in required_study_gates
            },
            "uncertainty": {
                "interval": None,
                "reason": "zero complete ratings",
            },
            "official_ready": human_study_complete,
        },
        "reporting_contract": {
            "metric_definitions": "docs/METRICS.md",
            "failure_denominators_included": True,
            "unavailable_intervals_are_explicit_nulls": True,
            "model_seeds_reported_per_trial": True,
            "session_group_split_policy_reported": True,
            "no_post_test_selection": True,
        },
        "duplex_transport": {
            "automated_websocket_regression_passed": transport_regression_passed,
            "evidence_class": transport.get("evidence_class") or "missing",
            "test_count": int(transport.get("tests") or 0),
            "bargein_preroll_samples": int(
                transport.get("automatic_bargein_preroll_samples") or 0
            ),
            "continued_capture_samples": int(transport.get("continued_capture_samples") or 0),
            "next_turn_input_samples": int(transport.get("next_turn_input_samples") or 0),
            "physical_browser_verified": False,
            "official_full_duplex_evidence": False,
            "claim_boundary": final_audit.get("claim_boundary") or {},
        },
        "submission_strategy": {
            "production_candidate": (
                "qwen4b_v2_cascade" if qwen4b_final["verified"] else "cascade"
            ),
            "production_candidate_evidence": (
                "automatic_same_family_llm_as_judge_qwen4b_prompt_v2_final_test"
                if qwen4b_final["verified"]
                else "real_service_group_disjoint_validation_mechanics"
            ),
            "direct_moshi_role": "experimental_negative_result_with_positive_learning_signal",
            "new_direct_training_before_deadline_recommended": False,
            "reason": (
                "The frozen Qwen3-4B prompt-v2 cascade passed every predeclared automatic "
                "semantic final-test gate after a passed development eligibility stage; "
                "v6.2 learned its train-only objective but failed every direct runtime row. "
                "A new direct run would risk the evidence freeze without a validated remedy."
            ),
        },
        "release": {
            "source_code_license_selected": source_license_selected,
            "license_files": license_files,
            "immutable_submission_tag_created": release_attestation["verified"],
            "submission_tag_attestation": release_attestation,
        },
        "required_to_complete": [text for gate, text in required.items() if not gates[gate]],
        "remaining_work_classification": {
            "active_evidence_gates": [text for gate, text in required.items() if not gates[gate]],
            "conditional_closeout_after_evidence": [
                "Reconcile the thesis manuscript and generated tables with final evidence.",
                "Freeze the final evidence bundle, commit/tag it, push it, and verify a fresh clone.",
            ],
            "submission_critical_now": [
                "Use the cascade as the submitted working system.",
                "Copy the exact positive and negative result tables into the thesis manuscript.",
                "Run the final repository audit, freeze the evidence snapshot, tag, and push.",
            ],
            "research_extension_not_deadline_critical": [
                "Redesign and validate a direct Persian Moshi training objective on a "
                "larger representative split."
            ],
            "waived_not_completed": [
                "40-row window listening QA",
                "24-row interaction listening QA",
                "24-pair assistant-audio listening QA",
            ],
            "platform_limited_not_a_scientific_gate": [
                "Private-repository main-branch protection requires an eligible GitHub plan."
            ],
        },
        "artifact_provenance": _artifact_provenance(project),
    }
    if out is not None:
        destination = Path(out)
        if not destination.is_absolute():
            destination = project / destination
        write_json(destination, payload)
    return payload


def render_evidence_summary(payload: dict[str, Any]) -> str:
    """Render the aggregate JSON as a compact, conservative Markdown report."""

    gates = payload.get("gates") or {}
    data = payload.get("data") or {}
    working = payload.get("working_system") or {}
    direct = payload.get("direct_moshi") or {}
    detector = payload.get("detector") or {}
    latency = payload.get("latency_and_hardware") or {}
    study = payload.get("human_study") or {}
    release = payload.get("release") or {}
    retention = payload.get("local_artifact_retention") or {}
    strategy = payload.get("submission_strategy") or {}
    transport = payload.get("duplex_transport") or {}
    cascade_analysis = payload.get("cascade_descriptive_analysis") or {}
    cascade_intelligibility = payload.get("cascade_intelligibility_proxy") or {}
    qwen4b_final = payload.get("qwen4b_v2_automatic_final_test") or {}
    v1_loss_evidence = direct.get("v1_automatic_heldout_loss_evidence") or {}
    v1_total_loss = v1_loss_evidence.get("selected_total_loss") or {}
    split_counts = data.get("split_counts") or {}
    lines = [
        "# Authoritative project evidence status",
        "",
        f"Generated: `{payload.get('generated_at')}`",
        "",
        f"**Verdict:** `{payload.get('verdict')}`. Thesis-ready: "
        f"**{str(bool(payload.get('thesis_ready'))).lower()}**.",
        "",
        "This table is generated from `EVIDENCE_STATUS.json`. Component and synthetic "
        "proxies are never promoted to official end-to-end evidence.",
        "",
        "## Acceptance gates",
        "",
        "| Gate | Passed |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {'yes' if passed else 'no'} |" for name, passed in gates.items())
    lines.extend(
        [
            "",
            "## Dataset and export",
            "",
            "| Measure | Value |",
            "|---|---:|",
            f"| Conversation source pairs | {data.get('conversation_pairs', 0):,} |",
            f"| Conversation source hours | {data.get('conversation_source_hours', 0):.3f} |",
            f"| Audited exported pairs | {data.get('exported_pairs', 0):,} |",
            f"| Audited exported hours | {data.get('exported_hours', 0):.3f} |",
            f"| Train / validation / test pairs | {split_counts.get('train', 0):,} / "
            f"{split_counts.get('val', 0):,} / {split_counts.get('test', 0):,} |",
            f"| Session-group split leaks | {data.get('group_split_leaks', 0)} |",
            f"| Machine audit failures | {sum((data.get('audit_failure_counts') or {}).values())} |",
            "| Split policy | source-session-group isolated |",
            f"| Automatic integrity audit | {'pass' if data.get('automatic_integrity_audit_passed') else 'fail'} |",
            f"| Human listening QA complete | {'yes' if data.get('human_review_complete') else 'no (waived)'} |",
            f"| Balanced v2 pairs (not used by completed Moshi runs) | {(data.get('balanced_v2') or {}).get('pairs', 0):,} |",
            f"| Balanced v2 source-pair hours | {(data.get('balanced_v2') or {}).get('source_pair_hours', 0):.3f} |",
            f"| Balanced v2 largest source share | {100 * (data.get('balanced_v2') or {}).get('largest_channel_share', 0):.1f}% |",
            "",
            "## Working speech-to-speech system",
            "",
            f"The real-service cascade passed {working.get('passed_rows', 0)}/"
            f"{working.get('panel_rows', 0)} fixed group-disjoint validation rows with "
            f"{working.get('fallback_rows', 0)} rule-fallback rows. Minimum reply Persian-script "
            f"fraction was {working.get('minimum_reply_persian_letter_fraction')}; minimum reply "
            f"audio RMS was {working.get('minimum_reply_audio_rms')}. This is a positive working "
            "user-turn prototype result and out-of-sample mechanics check, not semantic or "
            "population-level generalization, human quality, physical-target, or "
            "official browser-latency evidence.",
            "",
            "## Automatic semantic final test",
            "",
            f"The frozen Qwen3-4B prompt-v2 cascade result is "
            f"{'verified and passed' if qwen4b_final.get('verified') else 'not verified'} "
            f"on {qwen4b_final.get('rows', 0)} group-disjoint test rows with "
            f"{qwen4b_final.get('judge_calls_valid', 0)} valid judge calls and "
            f"{qwen4b_final.get('judge_calls_failed', 0)} failures. Mean relevance improved "
            f"from {qwen4b_final.get('base_relevance_mean')} for the frozen 0.5B baseline to "
            f"{qwen4b_final.get('candidate_relevance_mean')}; candidate coherence was "
            f"{qwen4b_final.get('candidate_coherence_mean')}. Relevance gain was "
            f"{qwen4b_final.get('relevance_gain')}, coherence gain "
            f"{qwen4b_final.get('coherence_gain')}, paired relevance win rate "
            f"{qwen4b_final.get('relevance_win_rate')}, and relevance-at-least-two rate "
            f"{qwen4b_final.get('relevance_at_least_two_rate')}. Every predeclared automatic "
            "engineering gate passed. This is an automatic same-family LLM-as-judge proxy, "
            "not a human or independent benchmark; it does not establish factuality, safety, "
            "naturalness, population usefulness, physical-4090 fit, or browser latency.",
            "",
            "## Privacy-safe descriptive error analysis",
            "",
            f"The hash-bound post-hoc analysis is "
            f"{'verified' if cascade_analysis.get('verified') else 'not verified'}. It found "
            f"{cascade_analysis.get('passed_rows', 0)}/{cascade_analysis.get('panel_rows', 0)} "
            "automatic mechanics successes and "
            f"{cascade_analysis.get('failed_rows', 0)} mechanics failures. Of the nine rows, "
            f"{(cascade_analysis.get('descriptive_risk_counts') or {}).get('transcript_over_1000_characters', 0)} "
            "had transcripts over 1,000 characters, "
            f"{(cascade_analysis.get('descriptive_risk_counts') or {}).get('reply_audio_over_8_seconds', 0)} "
            "had replies over 8 seconds, and "
            f"{(cascade_analysis.get('descriptive_risk_counts') or {}).get('full_turn_over_10_seconds', 0)} "
            "took over 10 seconds for complete generation. Transcript length and full-turn "
            f"time had descriptive Pearson r="
            f"{cascade_analysis.get('transcript_length_full_turn_pearson_r')}. This small, "
            "post-hoc association is not inferential, and full-turn time is not official "
            "first-audio latency. Plaintext was not read or emitted; this nine-row "
            "post-hoc analysis did not measure semantics. The separate frozen automatic "
            "semantic final test is reported above; pronunciation, naturalness, human "
            "quality, elderly performance, and population generalization remain unmeasured.",
            "",
            "## Automatic synthesized-speech intelligibility proxy",
            "",
            f"The predeclared NeMo round-trip measurement is "
            f"{'verified' if cascade_intelligibility.get('verified') else 'not verified'} "
            f"on {cascade_intelligibility.get('panel_rows', 0)} exact reproduced validation "
            f"outputs with {cascade_intelligibility.get('asr_failures', 0)} ASR failures. "
            f"Micro WER="
            f"{(cascade_intelligibility.get('word_error_rate') or {}).get('micro')} over "
            f"{(cascade_intelligibility.get('word_error_rate') or {}).get('reference_words', 0)} "
            f"reference words; micro CER="
            f"{(cascade_intelligibility.get('character_error_rate') or {}).get('micro')} over "
            f"{(cascade_intelligibility.get('character_error_rate') or {}).get('reference_characters', 0)} "
            "reference characters. No quality threshold was introduced after observation. "
            "This automatic single-voice ASR proxy does not establish human intelligibility, "
            "pronunciation, naturalness, semantic relevance, or population generalization.",
            "",
            "## Duplex transport",
            "",
            "The automated WebSocket regression "
            f"{'passed' if transport.get('automated_websocket_regression_passed') else 'is pending'} "
            f"with {transport.get('test_count', 0)} tests. It verifies "
            f"{transport.get('bargein_preroll_samples', 0):,} pre-roll samples plus "
            f"{transport.get('continued_capture_samples', 0):,} continued samples become a "
            f"{transport.get('next_turn_input_samples', 0):,}-sample next-turn input, together "
            "with acknowledgement ingestion, identity telemetry, cancellation, reconnect, "
            "state isolation, error recovery, and metrics-only no-WAV retention. Actual "
            "browser source-stop, microphone, and physical-target evidence remain pending.",
            "",
            "## Direct Moshi trials and ablations",
            "",
            "| Trial | Scope | Context | Rank | Seed | Embeddings trained | Candidates | "
            "Val chunks/candidate | Runtime pass/fail per 9 | Min val loss | Eligible |",
            "|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|",
        ]
    )
    for trial in direct.get("trials") or []:
        config = trial.get("configuration") or {}
        if config.get("selective_embeddings"):
            embeddings = ", ".join(config["selective_embeddings"])
        elif config.get("upstream_ft_embed"):
            embeddings = "upstream broad embedding switch"
        else:
            embeddings = "none"
        losses = [
            value for value in trial.get("fixed_validation_losses") or [] if value is not None
        ]
        minimum_loss = f"{min(losses):.6f}" if losses else "n/a"
        pass_counts = trial.get("runtime_panel_pass_counts") or []
        failure_counts = trial.get("runtime_panel_failure_counts") or []
        pass_fail = (
            ", ".join(
                f"{passed}/{failed}"
                for passed, failed in zip(pass_counts, failure_counts, strict=True)
            )
            if pass_counts
            else "n/a"
        )
        lines.append(
            f"| {trial.get('version')} | {trial.get('evidence_scope') or 'n/a'} | "
            f"{config.get('context_seconds') or 'n/a'} | "
            f"{config.get('lora_rank') or 'n/a'} | {config.get('seed') or 'n/a'} | "
            f"{embeddings} | {trial.get('candidate_count', 0)} | "
            f"{trial.get('fixed_validation_samples_per_candidate') or 'n/a'} | "
            f"{pass_fail} | {minimum_loss} | "
            f"{'yes' if trial.get('deployment_eligible') else 'no'} |"
        )
    lines.extend(
        [
            "",
            "V1 was selected under its frozen loss protocol but failed later autoregressive "
            "runtime validation. V2–v5 each failed closed with zero eligible checkpoints. "
            "V6.2 is a deliberately in-sample capacity diagnostic: its best post-step-50 text "
            "loss reduction was "
            f"{100.0 * float((direct.get('trials') or [{}])[-1].get('text_loss_reduction_fraction') or 0.0):.2f}% "
            "but every checkpoint passed 0/9 direct-runtime rows. The v2–v6.2 final-test "
            "firewalls remain closed.",
            "",
            f"V1 automatic held-out loss evidence used {v1_loss_evidence.get('rows', 0)} "
            f"rows / {v1_loss_evidence.get('chunks', 0)} chunks: selected total-loss mean "
            f"{v1_total_loss.get('mean')} with 95% CI "
            f"[{v1_total_loss.get('ci95_low')}, {v1_total_loss.get('ci95_high')}]. "
            "This is automatic loss evidence only and does not repair the runtime failure "
            "or support a perceptual/deployment claim.",
            "",
            "## Local artifact retention",
            "",
            f"The verified post-finalization cleanup reclaimed "
            f"{retention.get('observed_reclaimed_bytes', 0) / (1024**3):.2f} GiB and "
            f"retains {retention.get('representative_adapter_count', 0)} representative "
            "v1–v5 adapter tensors. Retained steps: "
            + "; ".join(
                f"{version}={steps}"
                for version, steps in (retention.get("representative_adapter_steps") or {}).items()
            )
            + ".",
            "",
            f"V6.2 separately retains {retention.get('current_v6_2_checkpoint_count', 0)} "
            "current diagnostic checkpoints; they postdate that cleanup receipt.",
            "",
            "All retained adapter hashes match committed evidence, and all scientific "
            "results, configurations, runtime outputs, and certificates remain. The full "
            "set of negative intermediate tensors is intentionally not retained; exact "
            "tensor recreation would require rerunning the frozen training recipes. "
            "Historical candidate counts and verdicts remain attested by their committed "
            "certificates.",
            "",
            "## Detector, latency, and study evidence",
            "",
            "| Area | Current evidence | Official gate |",
            "|---|---|---:|",
            f"| Detector | synthetic accuracy={detector.get('synthetic_accuracy')}; "
            f"recorded proxy n={(detector.get('recorded_proxy') or {}).get('n', 0)} / "
            f"{(detector.get('recorded_proxy') or {}).get('sessions', 0)} sessions, "
            f"accuracy={(detector.get('recorded_proxy') or {}).get('accuracy')}, "
            f"F1={(detector.get('recorded_proxy') or {}).get('interrupt_f1')}, "
            f"session CI={((detector.get('recorded_proxy') or {}).get('session_block_bootstrap_ci95') or {}).get('accuracy')}; "
            "automatic labels, not official ground truth | "
            f"{'pass' if detector.get('recorded_group_heldout_gate_passed') else 'pending'} |",
            f"| Hardware/latency | {latency.get('current_gpu') or 'unknown'}; "
            f"official E2E rows={latency.get('official_e2e_rows', 0)} | "
            f"{'pass' if gates.get('physical_12_to_24_gb_fit_and_live_latency') else 'pending'} |",
            f"| Human study | participants={study.get('participants', 0)}, aged 60+="
            f"{study.get('elderly_participants', 0)}, turns={study.get('turns', 0)}, "
            f"valid ratings={study.get('ratings', 0)}, complete ratings="
            f"{study.get('complete_ratings', 0)}, invalid rows="
            f"{study.get('invalid_rating_rows', 0)} | "
            f"{'pass' if study.get('official_ready') else 'pending'} |",
            f"| Project license | {', '.join(release.get('license_files') or []) or 'not selected'} | "
            f"{'pass' if release.get('source_code_license_selected') else 'pending'} |",
            f"| Immutable release | "
            f"{(release.get('submission_tag_attestation') or {}).get('tag') or 'not attested'} | "
            f"{'pass' if release.get('immutable_submission_tag_created') else 'pending'} |",
            "",
            "## Reporting contract",
            "",
            "Denominators and observed failures are shown above. Model seeds are reported "
            "per trial; the Moshi data split is frozen by `source_session_id`. Confidence "
            "intervals are reported where estimable, including event and session-block "
            "intervals for the recorded automatic-label proxy. Official latency, "
            "independently labeled detector, and human-study intervals remain null "
            "because their qualifying denominators are zero. Exact timing and acceptance "
            "definitions are in "
            "`docs/METRICS.md`.",
            "",
            "## Submission architecture decision",
            "",
            f"Production candidate: `{strategy.get('production_candidate')}`. Direct Moshi "
            f"role: `{strategy.get('direct_moshi_role')}`. Starting another direct-model "
            "training run before the deadline is not recommended because no validated "
            "corrective hypothesis remains, while the prompt-v2 cascade has passed both its "
            "eligibility stage and frozen automatic semantic final test.",
            "",
            "## Remaining requirements",
            "",
        ]
    )
    remaining = payload.get("required_to_complete") or []
    lines.extend(f"- {item}" for item in remaining)
    if not remaining:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "No claim should exceed these evidence classes. In particular, the H100 "
            "measurements are engineering/training evidence, not physical-4090 official "
            "latency evidence; synthetic accuracy and the recorded automatic-label proxy are "
            "not independently labeled official detector evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evidence_summary(path: str | Path, payload: dict[str, Any]) -> None:
    """Write the human-readable view paired with an evidence-status payload."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_evidence_summary(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
