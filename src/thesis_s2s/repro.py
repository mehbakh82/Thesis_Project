"""Reproducibility, upstream-lock, and physical-GPU readiness reports."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from thesis_s2s.config import load_yaml, project_root
from thesis_s2s.data.qa_policy import load_qa_waiver
from thesis_s2s.metrics import gpu_inventory, write_json

SHA256_RE = re.compile(r"^[0-9a-f]{40}$")
SNAPSHOT_DIRS = ("configs", "docs", "scripts", "src", "tests")
SNAPSHOT_FILES = (
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "requirements-moshi.lock",
    "third_party/UPSTREAMS.lock.json",
    "results/corpus_audit.json",
    "results/prepared_episode_audit.json",
    "results/prepared_reserve_tabaghe16_audit.json",
    "results/conversation_selection_audit.json",
    "results/diarization_smoke_h100.json",
    "results/hardware/current_preflight.json",
    "results/hardware/moshi_environment.json",
    "results/hardware/moshi_h100_smoke.json",
    "results/hardware/moshi_h100_profile_probe.json",
    "results/conversation_rights_report.json",
    "results/conversation_source_authorization_report_combined.json",
    "results/conversation_noise_report.json",
    "results/conversation_reserve_tabaghe16_noise_report.json",
    "results/conversation_yield_estimate.json",
    "results/conversation_reserve_tabaghe16_yield_estimate.json",
    "results/conversation_reserve_yield_selection.json",
    "results/conversation_yield_estimate_combined.json",
    "results/diarized_episode_audit.json",
    "results/diarized_episode_audit_combined.json",
    "results/diarized_episode_audit_combined_authorized.json",
    "results/conversation_audit.json",
    "results/moshi_export_report.json",
    "results/bargein/heldout_report.json",
    "results/bargein/recorded_heldout_report.json",
    "results/bargein/feature_heldout_report.json",
    "results/eval/human_study.json",
)
SNAPSHOT_GLOBS = (
    "models/piper/*",
    "checkpoints/llama_omni2_fa/*.pt",
    "results/bargein/*.pkl",
)
PACKAGE_NAMES = (
    "accelerate",
    "fastapi",
    "numpy",
    "peft",
    "pydantic",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
    "uvicorn",
    "soundfile",
    "piper-tts",
)
WINDOW_QA_CHECK_FIELDS = (
    "speaker_count_correct",
    "speaker_assignment_correct",
    "caption_acceptable",
    "overlap_annotation_correct",
)
INTERACTION_QA_CHECK_FIELDS = (
    "speakers_distinct_correct",
    "user_turn_boundary_correct",
    "response_turn_boundary_correct",
    "audible_overlap_correct",
)
YES_VALUES = {"1", "true", "yes", "y", "بله"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_upstream_lock(path: Path | None = None) -> dict:
    lock_path = Path(path or project_root() / "third_party" / "UPSTREAMS.lock.json")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("upstreams"), list):
        raise ValueError("unsupported upstream lock schema")
    names: set[str] = set()
    for entry in payload["upstreams"]:
        required = ("name", "repository", "revision", "license")
        if not all(isinstance(entry.get(key), str) and entry[key] for key in required):
            raise ValueError(f"upstream entry is incomplete: {entry!r}")
        if not SHA256_RE.fullmatch(entry["revision"]):
            raise ValueError(
                f"upstream revision must be a full 40-character commit: {entry['name']}"
            )
        if entry["name"] in names:
            raise ValueError(f"duplicate upstream name: {entry['name']}")
        names.add(entry["name"])
    return payload


def verify_upstream_lock(
    path: Path | None = None,
    *,
    checkouts_root: Path | None = None,
) -> dict:
    """Validate immutable pins and optionally compare local checkout HEADs."""

    payload = load_upstream_lock(path)
    checks = []
    for entry in payload["upstreams"]:
        check = {
            "name": entry["name"],
            "revision": entry["revision"],
            "pin_valid": True,
            "checkout_present": None,
            "checkout_matches": None,
        }
        if checkouts_root is not None:
            checkout = Path(checkouts_root) / entry["checkout_dir"]
            check["checkout_present"] = checkout.is_dir()
            if checkout.is_dir():
                try:
                    head = subprocess.run(
                        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    ).stdout.strip()
                    check["checkout_head"] = head
                    check["checkout_matches"] = head == entry["revision"]
                except (OSError, subprocess.SubprocessError) as exc:
                    check["checkout_error"] = str(exc)[:200]
                    check["checkout_matches"] = False
        checks.append(check)
    return {
        "schema_version": 1,
        "valid": all(row["pin_valid"] and row["checkout_matches"] is not False for row in checks),
        "checks": checks,
    }


def _conversation_status(root: Path) -> dict:
    final_path = root / "results" / "conversation_audit.json"
    staging_path = root / "results" / "diarized_episode_audit_combined_authorized.json"
    yield_path = root / "results" / "conversation_yield_estimate_combined.json"
    status: dict[str, object] = {
        "present": final_path.is_file(),
        "thesis_coverage_ok": False,
        "training_ready_under_qa_waiver": False,
        "path": str(final_path),
        "staging_audit_path": str(staging_path),
        "yield_estimate_path": str(yield_path),
    }
    if final_path.is_file():
        try:
            report = json.loads(final_path.read_text(encoding="utf-8"))
            status.update(
                {
                    "thesis_coverage_ok": report.get("thesis_coverage_ok") is True,
                    "training_ready_under_qa_waiver": report.get("training_ready_under_qa_waiver")
                    is True,
                    "qa_policy": report.get("qa_policy"),
                    "pairs": report.get("pairs"),
                    "hours": report.get("hours"),
                    "claims": report.get("claims") or {},
                }
            )
        except json.JSONDecodeError as exc:
            status["error"] = str(exc)

    if staging_path.is_file():
        try:
            staging = json.loads(staging_path.read_text(encoding="utf-8"))
            requirements = staging.get("requirements") or {}
            automatic = {
                key: value
                for key, value in requirements.items()
                if key != "manual_qa_sample_present"
            }
            status["staging"] = {
                "present": True,
                "windows": staging.get("windows"),
                "episodes": staging.get("episodes"),
                "automatic_multi_speaker_hours": staging.get("automatic_multi_speaker_hours"),
                "reference_aligned_hours": staging.get("reference_aligned_hours"),
                "automatic_gates_pass": bool(automatic) and all(automatic.values()),
                "manual_qa_complete": requirements.get("manual_qa_sample_present") is True,
            }
        except json.JSONDecodeError as exc:
            status["staging"] = {"present": True, "error": str(exc)}
    else:
        status["staging"] = {"present": False}

    if yield_path.is_file():
        try:
            estimate = json.loads(yield_path.read_text(encoding="utf-8"))
            status["yield_estimate"] = {
                "present": True,
                "pairs": estimate.get("estimated_pairs"),
                "hours": estimate.get("estimated_pair_hours"),
                "hours_100_to_200": estimate.get("hours_100_to_200") is True,
                "non_mutating": estimate.get("non_mutating_estimate") is True,
            }
        except json.JSONDecodeError as exc:
            status["yield_estimate"] = {"present": True, "error": str(exc)}
    else:
        status["yield_estimate"] = {"present": False}
    return status


def _json_report(path: Path) -> dict:
    """Load a JSON evidence object without turning malformed evidence into a pass."""

    status: dict[str, object] = {"path": str(path), "present": path.is_file()}
    if not path.is_file():
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        status["error"] = str(exc)[:200]
        return status
    if not isinstance(payload, dict):
        status["error"] = "expected a JSON object"
        return status
    status["report"] = payload
    return status


def _qa_sheet_status(path: Path, *, interaction: bool) -> dict:
    """Count complete reviewer decisions in an internal QA sheet."""

    status: dict[str, object] = {
        "path": str(path),
        "present": path.is_file(),
        "kind": "interaction_boundary" if interaction else "window",
        "rows": 0,
        "completed_rows": 0,
        "pending_or_invalid_rows": 0,
        "duplicate_ids": 0,
        "complete": False,
    }
    if not path.is_file():
        return status
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        status["error"] = str(exc)[:200]
        return status

    identifier = "candidate_id" if interaction else "window_id"
    check_fields = INTERACTION_QA_CHECK_FIELDS if interaction else WINDOW_QA_CHECK_FIELDS
    seen: set[str] = set()
    completed = 0
    duplicate_ids = 0
    passing_interruptions = 0
    status_counts: dict[str, int] = {}
    for row in rows:
        row_id = str(row.get(identifier) or "").strip()
        if row_id in seen or not row_id:
            duplicate_ids += 1
        seen.add(row_id)
        decision = str(row.get("review_status") or "").strip().lower()
        status_counts[decision or "blank"] = status_counts.get(decision or "blank", 0) + 1
        reviewer = str(row.get("reviewer_id") or "").strip()
        checks_supplied = all(str(row.get(field) or "").strip() for field in check_fields)
        if interaction:
            label = str(row.get("corrected_label") or "").strip().lower()
            complete = bool(reviewer) and (
                decision == "fail"
                or (
                    decision == "pass" and checks_supplied and label in {"interrupt", "backchannel"}
                )
            )
            if (
                complete
                and decision == "pass"
                and label == "interrupt"
                and all(
                    str(row.get(field) or "").strip().lower() in YES_VALUES
                    for field in check_fields
                )
            ):
                passing_interruptions += 1
        else:
            complete = (
                bool(reviewer)
                and decision in {"pass", "fail"}
                and (decision == "fail" or checks_supplied)
            )
        completed += int(complete)

    status.update(
        {
            "rows": len(rows),
            "completed_rows": completed,
            "pending_or_invalid_rows": len(rows) - completed,
            "duplicate_ids": duplicate_ids,
            "status_counts": status_counts,
            "complete": bool(rows) and completed == len(rows) and duplicate_ids == 0,
        }
    )
    if interaction:
        status["passing_interruption_rows"] = passing_interruptions
    return status


def _application_status(path: Path, sheet: dict, *, interaction: bool) -> dict:
    """Validate that a completed QA sheet was applied by the fail-closed pipeline."""

    loaded = _json_report(path)
    payload = loaded.pop("report", None)
    loaded["valid"] = False
    if not isinstance(payload, dict):
        return loaded
    counts = payload.get("counts") or {}
    rows = int(sheet.get("rows") or 0)
    common_valid = (
        payload.get("fail_closed") is True
        and sheet.get("complete") is True
        and counts.get("incomplete", 0) == 0
        and counts.get("qa_decisions") == rows
        and rows > 0
    )
    unknown_field = "unknown_candidate_ids" if interaction else "unknown_window_ids"
    valid = common_valid and counts.get(unknown_field, 0) == 0
    if interaction:
        valid = (
            valid
            and payload.get("qa_complete") is True
            and payload.get("verified_interruption_present") is True
        )
    loaded.update(
        {
            "valid": valid,
            "qa_complete": payload.get("qa_complete") if interaction else common_valid,
            "verified_interruption_present": (
                payload.get("verified_interruption_present") if interaction else None
            ),
            "counts": counts,
        }
    )
    return loaded


def _nvidia_smi() -> dict:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        fields = [
            "name",
            "driver_version",
            "memory_total_mib",
            "memory_used_mib",
            "memory_free_mib",
            "utilization_gpu_percent",
        ]
        rows = output.splitlines()
        devices = []
        for row in rows:
            values = [value.strip() for value in row.split(",")]
            if len(values) != len(fields):
                continue
            parsed: dict[str, object] = dict(zip(fields, values, strict=True))
            for field in fields[2:]:
                try:
                    parsed[field] = int(str(parsed[field]))
                except ValueError:
                    pass
            devices.append(parsed)
        return {
            "available": True,
            "fields": fields,
            "rows": rows,
            "devices": devices,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)[:200]}


def _base_model_status(root: Path, entry: dict | None, training_config: dict) -> dict:
    model_paths = training_config.get("moshi_paths") or {}
    configured = {
        Path(value).name: root / value
        for key in ("moshi_path", "mimi_path", "tokenizer_path")
        if isinstance((value := model_paths.get(key)), str) and value
    }
    checks = {}
    for filename, expected in ((entry or {}).get("weights") or {}).items():
        path = configured.get(filename)
        actual_bytes = path.stat().st_size if path is not None and path.is_file() else None
        actual_sha256 = sha256_file(path) if path is not None and path.is_file() else None
        checks[filename] = {
            "path": str(path) if path is not None else None,
            "expected_bytes": expected.get("bytes"),
            "actual_bytes": actual_bytes,
            "expected_sha256": expected.get("sha256"),
            "actual_sha256": actual_sha256,
            "matches_pin": actual_bytes == expected.get("bytes")
            and actual_sha256 == expected.get("sha256"),
        }
    config_value = model_paths.get("config_path")
    config_path = root / config_value if isinstance(config_value, str) else None
    config_present = config_path is not None and config_path.is_file()
    return {
        "revision": (entry or {}).get("revision"),
        "files": checks,
        "architecture_config": str(config_path) if config_path is not None else None,
        "architecture_config_present": config_present,
        "valid": bool(checks)
        and all(row["matches_pin"] for row in checks.values())
        and config_present,
    }


def gpu_preflight(out_json: Path | None = None) -> dict:
    """Report independent training and target-hardware evaluation readiness."""

    root = project_root()
    gpu = gpu_inventory()
    torch_info: dict[str, object] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        }
    except Exception as exc:  # pragma: no cover - depends on deployment
        torch_info = {"error": str(exc)[:200]}
    device = str(gpu.get("device") or "")
    physical_eligible = bool(gpu.get("official_size"))
    upstream = verify_upstream_lock(checkouts_root=root / "third_party" / "checkouts")
    conversation = _conversation_status(root)
    qa_waiver_path = root / "configs" / "conversation_qa_waiver.yaml"
    qa_waiver: dict | None = None
    qa_waiver_status: dict[str, object] = {
        "path": str(qa_waiver_path),
        "present": qa_waiver_path.is_file(),
        "valid": False,
    }
    if qa_waiver_path.is_file():
        try:
            qa_waiver = load_qa_waiver(qa_waiver_path)
            qa_waiver_status.update(qa_waiver or {})
        except (OSError, TypeError, ValueError) as exc:
            qa_waiver_status["error"] = str(exc)[:200]
    training_config = load_yaml(root / "configs" / "moshi_h100.yaml")
    locked_upstreams = load_upstream_lock()["upstreams"]
    moshi_entry = next(
        (entry for entry in locked_upstreams if entry.get("name") == "Moshi-Finetune"),
        None,
    )
    voice_entry = next(
        (entry for entry in locked_upstreams if entry.get("name") == "Mana-Persian-Piper"),
        None,
    )
    base_entry = next(
        (entry for entry in locked_upstreams if entry.get("name") == "Moshika-PyTorch-BF16"),
        None,
    )
    base_model = _base_model_status(root, base_entry, training_config)
    voice_weights = (voice_entry or {}).get("weights") or {}
    voice_file = str(voice_weights.get("file") or "")
    voice_path = root / "models" / "piper" / voice_file if voice_file else None
    voice_expected_sha256 = str(voice_weights.get("sha256") or "")
    voice_actual_sha256 = (
        sha256_file(voice_path) if voice_path is not None and voice_path.is_file() else None
    )
    voice_pinned = bool(
        voice_actual_sha256
        and voice_expected_sha256
        and voice_actual_sha256 == voice_expected_sha256
    )
    trainer_path = root / "third_party" / "checkouts" / "moshi-finetune" / "train.py"
    launcher_path = root / "scripts" / "moshi_train_entry.py"
    official_trainer_declared = bool(
        moshi_entry and str(moshi_entry.get("training_entrypoint") or "").strip()
    )
    reviewed_trainer_available = official_trainer_declared and trainer_path.is_file()

    environment_path = root / "results" / "hardware" / "moshi_environment.json"
    environment = _json_report(environment_path)
    environment_payload = environment.pop("report", None)
    environment_gates = (
        environment_payload.get("gates") if isinstance(environment_payload, dict) else {}
    ) or {}
    environment.update(
        {
            "valid": isinstance(environment_payload, dict)
            and environment_payload.get("valid") is True
            and bool(environment_gates)
            and all(value is True for value in environment_gates.values()),
            "gpu": environment_payload.get("gpu")
            if isinstance(environment_payload, dict)
            else None,
            "torch_cuda": environment_payload.get("torch_cuda")
            if isinstance(environment_payload, dict)
            else None,
            "gates": environment_gates,
        }
    )

    smoke_path = root / "results" / "hardware" / "moshi_h100_smoke.json"
    smoke = _json_report(smoke_path)
    smoke_payload = smoke.pop("report", None)
    smoke_artifacts = smoke_payload.get("artifacts") if isinstance(smoke_payload, dict) else {}
    smoke_artifacts = smoke_artifacts or {}
    smoke_hashes_current = (
        smoke_artifacts.get("environment_report_sha256") == sha256_file(environment_path)
        if environment_path.is_file()
        else False
    ) and (
        smoke_artifacts.get("source_config_sha256")
        == sha256_file(root / "configs" / "moshi_h100_smoke.yaml")
        if (root / "configs" / "moshi_h100_smoke.yaml").is_file()
        else False
    )
    smoke.update(
        {
            "valid": isinstance(smoke_payload, dict)
            and smoke_payload.get("status") == "passed"
            and smoke_payload.get("wiring_gate_passes") is True
            and smoke_payload.get("scientific_evidence") is False
            and smoke_hashes_current,
            "status": smoke_payload.get("status") if isinstance(smoke_payload, dict) else None,
            "wiring_gate_passes": smoke_payload.get("wiring_gate_passes")
            if isinstance(smoke_payload, dict)
            else None,
            "scientific_evidence": smoke_payload.get("scientific_evidence")
            if isinstance(smoke_payload, dict)
            else None,
            "hardware": smoke_payload.get("hardware") if isinstance(smoke_payload, dict) else None,
            "launcher": smoke_payload.get("launcher") if isinstance(smoke_payload, dict) else None,
            "recorded_hashes_match_current_inputs": smoke_hashes_current,
        }
    )

    window_sheet = _qa_sheet_status(
        root / "data" / "processed" / "manifests" / "conversation_manual_qa.csv",
        interaction=False,
    )
    interaction_sheet = _qa_sheet_status(
        root / "data" / "processed" / "manifests" / "conversation_interruption_qa.csv",
        interaction=True,
    )
    window_application = _application_status(
        root / "results" / "manual_qa_report.json",
        window_sheet,
        interaction=False,
    )
    interaction_application = _application_status(
        root / "results" / "interaction_qa_report.json",
        interaction_sheet,
        interaction=True,
    )
    export = _json_report(root / "results" / "moshi_export_report.json")
    export_payload = export.pop("report", None)
    export_requirements = (
        export_payload.get("requirements") if isinstance(export_payload, dict) else {}
    ) or {}
    export_waiver_requirements = (
        export_payload.get("waiver_requirements") if isinstance(export_payload, dict) else {}
    ) or {}
    export.update(
        {
            "valid": isinstance(export_payload, dict)
            and export_payload.get("final_training_ready") is True
            and bool(export_requirements)
            and all(value is True for value in export_requirements.values()),
            "final_training_ready": export_payload.get("final_training_ready")
            if isinstance(export_payload, dict)
            else None,
            "exported_pairs": export_payload.get("exported_pairs")
            if isinstance(export_payload, dict)
            else None,
            "exported_hours": export_payload.get("exported_hours")
            if isinstance(export_payload, dict)
            else None,
            "valid_under_qa_waiver": isinstance(export_payload, dict)
            and export_payload.get("training_ready_under_qa_waiver") is True
            and bool(export_waiver_requirements)
            and all(value is True for value in export_waiver_requirements.values()),
            "training_ready_under_qa_waiver": export_payload.get("training_ready_under_qa_waiver")
            if isinstance(export_payload, dict)
            else None,
            "qa_policy": export_payload.get("qa_policy")
            if isinstance(export_payload, dict)
            else None,
            "requirements": export_requirements,
            "waiver_requirements": export_waiver_requirements,
            "claims": export_payload.get("claims") if isinstance(export_payload, dict) else {},
        }
    )

    nvidia_smi = _nvidia_smi()
    devices = nvidia_smi.get("devices") or []
    current_device = devices[0] if devices else {}
    free_memory_mib = current_device.get("memory_free_mib")
    profile_probe = _json_report(root / "results" / "hardware" / "moshi_h100_profile_probe.json")
    profile_payload = profile_probe.pop("report", None)
    profile_hardware = (
        profile_payload.get("hardware") if isinstance(profile_payload, dict) else {}
    ) or {}
    profile_artifacts = (
        profile_payload.get("artifacts") if isinstance(profile_payload, dict) else {}
    ) or {}
    profile_inputs = {
        "probe_config_sha256": root / "configs" / "moshi_h100_profile_probe.yaml",
        "full_config_sha256": root / "configs" / "moshi_h100.yaml",
        "environment_report_sha256": environment_path,
        "export_report_sha256": root / "results" / "moshi_export_report.json",
    }
    profile_hashes_current = all(
        path.is_file() and profile_artifacts.get(key) == sha256_file(path)
        for key, path in profile_inputs.items()
    )
    profile_peak_gb = profile_hardware.get("peak_allocated_gb")
    profile_valid = (
        isinstance(profile_payload, dict)
        and profile_payload.get("status") == "passed"
        and profile_payload.get("full_profile_gate_passes") is True
        and profile_payload.get("scientific_evidence") is False
        and profile_hashes_current
    )
    try:
        if not isinstance(free_memory_mib, (int, float, str)):
            raise ValueError("current free-memory evidence is incomplete")
        free_memory_gb = round(float(free_memory_mib) / 1024, 3)
    except (TypeError, ValueError):
        free_memory_gb = None
    try:
        if not isinstance(profile_peak_gb, (int, float, str)):
            raise ValueError("profile peak-memory evidence is incomplete")
        required_with_margin_gb = round(float(profile_peak_gb) + 4.0, 3)
        current_headroom_sufficient = (
            profile_valid
            and free_memory_gb is not None
            and free_memory_gb >= required_with_margin_gb
        )
    except (TypeError, ValueError):
        required_with_margin_gb = None
        current_headroom_sufficient = False
    profile_probe.update(
        {
            "valid": profile_valid,
            "status": profile_payload.get("status") if isinstance(profile_payload, dict) else None,
            "peak_allocated_gb": profile_peak_gb,
            "recorded_hashes_match_current_inputs": profile_hashes_current,
        }
    )
    scheduling = {
        "full_profile_probe": profile_probe,
        "current_free_memory_gb": free_memory_gb,
        "required_free_memory_gb_including_4gb_margin": required_with_margin_gb,
        "current_headroom_sufficient": current_headroom_sufficient,
        "note": (
            "The 5-second rank-8 wiring smoke is a lower bound, not a memory certificate "
            "for the 20-second rank-64 embedding-tuning profile. Run the full-profile "
            "one-step probe after final data export and only while the shared H100 has "
            "sufficient free memory."
        ),
    }

    hardware_gates = {
        "cuda_available": torch_info.get("cuda_available") is True,
        "bf16_supported": torch_info.get("bf16_supported") is True,
    }
    infrastructure_gates = {
        "upstream_revisions_pinned": upstream["valid"],
        "reviewed_training_entrypoint_available": reviewed_trainer_available,
        "project_training_launcher_available": launcher_path.is_file(),
        "base_model_files_pinned": base_model["valid"],
        "assistant_voice_target_pinned": voice_pinned,
        "isolated_moshi_environment_valid": environment["valid"],
        "one_step_official_wiring_smoke_passes": smoke["valid"],
    }
    strict_data_gates = {
        "window_qa_sheet_complete": window_sheet["complete"],
        "window_qa_applied_fail_closed": window_application["valid"],
        "interaction_qa_sheet_complete": interaction_sheet["complete"],
        "interaction_qa_applied_with_verified_interruption": interaction_application["valid"],
        "conversational_data_audit_passes": conversation["thesis_coverage_ok"],
        "moshi_export_final_training_ready": export["valid"],
    }
    conversation_policy = conversation.get("qa_policy") or {}
    export_policy = export.get("qa_policy") or {}
    conversation_claims = conversation.get("claims") or {}
    export_claims = export.get("claims") or {}
    waiver_hash = qa_waiver.get("sha256") if qa_waiver is not None else None
    conversation_uses_current_waiver = isinstance(conversation_policy, dict) and (
        conversation_policy.get("sha256") == waiver_hash
    )
    export_uses_current_waiver = isinstance(export_policy, dict) and (
        export_policy.get("sha256") == waiver_hash
    )
    waiver_selected = qa_waiver is not None and (
        conversation_uses_current_waiver or export_uses_current_waiver
    )
    qa_waiver_status["selected_for_training"] = waiver_selected
    waiver_assets_preserved = all(
        path.is_file()
        for path in (
            root / "data" / "processed" / "manifests" / "conversation_manual_qa.csv",
            root / "data" / "processed" / "manifests" / "conversation_interruption_qa.csv",
            root / "docs" / "MANUAL_QA_FA.md",
            root / "docs" / "INTERRUPTION_QA_FA.md",
            root / "scripts" / "review_interaction_candidate.py",
        )
    )
    waiver_data_gates = {
        "documented_qa_waiver_valid": qa_waiver is not None,
        "best_practice_qa_assets_preserved": waiver_assets_preserved,
        "conversation_audit_ready_under_qa_waiver": conversation.get(
            "training_ready_under_qa_waiver"
        )
        is True,
        "conversation_audit_uses_current_waiver": conversation_uses_current_waiver,
        "moshi_export_ready_under_qa_waiver": export.get("valid_under_qa_waiver") is True,
        "moshi_export_uses_current_waiver": export_uses_current_waiver,
        "human_verification_claims_disabled": qa_waiver is not None
        and all(
            claims.get(key) is False
            for claims in (conversation_claims, export_claims)
            for key in (
                "human_verified_data",
                "human_verified_interruptions",
                "strict_thesis_data_coverage",
            )
        ),
    }
    training_data_policy = (
        qa_waiver["policy"] if waiver_selected and qa_waiver is not None else "strict"
    )
    data_gates = waiver_data_gates if waiver_selected else strict_data_gates
    training_gates = {**hardware_gates, **infrastructure_gates, **data_gates}
    strict_training_gates = {
        **hardware_gates,
        **infrastructure_gates,
        **strict_data_gates,
    }
    evaluation_gates = {
        "physical_gpu_12_to_24_gb": physical_eligible,
        "cuda_available": torch_info.get("cuda_available") is True,
    }
    failed_adaptation_gates = [name for name, passed in training_gates.items() if not passed]
    failed_strict_data_gates = [name for name, passed in strict_data_gates.items() if not passed]
    adaptation_run_ready = all(training_gates.values())
    strict_adaptation_run_ready = all(strict_training_gates.values())
    adaptation_launch_safe_now = adaptation_run_ready and current_headroom_sufficient
    report = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "nvidia_smi": nvidia_smi,
        "torch": torch_info,
        "is_rtx_4090": "4090" in device,
        "upstreams": upstream,
        "conversation_data": conversation,
        "human_review": {
            "window_sheet": window_sheet,
            "window_application": window_application,
            "interaction_sheet": interaction_sheet,
            "interaction_application": interaction_application,
        },
        "qa_waiver": qa_waiver_status,
        "moshi_environment": environment,
        "moshi_wiring_smoke": smoke,
        "moshi_export": export,
        "scheduling": scheduling,
        "training_config": str(root / "configs" / "moshi_h100.yaml"),
        "training_base_model": (training_config.get("moshi_paths") or {}).get("hf_repo_id"),
        "training_objective": {
            "model": "Moshika 7B LoRA via pinned Moshi-Finetune",
            "input": "timed stereo user/assistant conversations",
            "supervision": "assistant text tokens and Mimi assistant-speech tokens",
            "learns": "Persian response generation, response audio, turn timing, and overlap behavior",
            "not_asr": True,
        },
        "official_upstream_training_entrypoint_available": official_trainer_declared,
        "configured_trainer_entrypoint": str(trainer_path),
        "configured_project_launcher": str(launcher_path),
        "base_model": base_model,
        "assistant_voice_target": {
            "path": str(voice_path) if voice_path is not None else None,
            "expected_sha256": voice_expected_sha256 or None,
            "actual_sha256": voice_actual_sha256,
            "matches_pin": voice_pinned,
        },
        "hardware_gates": hardware_gates,
        "infrastructure_gates": infrastructure_gates,
        "training_data_policy": training_data_policy,
        "strict_data_gates": strict_data_gates,
        "waiver_data_gates": waiver_data_gates,
        "data_gates": data_gates,
        "training_gates": training_gates,
        "evaluation_gates": evaluation_gates,
        "strict_training_gates": strict_training_gates,
        "training_hardware_ready": all(hardware_gates.values()),
        "trainer_stack_ready": all(infrastructure_gates.values()),
        "training_data_ready": all(data_gates.values()),
        "strict_training_data_ready": all(strict_data_gates.values()),
        "training_data_ready_under_qa_waiver": waiver_selected and all(waiver_data_gates.values()),
        "evaluation_hardware_ready": all(evaluation_gates.values()),
        "adaptation_run_ready": adaptation_run_ready,
        "strict_adaptation_run_ready": strict_adaptation_run_ready,
        "adaptation_launch_safe_now": adaptation_launch_safe_now,
        "failed_adaptation_gates": failed_adaptation_gates,
        "failed_strict_data_gates": failed_strict_data_gates,
        "adaptation_status": (
            "ready_to_launch"
            if adaptation_launch_safe_now
            else "prerequisites_ready_waiting_for_safe_gpu_headroom"
            if adaptation_run_ready
            else "blocked_by_failed_training_gates"
        ),
        "note": (
            "The H100 is valid training hardware. A physical 12–24 GB GPU is required only "
            "for the final target-hardware fit and live-latency evidence. Moshi-Finetune is "
            "the pinned genuine response-audio trainer. Supervisor-approved internal "
            "training is recorded separately from raw-data redistribution, which remains "
            "disabled. The student-authorized QA waiver changes only limited internal "
            "training readiness; strict thesis readiness and all human-verification "
            "claims remain independently visible and false until the preserved reviews "
            "are actually completed."
        ),
    }
    if out_json is not None:
        write_json(out_json, report)
    return report


def _git_state(root: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"available": True, "commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)[:200]}


def release_snapshot(out_json: Path | None = None) -> dict:
    """Hash reproducibility-critical code, docs, configs, and evidence reports."""

    root = project_root()
    paths: set[Path] = set()
    for directory in SNAPSHOT_DIRS:
        base = root / directory
        if base.is_dir():
            paths.update(
                path
                for path in base.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not any(part.endswith(".egg-info") for part in path.parts)
                and path.suffix not in {".pyc", ".orig", ".rej"}
            )
    paths.update(root / name for name in SNAPSHOT_FILES if (root / name).is_file())
    for pattern in SNAPSHOT_GLOBS:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    files = {
        str(path.relative_to(root)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(paths)
    }
    packages: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git_state(root),
        "gpu": gpu_inventory(),
        "nvidia_smi": _nvidia_smi(),
        "packages": packages,
        "files": files,
        "file_count": len(files),
    }
    target = Path(out_json or root / "results" / "release" / "snapshot.json")
    write_json(target, report)
    return report
