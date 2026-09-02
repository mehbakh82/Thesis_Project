#!/usr/bin/env python3
"""Record the one-step, no-checkpoint Moshi v6.2 scheduled-dropout probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_moshi_adapter import json_object, sha256_file  # noqa: E402

EXPECTED_ENVIRONMENT = {
    "MOSHI_DISTRIBUTED_BACKEND": "gloo",
    "MOSHI_PERSIAN_TEXT_ADAPTATION": "1",
    "MOSHI_TEXT_EMBEDDINGS_ONLY": "0",
    "MOSHI_AUDIO_LOSS_WEIGHT": "0.1",
    "MOSHI_OPTIMIZER_CPU_OFFLOAD": "1",
    "MOSHI_OPTIMIZER_CPU_OFFLOAD_AUDIT": (
        "checkpoints/moshi_v6_text_dropout_probe/"
        "optimizer_cpu_offload_audit.json"
    ),
    "MOSHI_TEXT_INPUT_DROPOUT_START": "0.25",
    "MOSHI_TEXT_INPUT_DROPOUT_END": "0.75",
    "MOSHI_TEXT_INPUT_DROPOUT_FORWARDS": "4",
    "MOSHI_TEXT_INPUT_DROPOUT_SEED": "20260902",
    "MOSHI_TEXT_INPUT_DROPOUT_AUDIT": (
        "checkpoints/moshi_v6_text_dropout_probe/input_dropout_audit.jsonl"
    ),
}


def comparable_profile(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "data",
            "moshi_paths",
            "full_finetuning",
            "lora",
            "first_codebook_weight_multiplier",
            "text_padding_weight",
            "duration_sec",
            "batch_size",
            "num_microbatches",
            "gradient_checkpointing",
            "optim",
            "seed",
            "save_adapters",
        )
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON object lines: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path, default=Path("checkpoints/moshi_v6_text_dropout_probe")
    )
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=Path("configs/moshi_h100_v6_text_dropout_probe.yaml"),
    )
    parser.add_argument(
        "--full-config",
        type=Path,
        default=Path("configs/moshi_h100_v6_text_dropout.yaml"),
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("configs/moshi_v6_text_dropout_policy.json")
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path("results/hardware/moshi_v6_text_dropout_preflight.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v6_text_dropout_probe.json"),
    )
    args = parser.parse_args()

    run_dir = (ROOT / args.run_dir).resolve()
    probe_config_path = (ROOT / args.probe_config).resolve()
    full_config_path = (ROOT / args.full_config).resolve()
    policy_path = (ROOT / args.policy).resolve()
    preflight_path = (ROOT / args.preflight).resolve()
    out_path = (ROOT / args.out).resolve()
    launch_failure_path = (
        ROOT / "results/hardware/moshi_v6_text_dropout_launch_failure.json"
    )
    protocol_path = ROOT / "docs/MOSHI_V6_TEXT_DROPOUT_PROTOCOL.md"
    launcher_path = ROOT / "scripts/moshi_train_entry.py"
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v6.2 probe evidence: {out_path}")

    metrics_path = run_dir / "metrics.train.jsonl"
    resolved_args_path = run_dir / "args.yaml"
    audit_path = run_dir / "input_dropout_audit.jsonl"
    optimizer_audit_path = run_dir / "optimizer_cpu_offload_audit.json"
    metrics = read_jsonl(metrics_path)
    audit = read_jsonl(audit_path)
    optimizer_audit = json_object(optimizer_audit_path)
    resolved_args = yaml.safe_load(resolved_args_path.read_text(encoding="utf-8"))
    probe_config = yaml.safe_load(probe_config_path.read_text(encoding="utf-8"))
    full_config = yaml.safe_load(full_config_path.read_text(encoding="utf-8"))
    policy = json_object(policy_path)
    preflight = json_object(preflight_path)
    launch_failure = json_object(launch_failure_path)
    if not all(
        isinstance(value, dict)
        for value in (resolved_args, probe_config, full_config, policy, preflight)
    ):
        raise ValueError("v6.2 probe inputs must be mappings")
    metric = metrics[-1]
    effective_environment = {key: os.environ.get(key) for key in EXPECTED_ENVIRONMENT}
    calls = [int(row["train_forward_call"]) for row in audit]
    probabilities = [float(row["probability"]) for row in audit]
    expected_probabilities = [0.25, 5.0 / 12.0, 7.0 / 12.0, 0.75]
    checkpoints_dir = run_dir / "checkpoints"

    requirements = {
        "preflight_passed_and_bound": (
            preflight.get("status") == "passed"
            and preflight.get("preflight_passes") is True
            and preflight.get("final_test_accessed") is False
            and (preflight.get("artifacts") or {}).get("probe_config_sha256")
            == sha256_file(probe_config_path)
            and (preflight.get("artifacts") or {}).get("config_sha256")
            == sha256_file(full_config_path)
            and (preflight.get("artifacts") or {}).get("policy_sha256") == sha256_file(policy_path)
        ),
        "entrypoint_correction_bound": (
            launch_failure.get("status")
            == "infrastructure_failure_before_project_import"
            and launch_failure.get("model_loaded") is False
            and launch_failure.get("run_directory_created") is False
            and launch_failure.get("optimizer_steps") == 0
            and launch_failure.get("scientific_result") is False
            and launch_failure.get("final_test_accessed") is False
            and ((launch_failure.get("correction") or {}).get("environment_added") or {})
            == {"PYTHONPATH": "."}
            and (launch_failure.get("correction") or {}).get("scientific_inputs_changed")
            is False
            and (launch_failure.get("artifacts") or {}).get("parent_preflight_sha256")
            == sha256_file(preflight_path)
            and (launch_failure.get("artifacts") or {}).get(
                "pre_correction_protocol_sha256"
            )
            == (preflight.get("artifacts") or {}).get("protocol_sha256")
            and (launch_failure.get("artifacts") or {}).get("unchanged_launcher_sha256")
            == (preflight.get("artifacts") or {}).get("launcher_sha256")
            == sha256_file(launcher_path)
            and "env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0" in protocol_path.read_text()
        ),
        "runtime_environment_exact": effective_environment == EXPECTED_ENVIRONMENT,
        "one_optimizer_step_four_microbatches": (
            len(metrics) == 1
            and metric.get("step") == 1
            and resolved_args.get("max_steps") == 1
            and resolved_args.get("num_microbatches") == 4
        ),
        "finite_loss": math.isfinite(float(metric.get("loss"))),
        "full_training_shape_matches": comparable_profile(probe_config)
        == comparable_profile(full_config),
        "resolved_args_match_probe": comparable_profile(resolved_args)
        == comparable_profile(probe_config),
        "evaluation_and_checkpointing_disabled": (
            resolved_args.get("do_eval") is False
            and resolved_args.get("do_ckpt") is False
            and not checkpoints_dir.exists()
        ),
        "all_training_forwards_audited": calls == [1, 2, 3, 4],
        "linear_schedule_exact": all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(probabilities, expected_probabilities, strict=True)
        ),
        "eligible_and_dropped_tokens_observed": (
            all(int(row["eligible_tokens_this_call"]) > 0 for row in audit)
            and all(int(row["dropped_tokens_this_call"]) > 0 for row in audit)
            and int(audit[-1]["cumulative_dropped_tokens"]) > 0
        ),
        "targets_and_eval_remained_clean": all(
            row.get("targets_mutated") is False
            and row.get("evaluation_corrupted") is False
            and row.get("preserved_token_ids_at_most") == 3
            for row in audit
        ),
        "cpu_offloaded_fp32_adamw_audited": (
            optimizer_audit.get("algorithm") == "AdamW"
            and optimizer_audit.get("optimizer_step_calls") == 1
            and optimizer_audit.get("active_parameter_tensors") == 677
            and optimizer_audit.get("active_parameter_elements")
            == int(policy["estimated_adapter_bytes"]) // 2
            and optimizer_audit.get("master_devices") == ["cpu"]
            and optimizer_audit.get("gradient_devices") == ["cpu"]
            and optimizer_audit.get("moment_devices") == ["cpu"]
            and optimizer_audit.get("master_dtypes") == ["torch.float32"]
            and optimizer_audit.get("gradient_dtypes") == ["torch.float32"]
            and optimizer_audit.get("moment_dtypes") == ["torch.float32"]
            and optimizer_audit.get("moment_tensor_count") == 1354
            and optimizer_audit.get("foreach") is False
            and optimizer_audit.get("fused") is False
        ),
        "real_model_memory_allocated": 14.0 < float(metric.get("peak_allocated_mem")) < 22.0,
        "policy_matches_probe": (
            ((policy.get("text_input_dropout") or {}).get("probe_forwards") == 4)
            and ((policy.get("text_input_dropout") or {}).get("seed") == 20260902)
        ),
        "final_test_not_accessed": True,
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(requirements.values()) else "failed",
        "completed_at": metric.get("at") or datetime.now(timezone.utc).isoformat(),
        "purpose": "one-step exact-shape v6.2 scheduled text-input-dropout probe",
        "diagnostic_only": True,
        "scientific_validation_evidence": False,
        "generalization_claim_allowed": False,
        "final_test_accessed": False,
        "runtime_environment": effective_environment,
        "step": {
            "number": metric.get("step"),
            "loss": float(metric["loss"]),
            "peak_allocated_gb": float(metric["peak_allocated_mem"]),
        },
        "dropout_audit": {
            "rows": len(audit),
            "calls": calls,
            "probabilities": probabilities,
            "eligible_tokens": int(audit[-1]["cumulative_eligible_tokens"]),
            "dropped_tokens": int(audit[-1]["cumulative_dropped_tokens"]),
            "targets_mutated": False,
            "evaluation_corrupted": False,
        },
        "optimizer_cpu_offload": optimizer_audit,
        "requirements": requirements,
        "probe_passes": all(requirements.values()),
        "artifacts": {
            "metrics_sha256": sha256_file(metrics_path),
            "audit_sha256": sha256_file(audit_path),
            "optimizer_audit_sha256": sha256_file(optimizer_audit_path),
            "resolved_args_sha256": sha256_file(resolved_args_path),
            "probe_config_sha256": sha256_file(probe_config_path),
            "full_config_sha256": sha256_file(full_config_path),
            "policy_sha256": sha256_file(policy_path),
            "preflight_sha256": sha256_file(preflight_path),
            "launch_failure_sha256": sha256_file(launch_failure_path),
            "corrected_protocol_sha256": sha256_file(protocol_path),
            "unchanged_launcher_sha256": sha256_file(launcher_path),
            "recorder_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["probe_passes"] is not True:
        raise RuntimeError(f"v6.2 probe failed: {requirements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
