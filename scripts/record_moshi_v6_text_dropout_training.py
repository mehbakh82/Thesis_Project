#!/usr/bin/env python3
"""Attest a completed Moshi v6.2 scheduled text-input-dropout training run."""

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
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_moshi_adapter import json_object, sha256_file  # noqa: E402

EXPECTED_STEPS = [50, 100, 150, 200]
FULL_TEXT_PARAMETERS = {
    "depformer_text_emb.weight",
    "text_emb.weight",
    "text_linear.frozen_W.weight",
}
EXPECTED_ENVIRONMENT = {
    "MOSHI_DISTRIBUTED_BACKEND": "gloo",
    "MOSHI_PERSIAN_TEXT_ADAPTATION": "1",
    "MOSHI_TEXT_EMBEDDINGS_ONLY": "0",
    "MOSHI_AUDIO_LOSS_WEIGHT": "0.1",
    "MOSHI_OPTIMIZER_CPU_OFFLOAD": "1",
    "MOSHI_OPTIMIZER_CPU_OFFLOAD_AUDIT": (
        "checkpoints/moshi_v6_text_dropout/"
        "optimizer_cpu_offload_audit.json"
    ),
    "MOSHI_TEXT_INPUT_DROPOUT_START": "0.25",
    "MOSHI_TEXT_INPUT_DROPOUT_END": "0.75",
    "MOSHI_TEXT_INPUT_DROPOUT_FORWARDS": "800",
    "MOSHI_TEXT_INPUT_DROPOUT_SEED": "20260902",
    "MOSHI_TEXT_INPUT_DROPOUT_AUDIT": (
        "checkpoints/moshi_v6_text_dropout/input_dropout_audit.jsonl"
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def comparable_config(value: dict[str, Any]) -> dict[str, Any]:
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
            "max_steps",
            "gradient_checkpointing",
            "optim",
            "seed",
            "log_freq",
            "eval_freq",
            "do_eval",
            "do_ckpt",
            "ckpt_freq",
            "num_ckpt_keep",
            "save_adapters",
            "run_dir",
            "overwrite_run_dir",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/moshi_v6_text_dropout"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/moshi_h100_v6_text_dropout.yaml")
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
        "--probe",
        type=Path,
        default=Path("results/hardware/moshi_v6_text_dropout_probe.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v6_text_dropout_training.json"),
    )
    args = parser.parse_args()

    run_dir = (ROOT / args.run_dir).resolve()
    config_path = (ROOT / args.config).resolve()
    policy_path = (ROOT / args.policy).resolve()
    preflight_path = (ROOT / args.preflight).resolve()
    probe_path = (ROOT / args.probe).resolve()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v6.2 training evidence: {out_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved_args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(resolved_args, dict):
        raise ValueError("training and resolved configurations must be mappings")
    policy = json_object(policy_path)
    preflight = json_object(preflight_path)
    probe = json_object(probe_path)
    train_metrics_path = run_dir / "metrics.train.jsonl"
    eval_metrics_path = run_dir / "metrics.eval.jsonl"
    audit_path = run_dir / "input_dropout_audit.jsonl"
    train_metrics = read_jsonl(train_metrics_path)
    eval_metrics = read_jsonl(eval_metrics_path)
    audit = read_jsonl(audit_path)
    optimizer_audit_path = run_dir / "optimizer_cpu_offload_audit.json"
    optimizer_audit = json_object(optimizer_audit_path)
    environment = {key: os.environ.get(key) for key in EXPECTED_ENVIRONMENT}

    checkpoint_reports = []
    config_hashes = set()
    adapter_key_sets = []
    for step in EXPECTED_STEPS:
        consolidated = run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
        adapter_path = consolidated / "lora.safetensors"
        adapter_config_path = consolidated / "config.json"
        with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
            keys = set(adapter.keys())
        lora_keys = {key for key in keys if "lora" in key}
        full_parameter_keys = keys - lora_keys
        config_hash = sha256_file(adapter_config_path)
        config_hashes.add(config_hash)
        adapter_key_sets.append(keys)
        checkpoint_reports.append(
            {
                "step": step,
                "adapter_path": adapter_path.relative_to(ROOT).as_posix(),
                "adapter_bytes": adapter_path.stat().st_size,
                "adapter_sha256": sha256_file(adapter_path),
                "config_sha256": config_hash,
                "tensor_count": len(keys),
                "lora_tensor_count": len(lora_keys),
                "full_parameters": sorted(full_parameter_keys),
                "schema_exact": (
                    len(keys) == 677
                    and len(lora_keys) == 674
                    and full_parameter_keys == FULL_TEXT_PARAMETERS
                ),
            }
        )

    audit_calls = [int(row["train_forward_call"]) for row in audit]
    expected_audit_calls = [1, 200, 400, 600, 800]
    expected_probabilities = [0.25 + 0.5 * (call - 1) / 799 for call in expected_audit_calls]
    audit_probabilities = [float(row["probability"]) for row in audit]
    final_eligible = int(audit[-1]["cumulative_eligible_tokens"])
    final_dropped = int(audit[-1]["cumulative_dropped_tokens"])
    dropout_fraction = final_dropped / max(final_eligible, 1)

    requirements = {
        "preflight_and_probe_passed": (
            preflight.get("status") == "passed"
            and preflight.get("preflight_passes") is True
            and probe.get("status") == "passed"
            and probe.get("probe_passes") is True
            and preflight.get("final_test_accessed") is False
            and probe.get("final_test_accessed") is False
        ),
        "runtime_environment_exact": environment == EXPECTED_ENVIRONMENT,
        "resolved_args_exact": comparable_config(resolved_args) == comparable_config(config),
        "training_reached_200_with_finite_losses": (
            max(int(row["step"]) for row in train_metrics) == 200
            and all(math.isfinite(float(row["loss"])) for row in train_metrics)
        ),
        "raw_eval_steps_and_losses_exact": (
            [int(row["step"]) for row in eval_metrics] == EXPECTED_STEPS
            and all(math.isfinite(float(row["eval_loss"])) for row in eval_metrics)
        ),
        "all_four_checkpoints_exact": (
            [row["step"] for row in checkpoint_reports] == EXPECTED_STEPS
            and all(row["schema_exact"] for row in checkpoint_reports)
            and all(
                int(row["adapter_bytes"]) >= int(policy["estimated_adapter_bytes"])
                for row in checkpoint_reports
            )
            and len(config_hashes) == 1
            and all(keys == adapter_key_sets[0] for keys in adapter_key_sets)
        ),
        "scheduled_forwards_audited": audit_calls == expected_audit_calls,
        "scheduled_probabilities_exact": all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(audit_probabilities, expected_probabilities, strict=True)
        ),
        "corruption_executed_at_every_milestone": (
            all(int(row["eligible_tokens_this_call"]) > 0 for row in audit)
            and all(int(row["dropped_tokens_this_call"]) > 0 for row in audit)
            and 0.35 < dropout_fraction < 0.65
        ),
        "targets_audio_and_eval_unchanged": (
            (policy.get("text_input_dropout") or {}).get("audio_inputs_mutated") is False
            and all(
                row.get("targets_mutated") is False
                and row.get("evaluation_corrupted") is False
                and row.get("preserved_token_ids_at_most") == 3
                for row in audit
            )
        ),
        "cpu_offloaded_fp32_adamw_audited": (
            optimizer_audit.get("algorithm") == "AdamW"
            and optimizer_audit.get("optimizer_step_calls") == 200
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
        "final_test_not_accessed": True,
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(requirements.values()) else "failed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "moshi_v6_text_dropout",
        "diagnostic_only": True,
        "scientific_validation_evidence": False,
        "generalization_claim_allowed": False,
        "final_test_accessed": False,
        "runtime_environment": environment,
        "training": {
            "max_step": max(int(row["step"]) for row in train_metrics),
            "logged_train_rows": len(train_metrics),
            "raw_eval_steps": [int(row["step"]) for row in eval_metrics],
            "peak_allocated_gb": max(
                float(row.get("peak_allocated_mem") or 0.0) for row in train_metrics
            ),
            "final_logged_loss": float(train_metrics[-1]["loss"]),
        },
        "dropout_audit": {
            "calls": audit_calls,
            "probabilities": audit_probabilities,
            "cumulative_eligible_tokens": final_eligible,
            "cumulative_dropped_tokens": final_dropped,
            "observed_dropout_fraction": dropout_fraction,
            "targets_mutated": False,
            "audio_inputs_mutated": False,
            "evaluation_corrupted": False,
        },
        "optimizer_cpu_offload": optimizer_audit,
        "checkpoints": checkpoint_reports,
        "requirements": requirements,
        "training_passes": all(requirements.values()),
        "artifacts": {
            "config_sha256": sha256_file(config_path),
            "policy_sha256": sha256_file(policy_path),
            "preflight_sha256": sha256_file(preflight_path),
            "probe_sha256": sha256_file(probe_path),
            "resolved_args_sha256": sha256_file(run_dir / "args.yaml"),
            "train_metrics_sha256": sha256_file(train_metrics_path),
            "eval_metrics_sha256": sha256_file(eval_metrics_path),
            "dropout_audit_sha256": sha256_file(audit_path),
            "optimizer_audit_sha256": sha256_file(optimizer_audit_path),
            "recorder_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["training_passes"] is not True:
        raise RuntimeError(f"v6.2 training attestation failed: {requirements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
