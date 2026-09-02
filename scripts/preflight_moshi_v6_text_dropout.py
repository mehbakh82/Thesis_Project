#!/usr/bin/env python3
"""Fail-closed preflight for the Moshi v6.2 scheduled text-input-dropout diagnostic."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.preflight_moshi_v3 import (  # noqa: E402
    gpu_memory,
    json_object,
    sha256_file,
    write_report,
    yaml_object,
)

GIB = 1024**3
V6_RUNTIME_SHA256 = "fc3c394a76e8866baf40fe649a0ec5c78d0b857aa1307d3c2cbd9caf16961d6b"
V6_1_RUNTIME_SHA256 = "220e487e0a62e232c4ba479e87abdace4d3df4515d2a94c647371a882a4bc748"
FULL_TEXT_PARAMETERS = {
    "depformer_text_emb.weight",
    "text_emb.weight",
    "text_linear.frozen_W.weight",
}


def host_available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    raise RuntimeError("could not read MemAvailable from /proc/meminfo")


def comparable_training_config(value: dict[str, Any]) -> dict[str, Any]:
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
            "overwrite_run_dir",
        )
    }


def comparable_probe_shape(value: dict[str, Any]) -> dict[str, Any]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/moshi_h100_v6_text_dropout.yaml")
    )
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=Path("configs/moshi_h100_v6_text_dropout_probe.yaml"),
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("configs/moshi_v6_text_dropout_policy.json")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("docs/MOSHI_V6_TEXT_DROPOUT_PROTOCOL.md")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v6_text_dropout_preflight.json"),
    )
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    probe_config_path = (ROOT / args.probe_config).resolve()
    policy_path = (ROOT / args.policy).resolve()
    protocol_path = (ROOT / args.protocol).resolve()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v6.2 preflight evidence: {out_path}")

    baseline_config_path = ROOT / "configs/moshi_h100_v6_overfit.yaml"
    data_report_path = ROOT / "results/moshi_v6_overfit_data.json"
    v6_runtime_path = ROOT / "results/moshi_v6_overfit_runtime.json"
    v6_1_runtime_path = ROOT / "results/moshi_v6_text_greedy_runtime.json"
    v6_probe_path = ROOT / "results/hardware/moshi_v6_overfit_probe.json"
    launcher_path = ROOT / "scripts/moshi_train_entry.py"
    dropout_module_path = ROOT / "scripts/moshi_text_input_dropout.py"

    config = yaml_object(config_path)
    probe_config = yaml_object(probe_config_path)
    baseline_config = yaml_object(baseline_config_path)
    policy = json_object(policy_path)
    data_report = json_object(data_report_path)
    v6_runtime = json_object(v6_runtime_path)
    v6_1_runtime = json_object(v6_1_runtime_path)
    v6_probe = json_object(v6_probe_path)
    manifest = ROOT / str((config.get("data") or {}).get("train_data"))

    status = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    launch_commit = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gpu = gpu_memory()
    disk = shutil.disk_usage(ROOT)
    required_free_bytes = 4 * int(policy["estimated_adapter_bytes"]) + 2 * GIB
    host_available_bytes = host_available_memory_bytes()

    baseline_comparison = comparable_training_config(baseline_config)
    candidate_comparison = comparable_training_config(config)
    expected_environment = {
        "MOSHI_DISTRIBUTED_BACKEND": "gloo",
        "MOSHI_PERSIAN_TEXT_ADAPTATION": "1",
        "MOSHI_TEXT_EMBEDDINGS_ONLY": "0",
        "MOSHI_AUDIO_LOSS_WEIGHT": "0.1",
        "MOSHI_OPTIMIZER_CPU_OFFLOAD": "1",
        "MOSHI_TEXT_INPUT_DROPOUT_START": "0.25",
        "MOSHI_TEXT_INPUT_DROPOUT_END": "0.75",
        "MOSHI_TEXT_INPUT_DROPOUT_SEED": "20260902",
    }
    output_paths = (
        ROOT / "results/hardware/moshi_v6_text_dropout_probe.json",
        ROOT / "results/hardware/moshi_v6_text_dropout_training.json",
        ROOT / "results/moshi_v6_text_dropout_reevaluation.json",
        ROOT / "results/moshi_v6_text_dropout_runtime.json",
    )
    launcher_text = launcher_path.read_text(encoding="utf-8")
    dropout_text = dropout_module_path.read_text(encoding="utf-8")
    dropout_policy = policy.get("text_input_dropout") or {}
    resource_policy = policy.get("resource_execution_amendment") or {}

    requirements = {
        "launch_worktree_clean": not status.strip(),
        "parent_v6_exact_negative": (
            sha256_file(v6_runtime_path) == V6_RUNTIME_SHA256
            and v6_runtime.get("status") == "passed"
            and v6_runtime.get("diagnostic_outcome") == "negative"
            and v6_runtime.get("final_test_accessed") is False
        ),
        "parent_v6_1_exact_negative": (
            sha256_file(v6_1_runtime_path) == V6_1_RUNTIME_SHA256
            and v6_1_runtime.get("status") == "passed"
            and v6_1_runtime.get("diagnostic_outcome") == "negative"
            and v6_1_runtime.get("final_test_accessed") is False
        ),
        "controlled_training_config_matches_v6": candidate_comparison == baseline_comparison,
        "only_run_directory_differs_outside_comment": (
            config.get("run_dir") == "checkpoints/moshi_v6_text_dropout"
            and baseline_config.get("run_dir") == "checkpoints/moshi_v6_overfit"
        ),
        "probe_shape_matches_full": comparable_probe_shape(probe_config)
        == comparable_probe_shape(config),
        "probe_controls_exact": (
            probe_config.get("max_steps") == 1
            and probe_config.get("log_freq") == 1
            and probe_config.get("do_eval") is False
            and probe_config.get("do_ckpt") is False
            and probe_config.get("run_dir") == "checkpoints/moshi_v6_text_dropout_probe"
        ),
        "train_only_data_current": (
            data_report.get("passes") is True
            and data_report.get("final_test_accessed") is False
            and (data_report.get("output") or {}).get("rows") == 32
            and (data_report.get("output") or {}).get("manifest_sha256") == sha256_file(manifest)
            and (config.get("data") or {}).get("train_data")
            == (config.get("data") or {}).get("eval_data")
            and "test" not in manifest.as_posix().lower()
        ),
        "parameter_scope_and_objective_exact": (
            policy.get("mode") == "persian_text_head_adaptation_with_scheduled_text_input_dropout"
            and set(policy.get("trainable_full_parameters") or []) == FULL_TEXT_PARAMETERS
            and policy.get("frozen_audio_embeddings") is True
            and policy.get("audio_loss_weight") == 0.1
            and policy.get("expected_lora_tensor_count") == 674
            and policy.get("expected_total_adapter_tensor_count") == 677
            and policy.get("estimated_adapter_bytes") == 977709056
        ),
        "dropout_policy_exact": (
            dropout_policy.get("start_probability") == 0.25
            and dropout_policy.get("end_probability") == 0.75
            and dropout_policy.get("full_training_forwards") == 800
            and dropout_policy.get("probe_forwards") == 4
            and dropout_policy.get("seed") == 20260902
            and dropout_policy.get("replacement_token_id") == 3
            and dropout_policy.get("special_token_ids_preserved") == [0, 1, 2, 3]
            and dropout_policy.get("audio_inputs_mutated") is False
            and dropout_policy.get("targets_mutated") is False
            and dropout_policy.get("evaluation_corrupted") is False
            and (policy.get("launcher_environment") or {}) == expected_environment
        ),
        "optimizer_resource_amendment_exact": (
            resource_policy.get("model_shape_changed") is False
            and resource_policy.get("trainable_parameters_changed") is False
            and resource_policy.get("optimizer_algorithm_changed") is False
            and resource_policy.get("optimizer_hyperparameters_changed") is False
            and resource_policy.get("optimizer_dtype") == "float32"
            and resource_policy.get("optimizer_device") == "cpu"
            and resource_policy.get("foreach") is False
            and resource_policy.get("fused") is False
            and resource_policy.get("expected_active_parameter_elements")
            == int(policy["estimated_adapter_bytes"]) // 2
            and resource_policy.get("expected_optimizer_moment_tensors") == 1354
            and resource_policy.get("minimum_h100_free_mib") == 20 * 1024
            and resource_policy.get("minimum_host_available_bytes") == 12 * GIB
            and isinstance(resource_policy.get("kernel_numerics_caveat"), str)
        ),
        "launcher_and_hook_fail_closed": all(
            marker in launcher_text + dropout_text
            for marker in (
                "configure_scheduled_text_input_dropout",
                "scheduled text-input dropout requires Persian-text adaptation",
                "input_codes = codes.clone()",
                "if not model.training",
                "text_codes > TEXT_INPUT_PRESERVED_MAX_ID",
                "exceeded frozen train forwards",
                "targets_mutated",
                "evaluation_corrupted",
                "_configure_cpu_offloaded_adamw",
                "MOSHI_OPTIMIZER_CPU_OFFLOAD_AUDIT",
                "CPU-offloaded AdamW placement or dtype invariant failed",
                "moment_devices",
            )
        ),
        "protocol_frozen": protocol_path.is_file(),
        "prior_exact_profile_probe_passed": (
            v6_probe.get("status") == "passed"
            and v6_probe.get("probe_passes") is True
            and v6_probe.get("final_test_accessed") is False
            and (v6_probe.get("step") or {}).get("peak_allocated_gb") < 23.0
        ),
        "run_directories_absent": not (
            (ROOT / str(config.get("run_dir"))).exists()
            or (ROOT / str(probe_config.get("run_dir"))).exists()
        ),
        "result_outputs_absent": not any(path.exists() for path in output_paths),
        "disk_headroom_passes": disk.free >= required_free_bytes,
        "host_memory_headroom_passes": host_available_bytes
        >= int(resource_policy["minimum_host_available_bytes"]),
        "h100_headroom_passes": (
            gpu["name"] == "NVIDIA H100 NVL"
            and int(gpu["free_mib"]) >= int(resource_policy["minimum_h100_free_mib"])
        ),
        "final_test_not_accessed": True,
    }

    report = {
        "schema_version": 1,
        "status": "passed" if all(requirements.values()) else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "launch_commit": launch_commit,
        "diagnostic_only": True,
        "scientific_validation_evidence": False,
        "generalization_claim_allowed": False,
        "final_test_accessed": False,
        "experiment": {
            "name": "moshi_v6_text_dropout",
            "candidate_steps": [50, 100, 150, 200],
            "training_rows": 32,
            "training_forwards": 800,
        },
        "storage": {
            "free_bytes": disk.free,
            "required_free_bytes": required_free_bytes,
            "required_free_gib": round(required_free_bytes / GIB, 3),
        },
        "host_memory": {
            "available_bytes": host_available_bytes,
            "required_available_bytes": int(resource_policy["minimum_host_available_bytes"]),
        },
        "gpu": gpu,
        "requirements": requirements,
        "preflight_passes": all(requirements.values()),
        "artifacts": {
            "config_sha256": sha256_file(config_path),
            "probe_config_sha256": sha256_file(probe_config_path),
            "policy_sha256": sha256_file(policy_path),
            "protocol_sha256": sha256_file(protocol_path),
            "data_manifest_sha256": sha256_file(manifest),
            "launcher_sha256": sha256_file(launcher_path),
            "dropout_module_sha256": sha256_file(dropout_module_path),
            "v6_runtime_sha256": sha256_file(v6_runtime_path),
            "v6_1_runtime_sha256": sha256_file(v6_1_runtime_path),
            "prior_probe_sha256": sha256_file(v6_probe_path),
        },
    }
    write_report(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["preflight_passes"] is not True:
        raise RuntimeError(f"v6.2 preflight failed: {requirements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
