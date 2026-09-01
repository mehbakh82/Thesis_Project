#!/usr/bin/env python3
"""Validate and record the exact-shape Moshi v6 overfit one-step probe."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
FULL_TEXT_PARAMETERS = {
    "depformer_text_emb.weight",
    "text_emb.weight",
    "text_linear.frozen_W.weight",
}
EXPECTED_ENVIRONMENT = {
    "MOSHI_PERSIAN_TEXT_ADAPTATION": "1",
    "MOSHI_TEXT_EMBEDDINGS_ONLY": "0",
    "MOSHI_AUDIO_LOSS_WEIGHT": "0.1",
}


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def comparable_profile(value: dict) -> dict:
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
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/moshi_v6_overfit_probe"))
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=Path("configs/moshi_h100_v6_overfit_probe.yaml"),
    )
    parser.add_argument(
        "--full-config", type=Path, default=Path("configs/moshi_h100_v6_overfit.yaml")
    )
    parser.add_argument("--policy", type=Path, default=Path("configs/moshi_v6_overfit_policy.json"))
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path("results/hardware/moshi_v6_overfit_preflight.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v6_overfit_probe.json"),
    )
    args = parser.parse_args()

    run_dir = (ROOT / args.run_dir).resolve()
    probe_config_path = (ROOT / args.probe_config).resolve()
    full_config_path = (ROOT / args.full_config).resolve()
    policy_path = (ROOT / args.policy).resolve()
    preflight_path = (ROOT / args.preflight).resolve()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v6 probe evidence: {out_path}")
    metrics_path = run_dir / "metrics.train.jsonl"
    args_path = run_dir / "args.yaml"
    adapter_path = run_dir / "checkpoints/checkpoint_000001/consolidated/lora.safetensors"
    adapter_config_path = adapter_path.with_name("config.json")

    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resolved_args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    probe_config = yaml.safe_load(probe_config_path.read_text(encoding="utf-8"))
    full_config = yaml.safe_load(full_config_path.read_text(encoding="utf-8"))
    policy = json_object(policy_path)
    preflight = json_object(preflight_path)
    if len(metric_rows) != 1 or not all(
        isinstance(value, dict) for value in (resolved_args, probe_config, full_config)
    ):
        raise ValueError("v6 probe requires one metric row and valid YAML mappings")
    metric = metric_rows[0]
    with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
        adapter_keys = set(adapter.keys())
    lora_keys = {key for key in adapter_keys if "lora" in key}
    full_parameter_keys = adapter_keys - lora_keys
    effective_environment = {key: os.environ.get(key) for key in EXPECTED_ENVIRONMENT}

    requirements = {
        "preflight_passed": (
            preflight.get("status") == "passed"
            and preflight.get("preflight_passes") is True
            and preflight.get("test_access_started") is False
        ),
        "runtime_environment_exact": effective_environment == EXPECTED_ENVIRONMENT,
        "one_optimizer_step_completed": (
            metric.get("step") == 1 and resolved_args.get("max_steps") == 1
        ),
        "finite_loss": math.isfinite(float(metric.get("loss"))),
        "full_training_shape_matches": comparable_profile(probe_config)
        == comparable_profile(full_config),
        "resolved_args_match_probe": comparable_profile(resolved_args)
        == comparable_profile(probe_config),
        "evaluation_disabled": resolved_args.get("do_eval") is False,
        "checkpoint_present": adapter_path.is_file() and adapter_config_path.is_file(),
        "adapter_tensor_scope_exact": (
            len(lora_keys) == policy.get("expected_lora_tensor_count") == 674
            and full_parameter_keys == FULL_TEXT_PARAMETERS
            and len(adapter_keys) == policy.get("expected_total_adapter_tensor_count") == 677
        ),
        "adapter_size_matches_estimate": adapter_path.stat().st_size
        >= int(policy["estimated_adapter_bytes"]),
        "real_model_memory_allocated": float(metric.get("peak_allocated_mem") or 0.0) > 10.0,
        "preflight_bound_to_current_files": (
            (preflight.get("artifacts") or {}).get("config_sha256") == sha256_file(full_config_path)
            and (preflight.get("artifacts") or {}).get("probe_config_sha256")
            == sha256_file(probe_config_path)
            and (preflight.get("artifacts") or {}).get("policy_sha256") == sha256_file(policy_path)
        ),
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(requirements.values()) else "failed",
        "completed_at": metric.get("at") or datetime.now(timezone.utc).isoformat(),
        "purpose": "one-step exact-full-profile v6 in-sample diagnostic probe",
        "diagnostic_only": True,
        "scientific_validation_evidence": False,
        "persian_quality_claim_allowed": False,
        "final_test_accessed": False,
        "runtime_environment": effective_environment,
        "step": {
            "number": metric.get("step"),
            "loss": float(metric["loss"]),
            "peak_allocated_gb": float(metric["peak_allocated_mem"]),
        },
        "checkpoint": {
            "bytes": adapter_path.stat().st_size,
            "sha256": sha256_file(adapter_path),
            "tensor_count": len(adapter_keys),
            "lora_tensor_count": len(lora_keys),
            "full_parameters": sorted(full_parameter_keys),
        },
        "requirements": requirements,
        "probe_passes": all(requirements.values()),
        "artifacts": {
            "metrics_sha256": sha256_file(metrics_path),
            "resolved_args_sha256": sha256_file(args_path),
            "probe_config_sha256": sha256_file(probe_config_path),
            "full_config_sha256": sha256_file(full_config_path),
            "policy_sha256": sha256_file(policy_path),
            "preflight_sha256": sha256_file(preflight_path),
            "adapter_config_sha256": sha256_file(adapter_config_path),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["probe_passes"] is not True:
        raise RuntimeError(f"v6 overfit probe failed: {requirements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
