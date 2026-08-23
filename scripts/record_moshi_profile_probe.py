#!/usr/bin/env python3
"""Validate and record the exact-shape, one-step Moshi memory probe."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparable_profile(value: dict) -> dict:
    return {
        "duration_sec": value.get("duration_sec"),
        "batch_size": value.get("batch_size"),
        "num_microbatches": value.get("num_microbatches"),
        "gradient_checkpointing": value.get("gradient_checkpointing"),
        "full_finetuning": value.get("full_finetuning"),
        "lora": value.get("lora"),
        "first_codebook_weight_multiplier": value.get("first_codebook_weight_multiplier"),
        "text_padding_weight": value.get("text_padding_weight"),
    }


def main() -> None:
    metrics_path = ROOT / "checkpoints" / "moshi_h100_profile_probe" / "metrics.train.jsonl"
    args_path = ROOT / "checkpoints" / "moshi_h100_profile_probe" / "args.yaml"
    probe_config_path = ROOT / "configs" / "moshi_h100_profile_probe.yaml"
    full_config_path = ROOT / "configs" / "moshi_h100.yaml"
    environment_path = ROOT / "results" / "hardware" / "moshi_environment.json"
    export_path = ROOT / "results" / "moshi_export_report.json"

    environment = _json(environment_path)
    export = _json(export_path)
    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    probe_config = yaml.safe_load(probe_config_path.read_text(encoding="utf-8"))
    full_config = yaml.safe_load(full_config_path.read_text(encoding="utf-8"))
    if len(metric_rows) != 1 or not all(
        isinstance(value, dict) for value in (args, probe_config, full_config)
    ):
        raise ValueError("probe must contain one metric row and valid argument/config mappings")
    metric = metric_rows[0]
    requirements = {
        "environment_valid": environment.get("valid") is True,
        "final_training_export_valid": export.get("final_training_ready") is True,
        "one_optimizer_step_completed": metric.get("step") == 1 and args.get("max_steps") == 1,
        "finite_loss": math.isfinite(float(metric.get("loss"))),
        "full_training_shape_matches": _comparable_profile(probe_config)
        == _comparable_profile(full_config),
        "resolved_args_match_probe_config": _comparable_profile(args)
        == _comparable_profile(probe_config),
        "evaluation_disabled": args.get("do_eval") is False,
        "checkpointing_disabled": args.get("do_ckpt") is False,
        "real_model_memory_allocated": float(metric.get("peak_allocated_mem") or 0.0) > 10.0,
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(requirements.values()) else "failed",
        "completed_at": metric.get("at") or datetime.now(timezone.utc).isoformat(),
        "purpose": "one-step exact-full-profile memory and optimizer probe only",
        "scientific_evidence": False,
        "persian_training_claim_allowed": False,
        "hardware": {
            "gpu": environment.get("gpu"),
            "torch_cuda": environment.get("torch_cuda"),
            "peak_allocated_gb": round(float(metric["peak_allocated_mem"]), 3),
            "allocated_gb_after_step": round(float(metric["allocated_mem"]), 3),
        },
        "step": {
            "number": metric.get("step"),
            "loss": round(float(metric["loss"]), 6),
            "words_per_second": round(float(metric["wps"]), 3),
        },
        "profile": _comparable_profile(args),
        "requirements": requirements,
        "full_profile_gate_passes": all(requirements.values()),
        "artifacts": {
            "metrics_sha256": _sha256(metrics_path),
            "resolved_args_sha256": _sha256(args_path),
            "probe_config_sha256": _sha256(probe_config_path),
            "full_config_sha256": _sha256(full_config_path),
            "environment_report_sha256": _sha256(environment_path),
            "export_report_sha256": _sha256(export_path),
        },
        "note": (
            "This measures peak memory for one optimizer step with the full training shape. "
            "It is not convergence, Persian quality, held-out, or target-GPU evidence."
        ),
    }
    if not report["full_profile_gate_passes"]:
        raise RuntimeError(f"full-profile probe failed: {requirements}")
    output = ROOT / "results" / "hardware" / "moshi_h100_profile_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
