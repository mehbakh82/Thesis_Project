#!/usr/bin/env python3
"""Validate and record the non-scientific one-step Moshi smoke result."""

from __future__ import annotations

import hashlib
import json
import math
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


def main() -> None:
    fixture_path = ROOT / "data" / "processed" / "moshi_smoke" / "REPORT.json"
    metrics_path = ROOT / "checkpoints" / "moshi_smoke" / "metrics.train.jsonl"
    args_path = ROOT / "checkpoints" / "moshi_smoke" / "args.yaml"
    config_path = ROOT / "configs" / "moshi_h100_smoke.yaml"
    environment_path = ROOT / "results" / "hardware" / "moshi_environment.json"
    fixture = _json(fixture_path)
    environment = _json(environment_path)
    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if len(metric_rows) != 1 or not isinstance(args, dict):
        raise ValueError("smoke must contain exactly one metric row and one argument mapping")
    metric = metric_rows[0]
    requirements = {
        "fixture_is_non_scientific": fixture.get("scientific_evidence") is False,
        "environment_valid": environment.get("valid") is True,
        "one_optimizer_step_completed": metric.get("step") == 1 and args.get("max_steps") == 1,
        "finite_loss": math.isfinite(float(metric.get("loss"))),
        "real_model_memory_allocated": float(metric.get("peak_allocated_mem") or 0.0) > 10.0,
        "checkpointing_disabled": args.get("do_ckpt") is False,
        "evaluation_disabled": args.get("do_eval") is False,
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(requirements.values()) else "failed",
        "completed_at": metric.get("at"),
        "purpose": fixture.get("purpose"),
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
            "duration_sec": args.get("duration_sec"),
            "batch_size": args.get("batch_size"),
            "lora": args.get("lora"),
        },
        "launcher": {
            "entry_point": "scripts/moshi_train_entry.py",
            "world_size": 1,
            "backend": "gloo",
            "backend_scope": "single-GPU compatibility fallback only",
            "reason": (
                "The bundled NCCL 2.21.5 reproducibly SIGSEGVs in libnccl during "
                "a minimal one-rank communicator init on this host. Gloo CUDA "
                "all-reduce passed and the trainer's world-size-one path is unsharded."
            ),
            "command": (
                "MOSHI_DISTRIBUTED_BACKEND=gloo .venv-moshi/bin/torchrun "
                "--standalone --nproc-per-node 1 scripts/moshi_train_entry.py "
                "configs/moshi_h100_smoke.yaml"
            ),
        },
        "requirements": requirements,
        "wiring_gate_passes": all(requirements.values()),
        "artifacts": {
            "fixture_report_sha256": _sha256(fixture_path),
            "metrics_sha256": _sha256(metrics_path),
            "resolved_args_sha256": _sha256(args_path),
            "source_config_sha256": _sha256(config_path),
            "environment_report_sha256": _sha256(environment_path),
        },
        "note": (
            "This proves pinned model/Mimi loading, LoRA initialization, data "
            "tokenization, loss, backward pass, and optimizer wiring only. It is not "
            "Persian quality, convergence, held-out, or target-GPU evidence."
        ),
    }
    if not report["wiring_gate_passes"]:
        raise RuntimeError(f"smoke evidence failed: {requirements}")
    output = ROOT / "results" / "hardware" / "moshi_h100_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
