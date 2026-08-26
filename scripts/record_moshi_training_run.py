#!/usr/bin/env python3
"""Create a fail-closed certificate for the completed Moshi H100 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def git_blob(commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


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
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/moshi_fa"))
    parser.add_argument("--config", type=Path, default=Path("configs/moshi_h100.yaml"))
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("checkpoints/moshi_h100_train_5c8efc9.log"),
    )
    parser.add_argument("--launch-commit", default="5c8efc9")
    parser.add_argument("--service-unit", default="thesis-moshi-h100-train-5c8efc9.service")
    parser.add_argument("--invocation-id", default="d8b29e9c812243dc9916e4c9f87735c6")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_h100_training.json"),
    )
    args = parser.parse_args()

    run_dir = (ROOT / args.run_dir).resolve()
    config_path = (ROOT / args.config).resolve()
    log_path = (ROOT / args.log).resolve()
    out_path = (ROOT / args.out).resolve()
    args_path = run_dir / "args.yaml"
    train_metrics_path = run_dir / "metrics.train.jsonl"
    raw_eval_metrics_path = run_dir / "metrics.eval.jsonl"
    reevaluation_path = ROOT / "results/moshi_validation_reevaluation.json"
    selection_path = ROOT / "results/moshi_checkpoint_selection.json"
    validation_path = ROOT / "results/moshi_adapter_validation.json"
    heldout_path = ROOT / "results/moshi_heldout_model_eval.json"
    environment_path = ROOT / "results/hardware/moshi_environment.json"
    export_path = ROOT / "results/moshi_export_report.json"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved_args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(resolved_args, dict):
        raise ValueError("training config and resolved args must be mappings")
    train_rows = jsonl_objects(train_metrics_path)
    raw_eval_rows = jsonl_objects(raw_eval_metrics_path)
    reevaluation = json_object(reevaluation_path)
    selection = json_object(selection_path)
    validation = json_object(validation_path)
    heldout = json_object(heldout_path)
    environment = json_object(environment_path)
    export = json_object(export_path)

    max_steps = int(config["max_steps"])
    checkpoint_frequency = int(config["ckpt_freq"])
    expected_steps = list(range(checkpoint_frequency, max_steps + 1, checkpoint_frequency))
    expected_logged_steps = list(
        range(int(config["log_freq"]), max_steps + 1, int(config["log_freq"]))
    )
    candidate_by_step = {int(row["step"]): row for row in reevaluation["candidates"]}
    checkpoint_rows = []
    for step in expected_steps:
        consolidated = run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
        adapter_path = consolidated / "lora.safetensors"
        adapter_config_path = consolidated / "config.json"
        with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
            tensor_count = len(list(adapter.keys()))
        checkpoint_rows.append(
            {
                "step": step,
                "adapter_path": adapter_path.relative_to(ROOT).as_posix(),
                "adapter_bytes": adapter_path.stat().st_size,
                "adapter_sha256": sha256_file(adapter_path),
                "adapter_tensor_count": tensor_count,
                "config_path": adapter_config_path.relative_to(ROOT).as_posix(),
                "config_sha256": sha256_file(adapter_config_path),
            }
        )

    log_text = log_path.read_text(encoding="utf-8")
    first_timestamp = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \(UTC\)", log_text)
    elapsed_matches = re.findall(r" - (\d+):(\d{2}):(\d{2}) - ", log_text)
    if first_timestamp is None or not elapsed_matches:
        raise ValueError("training log lacks expected UTC timestamp/elapsed fields")
    hours, minutes, seconds = (int(value) for value in elapsed_matches[-1])
    wall_seconds = hours * 3600 + minutes * 60 + seconds
    launch_revision = subprocess.run(
        ["git", "rev-parse", args.launch_commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    launch_config = yaml.safe_load(git_blob(launch_revision, "configs/moshi_h100.yaml"))
    launch_launcher = git_blob(launch_revision, "scripts/moshi_train_entry.py")
    if not isinstance(launch_config, dict):
        raise ValueError("launch-commit training config is not a mapping")

    checkpoint_hashes_match_reevaluation = all(
        row["adapter_sha256"] == candidate_by_step[row["step"]]["adapter_sha256"]
        and row["config_sha256"] == candidate_by_step[row["step"]]["config_sha256"]
        for row in checkpoint_rows
    )
    requirements = {
        "launch_and_current_training_profiles_match": comparable_profile(launch_config)
        == comparable_profile(config)
        == comparable_profile(resolved_args),
        "all_expected_logged_steps_present": [int(row["step"]) for row in train_rows]
        == expected_logged_steps,
        "all_logged_training_losses_finite": all(
            math.isfinite(float(row["loss"])) for row in train_rows
        ),
        "configured_final_step_completed": int(train_rows[-1]["step"]) == max_steps,
        "service_log_records_clean_completion": all(
            marker in log_text
            for marker in (
                "step: 008000 - done (%): 100.0",
                "Done dumping checkpoint",
                "train - INFO - done!",
                "train - INFO - Closed everything!",
            )
        ),
        "all_expected_checkpoints_present": len(checkpoint_rows) == len(expected_steps),
        "all_checkpoint_schemas_match": all(
            row["adapter_tensor_count"] == 699 and row["adapter_bytes"] == 1013593144
            for row in checkpoint_rows
        ),
        "checkpoint_hashes_match_fixed_scope_reevaluation": checkpoint_hashes_match_reevaluation,
        "raw_validation_metrics_rejected": (
            any(not math.isfinite(float(row["eval_loss"])) for row in raw_eval_rows)
            and reevaluation.get("original_metrics_eligible_for_selection") is False
        ),
        "fixed_scope_reevaluation_passed": reevaluation.get("reevaluation_passes") is True,
        "checkpoint_selection_passed": selection.get("selection_passes") is True,
        "selected_adapter_validation_passed": validation.get("validation_passes") is True,
        "one_time_heldout_evaluation_passed": heldout.get("evaluation_passes") is True,
        "h100_environment_valid": (
            environment.get("valid") is True and "H100" in str(environment.get("gpu") or "")
        ),
        "training_export_valid_under_declared_qa_policy": export.get(
            "training_ready_under_qa_waiver"
        )
        is True,
    }
    passed = all(requirements.values())
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_training_evidence": True,
        "human_quality_claim_allowed": False,
        "strict_human_qa_claim_allowed": False,
        "training_policy": "limited internal training under the documented student QA waiver",
        "service": {
            "unit": args.service_unit,
            "invocation_id": args.invocation_id,
            "result": "success",
            "launch_commit": launch_revision,
            "started_at_utc": first_timestamp.group(1).replace(" ", "T") + "Z",
            "wall_seconds": wall_seconds,
            "wall_hms": f"{hours}:{minutes:02d}:{seconds:02d}",
        },
        "profile": comparable_profile(resolved_args),
        "training_metrics": {
            "logged_rows": len(train_rows),
            "first_logged_step": train_rows[0]["step"],
            "final_step": train_rows[-1]["step"],
            "final_logged_loss": train_rows[-1]["loss"],
            "minimum_logged_loss": min(float(row["loss"]) for row in train_rows),
            "maximum_peak_allocated_gb": max(
                float(row["peak_allocated_mem"]) for row in train_rows
            ),
            "final_average_words_per_second": train_rows[-1]["avg_wps"],
        },
        "raw_upstream_validation": {
            "eligible_for_selection": False,
            "first_nonfinite_step": next(
                int(row["step"])
                for row in raw_eval_rows
                if not math.isfinite(float(row["eval_loss"]))
            ),
            "diagnosis": reevaluation["correction_reason"],
        },
        "checkpoint_count": len(checkpoint_rows),
        "checkpoints": checkpoint_rows,
        "selected": selection["selected"],
        "requirements": requirements,
        "training_run_passes": passed,
        "artifacts": {
            "training_log_sha256": sha256_file(log_path),
            "training_log_bytes": log_path.stat().st_size,
            "training_config_sha256": sha256_file(config_path),
            "resolved_args_sha256": sha256_file(args_path),
            "train_metrics_sha256": sha256_file(train_metrics_path),
            "raw_eval_metrics_sha256": sha256_file(raw_eval_metrics_path),
            "launch_launcher_sha256": hashlib.sha256(launch_launcher).hexdigest(),
            "current_launcher_sha256": sha256_file(ROOT / "scripts/moshi_train_entry.py"),
            "reevaluation_sha256": sha256_file(reevaluation_path),
            "selection_sha256": sha256_file(selection_path),
            "adapter_validation_sha256": sha256_file(validation_path),
            "heldout_evaluation_sha256": sha256_file(heldout_path),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"H100 training certificate failed: {requirements}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
