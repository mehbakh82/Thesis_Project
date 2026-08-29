#!/usr/bin/env python3
"""Create the fail-closed certificate for the corrected Moshi v2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.record_moshi_training_run import comparable_profile  # noqa: E402
from scripts.validate_moshi_adapter import json_object, sha256_file  # noqa: E402


def jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def journal_for_invocation(invocation_id: str) -> str:
    result = subprocess.run(
        [
            "journalctl",
            f"_SYSTEMD_INVOCATION_ID={invocation_id}",
            "--no-pager",
            "--output=cat",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        raise RuntimeError(f"no journal records found for invocation {invocation_id}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("checkpoints/moshi_fa_100s_v2"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moshi_h100_v2.yaml"),
    )
    parser.add_argument(
        "--launch-commit",
        default="119f9e88319327a64fa2384dc1e6fceff2a65562",
    )
    parser.add_argument(
        "--service-unit",
        default="thesis-moshi-h100-v2-retry1-119f9e8.service",
    )
    parser.add_argument(
        "--invocation-id",
        default="4a07aff67c5b4e7884e4f5259d0ce0a3",
    )
    parser.add_argument(
        "--previous-invocation-id",
        default="4cba51d85eb94cc0a40b58cc9d0c1053",
    )
    parser.add_argument(
        "--previous-run-dir",
        type=Path,
        default=Path("checkpoints/moshi_fa_100s_v2_failed_enospc_20260827"),
    )
    parser.add_argument(
        "--reevaluation",
        type=Path,
        default=Path("results/moshi_v2_validation_reevaluation.json"),
    )
    parser.add_argument(
        "--runtime-panel",
        type=Path,
        default=Path("results/moshi_v2_runtime_candidates.json"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/moshi_v2_checkpoint_selection.json"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("results/moshi_v2_adapter_validation.json"),
    )
    parser.add_argument(
        "--final-test",
        type=Path,
        default=Path("results/moshi_v2_final_test.json"),
    )
    parser.add_argument(
        "--final-runtime",
        type=Path,
        default=Path("results/moshi_v2_final_runtime.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_h100_v2_training.json"),
    )
    args = parser.parse_args()

    run_dir = (ROOT / args.run_dir).resolve()
    config_path = (ROOT / args.config).resolve()
    previous_run_dir = (ROOT / args.previous_run_dir).resolve()
    previous_metrics_path = previous_run_dir / "metrics.train.jsonl"
    out_path = (ROOT / args.out).resolve()
    args_path = run_dir / "args.yaml"
    train_metrics_path = run_dir / "metrics.train.jsonl"
    raw_eval_metrics_path = run_dir / "metrics.eval.jsonl"
    reevaluation_path = (ROOT / args.reevaluation).resolve()
    runtime_panel_path = (ROOT / args.runtime_panel).resolve()
    selection_path = (ROOT / args.selection).resolve()
    validation_path = (ROOT / args.validation).resolve()
    final_test_path = (ROOT / args.final_test).resolve()
    final_runtime_path = (ROOT / args.final_runtime).resolve()
    environment_path = ROOT / "results/hardware/moshi_environment.json"
    export_path = ROOT / "results/moshi_export_report.json"
    split_path = ROOT / "results/moshi_v2_split.json"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved_args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(resolved_args, dict):
        raise ValueError("training config and resolved args must be mappings")
    train_rows = jsonl_objects(train_metrics_path)
    raw_eval_rows = jsonl_objects(raw_eval_metrics_path)
    previous_train_rows = jsonl_objects(previous_metrics_path)
    reevaluation = json_object(reevaluation_path)
    runtime_panel = json_object(runtime_panel_path)
    selection = json_object(selection_path)
    validation = json_object(validation_path)
    final_test = json_object(final_test_path)
    final_runtime = json_object(final_runtime_path)
    environment = json_object(environment_path)
    export = json_object(export_path)
    split = json_object(split_path)

    max_steps = int(config["max_steps"])
    checkpoint_frequency = int(config["ckpt_freq"])
    expected_steps = list(range(checkpoint_frequency, max_steps + 1, checkpoint_frequency))
    expected_logged_steps = list(
        range(int(config["log_freq"]), max_steps + 1, int(config["log_freq"]))
    )
    reevaluated_by_step = {
        int(candidate["step"]): candidate for candidate in reevaluation["candidates"]
    }
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

    journal_text = journal_for_invocation(args.invocation_id)
    previous_journal_text = journal_for_invocation(args.previous_invocation_id)
    first_timestamp = re.search(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \(UTC\)",
        journal_text,
        flags=re.MULTILINE,
    )
    elapsed_matches = re.findall(r" - (\d+):(\d{2}):(\d{2}) - ", journal_text)
    if first_timestamp is None or not elapsed_matches:
        raise ValueError("training journal lacks expected UTC timestamp/elapsed fields")
    hours, minutes, seconds = (int(value) for value in elapsed_matches[-1])
    wall_seconds = hours * 3600 + minutes * 60 + seconds
    launch_revision = subprocess.run(
        ["git", "rev-parse", args.launch_commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config_repo_path = config_path.relative_to(ROOT).as_posix()
    launch_config = yaml.safe_load(git_blob(launch_revision, config_repo_path))
    launch_launcher = git_blob(launch_revision, "scripts/moshi_train_entry.py")
    if not isinstance(launch_config, dict):
        raise ValueError("launch-commit training config is not a mapping")

    raw_nonfinite_steps = [
        int(row["step"]) for row in raw_eval_rows if not math.isfinite(float(row["eval_loss"]))
    ]
    checkpoint_hashes_match = all(
        row["adapter_sha256"] == reevaluated_by_step[row["step"]]["adapter_sha256"]
        and row["config_sha256"] == reevaluated_by_step[row["step"]]["config_sha256"]
        for row in checkpoint_rows
    )
    tensor_counts = {row["adapter_tensor_count"] for row in checkpoint_rows}
    adapter_sizes = {row["adapter_bytes"] for row in checkpoint_rows}
    requirements = {
        "failed_attempt_preserved_and_disclosed": (
            previous_run_dir.is_dir()
            and int(previous_train_rows[-1]["step"]) == 1800
            and all(math.isfinite(float(row["loss"])) for row in previous_train_rows)
            and "No space left on device" in previous_journal_text
            and "checkpoint_001800" in previous_journal_text
            and args.previous_invocation_id != args.invocation_id
        ),
        "launch_and_current_training_profiles_match": comparable_profile(launch_config)
        == comparable_profile(config)
        == comparable_profile(resolved_args),
        "all_expected_logged_steps_present": [int(row["step"]) for row in train_rows]
        == expected_logged_steps,
        "all_logged_training_losses_finite": all(
            math.isfinite(float(row["loss"])) for row in train_rows
        ),
        "configured_final_step_completed": int(train_rows[-1]["step"]) == max_steps,
        "service_journal_records_clean_completion": all(
            marker in journal_text
            for marker in (
                f"step: {max_steps:06d} - done (%): 100.0",
                "Done dumping checkpoint",
                "train - INFO - done!",
                "train - INFO - Closed everything!",
            )
        ),
        "all_expected_checkpoints_present": len(checkpoint_rows) == len(expected_steps),
        "all_checkpoint_schemas_match": len(tensor_counts) == 1 and len(adapter_sizes) == 1,
        "checkpoint_hashes_match_fixed_scope_reevaluation": checkpoint_hashes_match,
        "raw_validation_metrics_rejected": (
            reevaluation.get("original_metrics_eligible_for_selection") is False
            and (reevaluation.get("requirements") or {}).get("raw_validation_not_complete_scope")
            is True
        ),
        "fixed_scope_reevaluation_passed": reevaluation.get("reevaluation_passes") is True,
        "runtime_panel_evaluation_passed": runtime_panel.get("runtime_panel_evaluation_passes")
        is True,
        "checkpoint_selection_passed": selection.get("selection_passes") is True,
        "selected_adapter_validation_passed": validation.get("validation_passes") is True,
        "one_time_final_test_passed": final_test.get("evaluation_passes") is True,
        "final_test_not_used_for_selection": final_test.get("heldout_used_for_checkpoint_selection")
        is False,
        "final_runtime_diagnostics_passed": final_runtime.get("final_runtime_passes") is True,
        "final_runtime_not_used_for_selection": final_runtime.get("used_for_checkpoint_selection")
        is False,
        "h100_environment_valid": (
            environment.get("valid") is True and "H100" in str(environment.get("gpu") or "")
        ),
        "training_export_valid_under_declared_qa_policy": export.get(
            "training_ready_under_qa_waiver"
        )
        is True,
        "v2_group_split_passed": split.get("split_passes") is True,
    }
    passed = all(requirements.values())
    journal_bytes = journal_text.encode("utf-8")
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_training_evidence": True,
        "human_quality_claim_allowed": False,
        "strict_human_qa_claim_allowed": False,
        "training_policy": "limited internal training under the documented student QA waiver",
        "recovery": {
            "method": "clean step-zero retry; no optimizer-state-free pseudo-resume",
            "previous_invocation_id": args.previous_invocation_id,
            "previous_run_dir": previous_run_dir.relative_to(ROOT).as_posix(),
            "previous_final_logged_step": previous_train_rows[-1]["step"],
            "failure": "ENOSPC while creating checkpoint 001800",
            "previous_journal_sha256": hashlib.sha256(
                previous_journal_text.encode("utf-8")
            ).hexdigest(),
            "preserved": True,
        },
        "service": {
            "unit": args.service_unit,
            "invocation_id": args.invocation_id,
            "result": "success" if passed else "unverified",
            "launch_commit": launch_revision,
            "started_at_utc": first_timestamp.group(1).replace(" ", "T") + "Z",
            "wall_seconds": wall_seconds,
            "wall_hms": f"{hours}:{minutes:02d}:{seconds:02d}",
            "journal_sha256": hashlib.sha256(journal_bytes).hexdigest(),
            "journal_bytes": len(journal_bytes),
            "journal_retained_separately": False,
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
            "first_nonfinite_step": raw_nonfinite_steps[0] if raw_nonfinite_steps else None,
            "diagnosis": reevaluation["correction_reason"],
        },
        "checkpoint_count": len(checkpoint_rows),
        "checkpoint_schema": {
            "tensor_count": next(iter(tensor_counts)),
            "adapter_bytes": next(iter(adapter_sizes)),
        },
        "checkpoints": checkpoint_rows,
        "selected": selection["selected"],
        "requirements": requirements,
        "training_run_passes": passed,
        "artifacts": {
            "training_config_sha256": sha256_file(config_path),
            "resolved_args_sha256": sha256_file(args_path),
            "train_metrics_sha256": sha256_file(train_metrics_path),
            "raw_eval_metrics_sha256": sha256_file(raw_eval_metrics_path),
            "launch_launcher_sha256": hashlib.sha256(launch_launcher).hexdigest(),
            "current_launcher_sha256": sha256_file(ROOT / "scripts/moshi_train_entry.py"),
            "reevaluation_sha256": sha256_file(reevaluation_path),
            "runtime_panel_sha256": sha256_file(runtime_panel_path),
            "selection_sha256": sha256_file(selection_path),
            "adapter_validation_sha256": sha256_file(validation_path),
            "final_test_sha256": sha256_file(final_test_path),
            "final_runtime_sha256": sha256_file(final_runtime_path),
            "split_report_sha256": sha256_file(split_path),
            "previous_train_metrics_sha256": sha256_file(previous_metrics_path),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"H100 v2 training certificate failed: {requirements}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
