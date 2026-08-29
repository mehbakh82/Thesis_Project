#!/usr/bin/env python3
"""Wait for Moshi v2 training, then execute its frozen fail-closed pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def service_properties(unit: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "systemctl",
            "show",
            unit,
            "--property=ActiveState,SubState,Result,ExecMainStatus",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    properties = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pipeline_commands() -> list[dict[str, Any]]:
    moshi_python = ROOT / ".venv-moshi/bin/python"
    project_python = ROOT / ".venv/bin/python"
    return [
        {
            "name": "complete_validation_reevaluation",
            "output": ROOT / "results/moshi_v2_validation_reevaluation.json",
            "command": [
                moshi_python,
                ROOT / "scripts/reevaluate_moshi_checkpoints.py",
                "--training-config",
                "configs/moshi_h100_v2.yaml",
                "--run-dir",
                "checkpoints/moshi_fa_100s_v2",
                "--metrics-out",
                "checkpoints/moshi_fa_100s_v2/metrics.reeval.jsonl",
                "--report-out",
                "results/moshi_v2_validation_reevaluation.json",
                "--heldout-test-manifest",
                "data/processed/moshi_finetune_v2/test.jsonl",
            ],
        },
        {
            "name": "validation_runtime_panel",
            "output": ROOT / "results/moshi_v2_runtime_candidates.json",
            "command": [
                moshi_python,
                ROOT / "scripts/evaluate_moshi_v2_runtime_candidates.py",
            ],
        },
        {
            "name": "eligible_only_selection",
            "output": ROOT / "results/moshi_v2_checkpoint_selection.json",
            "command": [
                project_python,
                ROOT / "scripts/select_moshi_v2_checkpoint.py",
            ],
        },
        {
            "name": "selected_adapter_validation",
            "output": ROOT / "results/moshi_v2_adapter_validation.json",
            "command": [
                moshi_python,
                ROOT / "scripts/validate_moshi_adapter.py",
                "--selection",
                "results/moshi_v2_checkpoint_selection.json",
                "--training-config",
                "configs/moshi_h100_v2.yaml",
                "--out",
                "results/moshi_v2_adapter_validation.json",
                "--runtime-device",
                "cuda",
            ],
        },
        {
            "name": "one_time_objective_final_test",
            "output": ROOT / "results/moshi_v2_final_test.json",
            "command": [
                moshi_python,
                ROOT / "scripts/evaluate_moshi_v2_final_test.py",
            ],
        },
        {
            "name": "separate_final_runtime_diagnostics",
            "output": ROOT / "results/moshi_v2_final_runtime.json",
            "command": [
                moshi_python,
                ROOT / "scripts/evaluate_moshi_v2_final_runtime.py",
            ],
        },
        {
            "name": "training_and_pipeline_certificate",
            "output": ROOT / "results/hardware/moshi_h100_v2_training.json",
            "command": [
                moshi_python,
                ROOT / "scripts/record_moshi_v2_training_run.py",
            ],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-unit",
        default="thesis-moshi-h100-v2-retry1-119f9e8.service",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v2_posttraining_pipeline.json"),
    )
    args = parser.parse_args()
    if not 5.0 <= args.poll_seconds <= 60.0:
        raise ValueError("poll-seconds must be in [5, 60]")
    if not 1.0 <= args.timeout_hours <= 24.0:
        raise ValueError("timeout-hours must be in [1, 24]")
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite pipeline receipt: {out_path}")

    commands = pipeline_commands()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "waiting_for_training",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_unit": args.training_unit,
        "test_access_started": False,
        "stages": [],
        "requirements": {
            "training_service_succeeded": False,
            "stages_run_in_frozen_order": False,
        },
        "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_report(out_path, report)
    deadline = time.monotonic() + args.timeout_hours * 3600
    while True:
        properties = service_properties(args.training_unit)
        report["training_service"] = properties
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        write_report(out_path, report)
        if properties.get("ActiveState") not in {"active", "activating", "reloading"}:
            break
        if time.monotonic() >= deadline:
            report["status"] = "failed"
            report["failure"] = "training wait timeout"
            write_report(out_path, report)
            raise TimeoutError("timed out waiting for Moshi v2 training")
        time.sleep(args.poll_seconds)

    training_passed = (
        properties.get("Result") == "success"
        and properties.get("ExecMainStatus") == "0"
        and properties.get("ActiveState") == "inactive"
    )
    report["requirements"]["training_service_succeeded"] = training_passed
    if not training_passed:
        report["status"] = "failed"
        report["failure"] = "training service did not complete successfully"
        write_report(out_path, report)
        raise RuntimeError(f"training service failed: {properties}")

    free_bytes = shutil.disk_usage(ROOT).free
    report["pre_pipeline_free_bytes"] = free_bytes
    if free_bytes < 2 * 1024**3:
        report["status"] = "failed"
        report["failure"] = "less than 2 GiB free before evaluation"
        write_report(out_path, report)
        raise RuntimeError("insufficient disk space for v2 evaluation reports")

    report["status"] = "running"
    write_report(out_path, report)
    completed_names: list[str] = []
    for stage in commands:
        name = str(stage["name"])
        command = [str(value) for value in stage["command"]]
        output = Path(stage["output"])
        if output.exists():
            raise FileExistsError(f"refusing to overwrite stage output: {output}")
        if name.startswith(("one_time_", "separate_final_")):
            report["test_access_started"] = True
        stage_report: dict[str, Any] = {
            "name": name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "status": "running",
        }
        report["stages"].append(stage_report)
        write_report(out_path, report)
        result = subprocess.run(command, cwd=ROOT, check=False)
        stage_report["finished_at"] = datetime.now(timezone.utc).isoformat()
        stage_report["exit_code"] = result.returncode
        stage_report["status"] = "passed" if result.returncode == 0 else "failed"
        if output.is_file():
            stage_report["output"] = output.relative_to(ROOT).as_posix()
            stage_report["output_sha256"] = sha256_file(output)
        write_report(out_path, report)
        if result.returncode != 0:
            report["status"] = "failed"
            report["failure"] = f"stage failed: {name}"
            write_report(out_path, report)
            raise RuntimeError(f"v2 post-training stage failed: {name}")
        completed_names.append(name)

    expected_names = [str(stage["name"]) for stage in commands]
    report["requirements"]["stages_run_in_frozen_order"] = completed_names == expected_names
    report["status"] = "passed" if all(report["requirements"].values()) else "failed"
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_report(out_path, report)
    if report["status"] != "passed":
        raise RuntimeError("v2 post-training pipeline receipt failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
