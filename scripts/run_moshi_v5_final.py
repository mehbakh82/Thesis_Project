#!/usr/bin/env python3
"""Run Moshi v5 final stages only after an eligible selection is committed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_moshi_v2_posttraining import sha256_file, write_report  # noqa: E402
from scripts.validate_moshi_adapter import json_object  # noqa: E402


def final_commands() -> list[dict[str, Any]]:
    moshi_python = ROOT / ".venv-moshi/bin/python"
    return [
        {
            "name": "selected_adapter_validation",
            "test_access": False,
            "output": ROOT / "results/moshi_v5_adapter_validation.json",
            "command": [
                moshi_python,
                ROOT / "scripts/validate_moshi_adapter.py",
                "--selection",
                "results/moshi_v5_checkpoint_selection.json",
                "--training-config",
                "configs/moshi_h100_v5.yaml",
                "--out",
                "results/moshi_v5_adapter_validation.json",
                "--runtime-device",
                "cuda",
                "--embedding-policy",
                "configs/moshi_v5_embedding_policy.json",
            ],
        },
        {
            "name": "one_time_objective_final_test",
            "test_access": True,
            "output": ROOT / "results/moshi_v5_final_test.json",
            "command": [
                moshi_python,
                ROOT / "scripts/evaluate_moshi_v2_final_test.py",
                "--validation",
                "results/moshi_v5_adapter_validation.json",
                "--training-config",
                "configs/moshi_h100_v5.yaml",
                "--split-report",
                "results/moshi_v2_split.json",
                "--out",
                "results/moshi_v5_final_test.json",
                "--experiment-label",
                "v5",
            ],
        },
        {
            "name": "separate_complete_prompt_final_runtime_diagnostics",
            "test_access": True,
            "output": ROOT / "results/moshi_v5_final_runtime.json",
            "command": [
                moshi_python,
                ROOT / "scripts/evaluate_moshi_v2_final_runtime.py",
                "--validation",
                "results/moshi_v5_adapter_validation.json",
                "--training-config",
                "configs/moshi_h100_v5.yaml",
                "--split-report",
                "results/moshi_v2_split.json",
                "--out",
                "results/moshi_v5_final_runtime.json",
                "--experiment-label",
                "v5",
            ],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v5_final_pipeline.json"),
    )
    args = parser.parse_args()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v5 final receipt: {out_path}")

    selection_path = ROOT / "results/moshi_v5_checkpoint_selection.json"
    validation_pipeline_path = ROOT / "results/hardware/moshi_v5_validation_pipeline.json"
    training_certificate_path = ROOT / "results/hardware/moshi_h100_v5_training.json"
    embedding_policy_path = ROOT / "configs/moshi_v5_embedding_policy.json"
    selection = json_object(selection_path)
    validation_pipeline = json_object(validation_pipeline_path)
    training_certificate = json_object(training_certificate_path)
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("v5 final pipeline requires a selected checkpoint object")
    launch_commit = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    worktree_status = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_selection = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT}",
            "ls-files",
            "--error-unmatch",
            "--",
            selection_path.relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    preconditions = {
        "launch_worktree_clean": not worktree_status.strip(),
        "selection_committed_before_test_access": tracked_selection.returncode == 0,
        "eligible_v5_selection_frozen": (
            selection.get("schema_version") == 7
            and selection.get("status") == "passed"
            and selection.get("selection_passes") is True
            and isinstance(selected, dict)
            and selection.get("heldout_test_used_for_selection") is False
            and selection.get("exact_runtime_panel_indices_predeclared_before_training") is True
        ),
        "validation_pipeline_passed_without_test_access": (
            validation_pipeline.get("status") == "passed"
            and validation_pipeline.get("selection_outcome") == "eligible_checkpoint_selected"
            and validation_pipeline.get("test_access_started") is False
        ),
        "training_certificate_passed_without_test_access": (
            training_certificate.get("status") == "passed"
            and training_certificate.get("test_access_started") is False
            and training_certificate.get("selected") == selected
            and (training_certificate.get("artifacts") or {}).get("embedding_policy_sha256")
            == sha256_file(embedding_policy_path)
        ),
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"v5 final-pipeline preconditions failed: {preconditions}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "launch_commit": launch_commit,
        "test_access_started": False,
        "selected_step": int(selected["step"]),
        "preconditions": preconditions,
        "stages": [],
        "artifacts": {
            "selection_sha256": sha256_file(selection_path),
            "validation_pipeline_sha256": sha256_file(validation_pipeline_path),
            "training_certificate_sha256": sha256_file(training_certificate_path),
            "embedding_policy_sha256": sha256_file(embedding_policy_path),
            "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_report(out_path, report)
    for stage in final_commands():
        output = Path(stage["output"])
        if output.exists():
            raise FileExistsError(f"refusing to overwrite v5 final output: {output}")
        if stage["test_access"] is True:
            report["test_access_started"] = True
        command = [str(value) for value in stage["command"]]
        stage_report: dict[str, Any] = {
            "name": stage["name"],
            "test_access": stage["test_access"],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "status": "running",
        }
        report["stages"].append(stage_report)
        write_report(out_path, report)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        stage_report["finished_at"] = datetime.now(timezone.utc).isoformat()
        stage_report["exit_code"] = completed.returncode
        stage_report["status"] = "passed" if completed.returncode == 0 else "failed"
        if output.is_file():
            stage_report["output"] = output.relative_to(ROOT).as_posix()
            stage_report["output_sha256"] = sha256_file(output)
        write_report(out_path, report)
        if completed.returncode != 0:
            report["status"] = "failed"
            report["failure"] = f"stage failed: {stage['name']}"
            write_report(out_path, report)
            raise RuntimeError(report["failure"])

    report["status"] = "passed"
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["requirements"] = {
        "stages_run_in_frozen_order": [stage["name"] for stage in report["stages"]]
        == [stage["name"] for stage in final_commands()],
        "all_stages_passed": all(stage.get("status") == "passed" for stage in report["stages"]),
        "test_access_started_only_after_adapter_validation": (
            report["stages"][0]["name"] == "selected_adapter_validation"
            and report["stages"][0]["test_access"] is False
            and report["stages"][0]["status"] == "passed"
            and report["test_access_started"] is True
        ),
    }
    write_report(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
