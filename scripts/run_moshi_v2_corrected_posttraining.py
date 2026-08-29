#!/usr/bin/env python3
"""Continue Moshi v2 after the frozen complete-prompt protocol correction."""

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


def corrected_pipeline_commands() -> list[dict[str, Any]]:
    moshi_python = ROOT / ".venv-moshi/bin/python"
    project_python = ROOT / ".venv/bin/python"
    return [
        {
            "name": "corrected_complete_prompt_validation_runtime_panel",
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
            "name": "separate_complete_prompt_final_runtime_diagnostics",
            "output": ROOT / "results/moshi_v2_final_runtime.json",
            "command": [
                moshi_python,
                ROOT / "scripts/evaluate_moshi_v2_final_runtime.py",
            ],
        },
        {
            "name": "training_and_corrected_pipeline_certificate",
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
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v2_corrected_posttraining_pipeline.json"),
    )
    args = parser.parse_args()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite corrected pipeline receipt: {out_path}")

    reevaluation_path = ROOT / "results/moshi_v2_validation_reevaluation.json"
    superseded_runtime_path = (
        ROOT / "results/moshi_v2_runtime_candidates_truncated_prompt_invalid.json"
    )
    superseded_selection_path = (
        ROOT / "results/moshi_v2_checkpoint_selection_truncated_prompt_failed.json"
    )
    superseded_receipt_path = (
        ROOT / "results/hardware/moshi_v2_posttraining_truncated_prompt_failed.json"
    )
    correction_path = ROOT / "docs/MOSHI_V2_PROTOCOL_CORRECTION.md"
    required_paths = [
        reevaluation_path,
        superseded_runtime_path,
        superseded_selection_path,
        superseded_receipt_path,
        correction_path,
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing corrected-pipeline provenance: {missing}")

    reevaluation = json_object(reevaluation_path)
    superseded_runtime = json_object(superseded_runtime_path)
    superseded_selection = json_object(superseded_selection_path)
    superseded_receipt = json_object(superseded_receipt_path)
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
    preconditions = {
        "launch_worktree_clean": not worktree_status.strip(),
        "complete_validation_reevaluation_passed": (
            reevaluation.get("status") == "passed"
            and reevaluation.get("reevaluation_passes") is True
            and reevaluation.get("heldout_test_used") is False
        ),
        "superseded_runtime_preserved_and_ineligible": (
            superseded_runtime.get("schema_version") == 1
            and superseded_runtime.get("eligible_steps") == []
            and (superseded_runtime.get("protocol") or {}).get("panel_indices")
            == [0, 16, 32, 48, 64, 80, 96, 112, 130]
        ),
        "superseded_selection_failed_closed": (
            superseded_selection.get("status") == "failed"
            and superseded_selection.get("selection_passes") is False
            and superseded_selection.get("selected") is None
        ),
        "superseded_pipeline_stopped_before_test_access": (
            superseded_receipt.get("status") == "failed"
            and superseded_receipt.get("failure") == "stage failed: eligible_only_selection"
            and superseded_receipt.get("test_access_started") is False
        ),
        "correction_frozen_before_new_generation": True,
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"corrected-pipeline preconditions failed: {preconditions}")

    commands = corrected_pipeline_commands()
    existing_outputs = [
        Path(stage["output"]).relative_to(ROOT).as_posix()
        for stage in commands
        if Path(stage["output"]).exists()
    ]
    if existing_outputs:
        raise FileExistsError(f"refusing to overwrite corrected stage outputs: {existing_outputs}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "launch_commit": launch_commit,
        "test_access_started": False,
        "preconditions": preconditions,
        "stages": [],
        "requirements": {"stages_run_in_frozen_order": False},
        "artifacts": {
            "correction_document_sha256": sha256_file(correction_path),
            "reevaluation_sha256": sha256_file(reevaluation_path),
            "superseded_runtime_sha256": sha256_file(superseded_runtime_path),
            "superseded_selection_sha256": sha256_file(superseded_selection_path),
            "superseded_pipeline_receipt_sha256": sha256_file(superseded_receipt_path),
            "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_report(out_path, report)
    completed_names: list[str] = []
    for stage in commands:
        name = str(stage["name"])
        command = [str(value) for value in stage["command"]]
        output = Path(stage["output"])
        if name.startswith(("one_time_", "separate_complete_prompt_final_")):
            report["test_access_started"] = True
        stage_report: dict[str, Any] = {
            "name": name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "status": "running",
        }
        report["stages"].append(stage_report)
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        write_report(out_path, report)
        result = subprocess.run(command, cwd=ROOT, check=False)
        stage_report["finished_at"] = datetime.now(timezone.utc).isoformat()
        stage_report["exit_code"] = result.returncode
        stage_report["status"] = "passed" if result.returncode == 0 else "failed"
        if output.is_file():
            stage_report["output"] = output.relative_to(ROOT).as_posix()
            stage_report["output_sha256"] = sha256_file(output)
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        write_report(out_path, report)
        if result.returncode != 0:
            report["status"] = "failed"
            report["failure"] = f"stage failed: {name}"
            write_report(out_path, report)
            raise RuntimeError(f"corrected v2 post-training stage failed: {name}")
        completed_names.append(name)

    expected_names = [str(stage["name"]) for stage in commands]
    report["requirements"]["stages_run_in_frozen_order"] = completed_names == expected_names
    report["status"] = "passed" if all(report["requirements"].values()) else "failed"
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_report(out_path, report)
    if report["status"] != "passed":
        raise RuntimeError("corrected v2 post-training receipt failed")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
