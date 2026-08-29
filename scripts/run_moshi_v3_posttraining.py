#!/usr/bin/env python3
"""Run the Moshi v3 validation-only chain and stop before final-test access."""

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


def validation_commands() -> list[dict[str, Any]]:
    moshi_python = ROOT / ".venv-moshi/bin/python"
    project_python = ROOT / ".venv/bin/python"
    return [
        {
            "name": "complete_validation_reevaluation",
            "output": ROOT / "results/moshi_v3_validation_reevaluation.json",
            "command": [
                moshi_python,
                ROOT / "scripts/reevaluate_moshi_checkpoints.py",
                "--training-config",
                "configs/moshi_h100_v3.yaml",
                "--run-dir",
                "checkpoints/moshi_fa_100s_v3_no_embed",
                "--metrics-out",
                "checkpoints/moshi_fa_100s_v3_no_embed/metrics.reeval.jsonl",
                "--report-out",
                "results/moshi_v3_validation_reevaluation.json",
                "--heldout-test-manifest",
                "data/processed/moshi_finetune_v2/test.jsonl",
            ],
        },
        {
            "name": "predeclared_complete_prompt_runtime_panel",
            "output": ROOT / "results/moshi_v3_runtime_candidates.json",
            "command": [
                moshi_python,
                ROOT / "scripts/evaluate_moshi_v3_runtime_candidates.py",
            ],
        },
        {
            "name": "eligible_only_selection",
            "output": ROOT / "results/moshi_v3_checkpoint_selection.json",
            "command": [
                project_python,
                ROOT / "scripts/select_moshi_v2_checkpoint.py",
                "--config",
                "configs/moshi_h100_v3.yaml",
                "--reevaluation",
                "results/moshi_v3_validation_reevaluation.json",
                "--runtime",
                "results/moshi_v3_runtime_candidates.json",
                "--out",
                "results/moshi_v3_checkpoint_selection.json",
                "--protocol-mode",
                "predeclared-v3",
            ],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-invocation-id",
        default="0479e341d29944c3b6edf61298fc953f",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v3_validation_pipeline.json"),
    )
    args = parser.parse_args()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v3 validation receipt: {out_path}")

    config_path = ROOT / "configs/moshi_h100_v3.yaml"
    protocol_path = ROOT / "docs/MOSHI_V3_SELECTION_PROTOCOL.md"
    preflight_path = ROOT / "results/hardware/moshi_v3_preflight.json"
    probe_path = ROOT / "results/hardware/moshi_h100_v3_profile_probe.json"
    v2_finalization_path = ROOT / "results/hardware/moshi_v2_negative_selection_finalization.json"
    run_dir = ROOT / "checkpoints/moshi_fa_100s_v3_no_embed"
    preflight = json_object(preflight_path)
    probe = json_object(probe_path)
    v2_finalization = json_object(v2_finalization_path)
    train_metrics_path = run_dir / "metrics.train.jsonl"
    train_rows = [
        json.loads(line)
        for line in train_metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_steps = [100, 200, 300, 400, 500]
    checkpoint_paths = [
        run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated" / "lora.safetensors"
        for step in expected_steps
    ]
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
    journal = subprocess.run(
        [
            "journalctl",
            f"_SYSTEMD_INVOCATION_ID={args.training_invocation_id}",
            "--no-pager",
            "-o",
            "cat",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    final_outputs = (
        ROOT / "results/moshi_v3_adapter_validation.json",
        ROOT / "results/moshi_v3_final_test.json",
        ROOT / "results/moshi_v3_final_runtime.json",
    )
    preconditions = {
        "launch_worktree_clean": not worktree_status.strip(),
        "preflight_bound_to_config": (
            preflight.get("status") == "passed"
            and (preflight.get("artifacts") or {}).get("config_sha256") == sha256_file(config_path)
            and preflight.get("test_access_started") is False
        ),
        "profile_probe_bound_to_config": (
            probe.get("status") == "passed"
            and probe.get("full_profile_gate_passes") is True
            and (probe.get("artifacts") or {}).get("full_config_sha256") == sha256_file(config_path)
        ),
        "protocol_hash_bound_before_training": (
            (preflight.get("artifacts") or {}).get("protocol_sha256") == sha256_file(protocol_path)
        ),
        "v2_final_test_never_accessed": (
            v2_finalization.get("status") == "passed"
            and v2_finalization.get("test_access_started") is False
        ),
        "training_reached_step_500": (bool(train_rows) and int(train_rows[-1]["step"]) == 500),
        "training_completed_cleanly_in_journal": all(
            marker in journal
            for marker in (
                "step: 000500 - done (%): 100.0",
                "checkpoint_000500",
                "train - INFO - done!",
                "train - INFO - Closed everything!",
            )
        ),
        "all_five_checkpoints_present": all(path.is_file() for path in checkpoint_paths),
        "final_test_outputs_absent": not any(path.exists() for path in final_outputs),
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"v3 validation-pipeline preconditions failed: {preconditions}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "launch_commit": launch_commit,
        "training_invocation_id": args.training_invocation_id,
        "test_access_started": False,
        "preconditions": preconditions,
        "stages": [],
        "artifacts": {
            "config_sha256": sha256_file(config_path),
            "protocol_sha256": sha256_file(protocol_path),
            "preflight_sha256": sha256_file(preflight_path),
            "profile_probe_sha256": sha256_file(probe_path),
            "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_report(out_path, report)
    for stage in validation_commands():
        output = Path(stage["output"])
        if output.exists():
            raise FileExistsError(f"refusing to overwrite v3 stage output: {output}")
        command = [str(value) for value in stage["command"]]
        stage_report: dict[str, Any] = {
            "name": stage["name"],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "status": "running",
        }
        report["stages"].append(stage_report)
        write_report(out_path, report)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        stage_report["finished_at"] = datetime.now(timezone.utc).isoformat()
        stage_report["exit_code"] = completed.returncode
        if output.is_file():
            stage_report["output"] = output.relative_to(ROOT).as_posix()
            stage_report["output_sha256"] = sha256_file(output)

        if stage["name"] != "eligible_only_selection":
            stage_report["status"] = "passed" if completed.returncode == 0 else "failed"
            write_report(out_path, report)
            if completed.returncode != 0:
                report["status"] = "failed"
                report["failure"] = f"stage failed: {stage['name']}"
                write_report(out_path, report)
                raise RuntimeError(report["failure"])
            continue

        selection = json_object(output) if output.is_file() else {}
        eligible = (
            completed.returncode == 0
            and selection.get("status") == "passed"
            and selection.get("selection_passes") is True
            and isinstance(selection.get("selected"), dict)
        )
        expected_negative = (
            completed.returncode != 0
            and selection.get("status") == "failed"
            and selection.get("selection_passes") is False
            and selection.get("selected") is None
            and selection.get("eligible_candidate_count") == 0
        )
        stage_report["status"] = "passed" if eligible or expected_negative else "failed"
        stage_report["outcome"] = (
            "eligible_checkpoint_selected"
            if eligible
            else "scientific_negative_no_eligible_checkpoint"
            if expected_negative
            else "unexpected_selection_failure"
        )
        write_report(out_path, report)
        if not (eligible or expected_negative):
            report["status"] = "failed"
            report["failure"] = "unexpected eligible-only selection failure"
            write_report(out_path, report)
            raise RuntimeError(report["failure"])

        report["selection_outcome"] = stage_report["outcome"]
        report["selected_step"] = int(selection["selected"]["step"]) if eligible else None
        report["eligible_steps"] = selection.get("eligible_steps")
        report["scientific_negative_outcome"] = expected_negative

    report["status"] = "passed"
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["requirements"] = {
        "all_validation_stages_completed": all(
            stage.get("status") == "passed" for stage in report["stages"]
        ),
        "test_access_never_started": report["test_access_started"] is False,
    }
    report["next_stage"] = (
        "validate the selected adapter before one-time final-test access"
        if report["selected_step"] is not None
        else "v3 failed closed; design a new experiment without accessing the final test"
    )
    write_report(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
