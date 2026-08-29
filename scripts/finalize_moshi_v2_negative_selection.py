#!/usr/bin/env python3
"""Finalize the expected fail-closed Moshi v2 selection without touching test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FINAL_OUTPUTS = (
    "results/moshi_v2_adapter_validation.json",
    "results/moshi_v2_final_test.json",
    "results/moshi_v2_final_runtime.json",
    "results/hardware/moshi_h100_v2_training.json",
)


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("results/moshi_v2_runtime_candidates.json"),
    )
    parser.add_argument(
        "--failed-pipeline-receipt",
        type=Path,
        default=Path(
            "results/hardware/moshi_v2_corrected_posttraining_selection_environment_failed.json"
        ),
    )
    parser.add_argument(
        "--selection-out",
        type=Path,
        default=Path("results/moshi_v2_checkpoint_selection.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v2_negative_selection_finalization.json"),
    )
    args = parser.parse_args()

    runtime_path = (ROOT / args.runtime).resolve()
    failed_receipt_path = (ROOT / args.failed_pipeline_receipt).resolve()
    selection_path = (ROOT / args.selection_out).resolve()
    out_path = (ROOT / args.out).resolve()
    for path in (selection_path, out_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")

    final_paths = [ROOT / relative for relative in FINAL_OUTPUTS]
    existing_final_outputs = [
        path.relative_to(ROOT).as_posix() for path in final_paths if path.exists()
    ]
    if existing_final_outputs:
        raise RuntimeError(f"final-test outputs unexpectedly exist: {existing_final_outputs}")

    runtime = json_object(runtime_path)
    failed_receipt = json_object(failed_receipt_path)
    stages = failed_receipt.get("stages")
    if not isinstance(stages, list):
        raise ValueError("failed pipeline receipt has no stage list")
    runtime_stage = stages[0] if len(stages) > 0 and isinstance(stages[0], dict) else {}
    selection_stage = stages[1] if len(stages) > 1 and isinstance(stages[1], dict) else {}

    preconditions = {
        "corrected_runtime_complete": (
            runtime.get("schema_version") == 2
            and runtime.get("status") == "passed"
            and runtime.get("runtime_panel_evaluation_passes") is True
            and runtime.get("selection_performed") is False
            and runtime.get("candidate_count") == 20
        ),
        "no_runtime_candidate_eligible": runtime.get("eligible_steps") == [],
        "pipeline_failure_was_selection_environment_only": (
            failed_receipt.get("status") == "failed"
            and failed_receipt.get("failure") == "stage failed: eligible_only_selection"
            and failed_receipt.get("test_access_started") is False
            and runtime_stage.get("name") == "corrected_complete_prompt_validation_runtime_panel"
            and runtime_stage.get("status") == "passed"
            and runtime_stage.get("output_sha256") == sha256_file(runtime_path)
            and selection_stage.get("name") == "eligible_only_selection"
            and selection_stage.get("status") == "failed"
            and "output" not in selection_stage
        ),
        "final_test_outputs_absent": not existing_final_outputs,
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"negative-selection preconditions failed: {preconditions}")

    command = [
        sys.executable,
        str(ROOT / "scripts/select_moshi_v2_checkpoint.py"),
        "--runtime",
        runtime_path.relative_to(ROOT).as_posix(),
        "--out",
        selection_path.relative_to(ROOT).as_posix(),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    selector_stderr_last_line = (
        completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else None
    )
    selection = json_object(selection_path) if selection_path.is_file() else {}
    requirements = {
        "selector_failed_closed_as_expected": completed.returncode != 0,
        "negative_selection_report_written": selection_path.is_file(),
        "selection_schema_supported": selection.get("schema_version") == 4,
        "selection_status_failed": selection.get("status") == "failed",
        "selection_passes_false": selection.get("selection_passes") is False,
        "selected_checkpoint_absent": selection.get("selected") is None,
        "eligible_candidate_count_zero": selection.get("eligible_candidate_count") == 0,
        "eligible_steps_empty": selection.get("eligible_steps") == [],
        "heldout_test_not_used": selection.get("heldout_test_used_for_selection") is False,
        "selection_bound_to_corrected_runtime": (
            (selection.get("runtime_validation") or {}).get("sha256") == sha256_file(runtime_path)
        ),
        "final_test_outputs_still_absent": not any(path.exists() for path in final_paths),
    }
    status = "passed" if all(requirements.values()) else "failed"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_outcome": (
            "Moshi v2 failed closed: none of 20 checkpoints passed all nine unchanged "
            "official-runtime validation gates on complete user prompts."
        ),
        "scientific_negative_outcome_finalized": status == "passed",
        "test_access_started": False,
        "preconditions": preconditions,
        "requirements": requirements,
        "selector": {
            "command": [
                ".venv/bin/python",
                "scripts/select_moshi_v2_checkpoint.py",
                "--runtime",
                runtime_path.relative_to(ROOT).as_posix(),
                "--out",
                selection_path.relative_to(ROOT).as_posix(),
            ],
            "exit_code": completed.returncode,
            "expected_nonzero_exit": True,
            "stderr_last_line": selector_stderr_last_line,
        },
        "artifacts": {
            "runtime_path": runtime_path.relative_to(ROOT).as_posix(),
            "runtime_sha256": sha256_file(runtime_path),
            "failed_pipeline_receipt_path": failed_receipt_path.relative_to(ROOT).as_posix(),
            "failed_pipeline_receipt_sha256": sha256_file(failed_receipt_path),
            "selection_path": selection_path.relative_to(ROOT).as_posix()
            if selection_path.is_file()
            else None,
            "selection_sha256": sha256_file(selection_path) if selection_path.is_file() else None,
            "finalizer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "next_stage": (
            "Design a new predeclared training experiment; do not access the v2 final test."
            if status == "passed"
            else "Repair finalization evidence before any further experiment."
        ),
    }
    write_report(out_path, report)
    if status != "passed":
        raise RuntimeError(f"negative-selection finalization failed: {requirements}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
