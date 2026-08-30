#!/usr/bin/env python3
"""Select a Moshi v2 adapter only from official-runtime-eligible candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_moshi_adapter import json_object, project_path, sha256_file  # noqa: E402
from thesis_s2s.config import load_yaml, portable_project_values  # noqa: E402


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a selection report without importing the NumPy-backed metrics module."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def candidate_map(report: dict[str, Any], label: str) -> dict[int, dict[str, Any]]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not all(
        isinstance(candidate, dict) for candidate in candidates
    ):
        raise ValueError(f"{label} has no valid candidate list")
    result = {int(candidate["step"]): candidate for candidate in candidates}
    if len(result) != len(candidates):
        raise ValueError(f"{label} contains duplicate candidate steps")
    return result


def runtime_protocol_valid(runtime: dict[str, Any], protocol_mode: str) -> bool:
    protocol = runtime.get("protocol") or {}
    common = protocol.get("complete_user_turn_required") is True and protocol.get(
        "panel_indices"
    ) == [0, 11, 33, 51, 55, 74, 85, 87, 106]
    if protocol_mode == "corrected-v2":
        correction = runtime.get("protocol_correction") or {}
        return (
            common
            and correction.get(
                "frozen_after_training_before_corrected_generation_and_final_test_access"
            )
            is True
            and correction.get("selection_criterion_changed") is False
            and correction.get("eligibility_thresholds_changed") is False
        )
    if protocol_mode in {"predeclared-v3", "predeclared-v4"}:
        provenance = runtime.get("protocol_provenance") or {}
        version = protocol_mode.removeprefix("predeclared-")
        return (
            common
            and runtime.get("protocol_predeclared_before_training") is True
            and provenance.get(f"frozen_before_{version}_optimizer_step") is True
            and provenance.get("selection_criterion_changed_after_training") is False
            and provenance.get("eligibility_thresholds_changed_after_training") is False
        )
    raise ValueError(f"unsupported Moshi protocol mode: {protocol_mode}")


def select_checkpoint(
    *,
    config_path: Path,
    reevaluation_path: Path,
    runtime_path: Path,
    out_path: Path,
    root: Path = ROOT,
    protocol_mode: str = "corrected-v2",
) -> dict[str, Any]:
    config_path = config_path.resolve()
    reevaluation_path = reevaluation_path.resolve()
    runtime_path = runtime_path.resolve()
    out_path = out_path.resolve()
    config = load_yaml(config_path)
    reevaluation = json_object(reevaluation_path)
    runtime = json_object(runtime_path)
    predeclared_version = {
        "predeclared-v3": "v3",
        "predeclared-v4": "v4",
    }.get(protocol_mode)
    if protocol_mode not in {"corrected-v2", "predeclared-v3", "predeclared-v4"}:
        raise ValueError(f"unsupported Moshi protocol mode: {protocol_mode}")
    expected_runtime_schema = (
        {"v3": 3, "v4": 4}[predeclared_version]
        if predeclared_version is not None
        else 2
    )
    experiment_label = predeclared_version or "v2"
    is_predeclared = predeclared_version is not None
    max_steps = int(config["max_steps"])
    checkpoint_frequency = int(config["ckpt_freq"])
    expected_steps = list(range(checkpoint_frequency, max_steps + 1, checkpoint_frequency))
    loss_by_step = candidate_map(reevaluation, "reevaluation report")
    runtime_by_step = candidate_map(runtime, "runtime-panel report")

    preconditions = {
        "criterion_predeclared_before_training": True,
        "reevaluation_passed": (
            reevaluation.get("schema_version") == 1
            and reevaluation.get("status") == "passed"
            and reevaluation.get("reevaluation_passes") is True
        ),
        "heldout_test_not_used_for_loss": reevaluation.get("heldout_test_used") is False,
        "runtime_panel_evaluation_complete": (
            runtime.get("schema_version") == expected_runtime_schema
            and runtime.get("status") == "passed"
            and runtime.get("runtime_panel_evaluation_passes") is True
            and runtime.get("selection_performed") is False
        ),
        "candidate_steps_exact": (
            sorted(loss_by_step) == expected_steps and sorted(runtime_by_step) == expected_steps
        ),
        "profile_matches": (
            reevaluation.get("configured_max_steps") == max_steps
            and reevaluation.get("checkpoint_frequency") == checkpoint_frequency
        ),
        "runtime_bound_to_reevaluation": (
            (runtime.get("artifacts") or {}).get("reevaluation_sha256")
            == sha256_file(reevaluation_path)
        ),
        "complete_prompt_protocol_valid": runtime_protocol_valid(runtime, protocol_mode),
    }
    if not all(preconditions.values()):
        raise RuntimeError(
            f"Moshi {experiment_label} selection preconditions failed: {preconditions}"
        )

    candidates: list[dict[str, Any]] = []
    for step in expected_steps:
        loss_candidate = loss_by_step[step]
        runtime_candidate = runtime_by_step[step]
        adapter_path = project_path(
            root,
            runtime_candidate.get("adapter_path"),
            f"candidate {step} adapter",
        )
        config_artifact_path = project_path(
            root,
            runtime_candidate.get("config_path"),
            f"candidate {step} config",
        )
        loss = float(loss_candidate["eval_loss"])
        text_loss = float(loss_candidate["text_eval_loss"])
        audio_loss = float(loss_candidate["audio_eval_loss"])
        static_validation = runtime_candidate.get("static_adapter_validation")
        if not isinstance(static_validation, dict):
            raise ValueError(f"candidate {step} lacks static adapter validation")
        eligibility_requirements = {
            "complete_validation_losses_finite": all(
                math.isfinite(value) for value in (loss, text_loss, audio_loss)
            ),
            "complete_validation_scope": (
                loss_candidate.get("validation_scope") == "complete_fixed_manifest"
                and int(loss_candidate.get("sample_count") or 0) > 0
            ),
            "adapter_hashes_agree_and_are_current": (
                loss_candidate.get("adapter_sha256")
                == static_validation.get("adapter_sha256")
                == sha256_file(adapter_path)
            ),
            "config_hashes_agree_and_are_current": (
                loss_candidate.get("config_sha256")
                == static_validation.get("config_sha256")
                == sha256_file(config_artifact_path)
            ),
            "static_adapter_validation_passed": static_validation.get("passes") is True,
            "official_server_and_all_panel_gates_passed": runtime_candidate.get(
                "candidate_runtime_passes"
            )
            is True,
            "all_nine_panel_rows_passed": (
                runtime_candidate.get("panel_count") == 9
                and runtime_candidate.get("panel_pass_count") == 9
            ),
        }
        candidates.append(
            {
                "step": step,
                "eval_loss": loss,
                "text_eval_loss": text_loss,
                "audio_eval_loss": audio_loss,
                "validation_sample_count": loss_candidate.get("sample_count"),
                "validation_manifest_sha256": loss_candidate.get("validation_manifest_sha256"),
                "adapter_path": adapter_path.relative_to(root).as_posix(),
                "adapter_sha256": sha256_file(adapter_path),
                "adapter_bytes": adapter_path.stat().st_size,
                "config_path": config_artifact_path.relative_to(root).as_posix(),
                "config_sha256": sha256_file(config_artifact_path),
                "runtime_panel_pass_count": runtime_candidate.get("panel_pass_count"),
                "runtime_panel_count": runtime_candidate.get("panel_count"),
                "eligibility_requirements": eligibility_requirements,
                "autoregressive_eligible": all(eligibility_requirements.values()),
            }
        )

    eligible = [
        candidate for candidate in candidates if candidate["autoregressive_eligible"] is True
    ]
    selected = (
        min(
            eligible,
            key=lambda candidate: (float(candidate["eval_loss"]), int(candidate["step"])),
        )
        if eligible
        else None
    )
    selection_passes = selected is not None
    report: dict[str, Any] = {
        "schema_version": (
            {"v3": 5, "v4": 6}[predeclared_version]
            if predeclared_version is not None
            else 4
        ),
        "status": "passed" if selection_passes else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criterion_predeclared_before_training": True,
        "exact_runtime_panel_indices_predeclared_before_training": is_predeclared,
        "runtime_panel_corrected_after_training": not is_predeclared,
        "criterion": (
            "among candidates passing exact artifact validation and every unchanged automatic "
            "gate on all nine deterministically selected complete-prompt official-server "
            "validation rows, choose minimum finite mean "
            "eval_loss on the identical complete validation scope; exact ties choose the "
            "earlier step"
        ),
        "heldout_test_used_for_selection": False,
        "training_complete": True,
        "selection_input_corrected_after_training": not is_predeclared,
        "correction_changes_selection_criterion": False,
        "original_upstream_metrics_eligible_for_selection": False,
        "configured_max_steps": max_steps,
        "checkpoint_frequency": checkpoint_frequency,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "eligible_steps": [candidate["step"] for candidate in eligible],
        "candidates": candidates,
        "selected": selected,
        "validation_reevaluation": {
            "path": reevaluation_path.relative_to(root).as_posix(),
            "sha256": sha256_file(reevaluation_path),
            "requirements": preconditions,
        },
        "runtime_validation": {
            "path": runtime_path.relative_to(root).as_posix(),
            "sha256": sha256_file(runtime_path),
            "panel_indices": (runtime.get("protocol") or {}).get("panel_indices"),
            "panel_rule": (runtime.get("protocol") or {}).get("panel_rule"),
            "complete_user_turn_required": True,
            "protocol_correction": runtime.get("protocol_correction"),
            "protocol_provenance": runtime.get("protocol_provenance"),
            "all_panel_rows_must_pass": True,
        },
        "selection_passes": selection_passes,
        "next_stage": (
            f"validate the exact selected adapter, freeze its hash, then access the fresh "
            f"{experiment_label} final test exactly once; never revise this choice from test results"
            if selection_passes
            else (
                f"{experiment_label} fails closed because no predeclared candidate passed "
                "every eligibility gate"
            )
        ),
    }
    report = portable_project_values(report)
    write_report(out_path, report)
    if not selection_passes:
        raise RuntimeError(
            f"no Moshi {experiment_label} checkpoint passed every frozen eligibility gate"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moshi_h100_v2.yaml"),
    )
    parser.add_argument(
        "--reevaluation",
        type=Path,
        default=Path("results/moshi_v2_validation_reevaluation.json"),
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("results/moshi_v2_runtime_candidates.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/moshi_v2_checkpoint_selection.json"),
    )
    parser.add_argument(
        "--protocol-mode",
        choices=("corrected-v2", "predeclared-v3", "predeclared-v4"),
        default="corrected-v2",
    )
    args = parser.parse_args()
    report = select_checkpoint(
        config_path=ROOT / args.config,
        reevaluation_path=ROOT / args.reevaluation,
        runtime_path=ROOT / args.runtime,
        out_path=ROOT / args.out,
        protocol_mode=args.protocol_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
