#!/usr/bin/env python3
"""Select the Moshi adapter by the predeclared validation-loss rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from thesis_s2s.config import load_yaml, portable_project_values
from thesis_s2s.metrics import write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl_objects(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def select_checkpoint(
    run_dir: Path,
    config_path: Path,
    out_path: Path,
    reevaluation_path: Path,
) -> dict:
    config = load_yaml(config_path)
    max_steps = int(config["max_steps"])
    checkpoint_frequency = int(config["ckpt_freq"])
    expected_steps = set(range(checkpoint_frequency, max_steps + 1, checkpoint_frequency))
    training_rows = _jsonl_objects(run_dir / "metrics.train.jsonl")
    reevaluation = _json_object(reevaluation_path)
    rows = reevaluation.get("candidates")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("reevaluation report has no valid candidate list")
    report_steps = {int(row.get("step") or 0) for row in rows}
    report_sample_counts = {int(row.get("sample_count") or 0) for row in rows}
    report_manifest_hashes = {row.get("validation_manifest_sha256") for row in rows}
    reevaluation_requirements = {
        "schema_supported": reevaluation.get("schema_version") == 1,
        "reevaluation_passed": (
            reevaluation.get("status") == "passed"
            and reevaluation.get("reevaluation_passes") is True
        ),
        "test_split_not_used": reevaluation.get("heldout_test_used") is False,
        "selection_not_already_performed": reevaluation.get("checkpoint_selection_performed")
        is False,
        "criterion_unchanged": reevaluation.get("selection_criterion_changed") is False,
        "postrun_input_correction_disclosed": reevaluation.get(
            "selection_input_corrected_after_training"
        )
        is True,
        "original_metrics_rejected": reevaluation.get("original_metrics_eligible_for_selection")
        is False,
        "profile_matches": (
            reevaluation.get("configured_max_steps") == max_steps
            and reevaluation.get("checkpoint_frequency") == checkpoint_frequency
        ),
        "all_candidate_steps_present": report_steps == expected_steps,
        "identical_nonempty_scope": (
            len(report_sample_counts) == 1
            and next(iter(report_sample_counts), 0) > 0
            and len(report_manifest_hashes) == 1
            and None not in report_manifest_hashes
            and all(row.get("validation_scope") == "complete_fixed_manifest" for row in rows)
        ),
        "training_complete_with_finite_logged_losses": (
            max(int(row.get("step") or 0) for row in training_rows) == max_steps
            and all(math.isfinite(float(row.get("loss"))) for row in training_rows)
        ),
    }
    if not all(reevaluation_requirements.values()):
        raise RuntimeError(
            f"fixed-scope validation reevaluation is ineligible: {reevaluation_requirements}"
        )

    candidates = []
    for row in rows:
        step = int(row.get("step") or 0)
        loss = float(row.get("eval_loss"))
        if (
            step <= 0
            or step > max_steps
            or step % checkpoint_frequency != 0
            or not math.isfinite(loss)
        ):
            raise RuntimeError(f"invalid corrected validation candidate at step {step}")
        consolidated = run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
        adapter = consolidated / "lora.safetensors"
        adapter_config = consolidated / "config.json"
        if not adapter.is_file() or not adapter_config.is_file():
            raise RuntimeError(f"saved checkpoint artifact is absent at step {step}")
        if row.get("adapter_sha256") != sha256_file(adapter):
            raise RuntimeError(f"adapter changed after reevaluation at step {step}")
        if row.get("config_sha256") != sha256_file(adapter_config):
            raise RuntimeError(f"adapter config changed after reevaluation at step {step}")
        candidates.append(
            {
                "step": step,
                "eval_loss": loss,
                "text_eval_loss": row.get("text_eval_loss"),
                "audio_eval_loss": row.get("audio_eval_loss"),
                "validation_sample_count": row.get("sample_count"),
                "validation_manifest_sha256": row.get("validation_manifest_sha256"),
                "adapter_path": str(adapter),
                "adapter_sha256": sha256_file(adapter),
                "adapter_bytes": adapter.stat().st_size,
                "config_path": str(adapter_config),
                "config_sha256": sha256_file(adapter_config),
            }
        )
    expected_candidates = max_steps // checkpoint_frequency
    if len(candidates) != expected_candidates:
        raise RuntimeError(
            f"expected {expected_candidates} corrected checkpoint-aligned candidates, "
            f"found {len(candidates)}"
        )
    selected = min(candidates, key=lambda row: (float(row["eval_loss"]), int(row["step"])))
    report = {
        "schema_version": 2,
        "criterion_predeclared_before_training": True,
        "criterion": (
            "minimum finite mean eval_loss on one identical complete validation scope among "
            "every 500-step saved checkpoint; exact ties choose the earlier step"
        ),
        "heldout_test_used_for_selection": False,
        "training_complete": True,
        "selection_input_corrected_after_training": True,
        "correction_changes_selection_criterion": False,
        "original_upstream_metrics_eligible_for_selection": False,
        "configured_max_steps": max_steps,
        "checkpoint_frequency": checkpoint_frequency,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "validation_reevaluation": {
            "path": str(reevaluation_path),
            "sha256": sha256_file(reevaluation_path),
            "requirements": reevaluation_requirements,
        },
        "selection_passes": True,
        "next_stage": (
            "load the selected adapter in the pinned runtime, then evaluate the "
            "group-disjoint test split once; never revise this choice from test or "
            "subjective outputs"
        ),
    }
    report = portable_project_values(report)
    write_json(out_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/moshi_fa"))
    parser.add_argument("--config", type=Path, default=Path("configs/moshi_h100.yaml"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/moshi_checkpoint_selection.json"),
    )
    parser.add_argument(
        "--reevaluation",
        type=Path,
        default=Path("results/moshi_validation_reevaluation.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            select_checkpoint(args.run_dir, args.config, args.out, args.reevaluation),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
