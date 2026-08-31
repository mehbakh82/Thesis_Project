#!/usr/bin/env python3
"""Create the fail-closed certificate for the bounded Moshi v5 run."""

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
from scripts.record_moshi_v2_training_run import (  # noqa: E402
    git_blob,
    journal_for_invocation,
    jsonl_objects,
)
from scripts.validate_moshi_adapter import json_object, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("checkpoints/moshi_fa_100s_v5_codebook10"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moshi_h100_v5.yaml"),
    )
    parser.add_argument("--launch-commit", required=True)
    parser.add_argument("--service-unit", required=True)
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument(
        "--reevaluation",
        type=Path,
        default=Path("results/moshi_v5_validation_reevaluation.json"),
    )
    parser.add_argument(
        "--runtime-panel",
        type=Path,
        default=Path("results/moshi_v5_runtime_candidates.json"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/moshi_v5_checkpoint_selection.json"),
    )
    parser.add_argument(
        "--validation-pipeline",
        type=Path,
        default=Path("results/hardware/moshi_v5_validation_pipeline.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_h100_v5_training.json"),
    )
    args = parser.parse_args()

    run_dir = (ROOT / args.run_dir).resolve()
    config_path = (ROOT / args.config).resolve()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v5 training certificate: {out_path}")
    args_path = run_dir / "args.yaml"
    train_metrics_path = run_dir / "metrics.train.jsonl"
    raw_eval_metrics_path = run_dir / "metrics.eval.jsonl"
    reevaluation_path = (ROOT / args.reevaluation).resolve()
    runtime_path = (ROOT / args.runtime_panel).resolve()
    selection_path = (ROOT / args.selection).resolve()
    pipeline_path = (ROOT / args.validation_pipeline).resolve()
    protocol_path = ROOT / "docs/MOSHI_V5_SELECTION_PROTOCOL.md"
    preflight_path = ROOT / "results/hardware/moshi_v5_preflight.json"
    probe_path = ROOT / "results/hardware/moshi_h100_v5_profile_probe.json"
    policy_path = ROOT / "configs/moshi_v5_embedding_policy.json"
    v4_certificate_path = ROOT / "results/hardware/moshi_h100_v4_training.json"
    environment_path = ROOT / "results/hardware/moshi_environment.json"
    export_path = ROOT / "results/moshi_export_report.json"
    split_path = ROOT / "results/moshi_v2_split.json"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved_args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(resolved_args, dict):
        raise ValueError("training config and resolved args must be mappings")
    train_rows = jsonl_objects(train_metrics_path)
    raw_eval_rows = jsonl_objects(raw_eval_metrics_path)
    reevaluation = json_object(reevaluation_path)
    runtime = json_object(runtime_path)
    selection = json_object(selection_path)
    pipeline = json_object(pipeline_path)
    preflight = json_object(preflight_path)
    probe = json_object(probe_path)
    policy = json_object(policy_path)
    v4_certificate = json_object(v4_certificate_path)
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
    checkpoint_rows: list[dict[str, Any]] = []
    for step in expected_steps:
        consolidated = run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
        adapter_path = consolidated / "lora.safetensors"
        adapter_config_path = consolidated / "config.json"
        with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
            tensor_names = list(adapter.keys())
        lora_names = {name for name in tensor_names if "lora" in name}
        embedding_names = {name for name in tensor_names if "emb" in name}
        checkpoint_rows.append(
            {
                "step": step,
                "adapter_path": adapter_path.relative_to(ROOT).as_posix(),
                "adapter_bytes": adapter_path.stat().st_size,
                "adapter_sha256": sha256_file(adapter_path),
                "adapter_tensor_count": len(tensor_names),
                "lora_tensor_count": len(lora_names),
                "embedding_parameters": sorted(embedding_names),
                "selective_schema_exact": (
                    len(tensor_names) == 676
                    and len(lora_names) == 674
                    and embedding_names == {"depformer_text_emb.weight", "text_emb.weight"}
                    and set(tensor_names) == lora_names | embedding_names
                ),
                "config_path": adapter_config_path.relative_to(ROOT).as_posix(),
                "config_sha256": sha256_file(adapter_config_path),
            }
        )

    journal_text = journal_for_invocation(args.invocation_id)
    first_timestamp = re.search(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \(UTC\)",
        journal_text,
        flags=re.MULTILINE,
    )
    elapsed_matches = re.findall(r" - (\d+):(\d{2}):(\d{2}) - ", journal_text)
    if first_timestamp is None or not elapsed_matches:
        raise ValueError("v5 journal lacks expected UTC timestamp/elapsed fields")
    hours, minutes, seconds = (int(value) for value in elapsed_matches[-1])
    wall_seconds = hours * 3600 + minutes * 60 + seconds
    launch_revision = subprocess.run(
        ["git", "rev-parse", args.launch_commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    launch_config = yaml.safe_load(
        git_blob(launch_revision, config_path.relative_to(ROOT).as_posix())
    )
    launch_launcher = git_blob(launch_revision, "scripts/moshi_train_entry.py")
    launch_policy = git_blob(
        launch_revision,
        policy_path.relative_to(ROOT).as_posix(),
    )
    if not isinstance(launch_config, dict):
        raise ValueError("launch-commit v5 config is not a mapping")

    checkpoint_hashes_match = all(
        row["adapter_sha256"] == reevaluated_by_step[row["step"]]["adapter_sha256"]
        and row["config_sha256"] == reevaluated_by_step[row["step"]]["config_sha256"]
        for row in checkpoint_rows
    )
    tensor_counts = {row["adapter_tensor_count"] for row in checkpoint_rows}
    lora_tensor_counts = {row["lora_tensor_count"] for row in checkpoint_rows}
    embedding_parameter_sets = {tuple(row["embedding_parameters"]) for row in checkpoint_rows}
    adapter_sizes = {row["adapter_bytes"] for row in checkpoint_rows}
    selection_valid = (
        selection.get("schema_version") == 7
        and selection.get("heldout_test_used_for_selection") is False
        and (
            (
                selection.get("selection_passes") is True
                and isinstance(selection.get("selected"), dict)
                and int(selection.get("eligible_candidate_count") or 0) > 0
            )
            or (
                selection.get("selection_passes") is False
                and selection.get("selected") is None
                and selection.get("eligible_candidate_count") == 0
            )
        )
    )
    requirements = {
        "launch_and_current_profiles_match": comparable_profile(launch_config)
        == comparable_profile(config)
        == comparable_profile(resolved_args),
        "launch_commit_contains_exact_launcher_and_policy": (
            hashlib.sha256(launch_launcher).hexdigest()
            == sha256_file(ROOT / "scripts/moshi_train_entry.py")
            and hashlib.sha256(launch_policy).hexdigest() == sha256_file(policy_path)
        ),
        "all_expected_logged_steps_present": [int(row["step"]) for row in train_rows]
        == expected_logged_steps,
        "all_training_losses_finite": all(math.isfinite(float(row["loss"])) for row in train_rows),
        "all_raw_validation_losses_finite": all(
            math.isfinite(float(row["eval_loss"])) for row in raw_eval_rows
        ),
        "configured_final_step_completed": int(train_rows[-1]["step"]) == max_steps,
        "service_journal_records_clean_completion": all(
            marker in journal_text
            for marker in (
                "MOSHI_TEXT_EMBEDDINGS_ONLY_EFFECTIVE=depformer_text_emb.weight,text_emb.weight",
                "step: 000500 - done (%): 100.0",
                "checkpoint_000500",
                "train - INFO - done!",
                "train - INFO - Closed everything!",
            )
        ),
        "all_expected_checkpoints_present": len(checkpoint_rows) == len(expected_steps),
        "all_checkpoint_schemas_match": (
            tensor_counts == {676}
            and lora_tensor_counts == {674}
            and embedding_parameter_sets == {("depformer_text_emb.weight", "text_emb.weight")}
            and all(row["selective_schema_exact"] is True for row in checkpoint_rows)
            and len(adapter_sizes) == 1
        ),
        "checkpoint_hashes_match_reevaluation": checkpoint_hashes_match,
        "fixed_scope_reevaluation_passed": (
            reevaluation.get("status") == "passed"
            and reevaluation.get("reevaluation_passes") is True
            and reevaluation.get("heldout_test_used") is False
        ),
        "predeclared_runtime_panel_completed": (
            runtime.get("schema_version") == 5
            and runtime.get("status") == "passed"
            and runtime.get("runtime_panel_evaluation_passes") is True
            and runtime.get("protocol_predeclared_before_training") is True
        ),
        "eligible_only_selection_valid": selection_valid,
        "validation_pipeline_passed_without_test_access": (
            pipeline.get("status") == "passed"
            and pipeline.get("training_invocation_id") == args.invocation_id
            and pipeline.get("launch_commit") == launch_revision
            and pipeline.get("test_access_started") is False
            and (pipeline.get("requirements") or {}).get("test_access_never_started") is True
        ),
        "preflight_probe_and_selective_policy_passed": (
            preflight.get("status") == "passed"
            and probe.get("status") == "passed"
            and probe.get("full_profile_gate_passes") is True
            and policy.get("mode") == "text_embeddings_only"
            and set(policy.get("trainable_embedding_parameters") or [])
            == {"depformer_text_emb.weight", "text_emb.weight"}
            and (preflight.get("artifacts") or {}).get("embedding_policy_sha256")
            == sha256_file(policy_path)
            and (probe.get("embedding_policy") or {}).get("sha256") == sha256_file(policy_path)
        ),
        "v4_negative_finalized_without_test_access": (
            v4_certificate.get("status") == "passed"
            and v4_certificate.get("test_access_started") is False
            and v4_certificate.get("selection_outcome")
            == "scientific_negative_no_eligible_checkpoint"
        ),
        "protocol_hash_unchanged": (
            (preflight.get("artifacts") or {}).get("protocol_sha256") == sha256_file(protocol_path)
        ),
        "h100_environment_valid": (
            environment.get("valid") is True and "H100" in str(environment.get("gpu") or "")
        ),
        "training_export_valid_under_waiver": export.get("training_ready_under_qa_waiver") is True,
        "group_split_passed": split.get("split_passes") is True,
    }
    passed = all(requirements.values())
    journal_bytes = journal_text.encode("utf-8")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_training_evidence": True,
        "human_quality_claim_allowed": False,
        "test_access_started": False,
        "training_policy": "limited internal training under the documented student QA waiver",
        "experiment": {
            "name": "moshi_v5_codebook10",
            "changed_from_v4": {
                "only_changed_parameter": "first_codebook_weight_multiplier",
                "first_codebook_weight_multiplier": {"v4": 100.0, "v5": 10.0},
                "lora_rank": {"v4": 128, "v5": 128},
                "upstream_ft_embed": {"v4": False, "v5": False},
                "selective_embedding_parameters": {
                    "v4": ["depformer_text_emb.weight", "text_emb.weight"],
                    "v5": ["depformer_text_emb.weight", "text_emb.weight"],
                },
                "candidate_horizon": {"v4": 500, "v5": 500},
            },
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
            "journal_retained_by_systemd": True,
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
            "rows": len(raw_eval_rows),
            "all_finite": all(math.isfinite(float(row["eval_loss"])) for row in raw_eval_rows),
            "diagnosis": reevaluation.get("correction_reason"),
        },
        "checkpoint_count": len(checkpoint_rows),
        "checkpoint_schema": {
            "tensor_count": next(iter(tensor_counts)),
            "adapter_bytes": next(iter(adapter_sizes)),
        },
        "checkpoints": checkpoint_rows,
        "selection_outcome": pipeline.get("selection_outcome"),
        "selected": selection.get("selected"),
        "eligible_steps": selection.get("eligible_steps"),
        "requirements": requirements,
        "training_run_passes": passed,
        "artifacts": {
            "training_config_sha256": sha256_file(config_path),
            "resolved_args_sha256": sha256_file(args_path),
            "training_metrics_sha256": sha256_file(train_metrics_path),
            "raw_eval_metrics_sha256": sha256_file(raw_eval_metrics_path),
            "reevaluation_sha256": sha256_file(reevaluation_path),
            "runtime_panel_sha256": sha256_file(runtime_path),
            "selection_sha256": sha256_file(selection_path),
            "validation_pipeline_sha256": sha256_file(pipeline_path),
            "preflight_sha256": sha256_file(preflight_path),
            "profile_probe_sha256": sha256_file(probe_path),
            "protocol_sha256": sha256_file(protocol_path),
            "embedding_policy_sha256": sha256_file(policy_path),
            "v4_certificate_sha256": sha256_file(v4_certificate_path),
            "training_entry_sha256": sha256_file(ROOT / "scripts/moshi_train_entry.py"),
            "certificate_sha256": sha256_file(Path(__file__).resolve()),
        },
        "next_stage": (
            "validate the selected adapter before one-time final-test access"
            if selection.get("selection_passes") is True
            else "v5 is a finalized negative result; do not access its final test"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(f"v5 training certificate failed: {requirements}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
