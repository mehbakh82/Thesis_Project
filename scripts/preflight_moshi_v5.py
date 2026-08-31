#!/usr/bin/env python3
"""Fail-closed preflight for the bounded Moshi v5 experiment."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.preflight_moshi_v3 import (  # noqa: E402
    DTYPE_BYTES,
    GIB,
    OFFICIAL_EXAMPLE_SHA256,
    adapter_schema,
    experiment_invariants,
    gpu_memory,
    json_object,
    sha256_file,
    training_shape,
    write_report,
    yaml_object,
)

TEXT_EMBEDDINGS = {"depformer_text_emb.weight", "text_emb.weight"}
EXPECTED_AUDIO_EMBEDDING_COUNT = 23


def tensor_bytes(row: dict[str, Any]) -> int:
    dtype = str(row["dtype"])
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported safetensors dtype: {dtype}")
    return math.prod(int(value) for value in row["shape"]) * DTYPE_BYTES[dtype]


def selective_adapter_bytes(
    rank128_schema: list[dict[str, Any]],
    broad_embedding_schema: list[dict[str, Any]],
) -> dict[str, int]:
    lora_rows = [row for row in rank128_schema if "lora" in str(row["name"])]
    unexpected_rank128 = [row["name"] for row in rank128_schema if row not in lora_rows]
    text_rows = [row for row in broad_embedding_schema if str(row["name"]) in TEXT_EMBEDDINGS]
    if len(lora_rows) != 674 or unexpected_rank128:
        raise ValueError(
            "rank-128 source must contain exactly 674 LoRA tensors and no other tensors"
        )
    if {str(row["name"]) for row in text_rows} != TEXT_EMBEDDINGS:
        raise ValueError("broad source does not contain both exact text embeddings")
    lora_bytes = sum(tensor_bytes(row) for row in lora_rows)
    text_embedding_bytes = sum(tensor_bytes(row) for row in text_rows)
    return {
        "lora_bytes": lora_bytes,
        "text_embedding_bytes": text_embedding_bytes,
        "estimated_adapter_bytes": lora_bytes + text_embedding_bytes,
        "expected_tensor_count": len(lora_rows) + len(text_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/moshi_h100_v5.yaml"))
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=Path("configs/moshi_h100_v5_probe.yaml"),
    )
    parser.add_argument(
        "--embedding-policy",
        type=Path,
        default=Path("configs/moshi_v5_embedding_policy.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v5_preflight.json"),
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    probe_config_path = (ROOT / args.probe_config).resolve()
    policy_path = (ROOT / args.embedding_policy).resolve()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v5 preflight evidence: {out_path}")

    v4_config_path = ROOT / "configs/moshi_h100_v4.yaml"
    protocol_path = ROOT / "docs/MOSHI_V5_SELECTION_PROTOCOL.md"
    official_example_path = ROOT / "third_party/checkouts/moshi-finetune/example/moshi_7B.yaml"
    v4_selection_path = ROOT / "results/moshi_v4_checkpoint_selection.json"
    v4_runtime_path = ROOT / "results/moshi_v4_runtime_candidates.json"
    v4_certificate_path = ROOT / "results/hardware/moshi_h100_v4_training.json"
    split_path = ROOT / "results/moshi_v2_split.json"
    rank128_lora_source_path = (
        ROOT / "checkpoints/moshi_fa_100s_v3_no_embed/checkpoints/checkpoint_000400/"
        "consolidated/lora.safetensors"
    )
    v2_adapter_path = (
        ROOT / "checkpoints/moshi_fa_100s_v2/checkpoints/checkpoint_000400/"
        "consolidated/lora.safetensors"
    )
    launcher_path = ROOT / "scripts/moshi_train_entry.py"

    config = yaml_object(config_path)
    probe_config = yaml_object(probe_config_path)
    v4_config = yaml_object(v4_config_path)
    official_example = yaml_object(official_example_path)
    policy = json_object(policy_path)
    v4_selection = json_object(v4_selection_path)
    v4_runtime = json_object(v4_runtime_path)
    v4_certificate = json_object(v4_certificate_path)
    split = json_object(split_path)

    estimate = selective_adapter_bytes(
        adapter_schema(rank128_lora_source_path),
        adapter_schema(v2_adapter_path),
    )
    retained_adapter_count = 6
    transient_adapter_count = 1
    safety_bytes = 2 * GIB
    required_free_bytes = (
        estimate["estimated_adapter_bytes"] * (retained_adapter_count + transient_adapter_count)
        + safety_bytes
    )
    disk = shutil.disk_usage(ROOT)
    gpu = gpu_memory()
    worktree_status = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    launch_commit = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    lora = config.get("lora") or {}
    official_lora = official_example.get("lora") or {}
    split_outputs = split.get("outputs") or {}
    policy_names = set(policy.get("trainable_embedding_parameters") or [])
    frozen_audio_names = set(policy.get("frozen_audio_embedding_parameters") or [])
    launcher_text = launcher_path.read_text(encoding="utf-8")
    v4_pass_counts = [
        int(candidate.get("panel_pass_count") or 0)
        for candidate in v4_runtime.get("candidates") or []
    ]
    final_output_paths = (
        ROOT / "results/moshi_v5_checkpoint_selection.json",
        ROOT / "results/moshi_v5_adapter_validation.json",
        ROOT / "results/moshi_v5_final_test.json",
        ROOT / "results/moshi_v5_final_runtime.json",
    )
    prior_final_output_paths = (
        ROOT / "results/moshi_v4_adapter_validation.json",
        ROOT / "results/moshi_v4_final_test.json",
        ROOT / "results/moshi_v4_final_runtime.json",
    )
    requirements = {
        "launch_worktree_clean": not worktree_status.strip(),
        "v4_failed_closed_without_test_access": (
            v4_selection.get("status") == "failed"
            and v4_selection.get("eligible_candidate_count") == 0
            and v4_selection.get("selected") is None
            and v4_selection.get("heldout_test_used_for_selection") is False
            and v4_certificate.get("status") == "passed"
            and v4_certificate.get("test_access_started") is False
            and v4_certificate.get("selection_outcome")
            == "scientific_negative_no_eligible_checkpoint"
            and (v4_certificate.get("artifacts") or {}).get("selection_sha256")
            == sha256_file(v4_selection_path)
            and v4_pass_counts == [1, 0, 0, 1, 0]
        ),
        "single_objective_change_from_v4_exact": (
            {
                key: value
                for key, value in experiment_invariants(config).items()
                if key != "first_codebook_weight_multiplier"
            }
            == {
                key: value
                for key, value in experiment_invariants(v4_config).items()
                if key != "first_codebook_weight_multiplier"
            }
            and config.get("first_codebook_weight_multiplier") == 10.0
            and v4_config.get("first_codebook_weight_multiplier") == 100.0
            and (policy.get("objective_change") or {})
            == {
                "baseline_experiment": "v4",
                "changed_parameter": "first_codebook_weight_multiplier",
                "baseline_value": 100.0,
                "v5_value": 10.0,
                "all_other_training_profile_fields_unchanged": True,
            }
        ),
        "rank128_lora_and_broad_upstream_embedding_switch_unchanged": (
            lora == {"enable": True, "rank": 128, "scaling": 2.0, "ft_embed": False}
            and lora == (v4_config.get("lora") or {})
            and lora == official_lora
        ),
        "selective_embedding_policy_exact": (
            policy.get("schema_version") == 1
            and policy.get("mode") == "text_embeddings_only"
            and (policy.get("launcher_environment") or {})
            == {"name": "MOSHI_TEXT_EMBEDDINGS_ONLY", "required_value": "1"}
            and policy.get("upstream_lora_ft_embed") is False
            and policy_names == TEXT_EMBEDDINGS
            and len(frozen_audio_names) == EXPECTED_AUDIO_EMBEDDING_COUNT
            and not policy_names & frozen_audio_names
            and policy.get("expected_lora_tensor_count") == 674
            and policy.get("expected_total_adapter_tensor_count") == 676
        ),
        "launcher_implements_fail_closed_selective_scope": all(
            marker in launcher_text
            for marker in (
                "TEXT_EMBEDDING_PARAMETER_NAMES",
                "_configure_text_embeddings_only",
                "MOSHI_TEXT_EMBEDDINGS_ONLY must be exactly 0 or 1",
                "unexpected trainable embedding scope",
                "MOSHI_TEXT_EMBEDDINGS_ONLY_EFFECTIVE",
            )
        ),
        "bounded_candidate_set_exact": (
            config.get("max_steps") == 500
            and config.get("ckpt_freq") == 100
            and config.get("num_ckpt_keep") == 5
        ),
        "probe_shape_matches_full": training_shape(probe_config) == training_shape(config),
        "probe_controls_exact": (
            probe_config.get("max_steps") == 1
            and probe_config.get("do_eval") is False
            and probe_config.get("do_ckpt") is True
            and probe_config.get("ckpt_freq") == 1
            and probe_config.get("num_ckpt_keep") == 1
        ),
        "upstream_example_hash_current": (
            sha256_file(official_example_path) == OFFICIAL_EXAMPLE_SHA256
        ),
        "frozen_split_passed": (
            split.get("split_passes") is True
            and (split_outputs.get("train") or {}).get("path")
            == str((config.get("data") or {}).get("train_data"))
            and (split_outputs.get("validation") or {}).get("path")
            == str((config.get("data") or {}).get("eval_data"))
            and (split_outputs.get("final_test") or {}).get("rows") == 738
        ),
        "v5_run_directories_absent": not (
            (ROOT / str(config.get("run_dir"))).exists()
            or (ROOT / str(probe_config.get("run_dir"))).exists()
        ),
        "v5_final_outputs_absent": not any(path.exists() for path in final_output_paths),
        "prior_final_test_outputs_absent": not any(
            path.exists() for path in prior_final_output_paths
        ),
        "estimated_schema_exact": (
            estimate["lora_bytes"] == 775749632
            and estimate["text_embedding_bytes"] == 327690240
            and estimate["estimated_adapter_bytes"] == 1103439872
            and estimate["expected_tensor_count"] == 676
        ),
        "disk_headroom_passes": disk.free >= required_free_bytes,
        "h100_headroom_passes": (
            gpu["name"] == "NVIDIA H100 NVL" and int(gpu["free_mib"]) >= 30 * 1024
        ),
    }
    status = "passed" if all(requirements.values()) else "failed"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "launch_commit": launch_commit,
        "test_access_started": False,
        "experiment": {
            "name": "moshi_v5_codebook10",
            "candidate_steps": [100, 200, 300, 400, 500],
            "changed_from_v4": {
                "only_changed_parameter": "first_codebook_weight_multiplier",
                "first_codebook_weight_multiplier": {"v4": 100.0, "v5": 10.0},
                "lora": "unchanged rank-128 LoRA",
                "upstream_ft_embed": "unchanged false",
                "selective_text_embeddings": {
                    "v4": sorted(TEXT_EMBEDDINGS),
                    "v5": sorted(TEXT_EMBEDDINGS),
                },
            },
        },
        "storage": {
            **estimate,
            "estimated_adapter_gib": round(estimate["estimated_adapter_bytes"] / GIB, 3),
            "retained_adapter_count": retained_adapter_count,
            "transient_adapter_count": transient_adapter_count,
            "safety_gib": 2,
            "free_bytes": disk.free,
            "free_gib": round(disk.free / GIB, 3),
            "required_free_bytes": required_free_bytes,
            "required_free_gib": round(required_free_bytes / GIB, 3),
        },
        "gpu": gpu,
        "requirements": requirements,
        "preflight_passes": status == "passed",
        "artifacts": {
            "config": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(config_path),
            "probe_config": probe_config_path.relative_to(ROOT).as_posix(),
            "probe_config_sha256": sha256_file(probe_config_path),
            "embedding_policy": policy_path.relative_to(ROOT).as_posix(),
            "embedding_policy_sha256": sha256_file(policy_path),
            "protocol": protocol_path.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha256_file(protocol_path),
            "v4_selection_sha256": sha256_file(v4_selection_path),
            "v4_runtime_sha256": sha256_file(v4_runtime_path),
            "v4_certificate_sha256": sha256_file(v4_certificate_path),
            "split_report_sha256": sha256_file(split_path),
            "upstream_example_sha256": sha256_file(official_example_path),
            "rank128_source_adapter_sha256": sha256_file(rank128_lora_source_path),
            "broad_embedding_source_adapter_sha256": sha256_file(v2_adapter_path),
            "launcher_sha256": sha256_file(launcher_path),
            "preflight_sha256": sha256_file(Path(__file__).resolve()),
        },
        "next_stage": (
            "run the exact-shape one-step v5 probe with selective scope; do not access final-test data"
            if status == "passed"
            else "repair failed v5 preflight gates before any optimizer step"
        ),
    }
    write_report(out_path, report)
    if status != "passed":
        raise RuntimeError(f"Moshi v5 preflight failed: {requirements}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
