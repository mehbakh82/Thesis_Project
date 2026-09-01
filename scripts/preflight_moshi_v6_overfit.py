#!/usr/bin/env python3
"""Fail-closed preflight for the Moshi v6 train-only capacity diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.preflight_moshi_v3 import (  # noqa: E402
    DTYPE_BYTES,
    GIB,
    OFFICIAL_EXAMPLE_SHA256,
    adapter_schema,
    gpu_memory,
    json_object,
    sha256_file,
    training_shape,
    write_report,
    yaml_object,
)

FULL_TEXT_PARAMETERS = {
    "depformer_text_emb.weight",
    "text_emb.weight",
    "text_linear.frozen_W.weight",
}


def tensor_bytes(row: dict[str, Any]) -> int:
    dtype = str(row["dtype"])
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported safetensors dtype: {dtype}")
    return math.prod(int(value) for value in row["shape"]) * DTYPE_BYTES[dtype]


def expected_storage(
    lora_source: Path,
    embedding_source: Path,
    base_model: Path,
) -> dict[str, int]:
    lora_rows = [row for row in adapter_schema(lora_source) if "lora" in row["name"]]
    embedding_rows = [
        row
        for row in adapter_schema(embedding_source)
        if row["name"] in {"depformer_text_emb.weight", "text_emb.weight"}
    ]
    with safe_open(base_model, framework="pt", device="cpu") as model:
        if "text_linear.weight" not in model.keys():
            raise ValueError("base model has no independent text_linear.weight")
        output_row = {
            "name": "text_linear.frozen_W.weight",
            "dtype": model.get_slice("text_linear.weight").get_dtype(),
            "shape": model.get_slice("text_linear.weight").get_shape(),
        }
    if len(lora_rows) != 674:
        raise ValueError(f"expected 674 rank-128 LoRA tensors, found {len(lora_rows)}")
    if {row["name"] for row in embedding_rows} != {
        "depformer_text_emb.weight",
        "text_emb.weight",
    }:
        raise ValueError("embedding source lacks the exact two text embeddings")
    lora_bytes = sum(tensor_bytes(row) for row in lora_rows)
    embedding_bytes = sum(tensor_bytes(row) for row in embedding_rows)
    output_projection_bytes = tensor_bytes(output_row)
    return {
        "lora_bytes": lora_bytes,
        "text_embedding_bytes": embedding_bytes,
        "text_output_projection_bytes": output_projection_bytes,
        "estimated_adapter_bytes": lora_bytes + embedding_bytes + output_projection_bytes,
        "expected_tensor_count": len(lora_rows) + len(embedding_rows) + 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/moshi_h100_v6_overfit.yaml"))
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=Path("configs/moshi_h100_v6_overfit_probe.yaml"),
    )
    parser.add_argument("--policy", type=Path, default=Path("configs/moshi_v6_overfit_policy.json"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v6_overfit_preflight.json"),
    )
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    probe_config_path = (ROOT / args.probe_config).resolve()
    policy_path = (ROOT / args.policy).resolve()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v6 preflight evidence: {out_path}")

    protocol_path = ROOT / "docs/MOSHI_V6_OVERFIT_PROTOCOL.md"
    data_report_path = ROOT / "results/moshi_v6_overfit_data.json"
    official_example_path = ROOT / "third_party/checkouts/moshi-finetune/example/moshi_7B.yaml"
    launcher_path = ROOT / "scripts/moshi_train_entry.py"
    v5_selection_path = ROOT / "results/moshi_v5_checkpoint_selection.json"
    v5_certificate_path = ROOT / "results/hardware/moshi_h100_v5_training.json"
    lora_source_path = (
        ROOT / "checkpoints/moshi_fa_100s_v2/checkpoints/checkpoint_000400/"
        "consolidated/lora.safetensors"
    )
    embedding_source_path = (
        ROOT / "checkpoints/moshi_fa_100s_v2/checkpoints/checkpoint_000400/"
        "consolidated/lora.safetensors"
    )

    config = yaml_object(config_path)
    probe_config = yaml_object(probe_config_path)
    policy = json_object(policy_path)
    data_report = json_object(data_report_path)
    official_example = yaml_object(official_example_path)
    v5_selection = json_object(v5_selection_path)
    v5_certificate = json_object(v5_certificate_path)
    data_manifest = ROOT / str((config.get("data") or {}).get("train_data"))
    base_model_path = ROOT / str((config.get("moshi_paths") or {}).get("moshi_path"))
    estimate = expected_storage(lora_source_path, embedding_source_path, base_model_path)

    disk = shutil.disk_usage(ROOT)
    gpu = gpu_memory()
    retained_adapter_count = 5
    safety_bytes = 2 * GIB
    required_free_bytes = (
        estimate["estimated_adapter_bytes"] * retained_adapter_count + safety_bytes
    )
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
    launcher_text = launcher_path.read_text(encoding="utf-8")
    final_outputs = (
        ROOT / "results/moshi_v6_overfit_reevaluation.json",
        ROOT / "results/moshi_v6_overfit_runtime.json",
        ROOT / "results/moshi_v6_overfit_selection.json",
    )
    requirements = {
        "launch_worktree_clean": not worktree_status.strip(),
        "v5_failed_closed_without_final_test_access": (
            v5_selection.get("status") == "failed"
            and v5_selection.get("eligible_candidate_count") == 0
            and v5_selection.get("selected") is None
            and v5_selection.get("heldout_test_used_for_selection") is False
            and v5_certificate.get("status") == "passed"
            and v5_certificate.get("test_access_started") is False
        ),
        "train_only_data_freeze_passed": (
            data_report.get("passes") is True
            and data_report.get("scientific_scope") == "diagnostic_not_validation_or_final_test"
            and data_report.get("human_verified") is False
            and data_report.get("final_test_accessed") is False
            and (data_report.get("output") or {}).get("rows") == 32
            and (data_report.get("output") or {}).get("manifest")
            == data_manifest.relative_to(ROOT).as_posix()
            and (data_report.get("output") or {}).get("manifest_sha256")
            == sha256_file(data_manifest)
        ),
        "same_manifest_used_only_for_in_sample_train_eval": (
            (config.get("data") or {}).get("train_data")
            == (config.get("data") or {}).get("eval_data")
            and "test" not in str((config.get("data") or {}).get("train_data")).lower()
        ),
        "diagnostic_profile_exact": (
            lora
            == {
                "enable": True,
                "rank": 64,
                "scaling": 2.0,
                "ft_embed": False,
            }
            and {key: lora.get(key) for key in ("enable", "scaling", "ft_embed")}
            == {
                key: (official_example.get("lora") or {}).get(key)
                for key in ("enable", "scaling", "ft_embed")
            }
            and config.get("duration_sec") == 12
            and config.get("batch_size") == 1
            and config.get("num_microbatches") == 4
            and config.get("first_codebook_weight_multiplier") == 1.0
            and config.get("text_padding_weight") == 0.1
            and config.get("max_steps") == 200
            and config.get("ckpt_freq") == 50
            and config.get("num_ckpt_keep") == 4
        ),
        "probe_shape_matches_full": training_shape(probe_config) == training_shape(config),
        "probe_controls_exact": (
            probe_config.get("max_steps") == 1
            and probe_config.get("do_eval") is False
            and probe_config.get("do_ckpt") is True
            and probe_config.get("ckpt_freq") == 1
            and probe_config.get("num_ckpt_keep") == 1
        ),
        "parameter_and_objective_policy_exact": (
            policy.get("mode") == "persian_text_head_adaptation"
            and policy.get("diagnostic_only") is True
            and set(policy.get("trainable_full_parameters") or []) == FULL_TEXT_PARAMETERS
            and policy.get("full_text_output_projection") == "text_linear.frozen_W.weight"
            and policy.get("frozen_audio_embeddings") is True
            and policy.get("audio_loss_weight") == 0.1
            and policy.get("expected_lora_tensor_count") == 674
            and policy.get("expected_total_adapter_tensor_count") == 677
            and policy.get("estimated_adapter_bytes") == estimate["estimated_adapter_bytes"]
            and (policy.get("launcher_environment") or {})
            == {
                "MOSHI_DISTRIBUTED_BACKEND": "gloo",
                "MOSHI_PERSIAN_TEXT_ADAPTATION": "1",
                "MOSHI_TEXT_EMBEDDINGS_ONLY": "0",
                "MOSHI_AUDIO_LOSS_WEIGHT": "0.1",
            }
        ),
        "launcher_implements_fail_closed_v6_scope": all(
            marker in launcher_text
            for marker in (
                "PERSIAN_TEXT_PARAMETER_NAMES",
                "_configure_persian_text_adaptation",
                "unexpected trainable full-parameter scope",
                "unexpected trainable audio embeddings",
                "MOSHI_PERSIAN_TEXT_ADAPTATION_EFFECTIVE",
                "_configure_audio_loss_weight",
                "MOSHI_AUDIO_LOSS_WEIGHT_EFFECTIVE",
            )
        ),
        "adapter_storage_estimate_exact": (
            estimate["lora_bytes"] == 387874816
            and estimate["text_embedding_bytes"] == 327690240
            and estimate["text_output_projection_bytes"] == 262144000
            and estimate["estimated_adapter_bytes"] == 977709056
            and estimate["expected_tensor_count"] == 677
        ),
        "protocol_frozen": protocol_path.is_file(),
        "upstream_example_hash_current": sha256_file(official_example_path)
        == OFFICIAL_EXAMPLE_SHA256,
        "v6_run_directories_absent": not (
            (ROOT / str(config.get("run_dir"))).exists()
            or (ROOT / str(probe_config.get("run_dir"))).exists()
        ),
        "v6_result_outputs_absent": not any(path.exists() for path in final_outputs),
        "disk_headroom_passes": disk.free >= required_free_bytes,
        "h100_headroom_passes": (
            gpu["name"] == "NVIDIA H100 NVL" and int(gpu["free_mib"]) >= 25 * 1024
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed" if all(requirements.values()) else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "launch_commit": launch_commit,
        "diagnostic_only": True,
        "scientific_validation_evidence": False,
        "test_access_started": False,
        "experiment": {
            "name": "moshi_v6_overfit",
            "candidate_steps": [50, 100, 150, 200],
            "training_rows": 32,
            "scope": "in_sample_capacity_diagnostic",
        },
        "storage": {
            **estimate,
            "estimated_adapter_gib": round(estimate["estimated_adapter_bytes"] / GIB, 3),
            "retained_adapter_count": retained_adapter_count,
            "safety_gib": 2,
            "free_bytes": disk.free,
            "free_gib": round(disk.free / GIB, 3),
            "required_free_bytes": required_free_bytes,
            "required_free_gib": round(required_free_bytes / GIB, 3),
        },
        "gpu": gpu,
        "requirements": requirements,
        "preflight_passes": all(requirements.values()),
        "artifacts": {
            "config": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(config_path),
            "probe_config": probe_config_path.relative_to(ROOT).as_posix(),
            "probe_config_sha256": sha256_file(probe_config_path),
            "policy": policy_path.relative_to(ROOT).as_posix(),
            "policy_sha256": sha256_file(policy_path),
            "protocol": protocol_path.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha256_file(protocol_path),
            "data_report": data_report_path.relative_to(ROOT).as_posix(),
            "data_report_sha256": sha256_file(data_report_path),
            "data_manifest_sha256": sha256_file(data_manifest),
            "launcher_sha256": sha256_file(launcher_path),
            "base_model_sha256": sha256_file(base_model_path),
            "upstream_example_sha256": sha256_file(official_example_path),
        },
    }
    write_report(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["preflight_passes"] is not True:
        raise RuntimeError(f"v6 overfit preflight failed: {requirements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
