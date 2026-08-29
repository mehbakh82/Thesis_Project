#!/usr/bin/env python3
"""Fail-closed preflight for the bounded Moshi v3 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_EXAMPLE_SHA256 = "c574c424d1a078a89bcb7556978d8b4b8c160276aeebbda01ff77ca6f259ac48"
GIB = 1024**3
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def yaml_object(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def experiment_invariants(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key)
        for key in (
            "data",
            "moshi_paths",
            "full_finetuning",
            "first_codebook_weight_multiplier",
            "text_padding_weight",
            "duration_sec",
            "batch_size",
            "num_microbatches",
            "gradient_checkpointing",
            "optim",
            "seed",
            "log_freq",
            "eval_freq",
            "do_eval",
            "do_ckpt",
            "ckpt_freq",
            "save_adapters",
            "overwrite_run_dir",
        )
    }


def training_shape(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key)
        for key in (
            "data",
            "moshi_paths",
            "full_finetuning",
            "lora",
            "first_codebook_weight_multiplier",
            "text_padding_weight",
            "duration_sec",
            "batch_size",
            "num_microbatches",
            "gradient_checkpointing",
            "optim",
            "seed",
            "save_adapters",
        )
    }


def rank_scaled_adapter_bytes(
    tensors: list[dict[str, Any]],
    *,
    source_rank: int,
    target_rank: int,
) -> int:
    if source_rank <= 0 or target_rank <= 0:
        raise ValueError("LoRA ranks must be positive")
    total = 0
    for tensor in tensors:
        name = str(tensor["name"])
        if "lora" not in name:
            continue
        dtype = str(tensor["dtype"])
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported safetensors dtype: {dtype}")
        elements = math.prod(int(value) for value in tensor["shape"])
        total += elements * DTYPE_BYTES[dtype]
    if total <= 0:
        raise ValueError("source adapter contains no LoRA tensors")
    return math.ceil(total * target_rank / source_rank)


def adapter_schema(path: Path) -> list[dict[str, Any]]:
    with safe_open(path, framework="pt", device="cpu") as adapter:
        return [
            {
                "name": key,
                "dtype": adapter.get_slice(key).get_dtype(),
                "shape": adapter.get_slice(key).get_shape(),
            }
            for key in adapter.keys()
        ]


def gpu_memory() -> dict[str, int | str]:
    output = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(output) != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {len(output)}")
    name, total, used, free = [value.strip() for value in output[0].split(",")]
    return {
        "name": name,
        "total_mib": int(total),
        "used_mib": int(used),
        "free_mib": int(free),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/moshi_h100_v3.yaml"))
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=Path("configs/moshi_h100_v3_probe.yaml"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hardware/moshi_v3_preflight.json"),
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    probe_config_path = (ROOT / args.probe_config).resolve()
    out_path = (ROOT / args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite v3 preflight evidence: {out_path}")

    v2_config_path = ROOT / "configs/moshi_h100_v2.yaml"
    protocol_path = ROOT / "docs/MOSHI_V3_SELECTION_PROTOCOL.md"
    official_example_path = ROOT / "third_party/checkouts/moshi-finetune/example/moshi_7B.yaml"
    v2_selection_path = ROOT / "results/moshi_v2_checkpoint_selection.json"
    v2_finalization_path = ROOT / "results/hardware/moshi_v2_negative_selection_finalization.json"
    split_path = ROOT / "results/moshi_v2_split.json"
    source_adapter_path = (
        ROOT / "checkpoints/moshi_fa_100s_v2/checkpoints/checkpoint_000400/"
        "consolidated/lora.safetensors"
    )

    config = yaml_object(config_path)
    probe_config = yaml_object(probe_config_path)
    v2_config = yaml_object(v2_config_path)
    official_example = yaml_object(official_example_path)
    v2_selection = json_object(v2_selection_path)
    v2_finalization = json_object(v2_finalization_path)
    split = json_object(split_path)

    source_schema = adapter_schema(source_adapter_path)
    estimated_adapter_bytes = rank_scaled_adapter_bytes(
        source_schema,
        source_rank=int((v2_config.get("lora") or {})["rank"]),
        target_rank=int((config.get("lora") or {})["rank"]),
    )
    retained_adapter_count = 6
    transient_adapter_count = 1
    safety_bytes = 2 * GIB
    required_free_bytes = (
        estimated_adapter_bytes * (retained_adapter_count + transient_adapter_count) + safety_bytes
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

    v3_lora = config.get("lora") or {}
    official_lora = official_example.get("lora") or {}
    split_outputs = split.get("outputs") or {}
    final_output_paths = (
        ROOT / "results/moshi_v3_checkpoint_selection.json",
        ROOT / "results/moshi_v3_adapter_validation.json",
        ROOT / "results/moshi_v3_final_test.json",
        ROOT / "results/moshi_v3_final_runtime.json",
    )
    requirements = {
        "launch_worktree_clean": not worktree_status.strip(),
        "v2_failed_closed": (
            v2_selection.get("status") == "failed"
            and v2_selection.get("eligible_candidate_count") == 0
            and v2_selection.get("selected") is None
            and v2_selection.get("heldout_test_used_for_selection") is False
            and v2_finalization.get("status") == "passed"
            and v2_finalization.get("test_access_started") is False
        ),
        "same_train_validation_and_optimization_invariants_as_v2": (
            experiment_invariants(config) == experiment_invariants(v2_config)
        ),
        "upstream_embedding_preserving_lora_restored": (
            v3_lora
            == {
                "enable": True,
                "rank": 128,
                "scaling": 2.0,
                "ft_embed": False,
            }
            and v3_lora == official_lora
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
        "v3_run_directories_absent": not (
            (ROOT / str(config.get("run_dir"))).exists()
            or (ROOT / str(probe_config.get("run_dir"))).exists()
        ),
        "v3_final_outputs_absent": not any(path.exists() for path in final_output_paths),
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
            "name": "moshi_v3_embedding_preserving",
            "candidate_steps": [100, 200, 300, 400, 500],
            "changed_from_v2": {
                "lora.rank": {"v2": 64, "v3": 128},
                "lora.ft_embed": {"v2": True, "v3": False},
                "max_steps": {"v2": 2000, "v3_screen": 500},
            },
        },
        "storage": {
            "free_bytes": disk.free,
            "free_gib": round(disk.free / GIB, 3),
            "estimated_adapter_bytes": estimated_adapter_bytes,
            "estimated_adapter_gib": round(estimated_adapter_bytes / GIB, 3),
            "retained_adapter_count": retained_adapter_count,
            "transient_adapter_count": transient_adapter_count,
            "safety_gib": 2,
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
            "protocol": protocol_path.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha256_file(protocol_path),
            "v2_selection_sha256": sha256_file(v2_selection_path),
            "v2_finalization_sha256": sha256_file(v2_finalization_path),
            "split_report_sha256": sha256_file(split_path),
            "upstream_example_sha256": sha256_file(official_example_path),
            "source_adapter": source_adapter_path.relative_to(ROOT).as_posix(),
            "source_adapter_sha256": sha256_file(source_adapter_path),
            "preflight_sha256": sha256_file(Path(__file__).resolve()),
        },
        "next_stage": (
            "run the exact-shape one-step v3 probe; do not access final-test data"
            if status == "passed"
            else "repair failed v3 preflight gates before any optimizer step"
        ),
    }
    write_report(out_path, report)
    if status != "passed":
        raise RuntimeError(f"Moshi v3 preflight failed: {requirements}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
