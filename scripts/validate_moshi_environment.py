#!/usr/bin/env python3
"""Validate the isolated Moshi trainer, pinned assets, and project launcher."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import safetensors.torch
import torch
from finetune.args import TrainArgs
from moshi.models import loaders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "torchaudio", "triton", "moshi", "finetune", "sphn")
CHECKOUTS = {
    "Moshi": PROJECT_ROOT / "third_party" / "checkouts" / "moshi",
    "Moshi-Finetune": PROJECT_ROOT / "third_party" / "checkouts" / "moshi-finetune",
}


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def validate_base_model_schema(entry: dict) -> dict:
    destination = (
        PROJECT_ROOT
        / "hf_cache"
        / "pinned"
        / ("moshika-pytorch-bf16-" + str(entry["revision"])[:7])
    )
    model_path = destination / "model.safetensors"
    info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=entry["repository"].removeprefix("https://huggingface.co/"),
        moshi_weights=model_path,
        mimi_weights=destination / "tokenizer-e351c8d8-checkpoint125.safetensors",
        tokenizer=destination / "tokenizer_spm_32k_3.model",
        config_path=PROJECT_ROOT / "configs" / "moshika_7b_legacy.json",
    )
    model = info.get_moshi(device="meta", dtype=torch.bfloat16, load_weight=False)
    state = safetensors.torch.load_file(model_path, device="cpu")
    file_keys = len(state)
    incompatible = model.load_state_dict(state, strict=False, assign=True)
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    mimi = info.get_mimi(device="cpu")
    tokenizer = info.get_text_tokenizer()
    report = {
        "model_file": str(model_path),
        "file_tensor_keys": file_keys,
        "model_state_keys_after_legacy_expansion": len(model.state_dict()),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "meta_parameters": meta_parameters,
        "mimi": {
            "parameters": sum(parameter.numel() for parameter in mimi.parameters()),
            "sample_rate": mimi.sample_rate,
            "frame_rate": mimi.frame_rate,
            "num_codebooks": mimi.num_codebooks,
        },
        "sentencepiece": {
            "vocabulary_size": tokenizer.vocab_size(),
            "unknown_token_id": tokenizer.unk_id(),
        },
    }
    report["valid"] = (
        not any((report["missing_keys"], report["unexpected_keys"], report["meta_parameters"]))
        and report["mimi"]
        == {
            "parameters": 79308609,
            "sample_rate": 24000,
            "frame_rate": 12.5,
            "num_codebooks": 8,
        }
        and report["sentencepiece"]["vocabulary_size"] == 32000
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "hardware" / "moshi_environment.json",
    )
    args = parser.parse_args()
    lock = json.loads(
        (PROJECT_ROOT / "third_party" / "UPSTREAMS.lock.json").read_text(encoding="utf-8")
    )
    expected_revisions = {row["name"]: row["revision"] for row in lock["upstreams"]}
    base_entry = next(row for row in lock["upstreams"] if row["name"] == "Moshika-PyTorch-BF16")
    try:
        base_model_schema = validate_base_model_schema(base_entry)
    except Exception as exc:
        base_model_schema = {"valid": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    checkouts = {
        name: {
            "expected": expected_revisions.get(name),
            "actual": (actual := git_head(path)),
            "matches": actual == expected_revisions.get(name),
        }
        for name, path in CHECKOUTS.items()
    }
    legacy = json.loads(
        (PROJECT_ROOT / "configs" / "moshika_7b_legacy.json").read_text(encoding="utf-8")
    )
    model_type = legacy.pop("model_type", None)
    configs = {}
    config_filenames = (
        "moshi_h100.yaml",
        "moshi_h100_smoke.yaml",
        "moshi_h100_profile_probe.yaml",
    )
    for filename in config_filenames:
        parsed = TrainArgs.load(str(PROJECT_ROOT / "configs" / filename), drop_extra_fields=False)
        configs[filename] = {
            "max_steps": parsed.max_steps,
            "duration_sec": parsed.duration_sec,
            "lora": dataclasses.asdict(parsed.lora),
            "param_dtype": parsed.param_dtype,
        }
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gates = {
        "running_in_isolated_venv": Path(sys.prefix).resolve()
        == (PROJECT_ROOT / ".venv-moshi").resolve(),
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "architecture_matches_pinned_loader": model_type == "moshi"
        and legacy == loaders._lm_kwargs,
        "checkout_revisions_match": all(row["matches"] for row in checkouts.values()),
        "configs_parse": len(configs) == len(config_filenames),
        "model_assets_load_and_schema_match": base_model_schema["valid"] is True,
        "single_gpu_training_launcher_available": (
            CHECKOUTS["Moshi-Finetune"] / "train.py"
        ).is_file()
        and (PROJECT_ROOT / "scripts" / "moshi_train_entry.py").is_file(),
    }
    report = {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "environment": sys.prefix,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nccl": ".".join(str(part) for part in torch.cuda.nccl.version()),
        "distributed_backends": {
            "nccl_available": torch.distributed.is_nccl_available(),
            "gloo_available": torch.distributed.is_gloo_available(),
            "single_gpu_gloo_fallback": True,
        },
        "packages": packages,
        "checkouts": checkouts,
        "configs": configs,
        "base_model_schema": base_model_schema,
        "gates": gates,
        "valid": all(gates.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
