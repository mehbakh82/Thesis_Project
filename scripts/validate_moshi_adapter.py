#!/usr/bin/env python3
"""Fail-closed validation for the checkpoint selected after Moshi training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_CONFIG_ONLY_KEYS = {
    "lm_gen_config",
    "lora_name",
    "mimi_name",
    "model_type",
    "moshi_name",
    "tokenizer_name",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def normalise_lm_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(raw_config)
    for key in RAW_CONFIG_ONLY_KEYS:
        config.pop(key, None)
    return config


def compare_adapter_schema(
    expected: dict[str, dict[str, Any]],
    actual: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    shape_mismatches = sorted(
        key for key in expected_keys & actual_keys if expected[key]["shape"] != actual[key]["shape"]
    )
    dtype_mismatches = sorted(
        key for key in expected_keys & actual_keys if expected[key]["dtype"] != actual[key]["dtype"]
    )
    return {
        "expected_tensor_count": len(expected),
        "actual_tensor_count": len(actual),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "exact": not (missing or unexpected or shape_mismatches or dtype_mismatches),
    }


def selected_candidate_by_rule(candidates: list[object]) -> dict[str, Any]:
    valid_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("selection candidate must be an object")
        step = int(candidate.get("step") or 0)
        loss = float(candidate.get("eval_loss"))
        if step <= 0 or not math.isfinite(loss):
            raise ValueError("selection candidate has an invalid step/eval_loss")
        valid_candidates.append(candidate)
    if not valid_candidates:
        raise ValueError("selection report has no candidates")
    return min(
        valid_candidates,
        key=lambda candidate: (float(candidate["eval_loss"]), int(candidate["step"])),
    )


def project_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty project-relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be project-relative")
    unresolved = root / candidate
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the project root")
    if unresolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {candidate}")
    return resolved


def expected_adapter_schema(
    raw_config: dict[str, Any],
    *,
    ft_embed: bool,
) -> dict[str, dict[str, Any]]:
    import torch
    from moshi.models import loaders

    model = loaders.get_moshi_lm(
        None,
        lm_kwargs=normalise_lm_config(raw_config),
        device="meta",
        dtype=torch.bfloat16,
        lora_weights=None,
        fuse_lora=False,
    )
    schema = {
        name: {"shape": list(parameter.shape), "dtype": "BF16"}
        for name, parameter in model.named_parameters()
        if "lora" in name or (ft_embed and "emb" in name)
    }
    del model
    return schema


def read_adapter_schema(path: Path) -> tuple[dict[str, dict[str, Any]], bool, int]:
    import torch
    from safetensors import safe_open

    schema: dict[str, dict[str, Any]] = {}
    all_finite = True
    total_parameters = 0
    with safe_open(path, framework="pt", device="cpu") as adapter:
        for key in adapter.keys():
            tensor_slice = adapter.get_slice(key)
            shape = list(tensor_slice.get_shape())
            schema[key] = {
                "shape": shape,
                "dtype": tensor_slice.get_dtype(),
            }
            parameter_count = 1
            for dimension in shape:
                parameter_count *= dimension
            total_parameters += parameter_count
            tensor = adapter.get_tensor(key)
            if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all()):
                all_finite = False
    return schema, all_finite, total_parameters


def runtime_load(
    *,
    adapter_path: Path,
    config: dict[str, Any],
    base_model_path: Path,
    device: str,
) -> dict[str, Any]:
    import torch
    from moshi.models import loaders
    from moshi.modules.lora import LoRALinear

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime validation requested but CUDA is unavailable")
    model = loaders.get_moshi_lm(
        base_model_path,
        lm_kwargs=normalise_lm_config(config),
        device=device,
        dtype=torch.bfloat16,
        lora_weights=adapter_path,
        fuse_lora=False,
    )
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    devices = sorted({parameter.device.type for parameter in model.parameters()})
    lora_layers = sum(isinstance(module, LoRALinear) for module in model.modules())
    passed = not meta_parameters and devices == [device] and lora_layers > 0
    result = {
        "performed": True,
        "device": device,
        "fuse_lora": False,
        "meta_parameters": meta_parameters,
        "parameter_devices": devices,
        "lora_layer_count": lora_layers,
        "passed": passed,
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def validate_selected_adapter(
    *,
    selection_path: Path,
    training_config_path: Path,
    base_config_path: Path,
    out_path: Path,
    runtime_device: str | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    training_config_path = training_config_path.resolve()
    base_config_path = base_config_path.resolve()
    selection = json_object(selection_path)
    selected = selection.get("selected")
    candidates = selection.get("candidates")
    if not isinstance(selected, dict) or not isinstance(candidates, list):
        raise ValueError("selection report has no selected checkpoint/candidate list")

    reevaluation_metadata = selection.get("validation_reevaluation")
    if not isinstance(reevaluation_metadata, dict):
        raise ValueError("selection report has no validation reevaluation provenance")
    reevaluation_path = project_path(
        root,
        reevaluation_metadata.get("path"),
        "validation_reevaluation.path",
    )
    reevaluation = json_object(reevaluation_path)

    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    base_config = json_object(base_config_path)
    if not isinstance(training_config, dict):
        raise ValueError("training config must be a YAML mapping")
    lora_config = training_config.get("lora")
    if not isinstance(lora_config, dict):
        raise ValueError("training config has no LoRA mapping")

    adapter_path = project_path(root, selected.get("adapter_path"), "adapter_path")
    adapter_config_path = project_path(root, selected.get("config_path"), "config_path")
    saved_config = json_object(adapter_config_path)
    expected_saved_config = dict(base_config)
    expected_saved_config.update(
        {
            "lora": lora_config.get("enable"),
            "lora_rank": lora_config.get("rank"),
            "lora_scaling": lora_config.get("scaling"),
        }
    )

    expected_schema = expected_adapter_schema(
        saved_config,
        ft_embed=lora_config.get("ft_embed") is True,
    )
    actual_schema, all_values_finite, parameter_count = read_adapter_schema(adapter_path)
    schema_comparison = compare_adapter_schema(expected_schema, actual_schema)
    max_steps = int(training_config["max_steps"])
    checkpoint_frequency = int(training_config["ckpt_freq"])
    expected_candidates = max_steps // checkpoint_frequency
    selected_step = int(selected.get("step") or 0)
    candidate_steps = {int(candidate.get("step") or 0) for candidate in candidates}
    expected_candidate_steps = set(range(checkpoint_frequency, max_steps + 1, checkpoint_frequency))
    recomputed_selection = selected_candidate_by_rule(candidates)

    requirements = {
        "selection_schema_supported": selection.get("schema_version") == 2,
        "selection_passed": selection.get("selection_passes") is True,
        "criterion_predeclared_before_training": selection.get(
            "criterion_predeclared_before_training"
        )
        is True,
        "postrun_validation_input_correction_disclosed": selection.get(
            "selection_input_corrected_after_training"
        )
        is True,
        "selection_criterion_unchanged": selection.get("correction_changes_selection_criterion")
        is False,
        "original_upstream_metrics_rejected": selection.get(
            "original_upstream_metrics_eligible_for_selection"
        )
        is False,
        "fixed_scope_reevaluation_passed": (
            reevaluation.get("schema_version") == 1
            and reevaluation.get("status") == "passed"
            and reevaluation.get("reevaluation_passes") is True
            and reevaluation.get("heldout_test_used") is False
        ),
        "reevaluation_hash_matches_selection": reevaluation_metadata.get("sha256")
        == sha256_file(reevaluation_path),
        "training_complete": selection.get("training_complete") is True,
        "heldout_not_used_for_selection": selection.get("heldout_test_used_for_selection") is False,
        "selection_profile_matches_training_config": (
            selection.get("configured_max_steps") == max_steps
            and selection.get("checkpoint_frequency") == checkpoint_frequency
        ),
        "all_predeclared_candidates_present": (
            selection.get("candidate_count") == expected_candidates
            and len(candidates) == expected_candidates
            and candidate_steps == expected_candidate_steps
        ),
        "selected_entry_is_candidate": selected in candidates,
        "selected_entry_matches_frozen_rule": selected == recomputed_selection,
        "selected_step_checkpoint_aligned": (
            selected_step > 0
            and selected_step <= max_steps
            and selected_step % checkpoint_frequency == 0
        ),
        "adapter_bytes_match_selection": selected.get("adapter_bytes")
        == adapter_path.stat().st_size,
        "adapter_hash_matches_selection": selected.get("adapter_sha256")
        == sha256_file(adapter_path),
        "config_hash_matches_selection": selected.get("config_sha256")
        == sha256_file(adapter_config_path),
        "saved_config_matches_training": saved_config == expected_saved_config,
        "lora_enabled": saved_config.get("lora") is True,
        "lora_rank_matches": saved_config.get("lora_rank") == lora_config.get("rank"),
        "lora_scaling_matches": saved_config.get("lora_scaling") == lora_config.get("scaling"),
        "all_adapter_values_finite": all_values_finite,
        "all_adapter_names_intended": all(
            "lora" in key or (lora_config.get("ft_embed") is True and "emb" in key)
            for key in actual_schema
        ),
        "adapter_schema_exact": schema_comparison["exact"] is True,
    }
    static_passed = all(requirements.values())

    runtime_result: dict[str, Any] = {
        "performed": False,
        "device": None,
        "passed": None,
    }
    if runtime_device is not None:
        moshi_paths = training_config.get("moshi_paths")
        if not isinstance(moshi_paths, dict):
            raise ValueError("training config has no moshi_paths mapping")
        base_model_path = project_path(
            root,
            moshi_paths.get("moshi_path"),
            "moshi_paths.moshi_path",
        )
        try:
            runtime_result = runtime_load(
                adapter_path=adapter_path,
                config=saved_config,
                base_model_path=base_model_path,
                device=runtime_device,
            )
        except Exception as exc:
            runtime_result = {
                "performed": True,
                "device": runtime_device,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    validation_passed = static_passed and runtime_result.get("passed") is not False
    report = {
        "schema_version": 1,
        "status": "passed" if validation_passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "path": selection_path.relative_to(root).as_posix(),
            "sha256": sha256_file(selection_path),
            "selected_step": selected_step,
            "validation_reevaluation_path": reevaluation_path.relative_to(root).as_posix(),
            "validation_reevaluation_sha256": sha256_file(reevaluation_path),
            "criterion_predeclared_before_training": selection.get(
                "criterion_predeclared_before_training"
            )
            is True,
        },
        "adapter": {
            "path": adapter_path.relative_to(root).as_posix(),
            "bytes": adapter_path.stat().st_size,
            "sha256": sha256_file(adapter_path),
            "tensor_count": len(actual_schema),
            "parameter_count": parameter_count,
            "dtype_counts": {
                dtype: sum(row["dtype"] == dtype for row in actual_schema.values())
                for dtype in sorted({row["dtype"] for row in actual_schema.values()})
            },
        },
        "config": {
            "path": adapter_config_path.relative_to(root).as_posix(),
            "sha256": sha256_file(adapter_config_path),
            "lora_rank": saved_config.get("lora_rank"),
            "lora_scaling": saved_config.get("lora_scaling"),
            "embedding_finetuning": lora_config.get("ft_embed") is True,
        },
        "schema_comparison": schema_comparison,
        "requirements": requirements,
        "static_validation_passes": static_passed,
        "runtime_load": runtime_result,
        "validation_passes": validation_passed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not validation_passed:
        raise RuntimeError("selected Moshi adapter validation failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/moshi_checkpoint_selection.json"),
    )
    parser.add_argument("--training-config", type=Path, default=Path("configs/moshi_h100.yaml"))
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/moshika_7b_legacy.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/moshi_adapter_validation.json"),
    )
    parser.add_argument("--runtime-device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    report = validate_selected_adapter(
        selection_path=args.selection,
        training_config_path=args.training_config,
        base_config_path=args.base_config,
        out_path=args.out,
        runtime_device=args.runtime_device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
