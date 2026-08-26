#!/usr/bin/env python3
"""One-time group-disjoint Moshi adapter-on/off held-out loss evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_moshi_adapter import (  # noqa: E402
    json_object,
    normalise_lm_config,
    project_path,
    sha256_file,
)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("loss series must be nonempty and finite")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(len(values))
    return {
        "samples": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
        "minimum": min(values),
        "maximum": max(values),
    }


def paired_difference(
    left: list[float],
    right: list[float],
    *,
    definition: str,
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("paired loss series must have the same nonzero length")
    differences = [
        left_value - right_value for left_value, right_value in zip(left, right, strict=True)
    ]
    return {
        "definition": definition,
        **summarize(differences),
        "nonzero": any(difference != 0.0 for difference in differences),
    }


def expected_chunk_count(manifest_path: Path, duration_sec: float) -> tuple[int, int]:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("held-out manifest must contain JSON objects")
    chunks = sum(math.ceil(float(row["duration"]) / duration_sec) for row in rows)
    return len(rows), chunks


def build_eval_loader(
    *,
    mimi,
    tokenizer,
    model,
    test_manifest: Path,
    duration_sec: float,
) -> Iterator[Any]:
    from finetune.data.args import DataArgs
    from finetune.data.data_loader import build_data_loader
    from finetune.data.interleaver import InterleavedTokenizer, Interleaver

    interleaver = Interleaver(
        tokenizer,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi,
        interleaver,
        duration_sec=duration_sec,
    )
    data = DataArgs(
        train_data=str(test_manifest),
        eval_data=str(test_manifest),
        shuffle=False,
    )
    return build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=data,
        batch_size=1,
        seed=None,
        rank=0,
        world_size=1,
        is_eval=True,
    )


def evaluate_model(
    *,
    model,
    loader: Iterator[Any],
    first_codebook_weight_multiplier: float,
    text_padding_weight: float,
    mode: str,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    import torch
    from finetune.loss import compute_loss_with_mask

    text_losses: list[float] = []
    audio_losses: list[float] = []
    total_losses: list[float] = []
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    for batch in loader:
        with torch.no_grad():
            codes = batch.codes
            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(batch.condition_attributes)
            output = model(codes=codes, condition_tensors=condition_tensors)
            text_loss = compute_loss_with_mask(
                output.text_logits,
                codes[:, : model.audio_offset],
                output.text_mask,
                mode="text",
                text_padding_weight=text_padding_weight,
                text_padding_ids={
                    model.text_padding_token_id,
                    model.end_of_text_padding_id,
                },
            )
            audio_loss = compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=first_codebook_weight_multiplier,
            )
        text_value = float(text_loss.item())
        audio_value = float(audio_loss.item())
        text_losses.append(text_value)
        audio_losses.append(audio_value)
        total_losses.append(text_value + audio_value)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    series = {
        "text_loss": text_losses,
        "audio_loss": audio_losses,
        "total_loss": total_losses,
    }
    report = {
        "mode": mode,
        "samples": len(total_losses),
        "text_loss": summarize(text_losses),
        "audio_loss": summarize(audio_losses),
        "total_loss": summarize(total_losses),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(total_losses) / elapsed,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    return report, series


def negate_lora_up_projections(model) -> int:
    import torch

    changed = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_B.weight"):
                parameter.neg_()
                changed += 1
    if changed == 0:
        raise RuntimeError("selected model contains no LoRA up-projection parameters")
    return changed


def load_model(
    *,
    base_model_path: Path,
    raw_config: dict[str, Any],
    adapter_path: Path | None,
):
    import torch
    from moshi.models import loaders

    return loaders.get_moshi_lm(
        base_model_path,
        lm_kwargs=normalise_lm_config(raw_config),
        device="cuda",
        dtype=torch.bfloat16,
        lora_weights=adapter_path,
        fuse_lora=False,
    )


def evaluate_selected_adapter(
    *,
    validation_path: Path,
    training_config_path: Path,
    base_config_path: Path,
    export_report_path: Path,
    export_audit_path: Path,
    out_path: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    import sentencepiece
    import torch
    from moshi.models import loaders

    if not torch.cuda.is_available():
        raise RuntimeError("held-out Moshi evaluation requires CUDA")
    validation_path = validation_path.resolve()
    training_config_path = training_config_path.resolve()
    base_config_path = base_config_path.resolve()
    export_report_path = export_report_path.resolve()
    export_audit_path = export_audit_path.resolve()
    validation = json_object(validation_path)
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if not isinstance(training_config, dict):
        raise ValueError("training config must be a YAML mapping")
    base_config = json_object(base_config_path)
    export_report = json_object(export_report_path)
    export_audit = json_object(export_audit_path)

    adapter = validation.get("adapter")
    adapter_config = validation.get("config")
    selection_info = validation.get("selection")
    runtime_info = validation.get("runtime_load")
    if not all(
        isinstance(value, dict) for value in (adapter, adapter_config, selection_info, runtime_info)
    ):
        raise ValueError("adapter validation report is incomplete")
    adapter_path = project_path(root, adapter.get("path"), "validated adapter")
    adapter_config_path = project_path(
        root,
        adapter_config.get("path"),
        "validated adapter config",
    )
    selection_path = project_path(root, selection_info.get("path"), "selection report")
    selected_config = json_object(adapter_config_path)

    moshi_paths = training_config.get("moshi_paths")
    if not isinstance(moshi_paths, dict):
        raise ValueError("training config has no moshi_paths mapping")
    base_model_path = project_path(root, moshi_paths.get("moshi_path"), "base model")
    mimi_path = project_path(root, moshi_paths.get("mimi_path"), "Mimi model")
    tokenizer_path = project_path(root, moshi_paths.get("tokenizer_path"), "text tokenizer")
    data_config = training_config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("training config has no data mapping")
    test_manifest = project_path(
        root,
        "data/processed/moshi_finetune/test.jsonl",
        "held-out manifest",
    )

    duration_sec = float(training_config["duration_sec"])
    expected_rows, expected_chunks = expected_chunk_count(test_manifest, duration_sec)
    export_hashes = export_report.get("manifest_sha256")
    audit_requirements = export_audit.get("requirements")
    split_counts = export_audit.get("split_counts")
    if not all(
        isinstance(value, dict) for value in (export_hashes, audit_requirements, split_counts)
    ):
        raise ValueError("Moshi export/audit split evidence is incomplete")

    preconditions = {
        "adapter_validation_passed": validation.get("validation_passes") is True,
        "official_runtime_load_passed": (
            runtime_info.get("performed") is True and runtime_info.get("passed") is True
        ),
        "selection_hash_current": selection_info.get("sha256") == sha256_file(selection_path),
        "adapter_hash_current": adapter.get("sha256") == sha256_file(adapter_path),
        "adapter_config_hash_current": adapter_config.get("sha256")
        == sha256_file(adapter_config_path),
        "export_audit_passed": export_audit.get("audit_passes") is True,
        "test_split_group_isolated": (
            audit_requirements.get("session_group_splits_isolated") is True
            and export_audit.get("group_split_leaks") == 0
        ),
        "test_manifest_hash_current": export_hashes.get("test") == sha256_file(test_manifest),
        "test_row_count_matches_audit": split_counts.get("test") == expected_rows,
        "full_heldout_scope_nonempty": expected_rows > 0 and expected_chunks >= expected_rows,
        "test_manifest_not_training_or_validation": (
            data_config.get("train_data") != str(test_manifest.relative_to(root))
            and data_config.get("eval_data") != str(test_manifest.relative_to(root))
        ),
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"held-out evaluation preconditions failed: {preconditions}")

    dep_q = int(selected_config["dep_q"])
    mimi = loaders.get_mimi(mimi_path, num_codebooks=dep_q, device="cuda")
    mimi.eval()
    tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
    first_codebook_weight_multiplier = float(training_config["first_codebook_weight_multiplier"])
    text_padding_weight = float(training_config["text_padding_weight"])

    adapted_model = load_model(
        base_model_path=base_model_path,
        raw_config=selected_config,
        adapter_path=adapter_path,
    )
    adapted_report, adapted_series = evaluate_model(
        model=adapted_model,
        loader=build_eval_loader(
            mimi=mimi,
            tokenizer=tokenizer,
            model=adapted_model,
            test_manifest=test_manifest,
            duration_sec=duration_sec,
        ),
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        text_padding_weight=text_padding_weight,
        mode="selected_adapter",
    )
    perturbed_parameters = negate_lora_up_projections(adapted_model)
    perturbed_report, perturbed_series = evaluate_model(
        model=adapted_model,
        loader=build_eval_loader(
            mimi=mimi,
            tokenizer=tokenizer,
            model=adapted_model,
            test_manifest=test_manifest,
            duration_sec=duration_sec,
        ),
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        text_padding_weight=text_padding_weight,
        mode="selected_adapter_lora_B_sign_flipped",
    )
    del adapted_model
    gc.collect()
    torch.cuda.empty_cache()

    base_model = load_model(
        base_model_path=base_model_path,
        raw_config=base_config,
        adapter_path=None,
    )
    base_report, base_series = evaluate_model(
        model=base_model,
        loader=build_eval_loader(
            mimi=mimi,
            tokenizer=tokenizer,
            model=base_model,
            test_manifest=test_manifest,
            duration_sec=duration_sec,
        ),
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        text_padding_weight=text_padding_weight,
        mode="pinned_unadapted_base",
    )
    del base_model, mimi
    gc.collect()
    torch.cuda.empty_cache()

    sample_counts = {
        adapted_report["samples"],
        perturbed_report["samples"],
        base_report["samples"],
    }
    comparisons = {
        "base_minus_adapted": {
            loss_name: paired_difference(
                base_series[loss_name],
                adapted_series[loss_name],
                definition="positive means the selected adapter has lower loss",
            )
            for loss_name in ("text_loss", "audio_loss", "total_loss")
        },
        "perturbed_minus_adapted": {
            loss_name: paired_difference(
                perturbed_series[loss_name],
                adapted_series[loss_name],
                definition="nonzero proves the loaded LoRA path affects this loss",
            )
            for loss_name in ("text_loss", "audio_loss", "total_loss")
        },
    }
    requirements = {
        **preconditions,
        "all_modes_cover_full_identical_scope": sample_counts == {expected_chunks},
        "all_mode_losses_finite": all(
            math.isfinite(float(mode_report[loss_name]["mean"]))
            for mode_report in (adapted_report, perturbed_report, base_report)
            for loss_name in ("text_loss", "audio_loss", "total_loss")
        ),
        "lora_perturbation_applied": perturbed_parameters > 0,
        "adapter_changes_total_loss": comparisons["base_minus_adapted"]["total_loss"]["nonzero"]
        is True,
        "lora_perturbation_changes_total_loss": comparisons["perturbed_minus_adapted"][
            "total_loss"
        ]["nonzero"]
        is True,
    }
    passed = all(requirements.values())
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_evidence": True,
        "automatic_model_loss_evidence_only": True,
        "human_perceptual_claim_allowed": False,
        "heldout_used_for_checkpoint_selection": False,
        "test_protocol": {
            "split": "test",
            "manifest": test_manifest.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(test_manifest),
            "rows": expected_rows,
            "chunks": expected_chunks,
            "duration_sec": duration_sec,
            "batch_size": 1,
            "order": [
                "selected_adapter",
                "selected_adapter_lora_B_sign_flipped",
                "pinned_unadapted_base",
            ],
            "perturbation": "multiply every loaded lora_B.weight by -1 in memory",
        },
        "provenance": {
            "adapter_validation_path": validation_path.relative_to(root).as_posix(),
            "adapter_validation_sha256": sha256_file(validation_path),
            "selection_path": selection_path.relative_to(root).as_posix(),
            "selection_sha256": sha256_file(selection_path),
            "selected_adapter_sha256": sha256_file(adapter_path),
            "selected_config_sha256": sha256_file(adapter_config_path),
            "base_model_sha256": sha256_file(base_model_path),
            "mimi_sha256": sha256_file(mimi_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "export_report_sha256": sha256_file(export_report_path),
            "export_audit_sha256": sha256_file(export_audit_path),
        },
        "modes": {
            "selected_adapter": adapted_report,
            "selected_adapter_lora_B_sign_flipped": {
                **perturbed_report,
                "parameters_changed": perturbed_parameters,
            },
            "pinned_unadapted_base": base_report,
        },
        "paired_comparisons": comparisons,
        "requirements": requirements,
        "evaluation_passes": passed,
        "interpretation": (
            "Positive base-minus-adapted loss is favorable. This automatic objective "
            "evaluation does not establish Persian naturalness, pronunciation, response "
            "relevance, conversational success, or human preference."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("held-out Moshi adapter evaluation failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("results/moshi_adapter_validation.json"),
    )
    parser.add_argument("--training-config", type=Path, default=Path("configs/moshi_h100.yaml"))
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/moshika_7b_legacy.json"),
    )
    parser.add_argument(
        "--export-report",
        type=Path,
        default=Path("results/moshi_export_report.json"),
    )
    parser.add_argument(
        "--export-audit",
        type=Path,
        default=Path("results/moshi_export_audit.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/moshi_heldout_model_eval.json"),
    )
    args = parser.parse_args()
    report = evaluate_selected_adapter(
        validation_path=args.validation,
        training_config_path=args.training_config,
        base_config_path=args.base_config,
        export_report_path=args.export_report,
        export_audit_path=args.export_audit,
        out_path=args.out,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
