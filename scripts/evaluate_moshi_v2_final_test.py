#!/usr/bin/env python3
"""Evaluate the selected Moshi v2 adapter once on the frozen final test."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_moshi_adapter import (  # noqa: E402
    build_eval_loader,
    evaluate_model,
    expected_chunk_count,
    load_model,
    negate_lora_up_projections,
    paired_difference,
)
from scripts.reevaluate_moshi_checkpoints import cached_batch_iterator  # noqa: E402
from scripts.validate_moshi_adapter import (  # noqa: E402
    json_object,
    project_path,
    sha256_file,
)


def evaluate_final_test(
    *,
    validation_path: Path,
    training_config_path: Path,
    base_config_path: Path,
    split_report_path: Path,
    out_path: Path,
    root: Path = ROOT,
    experiment_label: str = "v2",
) -> dict[str, Any]:
    import sentencepiece
    import torch
    from moshi.models import loaders

    if experiment_label not in {"v2", "v3"}:
        raise ValueError(f"unsupported Moshi experiment label: {experiment_label}")
    expected_selection_schema = 5 if experiment_label == "v3" else 4
    adapted_mode = f"selected_{experiment_label}_adapter"
    perturbed_mode = f"{adapted_mode}_lora_B_sign_flipped"
    if not torch.cuda.is_available():
        raise RuntimeError(f"Moshi {experiment_label} final-test evaluation requires CUDA")
    paths = [
        validation_path,
        training_config_path,
        base_config_path,
        split_report_path,
        out_path,
    ]
    (
        validation_path,
        training_config_path,
        base_config_path,
        split_report_path,
        out_path,
    ) = [path.resolve() for path in paths]
    if out_path.exists():
        raise FileExistsError(
            f"refusing to repeat the one-time {experiment_label} final-test evaluation: {out_path}"
        )

    validation = json_object(validation_path)
    split_report = json_object(split_report_path)
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if not isinstance(training_config, dict):
        raise ValueError("training config must be a mapping")
    base_config = json_object(base_config_path)
    adapter = validation.get("adapter")
    adapter_config = validation.get("config")
    selection_info = validation.get("selection")
    runtime_info = validation.get("runtime_load")
    if not all(
        isinstance(value, dict) for value in (adapter, adapter_config, selection_info, runtime_info)
    ):
        raise ValueError("adapter validation report is incomplete")
    assert isinstance(adapter, dict)
    assert isinstance(adapter_config, dict)
    assert isinstance(selection_info, dict)
    assert isinstance(runtime_info, dict)

    adapter_path = project_path(root, adapter.get("path"), "validated adapter")
    adapter_config_path = project_path(
        root,
        adapter_config.get("path"),
        "validated adapter config",
    )
    selection_path = project_path(root, selection_info.get("path"), "selection report")
    selection = json_object(selection_path)
    selected_config = json_object(adapter_config_path)
    moshi_paths = training_config.get("moshi_paths")
    data_config = training_config.get("data")
    if not isinstance(moshi_paths, dict) or not isinstance(data_config, dict):
        raise ValueError("training config lacks moshi_paths or data mappings")
    base_model_path = project_path(root, moshi_paths.get("moshi_path"), "base model")
    mimi_path = project_path(root, moshi_paths.get("mimi_path"), "Mimi model")
    tokenizer_path = project_path(root, moshi_paths.get("tokenizer_path"), "text tokenizer")

    outputs = split_report.get("outputs")
    sessions = split_report.get("sessions")
    if not isinstance(outputs, dict) or not isinstance(sessions, dict):
        raise ValueError("v2 split report is incomplete")
    train_split = outputs.get("train")
    validation_split = outputs.get("validation")
    final_test_split = outputs.get("final_test")
    if not all(
        isinstance(value, dict) for value in (train_split, validation_split, final_test_split)
    ):
        raise ValueError("v2 split output evidence is incomplete")
    assert isinstance(train_split, dict)
    assert isinstance(validation_split, dict)
    assert isinstance(final_test_split, dict)
    test_manifest = project_path(
        root,
        final_test_split.get("path"),
        "v2 final-test manifest",
    )
    expected_rows, expected_chunks = expected_chunk_count(
        test_manifest,
        float(training_config["duration_sec"]),
    )
    selection_reevaluation = selection.get("validation_reevaluation")
    selection_runtime = selection.get("runtime_validation")
    if not isinstance(selection_reevaluation, dict) or not isinstance(selection_runtime, dict):
        raise ValueError("v2 selection provenance is incomplete")
    reevaluation_path = project_path(
        root,
        selection_reevaluation.get("path"),
        "v2 validation reevaluation",
    )
    runtime_validation_path = project_path(
        root,
        selection_runtime.get("path"),
        "v2 runtime-panel evaluation",
    )
    reevaluation = json_object(reevaluation_path)
    runtime_validation = json_object(runtime_validation_path)
    runtime_candidates = runtime_validation.get("candidates")
    if not isinstance(runtime_candidates, list):
        raise ValueError("v2 runtime-panel candidate evidence is incomplete")
    frozen_validation_hash = validation_split.get("sha256")
    preconditions = {
        "adapter_validation_passed": validation.get("validation_passes") is True,
        "official_cuda_loader_passed": (
            runtime_info.get("performed") is True and runtime_info.get("passed") is True
        ),
        "selection_schema_supported": (
            selection.get("schema_version") == expected_selection_schema
        ),
        "complete_prompt_protocol_disclosed": (
            (
                experiment_label == "v2"
                and selection.get("runtime_panel_corrected_after_training") is True
                and selection.get("exact_runtime_panel_indices_predeclared_before_training")
                is False
            )
            or (
                experiment_label == "v3"
                and selection.get("runtime_panel_corrected_after_training") is False
                and selection.get("exact_runtime_panel_indices_predeclared_before_training") is True
            )
        ),
        "selection_passed_before_test_access": selection.get("selection_passes") is True,
        "test_not_used_for_checkpoint_selection": selection.get("heldout_test_used_for_selection")
        is False,
        "selection_hash_current": selection_info.get("sha256") == sha256_file(selection_path),
        "adapter_hash_current": adapter.get("sha256") == sha256_file(adapter_path),
        "adapter_config_hash_current": adapter_config.get("sha256")
        == sha256_file(adapter_config_path),
        "group_split_passed": split_report.get("split_passes") is True,
        "session_groups_pairwise_disjoint": sessions.get("all_pairwise_disjoint") is True,
        "reevaluation_hash_current": selection_reevaluation.get("sha256")
        == sha256_file(reevaluation_path),
        "runtime_panel_hash_current": selection_runtime.get("sha256")
        == sha256_file(runtime_validation_path),
        "training_manifest_matches_frozen_split": (
            data_config.get("train_data") == train_split.get("path")
            and train_split.get("sha256")
            == sha256_file(project_path(root, train_split.get("path"), "v2 train manifest"))
        ),
        "validation_manifest_matches_frozen_split": (
            data_config.get("eval_data") == validation_split.get("path")
            and validation_split.get("sha256")
            == sha256_file(
                project_path(root, validation_split.get("path"), "v2 validation manifest")
            )
        ),
        "final_test_manifest_hash_current": final_test_split.get("sha256")
        == sha256_file(test_manifest),
        "final_test_row_count_current": final_test_split.get("rows") == expected_rows,
        "final_test_not_training_or_validation": final_test_split.get("path")
        not in {data_config.get("train_data"), data_config.get("eval_data")},
        "complete_final_test_nonempty": expected_rows > 0 and expected_chunks >= expected_rows,
        "selection_inputs_are_validation_only": (
            sha256_file(test_manifest) != frozen_validation_hash
            and (reevaluation.get("validation_scope") or {}).get("manifest_sha256")
            == frozen_validation_hash
            and all(
                (candidate.get("input") or {}).get("manifest_sha256") == frozen_validation_hash
                for candidate in runtime_candidates
                if isinstance(candidate, dict)
            )
        ),
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"{experiment_label} final-test preconditions failed: {preconditions}")

    duration_sec = float(training_config["duration_sec"])
    dep_q = int(selected_config["dep_q"])
    mimi = loaders.get_mimi(mimi_path, num_codebooks=dep_q, device="cuda")
    mimi.eval()
    tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
    adapted_model = load_model(
        base_model_path=base_model_path,
        raw_config=selected_config,
        adapter_path=adapter_path,
    )
    cached_batches: list[tuple[Any, Any]] = []
    cache_started = time.monotonic()
    for index, batch in enumerate(
        build_eval_loader(
            mimi=mimi,
            tokenizer=tokenizer,
            model=adapted_model,
            test_manifest=test_manifest,
            duration_sec=duration_sec,
        ),
        start=1,
    ):
        cached_batches.append((batch.codes.to(device="cpu"), batch.condition_attributes))
        if index % 100 == 0:
            print(f"cached final-test chunks: {index}/{expected_chunks}", flush=True)
    cache_seconds = time.monotonic() - cache_started
    if len(cached_batches) != expected_chunks:
        raise RuntimeError(
            f"final-test loader yielded {len(cached_batches)} chunks; expected {expected_chunks}"
        )
    del mimi, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    first_codebook_weight_multiplier = float(training_config["first_codebook_weight_multiplier"])
    text_padding_weight = float(training_config["text_padding_weight"])
    adapted_report, adapted_series = evaluate_model(
        model=adapted_model,
        loader=cached_batch_iterator(cached_batches),
        mode=adapted_mode,
        measure_target_sensitivity=True,
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        text_padding_weight=text_padding_weight,
    )
    perturbed_parameters = negate_lora_up_projections(adapted_model)
    perturbed_report, perturbed_series = evaluate_model(
        model=adapted_model,
        loader=cached_batch_iterator(cached_batches),
        mode=perturbed_mode,
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        text_padding_weight=text_padding_weight,
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
        loader=cached_batch_iterator(cached_batches),
        mode="pinned_unadapted_base",
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        text_padding_weight=text_padding_weight,
    )
    del base_model, cached_batches
    gc.collect()
    torch.cuda.empty_cache()

    comparisons = {
        "base_minus_adapted": {
            loss_name: paired_difference(
                base_series[loss_name],
                adapted_series[loss_name],
                definition=(
                    f"positive means the selected {experiment_label} adapter has lower loss"
                ),
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
    target_sensitivity = adapted_report.get("target_sensitivity") or {}
    sample_counts = {
        adapted_report["samples"],
        perturbed_report["samples"],
        base_report["samples"],
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
        "text_target_sensitivity_proven": target_sensitivity.get("text_loss_delta") != 0.0,
        "audio_target_sensitivity_proven": target_sensitivity.get("audio_loss_delta") != 0.0,
        "adapter_changes_total_loss": comparisons["base_minus_adapted"]["total_loss"]["nonzero"]
        is True,
        "lora_perturbation_changes_total_loss": comparisons["perturbed_minus_adapted"][
            "total_loss"
        ]["nonzero"]
        is True,
    }
    passed = all(requirements.values())
    total_improvement = comparisons["base_minus_adapted"]["total_loss"]
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_evidence": True,
        "automatic_model_loss_evidence_only": True,
        "human_perceptual_claim_allowed": False,
        "heldout_used_for_checkpoint_selection": False,
        "one_time_final_test": True,
        "test_protocol": {
            "split": f"{experiment_label}_final_test",
            "manifest": test_manifest.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(test_manifest),
            "rows": expected_rows,
            "chunks": expected_chunks,
            "duration_sec": duration_sec,
            "batch_size": 1,
            "order": [
                adapted_mode,
                perturbed_mode,
                "pinned_unadapted_base",
            ],
            "selection_frozen_before_access": True,
        },
        "provenance": {
            "adapter_validation_sha256": sha256_file(validation_path),
            "selection_sha256": sha256_file(selection_path),
            "split_report_sha256": sha256_file(split_report_path),
            "selected_adapter_sha256": sha256_file(adapter_path),
            "selected_config_sha256": sha256_file(adapter_config_path),
            "base_model_sha256": sha256_file(base_model_path),
            "mimi_sha256": sha256_file(mimi_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
        },
        "modes": {
            adapted_mode: adapted_report,
            perturbed_mode: {
                **perturbed_report,
                "parameters_changed": perturbed_parameters,
            },
            "pinned_unadapted_base": base_report,
        },
        "paired_comparisons": comparisons,
        "requirements": requirements,
        "evaluation_passes": passed,
        "scientific_outcome": {
            "adapted_total_loss_lower_than_base": total_improvement["mean"] > 0.0,
            "adapted_total_loss_improvement_ci95_above_zero": total_improvement["ci95_low"] > 0.0,
        },
        "timing": {"final_test_token_cache_seconds": cache_seconds},
        "interpretation": (
            "Positive base-minus-adapted loss is favorable. This one-time objective test does "
            "not establish Persian naturalness, pronunciation, relevance, or human preference."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"Moshi {experiment_label} final-test integrity evaluation failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("results/moshi_v2_adapter_validation.json"),
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/moshi_h100_v2.yaml"),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/moshika_7b_legacy.json"),
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        default=Path("results/moshi_v2_split.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/moshi_v2_final_test.json"),
    )
    parser.add_argument(
        "--experiment-label",
        choices=("v2", "v3"),
        default="v2",
    )
    args = parser.parse_args()
    report = evaluate_final_test(
        validation_path=ROOT / args.validation,
        training_config_path=ROOT / args.training_config,
        base_config_path=ROOT / args.base_config,
        split_report_path=ROOT / args.split_report,
        out_path=ROOT / args.out,
        experiment_label=args.experiment_label,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
