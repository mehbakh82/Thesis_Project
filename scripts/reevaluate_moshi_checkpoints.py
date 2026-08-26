#!/usr/bin/env python3
"""Reevaluate every saved Moshi adapter on one identical validation scope."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections.abc import Iterator
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
)
from scripts.validate_moshi_adapter import json_object, sha256_file  # noqa: E402

UPSTREAM_EVAL_LIMIT = 40


def simulate_upstream_eval_consumption(
    total_chunks: int,
    evaluation_steps: list[int],
    *,
    limit: int = UPSTREAM_EVAL_LIMIT,
) -> list[dict[str, int]]:
    """Model the pinned evaluator's persistent-iterator and off-by-one behavior."""

    if total_chunks < 1 or limit < 1 or not evaluation_steps:
        raise ValueError("positive chunks, limit, and evaluation steps are required")
    remaining = total_chunks
    rows: list[dict[str, int]] = []
    for step in evaluation_steps:
        processed = min(limit, remaining)
        discarded = int(remaining > limit)
        consumed = processed + discarded
        rows.append(
            {
                "step": step,
                "processed": processed,
                "discarded_by_off_by_one_break": discarded,
                "consumed": consumed,
                "remaining_after": remaining - consumed,
            }
        )
        remaining -= consumed
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def copy_adapter_parameters(model, adapter_path: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    parameters = dict(model.named_parameters())
    keys: list[str] = []
    nonfinite: list[str] = []
    shape_mismatches: list[str] = []
    with torch.no_grad(), safe_open(adapter_path, framework="pt", device="cpu") as adapter:
        keys = list(adapter.keys())
        unexpected = sorted(set(keys) - set(parameters))
        if unexpected:
            raise RuntimeError(f"adapter has parameters absent from model: {unexpected[:5]}")
        for key in keys:
            value = adapter.get_tensor(key)
            parameter = parameters[key]
            if value.shape != parameter.shape:
                shape_mismatches.append(key)
                continue
            if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
                nonfinite.append(key)
                continue
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    if shape_mismatches or nonfinite:
        raise RuntimeError(
            f"adapter copy failed: shape_mismatches={shape_mismatches[:5]}, "
            f"nonfinite={nonfinite[:5]}"
        )
    return {
        "tensor_count": len(keys),
        "all_values_finite": not nonfinite,
        "shape_mismatches": shape_mismatches,
    }


def cached_batch_iterator(cached_batches: list[tuple[Any, Any]]) -> Iterator[Any]:
    from finetune.data.interleaver import Batch

    for codes, condition_attributes in cached_batches:
        yield Batch(
            codes=codes.to(device="cuda", non_blocking=True),
            condition_attributes=condition_attributes,
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def reevaluate_checkpoints(
    *,
    training_config_path: Path,
    run_dir: Path,
    metrics_out: Path,
    report_out: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    import sentencepiece
    import torch
    from moshi.models import loaders

    training_config_path = training_config_path.resolve()
    run_dir = run_dir.resolve()
    metrics_out = metrics_out.resolve()
    report_out = report_out.resolve()
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if not isinstance(training_config, dict):
        raise ValueError("training config must be a mapping")
    data_config = training_config.get("data")
    moshi_paths = training_config.get("moshi_paths")
    if not isinstance(data_config, dict) or not isinstance(moshi_paths, dict):
        raise ValueError("training config must contain data and moshi_paths mappings")

    max_steps = int(training_config["max_steps"])
    checkpoint_frequency = int(training_config["ckpt_freq"])
    expected_steps = list(range(checkpoint_frequency, max_steps + 1, checkpoint_frequency))
    duration_sec = float(training_config["duration_sec"])
    validation_manifest = (root / str(data_config["eval_data"])).resolve()
    heldout_test_manifest = (root / "data/processed/moshi_finetune/test.jsonl").resolve()
    base_model_path = (root / str(moshi_paths["moshi_path"])).resolve()
    mimi_path = (root / str(moshi_paths["mimi_path"])).resolve()
    tokenizer_path = (root / str(moshi_paths["tokenizer_path"])).resolve()
    raw_train_metrics_path = run_dir / "metrics.train.jsonl"
    raw_eval_metrics_path = run_dir / "metrics.eval.jsonl"

    train_rows = load_jsonl(raw_train_metrics_path)
    raw_eval_rows = load_jsonl(raw_eval_metrics_path)
    train_losses = [float(row["loss"]) for row in train_rows]
    raw_eval_steps = [int(row["step"]) for row in raw_eval_rows]
    raw_eval_losses = [float(row["eval_loss"]) for row in raw_eval_rows]
    expected_rows, expected_chunks = expected_chunk_count(validation_manifest, duration_sec)
    consumption = simulate_upstream_eval_consumption(expected_chunks, raw_eval_steps)
    predicted_empty_steps = [row["step"] for row in consumption if row["processed"] == 0]
    observed_nonfinite_steps = [
        step
        for step, loss in zip(raw_eval_steps, raw_eval_losses, strict=True)
        if not math.isfinite(loss)
    ]
    checkpoint_paths = {
        step: run_dir
        / "checkpoints"
        / f"checkpoint_{step:06d}"
        / "consolidated"
        / "lora.safetensors"
        for step in expected_steps
    }
    config_paths = {step: path.with_name("config.json") for step, path in checkpoint_paths.items()}

    preconditions = {
        "training_reached_configured_final_step": max(int(row["step"]) for row in train_rows)
        == max_steps,
        "all_logged_training_losses_finite": all(math.isfinite(loss) for loss in train_losses),
        "raw_evaluation_reached_configured_final_step": max(raw_eval_steps) == max_steps,
        "all_expected_checkpoints_present": all(
            checkpoint_paths[step].is_file() and config_paths[step].is_file()
            for step in expected_steps
        ),
        "raw_nonfinite_validation_observed": bool(observed_nonfinite_steps),
        "upstream_exhaustion_predicts_nonfinite_onset": (
            bool(predicted_empty_steps)
            and bool(observed_nonfinite_steps)
            and predicted_empty_steps[0] == observed_nonfinite_steps[0]
        ),
        "heldout_test_not_used": (
            validation_manifest != heldout_test_manifest
            and sha256_file(validation_manifest) != sha256_file(heldout_test_manifest)
        ),
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"checkpoint reevaluation preconditions failed: {preconditions}")

    first_config = json_object(config_paths[expected_steps[0]])
    model = load_model(
        base_model_path=base_model_path,
        raw_config=first_config,
        adapter_path=checkpoint_paths[expected_steps[0]],
    )
    dep_q = int(first_config["dep_q"])
    mimi = loaders.get_mimi(mimi_path, num_codebooks=dep_q, device="cuda")
    mimi.eval()
    tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

    cached_batches: list[tuple[Any, Any]] = []
    cache_started = time.monotonic()
    for index, batch in enumerate(
        build_eval_loader(
            mimi=mimi,
            tokenizer=tokenizer,
            model=model,
            test_manifest=validation_manifest,
            duration_sec=duration_sec,
        ),
        start=1,
    ):
        cached_batches.append((batch.codes.to(device="cpu"), batch.condition_attributes))
        if index % 100 == 0:
            print(f"cached validation chunks: {index}/{expected_chunks}", flush=True)
    cache_elapsed = time.monotonic() - cache_started
    if len(cached_batches) != expected_chunks:
        raise RuntimeError(
            f"validation loader yielded {len(cached_batches)} chunks; expected {expected_chunks}"
        )
    del mimi, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    first_codebook_weight_multiplier = float(training_config["first_codebook_weight_multiplier"])
    text_padding_weight = float(training_config["text_padding_weight"])
    candidates: list[dict[str, Any]] = []
    expected_adapter_keys: set[str] | None = None
    expected_config_hash: str | None = None
    reevaluation_started = time.monotonic()
    for candidate_index, step in enumerate(expected_steps, start=1):
        adapter_path = checkpoint_paths[step]
        adapter_config_path = config_paths[step]
        adapter_config_hash = sha256_file(adapter_config_path)
        if expected_config_hash is None:
            expected_config_hash = adapter_config_hash
        elif adapter_config_hash != expected_config_hash:
            raise RuntimeError("saved adapter configs differ across checkpoint candidates")
        copy_result = copy_adapter_parameters(model, adapter_path)
        from safetensors import safe_open

        with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
            adapter_keys = set(adapter.keys())
        if expected_adapter_keys is None:
            expected_adapter_keys = adapter_keys
        elif adapter_keys != expected_adapter_keys:
            raise RuntimeError("adapter tensor keys differ across checkpoint candidates")

        mode_report, _ = evaluate_model(
            model=model,
            loader=cached_batch_iterator(cached_batches),
            first_codebook_weight_multiplier=first_codebook_weight_multiplier,
            text_padding_weight=text_padding_weight,
            mode=f"checkpoint_{step:06d}_fixed_full_validation",
        )
        candidate = {
            "schema_version": 1,
            "step": step,
            "eval_loss": mode_report["total_loss"]["mean"],
            "text_eval_loss": mode_report["text_loss"]["mean"],
            "audio_eval_loss": mode_report["audio_loss"]["mean"],
            "sample_count": mode_report["samples"],
            "validation_scope": "complete_fixed_manifest",
            "validation_manifest_sha256": sha256_file(validation_manifest),
            "heldout_test_used": False,
            "adapter_sha256": sha256_file(adapter_path),
            "config_sha256": adapter_config_hash,
            "adapter_tensor_count": copy_result["tensor_count"],
            "elapsed_seconds": mode_report["elapsed_seconds"],
            "peak_allocated_gb": mode_report["peak_allocated_gb"],
        }
        candidates.append(candidate)
        write_jsonl_atomic(metrics_out, candidates)
        print(
            f"reevaluated checkpoint {candidate_index}/{len(expected_steps)}: "
            f"step={step} eval_loss={candidate['eval_loss']:.9f}",
            flush=True,
        )

    reevaluation_elapsed = time.monotonic() - reevaluation_started
    reevaluation_requirements = {
        **preconditions,
        "all_candidates_reevaluated": len(candidates) == len(expected_steps),
        "candidate_steps_exact": [row["step"] for row in candidates] == expected_steps,
        "identical_complete_scope": all(
            row["sample_count"] == expected_chunks
            and row["validation_scope"] == "complete_fixed_manifest"
            and row["validation_manifest_sha256"] == sha256_file(validation_manifest)
            for row in candidates
        ),
        "all_corrected_losses_finite": all(
            math.isfinite(float(row[key]))
            for row in candidates
            for key in ("eval_loss", "text_eval_loss", "audio_eval_loss")
        ),
        "all_adapter_schemas_identical": expected_adapter_keys is not None,
        "all_saved_configs_identical": expected_config_hash is not None,
    }
    passed = all(reevaluation_requirements.values())
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_validation_evidence": True,
        "heldout_test_used": False,
        "checkpoint_selection_performed": False,
        "selection_criterion_changed": False,
        "selection_input_corrected_after_training": True,
        "correction_reason": (
            "The pinned upstream evaluator reused one finite iterator and fetched one extra "
            "batch at its 40-sample break. It therefore evaluated different subsets, exhausted "
            "all 682 validation chunks after step 4250, and divided by zero from step 4500."
        ),
        "original_metrics_eligible_for_selection": False,
        "corrected_protocol": (
            "Evaluate every predeclared 500-step saved checkpoint on the same complete, "
            "ordered validation manifest; choose later using the already-frozen minimum mean "
            "validation-loss rule."
        ),
        "validation_scope": {
            "manifest": validation_manifest.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(validation_manifest),
            "rows": expected_rows,
            "chunks": expected_chunks,
            "duration_sec": duration_sec,
            "batch_size": 1,
            "order": "manifest order, identical cached Mimi/text tokens for every checkpoint",
        },
        "upstream_failure_diagnosis": {
            "evaluation_limit": UPSTREAM_EVAL_LIMIT,
            "observed_first_nonfinite_step": observed_nonfinite_steps[0],
            "predicted_first_empty_step": predicted_empty_steps[0],
            "nonfinite_steps": observed_nonfinite_steps,
            "consumption_by_evaluation": consumption,
        },
        "configured_max_steps": max_steps,
        "checkpoint_frequency": checkpoint_frequency,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "timing": {
            "validation_token_cache_seconds": cache_elapsed,
            "checkpoint_reevaluation_seconds": reevaluation_elapsed,
        },
        "requirements": reevaluation_requirements,
        "reevaluation_passes": passed,
        "artifacts": {
            "training_config": training_config_path.relative_to(root).as_posix(),
            "training_config_sha256": sha256_file(training_config_path),
            "resolved_training_args": (run_dir / "args.yaml").relative_to(root).as_posix(),
            "resolved_training_args_sha256": sha256_file(run_dir / "args.yaml"),
            "raw_train_metrics_sha256": sha256_file(raw_train_metrics_path),
            "raw_eval_metrics_sha256": sha256_file(raw_eval_metrics_path),
            "corrected_metrics": metrics_out.relative_to(root).as_posix(),
            "corrected_metrics_sha256": sha256_file(metrics_out),
            "upstream_eval_sha256": sha256_file(
                root / "third_party/checkouts/moshi-finetune/finetune/eval.py"
            ),
            "upstream_dataset_sha256": sha256_file(
                root / "third_party/checkouts/moshi-finetune/finetune/data/dataset.py"
            ),
            "reevaluator_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("fixed-scope checkpoint reevaluation failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", type=Path, default=Path("configs/moshi_h100.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/moshi_fa"))
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("checkpoints/moshi_fa/metrics.reeval.jsonl"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("results/moshi_validation_reevaluation.json"),
    )
    args = parser.parse_args()
    report = reevaluate_checkpoints(
        training_config_path=args.training_config,
        run_dir=args.run_dir,
        metrics_out=args.metrics_out,
        report_out=args.report_out,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
