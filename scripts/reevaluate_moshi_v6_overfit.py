#!/usr/bin/env python3
"""Reevaluate v6 checkpoints on the identical 32-row in-sample diagnostic scope."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_moshi_adapter import (  # noqa: E402
    build_eval_loader,
    evaluate_model,
    expected_chunk_count,
    load_model,
)
from scripts.reevaluate_moshi_checkpoints import (  # noqa: E402
    cached_batch_iterator,
    copy_adapter_parameters,
    write_jsonl_atomic,
)
from scripts.validate_moshi_adapter import json_object, sha256_file  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/moshi_h100_v6_overfit.yaml"),
    )
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/moshi_v6_overfit"))
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("checkpoints/moshi_v6_overfit/metrics.reeval.jsonl"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("results/moshi_v6_overfit_reevaluation.json"),
    )
    parser.add_argument("--experiment-label", default="v6_overfit")
    args = parser.parse_args()

    training_config_path = (ROOT / args.training_config).resolve()
    run_dir = (ROOT / args.run_dir).resolve()
    metrics_out = (ROOT / args.metrics_out).resolve()
    report_out = (ROOT / args.report_out).resolve()
    if report_out.exists() or metrics_out.exists():
        raise FileExistsError("refusing to overwrite frozen v6 reevaluation outputs")
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if not isinstance(training_config, dict):
        raise ValueError("training config must be a mapping")
    data_config = training_config.get("data") or {}
    moshi_paths = training_config.get("moshi_paths") or {}
    manifest = (ROOT / str(data_config.get("eval_data"))).resolve()
    if data_config.get("eval_data") != data_config.get("train_data"):
        raise ValueError("v6 diagnostic requires identical train and evaluation manifests")
    if "test" in manifest.as_posix().lower():
        raise ValueError("v6 diagnostic manifest name must not contain test")
    duration_sec = float(training_config["duration_sec"])
    expected_rows, expected_chunks = expected_chunk_count(manifest, duration_sec)
    if expected_rows != expected_chunks or expected_rows != 32:
        raise RuntimeError(
            f"expected exactly 32 single-chunk in-sample rows, got rows={expected_rows}, "
            f"chunks={expected_chunks}"
        )
    expected_steps = [50, 100, 150, 200]
    checkpoint_paths = {
        step: run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated/lora.safetensors"
        for step in expected_steps
    }
    config_paths = {step: path.with_name("config.json") for step, path in checkpoint_paths.items()}
    raw_train_metrics_path = run_dir / "metrics.train.jsonl"
    raw_eval_metrics_path = run_dir / "metrics.eval.jsonl"
    train_metrics = read_jsonl(raw_train_metrics_path)
    raw_eval_metrics = read_jsonl(raw_eval_metrics_path)
    preconditions = {
        "training_reached_step_200": max(int(row["step"]) for row in train_metrics) == 200,
        "all_training_losses_finite": all(
            math.isfinite(float(row["loss"])) for row in train_metrics
        ),
        "raw_eval_steps_exact": [int(row["step"]) for row in raw_eval_metrics] == expected_steps,
        "raw_eval_losses_finite": all(
            math.isfinite(float(row["eval_loss"])) for row in raw_eval_metrics
        ),
        "all_expected_checkpoints_present": all(
            checkpoint_paths[step].is_file() and config_paths[step].is_file()
            for step in expected_steps
        ),
        "scope_is_train_only_in_sample": (
            expected_rows == 32 and data_config.get("train_data") == data_config.get("eval_data")
        ),
        "final_test_not_opened_or_used": True,
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"v6 reevaluation preconditions failed: {preconditions}")

    import sentencepiece
    import torch
    from moshi.models import loaders

    first_config = json_object(config_paths[50])
    base_model_path = (ROOT / str(moshi_paths["moshi_path"])).resolve()
    mimi_path = (ROOT / str(moshi_paths["mimi_path"])).resolve()
    tokenizer_path = (ROOT / str(moshi_paths["tokenizer_path"])).resolve()
    model = load_model(
        base_model_path=base_model_path,
        raw_config=first_config,
        adapter_path=checkpoint_paths[50],
    )
    dep_q = int(first_config["dep_q"])
    mimi = loaders.get_mimi(mimi_path, num_codebooks=dep_q, device="cuda")
    mimi.eval()
    tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
    cached_batches = []
    for batch in build_eval_loader(
        mimi=mimi,
        tokenizer=tokenizer,
        model=model,
        test_manifest=manifest,
        duration_sec=duration_sec,
    ):
        cached_batches.append((batch.codes.to(device="cpu"), batch.condition_attributes))
    if len(cached_batches) != expected_chunks:
        raise RuntimeError(
            f"v6 cache has {len(cached_batches)} batches; expected {expected_chunks}"
        )
    del mimi, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    candidates = []
    expected_keys = None
    expected_config_hash = None
    for step in expected_steps:
        config_hash = sha256_file(config_paths[step])
        if expected_config_hash is None:
            expected_config_hash = config_hash
        elif config_hash != expected_config_hash:
            raise RuntimeError("v6 saved configs differ across candidates")
        copy_result = copy_adapter_parameters(model, checkpoint_paths[step])
        from safetensors import safe_open

        with safe_open(checkpoint_paths[step], framework="pt", device="cpu") as adapter:
            keys = set(adapter.keys())
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise RuntimeError("v6 adapter schemas differ across candidates")
        mode_report, _ = evaluate_model(
            model=model,
            loader=cached_batch_iterator(cached_batches),
            first_codebook_weight_multiplier=float(
                training_config["first_codebook_weight_multiplier"]
            ),
            text_padding_weight=float(training_config["text_padding_weight"]),
            mode=f"{args.experiment_label}_step_{step:06d}",
        )
        candidate = {
            "step": step,
            "eval_loss": mode_report["total_loss"]["mean"],
            "text_eval_loss": mode_report["text_loss"]["mean"],
            "audio_eval_loss": mode_report["audio_loss"]["mean"],
            "sample_count": mode_report["samples"],
            "validation_scope": "train_only_in_sample_fixed_manifest",
            "validation_manifest_sha256": sha256_file(manifest),
            "heldout_test_used": False,
            "final_test_opened": False,
            "adapter_sha256": sha256_file(checkpoint_paths[step]),
            "config_sha256": config_hash,
            "adapter_tensor_count": copy_result["tensor_count"],
            "elapsed_seconds": mode_report["elapsed_seconds"],
            "peak_allocated_gb": mode_report["peak_allocated_gb"],
        }
        candidates.append(candidate)
        write_jsonl_atomic(metrics_out, candidates)
        print(
            f"{args.experiment_label} reevaluation step={step} text_loss={candidate['text_eval_loss']:.9f}",
            flush=True,
        )

    requirements = {
        **preconditions,
        "all_candidates_reevaluated": [candidate["step"] for candidate in candidates]
        == expected_steps,
        "identical_32_row_scope": all(
            candidate["sample_count"] == 32
            and candidate["validation_manifest_sha256"] == sha256_file(manifest)
            for candidate in candidates
        ),
        "all_losses_finite": all(
            math.isfinite(float(candidate[key]))
            for candidate in candidates
            for key in ("eval_loss", "text_eval_loss", "audio_eval_loss")
        ),
        "all_adapter_schemas_identical": expected_keys is not None,
        "all_saved_configs_identical": expected_config_hash is not None,
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(requirements.values()) else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment_label,
        "diagnostic_only": True,
        "scientific_validation_evidence": False,
        "in_sample_training_evidence": True,
        "heldout_test_used": False,
        "final_test_opened": False,
        "checkpoint_selection_performed": False,
        "scope": {
            "role": "train_only_in_sample_capacity_diagnostic",
            "manifest": manifest.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(manifest),
            "rows": expected_rows,
            "chunks": expected_chunks,
            "duration_sec": duration_sec,
        },
        "candidates": candidates,
        "requirements": requirements,
        "reevaluation_passes": all(requirements.values()),
        "artifacts": {
            "training_config_sha256": sha256_file(training_config_path),
            "resolved_args_sha256": sha256_file(run_dir / "args.yaml"),
            "raw_train_metrics_sha256": sha256_file(raw_train_metrics_path),
            "raw_eval_metrics_sha256": sha256_file(raw_eval_metrics_path),
            "corrected_metrics_sha256": sha256_file(metrics_out),
            "reevaluator_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["reevaluation_passes"] is not True:
        raise RuntimeError("v6 in-sample reevaluation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
