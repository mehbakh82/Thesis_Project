#!/usr/bin/env python3
"""Post-hoc, hash-locked Shenava screen on the opened ASR v1 development panel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_asr_candidates_v1 import (  # noqa: E402
    CURRENT_ARM,
    SOURCE_SHA256,
    arm_summary,
    atomic_write,
    audio_path,
    evaluate_arm,
    load_source,
    model_identity,
    nemo_identity,
    paired_comparison,
    panel_receipt,
    select_panels,
)
from scripts.evaluate_cascade_intelligibility import text_error_counts  # noqa: E402
from scripts.evaluate_cascade_real_service_v4 import (  # noqa: E402
    script_statistics,
    sha256_file,
    sha256_text,
)
from thesis_s2s.data.verbatim import verbatim_normalize  # noqa: E402

CHALLENGER_ARM = "shenava-koochik-v1.0-sherpa-onnx"
SHENAVA_REPOSITORY = "Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx"
SHENAVA_REVISION = "f063f38cb38fe02df39887ac65c441b755ab25a2"
SHENAVA_MODEL_SHA256 = "6a564b5541920ce1c37bbc91d22e4b3a6838648b9b327eb88997e8db1f90950d"
ASR_V1_PLAN_SHA256 = "ca73a72c5f27b9459ccdf4298c282fb153af3b03c2709dcb23bbb89bb542f0ef"
PROTOCOL = ROOT / "docs" / "ASR_SHENAVA_SUPPLEMENTAL_V1.md"
ASR_V1_PLAN = ROOT / "results" / "eval" / "asr_candidates_v1_plan.json"
ASR_V1_EVALUATOR = ROOT / "scripts" / "evaluate_asr_candidates_v1.py"


class ShenavaAsr:
    def __init__(self, model_dir: Path) -> None:
        import sherpa_onnx

        model = model_dir / "model.onnx"
        tokens = model_dir / "tokens.txt"
        if sha256_file(model) != SHENAVA_MODEL_SHA256:
            raise RuntimeError("Shenava model hash drift")
        started = time.perf_counter()
        self.recognizer: Any = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(model),
            tokens=str(tokens),
            num_threads=4,
        )
        self.load_seconds = time.perf_counter() - started

    def transcribe(self, path: Path) -> tuple[str, float]:
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        waveform = np.asarray(audio, dtype=np.float32)
        if sample_rate != 16_000 or waveform.ndim != 1:
            raise ValueError("expected mono 16 kHz user audio")
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, waveform)
        started = time.perf_counter()
        self.recognizer.decode_stream(stream)
        elapsed = time.perf_counter() - started
        return str(stream.result.text), elapsed


def evaluate_shenava(
    panel: list[dict[str, Any]], model_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    backend = ShenavaAsr(model_dir)
    samples: list[dict[str, Any]] = []
    for row in panel:
        path = audio_path(row)
        reference = verbatim_normalize(str(row["text"]))
        duration = float(row.get("user_duration") or row.get("duration") or 0.0)
        sample: dict[str, Any] = {
            "row_hash": sha256_text(str(row["utt_id"])),
            "session_hash": sha256_text(str(row["session_id"])),
            "channel": str(row["session_id"]).split("/", 1)[0],
            "audio_sha256": sha256_file(path),
            "reference_sha256": sha256_text(reference),
            "duration_seconds": round(duration, 6),
        }
        try:
            raw, elapsed = backend.transcribe(path)
            hypothesis = verbatim_normalize(raw)
            sample.update(text_error_counts(reference, hypothesis))
            sample.update(
                {
                    "hypothesis_sha256": sha256_text(hypothesis),
                    "empty_hypothesis": not bool(hypothesis),
                    "language": "Persian" if script_statistics(hypothesis)["persian_letters"] else None,
                    "elapsed_seconds": round(elapsed, 6),
                    "real_time_factor": round(elapsed / duration, 6),
                }
            )
        except Exception as exc:
            sample.update(text_error_counts(reference, ""))
            sample.update(
                {
                    "hypothesis_sha256": sha256_text(""),
                    "empty_hypothesis": True,
                    "failure_type": type(exc).__name__,
                }
            )
        samples.append(sample)
    return samples, {
        "arm": CHALLENGER_ARM,
        "load_seconds": round(backend.load_seconds, 6),
        "runtime": "sherpa-onnx",
        "num_threads": 4,
        "persian_itn_applied": False,
    }


def screen_decision(
    incumbent: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    comparison = paired_comparison(incumbent, challenger, CHALLENGER_ARM)
    both_valid = all(summaries[arm]["mechanically_valid"] for arm in (CURRENT_ARM, CHALLENGER_ARM))
    advances = bool(
        both_valid
        and comparison["cer_absolute_reduction"] >= 0.02
        and comparison["wer_absolute_reduction"] >= 0.0
        and comparison["challenger_minus_incumbent_cer_bootstrap_95"][1] < 0.0
    )
    return {
        "status": "new_confirmation_required" if advances else "incumbent_retained",
        "screen_winner": CHALLENGER_ARM if advances else CURRENT_ARM,
        "challenger_advances_to_new_confirmation": advances,
        "production_promotion_allowed": False,
        "pairwise_against_incumbent": comparison,
        "thresholds": {
            "minimum_cer_absolute_reduction": 0.02,
            "minimum_wer_absolute_reduction": 0.0,
            "maximum_cer_difference_ci95_upper": 0.0,
        },
    }


def runtime_packages() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("numpy", "sherpa-onnx", "soundfile")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("plan", "screen"), required=True)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/processed/manifests/conversations.jsonl"
    )
    parser.add_argument(
        "--shenava-dir",
        type=Path,
        default=ROOT / "models" / "Shenava-Koochik-v1.0-sherpa-onnx",
    )
    parser.add_argument("--nemo-url", default="http://127.0.0.1:8090")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--plan-out",
        type=Path,
        default=ROOT / "results/eval/asr_shenava_supplemental_v1_plan.json",
    )
    parser.add_argument(
        "--screen-out",
        type=Path,
        default=ROOT / "results/eval/asr_shenava_supplemental_v1_screen.json",
    )
    args = parser.parse_args()

    if sha256_file(ASR_V1_PLAN) != ASR_V1_PLAN_SHA256:
        raise RuntimeError("ASR v1 plan hash drift")
    rows = load_source(args.source)
    development_panel = select_panels(rows)["development"]
    recorded_v1 = json.loads(ASR_V1_PLAN.read_text(encoding="utf-8"))
    receipt = panel_receipt(development_panel)
    if receipt != recorded_v1["development"]:
        raise RuntimeError("ASR v1 development panel drift")

    identities = {
        CURRENT_ARM: nemo_identity(args.nemo_url, args.timeout),
        CHALLENGER_ARM: {
            **model_identity(args.shenava_dir, SHENAVA_REPOSITORY),
            "revision": SHENAVA_REVISION,
            "model_sha256": SHENAVA_MODEL_SHA256,
            "license": "Apache-2.0",
        },
    }
    plan = {
        "schema_version": 1,
        "stage": "preinference_posthoc_screen_plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": SOURCE_SHA256,
        "asr_v1_plan_sha256": ASR_V1_PLAN_SHA256,
        "asr_v1_evaluator_sha256": sha256_file(ASR_V1_EVALUATOR),
        "protocol_sha256": sha256_file(PROTOCOL),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "development": receipt,
        "model_identities": identities,
        "runtime_packages": runtime_packages(),
        "posthoc_not_predeclared": True,
        "final_panel_accessed": False,
        "plaintext_retained": False,
    }
    if args.stage == "plan":
        atomic_write(args.plan_out, plan)
        print(json.dumps({"status": "planned", "development": receipt}, indent=2))
        return 0

    if not args.plan_out.is_file():
        raise FileNotFoundError("run --stage plan before screening")
    recorded_plan = json.loads(args.plan_out.read_text(encoding="utf-8"))
    locked_keys = (
        "source_sha256",
        "asr_v1_plan_sha256",
        "asr_v1_evaluator_sha256",
        "protocol_sha256",
        "evaluator_sha256",
        "development",
        "model_identities",
        "runtime_packages",
        "posthoc_not_predeclared",
        "final_panel_accessed",
    )
    if any(recorded_plan.get(key) != plan.get(key) for key in locked_keys):
        raise RuntimeError("supplemental plan or code/protocol/model/source drift")

    os.environ["ASR_TIMEOUT_SECONDS"] = str(args.timeout)
    incumbent_samples, incumbent_runtime = evaluate_arm(
        CURRENT_ARM,
        development_panel,
        None,
        args.nemo_url,
        args.timeout,
    )
    challenger_samples, challenger_runtime = evaluate_shenava(
        development_panel, args.shenava_dir
    )
    samples = {CURRENT_ARM: incumbent_samples, CHALLENGER_ARM: challenger_samples}
    summaries = {arm: arm_summary(arm_samples) for arm, arm_samples in samples.items()}
    decision = screen_decision(incumbent_samples, challenger_samples, summaries)
    report = {
        "schema_version": 1,
        "stage": "posthoc_development_screen",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if summaries[CURRENT_ARM]["mechanically_valid"] else "failed",
        "plan_sha256": sha256_file(args.plan_out),
        "source_sha256": SOURCE_SHA256,
        "protocol_sha256": sha256_file(PROTOCOL),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "panel": receipt,
        "runtime_identities": {
            CURRENT_ARM: incumbent_runtime,
            CHALLENGER_ARM: challenger_runtime,
        },
        "summary": summaries,
        "decision": decision,
        "samples": samples,
        "claim_boundary": {
            "posthoc_candidate_screen": True,
            "candidate_predeclared_in_asr_v1": False,
            "automatic_caption_references": True,
            "shenava_training_overlap_known": False,
            "production_promotion_allowed": False,
            "final_panel_accessed": False,
            "plaintext_retained": False,
        },
    }
    atomic_write(args.screen_out, report)
    print(json.dumps({"status": report["status"], "decision": decision}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
