#!/usr/bin/env python3
"""Run the frozen group-disjoint NeMo -> Qwen -> Piper validation panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_cascade_real_service_v4 import (  # noqa: E402
    atomic_write,
    health_probe,
    percentile,
    qwen_revision,
    script_statistics,
    sha256_file,
    sha256_text,
)
from thesis_s2s import SAMPLE_RATE  # noqa: E402
from thesis_s2s.metrics import gpu_inventory  # noqa: E402
from thesis_s2s.runtime.cascade import CascadeTalker, TextResponder  # noqa: E402

PANEL_INDICES = (0, 16, 32, 48, 65, 81, 97, 113, 130)
EXPECTED_ROWS = 131
EXPECTED_CHANNELS = 2
USER_CHANNEL_INDEX = 1
MIN_TEXT_LENGTH = 4
MIN_PERSIAN_LETTER_FRACTION = 0.8
MIN_AUDIO_SAMPLES = 1600
MIN_AUDIO_RMS = 1e-3
MANIFEST_SHA256 = "a2762495830c82ce12d3dc4cc8469e2110ae21a77ea91f9b70498a6812e187a7"
PARENT_RESULT_SHA256 = "d6c166c7bb8d93f332c277bdb4ac1fef5bfdfe080985025d7ec0072cda738b36"
EXPORT_AUDIT_SHA256 = "9f4fdf6301ed7d63713df6e4818c41e0a4bfefbe0a14e88e0faaba6ef6d5ee00"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def read_user_channel(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    """Read audited user channel 1 from an assistant-0/user-1 stereo export."""

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if channels != EXPECTED_CHANNELS or sample_width != 2 or sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"expected 16 kHz, 16-bit stereo pair, found "
            f"{sample_rate} Hz/{sample_width * 8}-bit/{channels} channels: {path}"
        )
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
    user = pcm[:, USER_CHANNEL_INDEX].astype(np.float32) / 32768.0
    return user, {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frames,
        "selected_user_channel": USER_CHANNEL_INDEX,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/moshi_finetune/val.jsonl"),
    )
    parser.add_argument("--asr-url", default="http://127.0.0.1:8090")
    parser.add_argument("--asr-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--piper-model", type=Path, default=Path("models/piper/fa_IR-mana-medium.onnx")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("docs/CASCADE_VALIDATION_PROTOCOL.md")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/eval/cascade_real_service_validation_panel.json"),
    )
    args = parser.parse_args()

    manifest_path = (ROOT / args.manifest).resolve()
    piper_path = (ROOT / args.piper_model).resolve()
    protocol_path = (ROOT / args.protocol).resolve()
    output_path = (ROOT / args.out).resolve()
    piper_config_path = piper_path.with_suffix(piper_path.suffix + ".json")
    parent_result_path = (
        ROOT / "results/eval/cascade_real_service_train_panel_v4.json"
    ).resolve()
    export_audit_path = (ROOT / "results/moshi_export_audit.json").resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    if "test" in manifest_path.name.lower() or "test" in manifest_path.parent.name.lower():
        raise ValueError("test manifests are forbidden for this validation panel")
    for required in (
        manifest_path,
        piper_path,
        piper_config_path,
        protocol_path,
        parent_result_path,
        export_audit_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("validation manifest hash drift")
    if sha256_file(parent_result_path) != PARENT_RESULT_SHA256:
        raise RuntimeError("parent cascade result hash drift")
    if sha256_file(export_audit_path) != EXPORT_AUDIT_SHA256:
        raise RuntimeError("export audit hash drift")

    rows = load_jsonl(manifest_path)
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    export_audit = json.loads(export_audit_path.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} validation rows, found {len(rows)}")

    os.environ["ASR_API_URL"] = args.asr_url
    os.environ["ASR_TIMEOUT_SECONDS"] = str(args.asr_timeout_seconds)
    os.environ["TEXT_LLM_ENABLED"] = "1"
    os.environ["TEXT_LLM_MODEL"] = args.llm_model
    os.environ["PIPER_MODEL"] = str(piper_path)

    health = health_probe(args.asr_url, args.asr_timeout_seconds)
    model_started = time.perf_counter()
    responder = TextResponder(args.llm_model)
    model_load_seconds = round(time.perf_counter() - model_started, 3)
    talker = CascadeTalker(responder)
    samples: list[dict[str, Any]] = []
    latencies: list[float] = []

    for panel_position, row_index in enumerate(PANEL_INDICES):
        row = rows[row_index]
        audio_path = Path(str(row["path"])).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        actual_audio_sha256 = sha256_file(audio_path)
        user_audio, audio_format = read_user_channel(audio_path)
        started = time.perf_counter()
        reply_audio = np.asarray(talker.reply_audio(user_audio), dtype=np.float32)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        latencies.append(elapsed_ms)

        transcript_stats = script_statistics(talker.last_transcript)
        reply_stats = script_statistics(talker.last_reply_text)
        audio_finite = bool(np.isfinite(reply_audio).all())
        audio_rms = (
            float(np.sqrt(np.mean(np.square(reply_audio, dtype=np.float64))))
            if reply_audio.size and audio_finite
            else 0.0
        )
        audio_peak = float(np.max(np.abs(reply_audio))) if reply_audio.size and audio_finite else 0.0
        requirements = {
            "source_hash_matches_manifest": actual_audio_sha256 == row.get("sha256"),
            "audited_user_channel_selected": audio_format["selected_user_channel"]
            == USER_CHANNEL_INDEX,
            "asr_succeeded": talker.last_asr_error is None,
            "transcript_has_minimum_length": transcript_stats["characters"] >= MIN_TEXT_LENGTH,
            "transcript_is_persian": transcript_stats["persian_letter_fraction"]
            >= MIN_PERSIAN_LETTER_FRACTION,
            "qwen_backend_exact": talker.responder_backend == args.llm_model,
            "qwen_initialization_succeeded": talker.responder_initialization_error is None,
            "qwen_generation_succeeded_without_fallback": (
                talker.last_responder_fallback_used is False
                and talker.last_responder_error is None
            ),
            "qwen_generation_attempts_bounded": (
                talker.last_responder_generation_attempts in {1, 2}
                and talker.last_responder_language_retry_used
                == (talker.last_responder_generation_attempts == 2)
            ),
            "reply_has_minimum_length": reply_stats["characters"] >= MIN_TEXT_LENGTH,
            "reply_is_persian": reply_stats["persian_letter_fraction"]
            >= MIN_PERSIAN_LETTER_FRACTION,
            "piper_backend_exact": talker.backend == "piper",
            "reply_audio_is_finite": audio_finite,
            "reply_audio_has_minimum_length": reply_audio.size >= MIN_AUDIO_SAMPLES,
            "reply_audio_is_non_silent": audio_rms >= MIN_AUDIO_RMS,
            "reply_audio_is_normalized": audio_peak <= 1.0,
        }
        samples.append(
            {
                "panel_position": panel_position,
                "manifest_row_index": row_index,
                "source_audio_sha256": actual_audio_sha256,
                "source_audio_format": audio_format,
                "transcript_sha256": sha256_text(talker.last_transcript),
                "transcript_statistics": transcript_stats,
                "reply_text_sha256": sha256_text(talker.last_reply_text),
                "reply_statistics": reply_stats,
                "reply_audio_samples": int(reply_audio.size),
                "reply_audio_duration_seconds": round(reply_audio.size / SAMPLE_RATE, 6),
                "reply_audio_rms": round(audio_rms, 6),
                "reply_audio_peak": round(audio_peak, 6),
                "full_turn_generation_ms": elapsed_ms,
                "asr_error": talker.last_asr_error,
                "responder_backend": talker.responder_backend,
                "responder_fallback_used": talker.last_responder_fallback_used,
                "responder_error": talker.last_responder_error,
                "responder_generation_attempts": talker.last_responder_generation_attempts,
                "responder_language_retry_used": talker.last_responder_language_retry_used,
                "tts_backend": talker.backend,
                "requirements": requirements,
                "passes": all(requirements.values()),
            }
        )

    audit_requirements = export_audit.get("requirements") or {}
    requirements = {
        "validation_manifest_exact": (
            len(rows) == EXPECTED_ROWS and sha256_file(manifest_path) == MANIFEST_SHA256
        ),
        "fixed_panel_exact": tuple(
            sample["manifest_row_index"] for sample in samples
        )
        == PANEL_INDICES,
        "asr_health_passes": health["passes"] is True,
        "qwen_backend_loaded": responder.backend == args.llm_model,
        "qwen_initialization_succeeded": responder.initialization_error is None,
        "parent_result_preserved_and_bound": (
            parent_result.get("status") == "passed"
            and (parent_result.get("claim") or {}).get("positive_working_result") is True
        ),
        "export_audit_preserved_and_bound": (
            export_audit.get("audit_passes") is True
            and (export_audit.get("split_counts") or {}).get("val") == EXPECTED_ROWS
            and export_audit.get("group_split_leaks") == 0
            and audit_requirements.get("session_group_splits_isolated") is True
        ),
        "audited_user_channel_one_used": all(
            (sample.get("source_audio_format") or {}).get("selected_user_channel")
            == USER_CHANNEL_INDEX
            for sample in samples
        ),
        "all_nine_samples_pass": len(samples) == len(PANEL_INDICES)
        and all(sample["passes"] for sample in samples),
        "final_test_not_accessed": True,
    }
    passed = all(requirements.values())
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "evidence_class": "real_service_group_disjoint_validation_mechanics",
        "claim": {
            "positive_working_user_turn_result": passed,
            "group_disjoint_validation_execution": passed,
            "scientific_generalization_result": False,
            "semantic_relevance_result": False,
            "human_quality_result": False,
            "official_end_to_end_latency_result": False,
            "description": (
                "Frozen NeMo ASR -> local Qwen -> Piper execution on audited user channel 1 "
                "from a source-session-group-isolated validation panel."
            ),
        },
        "correction": {
            "parent_result": "results/eval/cascade_real_service_train_panel_v4.json",
            "parent_input_channel": 0,
            "audited_user_channel": USER_CHANNEL_INDEX,
            "interpretation": (
                "The parent remains component-chain evidence; this validation corrects "
                "the input channel before any validation-row execution."
            ),
        },
        "panel": {
            "manifest": str(args.manifest),
            "manifest_rows": len(rows),
            "indices": list(PANEL_INDICES),
            "selection": "predeclared floor-spaced indices over 131 validation rows",
            "split_unit": "source_session_id",
            "group_split_leaks": export_audit.get("group_split_leaks"),
            "selected_user_channel": USER_CHANNEL_INDEX,
            "final_test_accessed": False,
        },
        "components": {
            "asr": {"backend": "nemo-soroush-http", "base_url": args.asr_url, "health": health},
            "responder": {
                "backend": responder.backend,
                "requested_model": args.llm_model,
                "revision": qwen_revision(args.llm_model),
                "device": responder.device,
                "model_class": type(responder.model).__name__ if responder.model is not None else None,
                "model_load_seconds": model_load_seconds,
                "initialization_error": responder.initialization_error,
            },
            "tts": {"backend_required": "piper", "model": str(args.piper_model)},
        },
        "hardware": gpu_inventory(),
        "thresholds": {
            "minimum_text_characters": MIN_TEXT_LENGTH,
            "minimum_persian_letter_fraction": MIN_PERSIAN_LETTER_FRACTION,
            "minimum_reply_audio_samples": MIN_AUDIO_SAMPLES,
            "minimum_reply_audio_rms": MIN_AUDIO_RMS,
            "maximum_qwen_generation_attempts": 2,
        },
        "timing_descriptive_only": {
            "metric": "full_turn_generation_ms",
            "n": len(latencies),
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "max_ms": round(max(latencies), 3),
        },
        "samples": samples,
        "requirements": requirements,
        "artifacts": {
            "manifest_sha256": sha256_file(manifest_path),
            "export_audit_sha256": sha256_file(export_audit_path),
            "parent_result_sha256": sha256_file(parent_result_path),
            "piper_model_sha256": sha256_file(piper_path),
            "piper_config_sha256": sha256_file(piper_config_path),
            "protocol_sha256": sha256_file(protocol_path),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "runtime_cascade_sha256": sha256_file(ROOT / "src/thesis_s2s/runtime/cascade.py"),
            "runtime_tts_sha256": sha256_file(ROOT / "src/thesis_s2s/runtime/tts.py"),
        },
    }
    atomic_write(output_path, report)
    print(json.dumps({"output": str(output_path), "status": report["status"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
