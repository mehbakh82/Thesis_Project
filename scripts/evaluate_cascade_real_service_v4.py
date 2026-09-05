#!/usr/bin/env python3
"""Run v4 of the frozen train-only NeMo -> Qwen -> Piper acceptance panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis_s2s import SAMPLE_RATE  # noqa: E402
from thesis_s2s.metrics import gpu_inventory  # noqa: E402
from thesis_s2s.runtime.cascade import CascadeTalker, TextResponder  # noqa: E402

PANEL_INDICES = (0, 3, 7, 11, 15, 19, 23, 27, 31)
EXPECTED_ROWS = 32
EXPECTED_CHANNELS = 2
MIN_TEXT_LENGTH = 4
MIN_PERSIAN_LETTER_FRACTION = 0.8
MIN_AUDIO_SAMPLES = 1600
MIN_AUDIO_RMS = 1e-3
PARENT_RESULT_SHA256 = "04b22fb4f4738bc238c9b1c647030176a3ece21aa427ebd91cb9fa08a16cd8a4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def script_statistics(text: str) -> dict[str, int | float]:
    letters = [character for character in text if character.isalpha()]
    persian_letters = [
        character for character in letters if "ARABIC" in unicodedata.name(character, "")
    ]
    return {
        "characters": len(text),
        "letters": len(letters),
        "persian_letters": len(persian_letters),
        "persian_letter_fraction": round(len(persian_letters) / max(1, len(letters)), 6),
    }


def read_user_channel(path: Path) -> tuple[np.ndarray, dict[str, int]]:
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
    user = pcm[:, 0].astype(np.float32) / 32768.0
    return user, {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frames,
    }


def percentile(values: list[float], q: float) -> float:
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 3)


def qwen_revision(model_name: str) -> str | None:
    cache_name = "models--" + model_name.replace("/", "--")
    reference = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "refs" / "main"
    if not reference.is_file():
        return None
    revision = reference.read_text(encoding="utf-8").strip()
    return revision or None


def health_probe(base_url: str, timeout_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with urlopen(base_url.rstrip("/") + "/health", timeout=timeout_seconds) as response:
            body = response.read()
            return {
                "http_status": response.status,
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "passes": response.status == 200,
            }
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "passes": False,
        }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/moshi_finetune_v6_overfit/train.jsonl"),
    )
    parser.add_argument("--asr-url", default="http://127.0.0.1:8090")
    parser.add_argument("--asr-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--piper-model", type=Path, default=Path("models/piper/fa_IR-mana-medium.onnx")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("docs/CASCADE_REAL_SERVICE_PROTOCOL_V4.md")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/eval/cascade_real_service_train_panel_v4.json")
    )
    args = parser.parse_args()

    manifest_path = (ROOT / args.manifest).resolve()
    piper_path = (ROOT / args.piper_model).resolve()
    protocol_path = (ROOT / args.protocol).resolve()
    output_path = (ROOT / args.out).resolve()
    piper_config_path = piper_path.with_suffix(piper_path.suffix + ".json")
    parent_result_path = (
        ROOT / "results/eval/cascade_real_service_train_panel_v3.json"
    ).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    if "test" in manifest_path.name.lower() or "test" in manifest_path.parent.name.lower():
        raise ValueError("sealed test manifests are forbidden for this acceptance panel")
    for required in (
        manifest_path,
        piper_path,
        piper_config_path,
        protocol_path,
        parent_result_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    rows = load_jsonl(manifest_path)
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} train-only rows, found {len(rows)}")

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
                "responder_language_retry_used": (
                    talker.last_responder_language_retry_used
                ),
                "tts_backend": talker.backend,
                "requirements": requirements,
                "passes": all(requirements.values()),
            }
        )

    requirements = {
        "train_only_manifest_exact_size": len(rows) == EXPECTED_ROWS,
        "fixed_panel_exact": tuple(sample["manifest_row_index"] for sample in samples)
        == PANEL_INDICES,
        "asr_health_passes": health["passes"] is True,
        "qwen_backend_loaded": responder.backend == args.llm_model,
        "qwen_initialization_succeeded": responder.initialization_error is None,
        "parent_result_preserved_and_bound": (
            parent_result.get("status") == "failed"
            and sha256_file(parent_result_path) == PARENT_RESULT_SHA256
        ),
        "all_nine_samples_pass": len(samples) == len(PANEL_INDICES)
        and all(sample["passes"] for sample in samples),
        "final_test_not_accessed": True,
    }
    report = {
        "schema_version": 4,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(requirements.values()) else "failed",
        "evidence_class": "real_service_train_only_working_demo",
        "claim": {
            "positive_working_result": all(requirements.values()),
            "scientific_generalization_result": False,
            "official_end_to_end_latency_result": False,
            "description": "Real NeMo ASR -> local Qwen -> Piper execution on a fixed train-only panel.",
            "limitations": [
                "The panel is in-sample and cannot establish generalization.",
                "No independent human naturalness or task-success labels were collected.",
                "Full-turn generation time is not the thesis T_first_audio browser metric.",
                "The physical GPU is outside the official 12-24 GB evaluation class when reported as such.",
            ],
        },
        "versioned_correction": {
            "parent_result": "results/eval/cascade_real_service_train_panel_v3.json",
            "parent_status": parent_result.get("status"),
            "changes": [
                "retry Qwen once with a stricter Persian-only instruction after a script violation",
            ],
        },
        "panel": {
            "manifest": str(args.manifest),
            "manifest_rows": len(rows),
            "indices": list(PANEL_INDICES),
            "selection": "predeclared evenly spread indices over the 32-row v6 train-only set",
            "sealed_final_test_accessed": False,
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
            "piper_model_sha256": sha256_file(piper_path),
            "piper_config_sha256": sha256_file(piper_config_path),
            "protocol_sha256": sha256_file(protocol_path),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "runtime_cascade_sha256": sha256_file(ROOT / "src/thesis_s2s/runtime/cascade.py"),
            "runtime_tts_sha256": sha256_file(ROOT / "src/thesis_s2s/runtime/tts.py"),
            "parent_result_sha256": sha256_file(parent_result_path),
        },
    }
    atomic_write(output_path, report)
    print(json.dumps({"output": str(output_path), "status": report["status"]}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
