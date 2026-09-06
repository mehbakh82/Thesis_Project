#!/usr/bin/env python3
"""Evaluate the frozen compact responder LoRA with privacy-safe semantic evidence."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_cascade_real_service_v4 import (  # noqa: E402
    atomic_write,
    qwen_revision,
    script_statistics,
    sha256_file,
    sha256_text,
)
from scripts.evaluate_cascade_semantic_proxy import (  # noqa: E402
    JUDGE_CONTAINER,
    JUDGE_IMAGE_ID,
    JUDGE_IMAGE_REF,
    JUDGE_MODEL_ROOT,
    JUDGE_REVISION,
    JUDGE_SERVED_MODEL,
    SCORE_KEYS,
    _http_json,
    inspect_judge_container,
    judge_service_identity,
    parse_score_content,
)
from scripts.evaluate_cascade_semantic_proxy import (  # noqa: E402
    SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT,
)
from scripts.evaluate_cascade_validation import (  # noqa: E402
    read_user_channel,
)
from scripts.train_responder_lora import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    DEV_SESSION_HASHES,
    SOURCE_MANIFEST_SHA256,
    SPLIT_SEED,
    VAL_EXPORT_SHA256,
    adapter_tree,
    session_order_key,
)
from thesis_s2s.data.verbatim import verbatim_normalize  # noqa: E402
from thesis_s2s.metrics import gpu_inventory  # noqa: E402
from thesis_s2s.runtime.cascade import TextResponder, _wav_bytes, asr_http  # noqa: E402
from thesis_s2s.runtime.tts import synthesize  # noqa: E402

SEED = 20260906
EXPECTED_VAL_ROWS = 131
EXPECTED_FINAL_ROWS = 40
REPEATS = 2
ARMS = ("base", "lora", "reference")
EXPECTED_JUDGE_CALLS = EXPECTED_FINAL_ROWS * len(ARMS) * REPEATS
PREVIOUSLY_OBSERVED_INDICES = (0, 16, 32, 48, 65, 81, 97, 113, 130)
FINAL_PANEL_INDICES = (
    20,
    21,
    28,
    29,
    30,
    31,
    34,
    36,
    38,
    39,
    41,
    42,
    46,
    49,
    51,
    52,
    55,
    58,
    59,
    85,
    86,
    87,
    89,
    90,
    91,
    93,
    94,
    102,
    104,
    105,
    106,
    109,
    111,
    114,
    117,
    119,
    120,
    123,
    124,
    127,
)
MIN_RELEVANCE_MEAN = 2.0
MIN_COHERENCE_MEAN = 2.5
MIN_RELEVANCE_GAIN = 0.5
MIN_COHERENCE_GAIN = 0.0
MIN_RELEVANCE_WIN_RATE = 0.60
MIN_RELEVANCE_AT_LEAST_TWO_RATE = 0.70
PIPER_MODEL_SHA256 = "e390c0e74ba71fd97c49ba662ee0c6e1724b462ba2d4561698af4f564840f126"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def load_validation_source_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "val":
                rows.append(row)
    return rows


def expected_final_indices(source_rows: list[dict[str, Any]]) -> tuple[int, ...]:
    sessions = sorted({str(row["session_id"]) for row in source_rows}, key=session_order_key)
    if len(sessions) != 5:
        raise ValueError("expected exactly five validation sessions")
    if tuple(sha256_text(session) for session in sessions[:2]) != DEV_SESSION_HASHES:
        raise ValueError("development-session drift")
    final_sessions = set(sessions[2:])
    excluded = set(PREVIOUSLY_OBSERVED_INDICES)
    candidates = [
        index
        for index, row in enumerate(source_rows)
        if str(row["session_id"]) in final_sessions and index not in excluded
    ]
    candidates.sort(
        key=lambda index: sha256_text(SPLIT_SEED + "\0" + str(source_rows[index]["utt_id"]))
    )
    return tuple(sorted(candidates[:EXPECTED_FINAL_ROWS]))


def blind_arm_order(row_index: int) -> tuple[str, ...]:
    return tuple(
        sorted(
            ARMS,
            key=lambda arm: sha256_text(f"{SEED}\0{row_index}\0{arm}"),
        )
    )


def judge_pair(
    base_url: str,
    *,
    transcript: str,
    reply: str,
    timeout: float,
) -> tuple[dict[str, int], dict[str, Any]]:
    user_payload = json.dumps(
        {"user_utterance": transcript, "assistant_reply": reply},
        ensure_ascii=False,
        sort_keys=True,
    )
    request_payload = {
        "model": JUDGE_SERVED_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": SEED,
        "max_tokens": 128,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = _http_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        payload=request_payload,
        timeout=timeout,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("judge must return exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("judge choice has no message")
    return parse_score_content(message.get("content")), {
        "response_model": response.get("model"),
        "system_fingerprint": response.get("system_fingerprint"),
        "finish_reason": choices[0].get("finish_reason"),
        "request_sha256": sha256_text(JUDGE_SYSTEM_PROMPT + "\n" + user_payload),
    }


def dimension_summary(values: list[int]) -> dict[str, Any]:
    return {
        "mean": round(float(np.mean(values)), 6),
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "min": min(values),
        "max": max(values),
        "histogram": {str(score): values.count(score) for score in range(5)},
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": len(samples),
        "judge_calls_expected": EXPECTED_JUDGE_CALLS,
        "judge_calls_valid": sum(
            len(arm["scores"]) for sample in samples for arm in sample["arms"].values()
        ),
    }
    summary["judge_calls_failed"] = summary["judge_calls_expected"] - summary["judge_calls_valid"]
    for arm_name in ARMS:
        arm_summary: dict[str, Any] = {}
        for dimension in SCORE_KEYS:
            values = [
                int(score[dimension])
                for sample in samples
                for score in sample["arms"][arm_name]["scores"]
            ]
            arm_summary[dimension] = dimension_summary(values)
        arm_summary["repeat_exact_agreement_rows"] = sum(
            len(sample["arms"][arm_name]["scores"]) == REPEATS
            and sample["arms"][arm_name]["scores"][0] == sample["arms"][arm_name]["scores"][1]
            for sample in samples
        )
        arm_summary["repeat_exact_agreement_rate"] = round(
            arm_summary["repeat_exact_agreement_rows"] / max(1, len(samples)), 6
        )
        summary[arm_name] = arm_summary
    relevance_wins = 0
    relevance_at_least_two = 0
    for sample in samples:
        base_values = [score["relevance"] for score in sample["arms"]["base"]["scores"]]
        lora_values = [score["relevance"] for score in sample["arms"]["lora"]["scores"]]
        if np.median(lora_values) > np.median(base_values):
            relevance_wins += 1
        if np.median(lora_values) >= 2:
            relevance_at_least_two += 1
    summary["paired"] = {
        "lora_minus_base_relevance_mean": round(
            summary["lora"]["relevance"]["mean"] - summary["base"]["relevance"]["mean"],
            6,
        ),
        "lora_minus_base_coherence_mean": round(
            summary["lora"]["coherence"]["mean"] - summary["base"]["coherence"]["mean"],
            6,
        ),
        "lora_relevance_win_rows": relevance_wins,
        "lora_relevance_win_rate": round(relevance_wins / max(1, len(samples)), 6),
        "lora_relevance_at_least_two_rows": relevance_at_least_two,
        "lora_relevance_at_least_two_rate": round(relevance_at_least_two / max(1, len(samples)), 6),
    }
    return summary


def engineering_gate(summary: dict[str, Any], mechanically_valid: bool) -> dict[str, bool]:
    return {
        "mechanically_valid": mechanically_valid,
        "lora_relevance_mean_at_least_2_0": (
            summary["lora"]["relevance"]["mean"] >= MIN_RELEVANCE_MEAN
        ),
        "lora_coherence_mean_at_least_2_5": (
            summary["lora"]["coherence"]["mean"] >= MIN_COHERENCE_MEAN
        ),
        "relevance_gain_at_least_0_50": (
            summary["paired"]["lora_minus_base_relevance_mean"] >= MIN_RELEVANCE_GAIN
        ),
        "coherence_gain_nonnegative": (
            summary["paired"]["lora_minus_base_coherence_mean"] >= MIN_COHERENCE_GAIN
        ),
        "relevance_win_rate_at_least_0_60": (
            summary["paired"]["lora_relevance_win_rate"] >= MIN_RELEVANCE_WIN_RATE
        ),
        "relevance_at_least_two_rate_at_least_0_70": (
            summary["paired"]["lora_relevance_at_least_two_rate"] >= MIN_RELEVANCE_AT_LEAST_TWO_RATE
        ),
    }


def _load_responder(
    adapter_path: Path | None = None, adapter_hash: str | None = None
) -> TextResponder:
    responder = TextResponder(BASE_MODEL)
    if responder.model is None or responder.tokenizer is None or responder.backend != BASE_MODEL:
        raise RuntimeError(f"base responder failed to load: {responder.initialization_error}")
    if adapter_path is not None:
        from peft import PeftModel

        responder.model = PeftModel.from_pretrained(
            responder.model, adapter_path, local_files_only=True
        ).eval()
        responder.backend = f"{BASE_MODEL}+LoRA:{str(adapter_hash)[:16]}"
    return responder


def _unload_responder(responder: TextResponder) -> None:
    import torch

    responder.model = None
    responder.tokenizer = None
    del responder
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/processed/manifests/conversations.jsonl"),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("data/processed/moshi_finetune/val.jsonl"),
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=Path("results/training/responder_lora_v1.json"),
    )
    parser.add_argument("--protocol", type=Path, default=Path("docs/RESPONDER_LORA_PROTOCOL.md"))
    parser.add_argument("--trainer", type=Path, default=Path("scripts/train_responder_lora.py"))
    parser.add_argument(
        "--piper-model", type=Path, default=Path("models/piper/fa_IR-mana-medium.onnx")
    )
    parser.add_argument("--asr-url", default="http://127.0.0.1:8090")
    parser.add_argument("--asr-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8003")
    parser.add_argument("--judge-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--judge-container", default=JUDGE_CONTAINER)
    parser.add_argument(
        "--out", type=Path, default=Path("results/eval/responder_lora_semantic_proxy.json")
    )
    args = parser.parse_args()

    source_path = (ROOT / args.source_manifest).resolve()
    val_path = (ROOT / args.val_manifest).resolve()
    training_report_path = (ROOT / args.training_report).resolve()
    protocol_path = (ROOT / args.protocol).resolve()
    trainer_path = (ROOT / args.trainer).resolve()
    piper_path = (ROOT / args.piper_model).resolve()
    piper_config_path = piper_path.with_suffix(piper_path.suffix + ".json")
    output_path = (ROOT / args.out).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    if "test" in val_path.name.lower() or "test" in val_path.parent.name.lower():
        raise ValueError("test manifests are forbidden")
    for required in (
        source_path,
        val_path,
        training_report_path,
        protocol_path,
        trainer_path,
        piper_path,
        piper_config_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if sha256_file(source_path) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("source manifest hash drift")
    if sha256_file(val_path) != VAL_EXPORT_SHA256:
        raise RuntimeError("validation manifest hash drift")
    if sha256_file(piper_path) != PIPER_MODEL_SHA256:
        raise RuntimeError("Piper model hash drift")

    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    selected_relative = str(
        (training_report.get("selection") or {}).get("selected_adapter_relative_path") or ""
    )
    adapter_path = (ROOT / selected_relative).resolve()
    if not selected_relative or not adapter_path.is_dir() or not adapter_path.is_relative_to(ROOT):
        raise RuntimeError("selected adapter path is invalid")
    adapter_files, adapter_hash = adapter_tree(adapter_path)
    expected_adapter_hash = (training_report.get("selection") or {}).get(
        "selected_adapter_tree_sha256"
    )

    source_rows = load_validation_source_rows(source_path)
    exported_rows = load_jsonl(val_path)
    if len(source_rows) != EXPECTED_VAL_ROWS or len(exported_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError("validation row-count drift")
    for source, exported in zip(source_rows, exported_rows, strict=True):
        sidecar = json.loads(Path(str(exported["path"])).with_suffix(".json").read_text())
        if sidecar.get("source_pair_id") != source.get("utt_id"):
            raise RuntimeError("source/exported validation order drift")
    if expected_final_indices(source_rows) != FINAL_PANEL_INDICES:
        raise RuntimeError("final panel-selection drift")

    container_identity = inspect_judge_container(args.judge_container)
    service_identity = judge_service_identity(args.judge_url, timeout=args.judge_timeout_seconds)
    judge_identity_validity = {
        "judge_container_running": container_identity["running"] is True,
        "judge_image_id_exact": container_identity["image_id"] == JUDGE_IMAGE_ID,
        "judge_image_ref_exact": container_identity["image_ref"] == JUDGE_IMAGE_REF,
        "judge_container_model_root_exact": container_identity["model_root"] == JUDGE_MODEL_ROOT,
        "judge_container_revision_exact": container_identity["revision"] == JUDGE_REVISION,
        "judge_container_served_name_exact": container_identity["served_model_name"]
        == JUDGE_SERVED_MODEL,
        "judge_service_model_root_exact": service_identity["model_root"] == JUDGE_MODEL_ROOT,
        "judge_service_served_name_exact": service_identity["served_model_name"]
        == JUDGE_SERVED_MODEL,
    }
    if not all(judge_identity_validity.values()):
        raise RuntimeError("judge identity failed before project text submission")

    os.environ["ASR_API_URL"] = args.asr_url
    os.environ["ASR_TIMEOUT_SECONDS"] = str(args.asr_timeout_seconds)
    os.environ["TEXT_LLM_ENABLED"] = "1"
    os.environ["PIPER_MODEL"] = str(piper_path)
    runtime_rows: list[dict[str, Any]] = []
    for row_index in FINAL_PANEL_INDICES:
        exported = exported_rows[row_index]
        audio_path = Path(str(exported["path"])).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        user_audio, audio_format = read_user_channel(audio_path)
        asr_result = asr_http(_wav_bytes(user_audio), args.asr_url)
        transcript = verbatim_normalize(
            str(asr_result.get("persian") or asr_result.get("text") or "")
        )
        if asr_result.get("error") or not transcript:
            raise RuntimeError(f"ASR failure at locked row {row_index}")
        runtime_rows.append(
            {
                "row_index": row_index,
                "audio_path": audio_path,
                "audio_format": audio_format,
                "transcript": transcript,
                "reference": str(source_rows[row_index]["response_text"]).strip(),
            }
        )

    base_responder = _load_responder()
    base_backend = base_responder.backend
    for row in runtime_rows:
        row["base"] = base_responder.reply(row["transcript"])
        row["base_fallback"] = base_responder.last_fallback_used
    _unload_responder(base_responder)

    lora_responder = _load_responder(adapter_path, adapter_hash)
    lora_backend = lora_responder.backend
    for row in runtime_rows:
        row["lora"] = lora_responder.reply(row["transcript"])
        row["lora_fallback"] = lora_responder.last_fallback_used
        audio, backend = synthesize(row["lora"])
        row["lora_tts_backend"] = backend
        row["lora_audio_samples"] = int(audio.size)
        row["lora_audio_rms"] = float(np.sqrt(np.mean(np.square(audio))))
    _unload_responder(lora_responder)

    samples: list[dict[str, Any]] = []
    for row in runtime_rows:
        arm_payloads = {name: row[name] for name in ARMS}
        arm_reports: dict[str, Any] = {}
        arm_order = blind_arm_order(int(row["row_index"]))
        for arm_name in arm_order:
            scores: list[dict[str, int]] = []
            metadata: list[dict[str, Any]] = []
            errors: list[str | None] = []
            for _ in range(REPEATS):
                try:
                    score, call_metadata = judge_pair(
                        args.judge_url,
                        transcript=row["transcript"],
                        reply=arm_payloads[arm_name],
                        timeout=args.judge_timeout_seconds,
                    )
                    scores.append(score)
                    metadata.append(call_metadata)
                    errors.append(None)
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            arm_reports[arm_name] = {
                "reply_sha256": sha256_text(arm_payloads[arm_name]),
                "script_statistics": script_statistics(arm_payloads[arm_name]),
                "scores": scores,
                "judge_metadata": metadata,
                "judge_errors": errors,
                "repeat_exact_agreement": len(scores) == REPEATS and scores[0] == scores[1],
            }
        samples.append(
            {
                "manifest_row_index": row["row_index"],
                "source_audio_sha256": sha256_file(row["audio_path"]),
                "input_transcript_sha256": sha256_text(row["transcript"]),
                "selected_user_channel": row["audio_format"]["selected_user_channel"],
                "arm_order_sha256": sha256_text("\n".join(arm_order)),
                "arms": arm_reports,
                "base_fallback_used": row["base_fallback"],
                "lora_fallback_used": row["lora_fallback"],
                "lora_tts_backend": row["lora_tts_backend"],
                "lora_audio_samples": row["lora_audio_samples"],
                "lora_audio_rms": round(row["lora_audio_rms"], 8),
            }
        )

    summary = summarize(samples)
    validity = {
        "training_report_completed": training_report.get("status") == "completed",
        "training_report_all_validity_gates_passed": all(
            (training_report.get("validity_requirements") or {}).values()
        ),
        "training_protocol_hash_current": (
            (training_report.get("artifacts") or {}).get("protocol_sha256")
            == sha256_file(protocol_path)
        ),
        "training_script_hash_current": (
            (training_report.get("artifacts") or {}).get("trainer_sha256")
            == sha256_file(trainer_path)
        ),
        "adapter_tree_hash_current": adapter_hash == expected_adapter_hash,
        "adapter_file_manifest_nonempty": bool(adapter_files),
        "source_manifest_hash_exact": sha256_file(source_path) == SOURCE_MANIFEST_SHA256,
        "validation_manifest_hash_exact": sha256_file(val_path) == VAL_EXPORT_SHA256,
        "base_revision_exact": qwen_revision(BASE_MODEL) == BASE_REVISION,
        "panel_indices_exact": tuple(sample["manifest_row_index"] for sample in samples)
        == FINAL_PANEL_INDICES,
        "previously_observed_rows_excluded": not bool(
            set(FINAL_PANEL_INDICES) & set(PREVIOUSLY_OBSERVED_INDICES)
        ),
        "exactly_forty_rows_measured": len(samples) == EXPECTED_FINAL_ROWS,
        "all_240_judge_calls_valid": summary["judge_calls_valid"] == EXPECTED_JUDGE_CALLS
        and summary["judge_calls_failed"] == 0,
        "all_judge_responses_finished": all(
            metadata["finish_reason"] == "stop" and metadata["response_model"] == JUDGE_SERVED_MODEL
            for sample in samples
            for arm in sample["arms"].values()
            for metadata in arm["judge_metadata"]
        ),
        "base_backend_exact": base_backend == BASE_MODEL,
        "lora_backend_exact": lora_backend == f"{BASE_MODEL}+LoRA:{adapter_hash[:16]}",
        "no_base_fallback": all(sample["base_fallback_used"] is False for sample in samples),
        "no_lora_fallback": all(sample["lora_fallback_used"] is False for sample in samples),
        "all_base_replies_persian_script": all(
            sample["arms"]["base"]["script_statistics"]["persian_letter_fraction"] >= 0.8
            for sample in samples
        ),
        "all_lora_replies_persian_script": all(
            sample["arms"]["lora"]["script_statistics"]["persian_letter_fraction"] >= 0.8
            for sample in samples
        ),
        "piper_backend_exact_for_all_lora_replies": all(
            sample["lora_tts_backend"] == "piper" for sample in samples
        ),
        "all_lora_audio_nonempty_finite": all(
            sample["lora_audio_samples"] >= 1600
            and math.isfinite(sample["lora_audio_rms"])
            and sample["lora_audio_rms"] >= 1e-3
            for sample in samples
        ),
        "audited_user_channel_used": all(
            sample["selected_user_channel"] == 1 for sample in samples
        ),
        "test_not_used": True,
        **judge_identity_validity,
    }
    mechanically_valid = all(validity.values())
    gate = engineering_gate(summary, mechanically_valid)
    passed = all(gate.values())
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else ("valid_negative" if mechanically_valid else "invalid"),
        "evidence_class": "automatic_same_family_llm_as_judge_paired_responder_proxy",
        "claim": {
            "measurement_valid": mechanically_valid,
            "predeclared_automatic_engineering_gate_passed": passed,
            "compact_responder_automatic_semantic_improvement": passed,
            "human_semantic_result": False,
            "independent_dialogue_benchmark_result": False,
            "factuality_or_safety_result": False,
            "physical_4090_result": False,
            "test_result": False,
        },
        "panel": {
            "split": "validation",
            "split_unit": "source_session_id",
            "manifest_rows": len(exported_rows),
            "indices": list(FINAL_PANEL_INDICES),
            "sample_count": len(samples),
            "previously_observed_indices_excluded": list(PREVIOUSLY_OBSERVED_INDICES),
            "test_used": False,
        },
        "responders": {
            "base": {"backend": base_backend, "revision": qwen_revision(BASE_MODEL)},
            "lora": {
                "backend": lora_backend,
                "base_revision": qwen_revision(BASE_MODEL),
                "selected_epoch": (training_report.get("selection") or {}).get("selected_epoch"),
                "adapter_tree_sha256": adapter_hash,
            },
            "reference": {
                "type": "automatically_aligned_next_speaker_turn",
                "human_verified": False,
                "gold_answer": False,
            },
        },
        "judge": {
            **container_identity,
            **service_identity,
            "endpoint_scope": "loopback",
            "temperature": 0,
            "top_p": 1,
            "seed": SEED,
            "max_tokens": 128,
            "reasoning_enabled": False,
            "response_format": "json_object",
            "repeats_per_arm_row": REPEATS,
            "blinded_arm_order": True,
            "system_prompt_sha256": sha256_text(JUDGE_SYSTEM_PROMPT),
            "system_fingerprints": sorted(
                {
                    str(metadata["system_fingerprint"])
                    for sample in samples
                    for arm in sample["arms"].values()
                    for metadata in arm["judge_metadata"]
                }
            ),
        },
        "thresholds": {
            "lora_relevance_mean_min": MIN_RELEVANCE_MEAN,
            "lora_coherence_mean_min": MIN_COHERENCE_MEAN,
            "lora_minus_base_relevance_mean_min": MIN_RELEVANCE_GAIN,
            "lora_minus_base_coherence_mean_min": MIN_COHERENCE_GAIN,
            "lora_relevance_win_rate_min": MIN_RELEVANCE_WIN_RATE,
            "lora_relevance_at_least_two_rate_min": MIN_RELEVANCE_AT_LEAST_TWO_RATE,
        },
        "aggregate": summary,
        "automatic_engineering_gate": gate,
        "samples": samples,
        "validity_requirements": validity,
        "hardware": gpu_inventory(),
        "artifacts": {
            "source_manifest_sha256": sha256_file(source_path),
            "validation_manifest_sha256": sha256_file(val_path),
            "training_report_sha256": sha256_file(training_report_path),
            "adapter_tree_sha256": adapter_hash,
            "protocol_sha256": sha256_file(protocol_path),
            "trainer_sha256": sha256_file(trainer_path),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "piper_model_sha256": sha256_file(piper_path),
        },
        "limitations": [
            "This is an automatic LLM-as-judge proxy, not human semantic evaluation.",
            "Judge and responder use different weights but are from the Qwen family.",
            "The rubric has not been calibrated against independent Persian ratings.",
            "Reference turns are automatically aligned and not human-verified gold answers.",
            "The panel excludes previously observed rows but contains only three source sessions.",
            "Scores do not establish factuality, safety, naturalness, or population usefulness.",
            "The source license is unverified; the adapter remains local and unredistributed.",
        ],
        "privacy": {
            "plaintext_input_transcripts_stored": False,
            "plaintext_base_replies_stored": False,
            "plaintext_lora_replies_stored": False,
            "plaintext_reference_replies_stored": False,
            "judge_rationales_stored": False,
            "audio_stored": False,
            "project_text_sent_beyond_loopback": False,
            "adapter_committed_or_uploaded": False,
        },
    }
    atomic_write(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": summary["rows"],
                "judge_calls_valid": summary["judge_calls_valid"],
                "base_relevance": summary["base"]["relevance"]["mean"],
                "lora_relevance": summary["lora"]["relevance"]["mean"],
                "lora_coherence": summary["lora"]["coherence"]["mean"],
                "relevance_gain": summary["paired"]["lora_minus_base_relevance_mean"],
                "gate_passed": passed,
                "output": output_path.relative_to(ROOT).as_posix(),
            }
        )
    )
    return 0 if mechanically_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
