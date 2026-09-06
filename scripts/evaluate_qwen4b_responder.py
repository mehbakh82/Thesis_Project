#!/usr/bin/env python3
"""Evaluate the frozen local Qwen3-4B cascade responder on unused validation rows."""

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
    inspect_judge_container,
    judge_service_identity,
)
from scripts.evaluate_cascade_validation import read_user_channel  # noqa: E402
from scripts.evaluate_responder_lora import (  # noqa: E402
    judge_pair,
    load_jsonl,
    load_validation_source_rows,
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
EXPECTED_ROWS = 40
REPEATS = 2
ARMS = ("base", "qwen4b", "reference")
EXPECTED_JUDGE_CALLS = EXPECTED_ROWS * len(ARMS) * REPEATS
PRIOR_INDICES = (
    0,
    16,
    20,
    21,
    28,
    29,
    30,
    31,
    32,
    34,
    36,
    38,
    39,
    41,
    42,
    46,
    48,
    49,
    51,
    52,
    55,
    58,
    59,
    65,
    81,
    85,
    86,
    87,
    89,
    90,
    91,
    93,
    94,
    97,
    102,
    104,
    105,
    106,
    109,
    111,
    113,
    114,
    117,
    119,
    120,
    123,
    124,
    127,
    130,
)
PANEL_INDICES = (
    17,
    18,
    19,
    22,
    25,
    26,
    27,
    35,
    37,
    43,
    44,
    45,
    50,
    53,
    56,
    57,
    82,
    83,
    84,
    88,
    92,
    95,
    96,
    98,
    99,
    100,
    101,
    103,
    107,
    108,
    110,
    112,
    115,
    116,
    118,
    121,
    125,
    126,
    128,
    129,
)
RESIDUAL_INDICES = (23, 24, 33, 40, 47, 54, 122)
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_TREE_SHA256 = "cde447f1326f10c4126061914c57c3664551649286ad6411bffe3d1aa3e3b978"
MODEL_FILE_COUNT = 28
MODEL_BYTES = 8_060_919_167
MODEL_LICENSE = "Apache-2.0"
MODEL_LICENSE_SHA256 = "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
PIPER_MODEL_SHA256 = "e390c0e74ba71fd97c49ba662ee0c6e1724b462ba2d4561698af4f564840f126"
PRIMARY_PROMPT = (
    "شما دستیار گفت‌وگوی صوتی فارسی هستید. به پرسش یا منظور اصلی کاربر، مستقیم، "
    "مرتبط، طبیعی و حداکثر در دو جمله کوتاه پاسخ بده. فقط از خط فارسی استفاده کن "
    "و هیچ حرف لاتین، فهرست یا توضیح اضافه ننویس."
)
RETRY_PROMPT = (
    "فقط در یک یا دو جمله کوتاه فارسی به منظور اصلی کاربر پاسخ مستقیم و مرتبط بده. "
    "هیچ حرف لاتین، کد، فهرست یا توضیح حاشیه‌ای ننویس."
)
MIN_RELEVANCE_MEAN = 2.0
MIN_COHERENCE_MEAN = 2.5
MIN_RELEVANCE_GAIN = 0.5
MIN_COHERENCE_GAIN = 0.0
MIN_RELEVANCE_WIN_RATE = 0.60
MIN_RELEVANCE_AT_LEAST_TWO_RATE = 0.70


def expected_panel_indices(source_rows: list[dict[str, Any]]) -> tuple[int, ...]:
    sessions = sorted({str(row["session_id"]) for row in source_rows}, key=session_order_key)
    if len(sessions) != 5:
        raise ValueError("expected exactly five validation sessions")
    if tuple(sha256_text(session) for session in sessions[:2]) != DEV_SESSION_HASHES:
        raise ValueError("development-session drift")
    final_sessions = set(sessions[2:])
    prior = set(PRIOR_INDICES)
    candidates = [
        index
        for index, row in enumerate(source_rows)
        if str(row["session_id"]) in final_sessions and index not in prior
    ]
    candidates.sort(
        key=lambda index: sha256_text(
            SPLIT_SEED + "\0" + str(source_rows[index]["utt_id"])
        )
    )
    return tuple(sorted(candidates[:EXPECTED_ROWS]))


def arm_order(row_index: int) -> tuple[str, ...]:
    return tuple(
        sorted(ARMS, key=lambda arm: sha256_text(f"qwen4b\0{SEED}\0{row_index}\0{arm}"))
    )


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
            len(arm["scores"])
            for sample in samples
            for arm in sample["arms"].values()
        ),
    }
    summary["judge_calls_failed"] = (
        summary["judge_calls_expected"] - summary["judge_calls_valid"]
    )
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
            and sample["arms"][arm_name]["scores"][0]
            == sample["arms"][arm_name]["scores"][1]
            for sample in samples
        )
        arm_summary["repeat_exact_agreement_rate"] = round(
            arm_summary["repeat_exact_agreement_rows"] / max(1, len(samples)), 6
        )
        summary[arm_name] = arm_summary
    wins = 0
    at_least_two = 0
    for sample in samples:
        base = [score["relevance"] for score in sample["arms"]["base"]["scores"]]
        candidate = [
            score["relevance"] for score in sample["arms"]["qwen4b"]["scores"]
        ]
        wins += int(np.median(candidate) > np.median(base))
        at_least_two += int(np.median(candidate) >= 2)
    summary["paired"] = {
        "qwen4b_minus_base_relevance_mean": round(
            summary["qwen4b"]["relevance"]["mean"]
            - summary["base"]["relevance"]["mean"],
            6,
        ),
        "qwen4b_minus_base_coherence_mean": round(
            summary["qwen4b"]["coherence"]["mean"]
            - summary["base"]["coherence"]["mean"],
            6,
        ),
        "qwen4b_relevance_win_rows": wins,
        "qwen4b_relevance_win_rate": round(wins / max(1, len(samples)), 6),
        "qwen4b_relevance_at_least_two_rows": at_least_two,
        "qwen4b_relevance_at_least_two_rate": round(
            at_least_two / max(1, len(samples)), 6
        ),
    }
    return summary


def engineering_gate(summary: dict[str, Any], valid: bool) -> dict[str, bool]:
    return {
        "mechanically_valid": valid,
        "qwen4b_relevance_mean_at_least_2_0": (
            summary["qwen4b"]["relevance"]["mean"] >= MIN_RELEVANCE_MEAN
        ),
        "qwen4b_coherence_mean_at_least_2_5": (
            summary["qwen4b"]["coherence"]["mean"] >= MIN_COHERENCE_MEAN
        ),
        "relevance_gain_at_least_0_50": (
            summary["paired"]["qwen4b_minus_base_relevance_mean"] >= MIN_RELEVANCE_GAIN
        ),
        "coherence_gain_nonnegative": (
            summary["paired"]["qwen4b_minus_base_coherence_mean"] >= MIN_COHERENCE_GAIN
        ),
        "relevance_win_rate_at_least_0_60": (
            summary["paired"]["qwen4b_relevance_win_rate"] >= MIN_RELEVANCE_WIN_RATE
        ),
        "relevance_at_least_two_rate_at_least_0_70": (
            summary["paired"]["qwen4b_relevance_at_least_two_rate"]
            >= MIN_RELEVANCE_AT_LEAST_TWO_RATE
        ),
    }


class Qwen4BResponder:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16
        ).to("cuda").eval()
        self.backend = f"{MODEL_NAME}@{MODEL_REVISION}"
        self.last_attempts = 0
        self.last_language_retry_used = False
        self.last_error: str | None = None
        self.cuda_allocated_bytes = int(torch.cuda.memory_allocated())

    def reply(self, user_text: str) -> str:
        import torch

        self.last_attempts = 0
        self.last_language_retry_used = False
        self.last_error = None
        for index, prompt in enumerate((PRIMARY_PROMPT, RETRY_PROMPT)):
            self.last_attempts += 1
            encoded = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_text},
                ],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                output = self.model.generate(**encoded, max_new_tokens=64, do_sample=False)
            answer = str(
                self.tokenizer.decode(
                    output[0, encoded["input_ids"].shape[-1] :], skip_special_tokens=True
                )
            ).strip()
            if answer and script_statistics(answer)["persian_letter_fraction"] >= 0.8:
                return answer
            self.last_error = "response failed Persian-script constraint"
            self.last_language_retry_used = index == 0
        raise RuntimeError(self.last_error)

    def unload(self) -> None:
        import torch

        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/processed/manifests/conversations.jsonl"),
    )
    parser.add_argument(
        "--val-manifest", type=Path, default=Path("data/processed/moshi_finetune/val.jsonl")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("docs/QWEN4B_CASCADE_PROTOCOL.md")
    )
    parser.add_argument(
        "--piper-model", type=Path, default=Path("models/piper/fa_IR-mana-medium.onnx")
    )
    parser.add_argument("--asr-url", default="http://127.0.0.1:8090")
    parser.add_argument("--asr-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8003")
    parser.add_argument("--judge-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--judge-container", default=JUDGE_CONTAINER)
    parser.add_argument(
        "--out", type=Path, default=Path("results/eval/qwen4b_responder_semantic_proxy.json")
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    source_path = (ROOT / args.source_manifest).resolve()
    val_path = (ROOT / args.val_manifest).resolve()
    protocol_path = (ROOT / args.protocol).resolve()
    piper_path = (ROOT / args.piper_model).resolve()
    piper_config_path = piper_path.with_suffix(piper_path.suffix + ".json")
    output_path = (ROOT / args.out).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    if "test" in val_path.name.lower() or "test" in val_path.parent.name.lower():
        raise ValueError("test manifests are forbidden")
    for required in (
        model_dir,
        source_path,
        val_path,
        protocol_path,
        piper_path,
        piper_config_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    model_files, model_tree_hash = adapter_tree(model_dir)
    model_bytes = sum(int(row["bytes"]) for row in model_files)
    metadata_path = model_dir / ".cache/huggingface/download/config.json.metadata"
    model_revision = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if (
        model_tree_hash != MODEL_TREE_SHA256
        or len(model_files) != MODEL_FILE_COUNT
        or model_bytes != MODEL_BYTES
        or model_revision != MODEL_REVISION
        or sha256_file(model_dir / "LICENSE") != MODEL_LICENSE_SHA256
    ):
        raise RuntimeError("Qwen3-4B model identity drift")
    if sha256_file(source_path) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("source manifest hash drift")
    if sha256_file(val_path) != VAL_EXPORT_SHA256:
        raise RuntimeError("validation manifest hash drift")
    if sha256_file(piper_path) != PIPER_MODEL_SHA256:
        raise RuntimeError("Piper model hash drift")

    source_rows = load_validation_source_rows(source_path)
    exported_rows = load_jsonl(val_path)
    if len(source_rows) != EXPECTED_VAL_ROWS or len(exported_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError("validation row-count drift")
    for source, exported in zip(source_rows, exported_rows, strict=True):
        sidecar = json.loads(Path(str(exported["path"])).with_suffix(".json").read_text())
        if sidecar.get("source_pair_id") != source.get("utt_id"):
            raise RuntimeError("source/exported validation order drift")
    if expected_panel_indices(source_rows) != PANEL_INDICES:
        raise RuntimeError("panel-selection drift")
    if set(PANEL_INDICES) & set(PRIOR_INDICES):
        raise RuntimeError("previously observed row entered the panel")

    container_identity = inspect_judge_container(args.judge_container)
    service_identity = judge_service_identity(
        args.judge_url, timeout=args.judge_timeout_seconds
    )
    judge_identity_validity = {
        "judge_container_running": container_identity["running"] is True,
        "judge_image_id_exact": container_identity["image_id"] == JUDGE_IMAGE_ID,
        "judge_image_ref_exact": container_identity["image_ref"] == JUDGE_IMAGE_REF,
        "judge_container_model_root_exact": container_identity["model_root"]
        == JUDGE_MODEL_ROOT,
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
    for row_index in PANEL_INDICES:
        exported = exported_rows[row_index]
        audio_path = Path(str(exported["path"])).resolve()
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

    base_responder = TextResponder(BASE_MODEL)
    if base_responder.backend != BASE_MODEL or base_responder.model is None:
        raise RuntimeError(f"base responder failed to load: {base_responder.initialization_error}")
    for row in runtime_rows:
        row["base"] = base_responder.reply(row["transcript"])
        row["base_fallback"] = base_responder.last_fallback_used
    base_responder.model = None
    base_responder.tokenizer = None
    del base_responder
    gc.collect()
    import torch

    torch.cuda.empty_cache()

    candidate = Qwen4BResponder(model_dir)
    candidate_cuda_bytes = candidate.cuda_allocated_bytes
    for row in runtime_rows:
        row["qwen4b"] = candidate.reply(row["transcript"])
        row["qwen4b_attempts"] = candidate.last_attempts
        row["qwen4b_language_retry_used"] = candidate.last_language_retry_used
        audio, backend = synthesize(row["qwen4b"])
        row["qwen4b_tts_backend"] = backend
        row["qwen4b_audio_samples"] = int(audio.size)
        row["qwen4b_audio_rms"] = float(np.sqrt(np.mean(np.square(audio))))
    candidate.unload()

    samples: list[dict[str, Any]] = []
    for row in runtime_rows:
        payloads = {name: row[name] for name in ARMS}
        order = arm_order(int(row["row_index"]))
        arm_reports: dict[str, Any] = {}
        for arm_name in order:
            scores: list[dict[str, int]] = []
            metadata: list[dict[str, Any]] = []
            errors: list[str | None] = []
            for _ in range(REPEATS):
                try:
                    score, call_metadata = judge_pair(
                        args.judge_url,
                        transcript=row["transcript"],
                        reply=payloads[arm_name],
                        timeout=args.judge_timeout_seconds,
                    )
                    scores.append(score)
                    metadata.append(call_metadata)
                    errors.append(None)
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            arm_reports[arm_name] = {
                "reply_sha256": sha256_text(payloads[arm_name]),
                "script_statistics": script_statistics(payloads[arm_name]),
                "scores": scores,
                "judge_metadata": metadata,
                "judge_errors": errors,
                "repeat_exact_agreement": len(scores) == REPEATS
                and scores[0] == scores[1],
            }
        samples.append(
            {
                "manifest_row_index": row["row_index"],
                "source_audio_sha256": sha256_file(row["audio_path"]),
                "input_transcript_sha256": sha256_text(row["transcript"]),
                "selected_user_channel": row["audio_format"]["selected_user_channel"],
                "arm_order_sha256": sha256_text("\n".join(order)),
                "arms": arm_reports,
                "base_fallback_used": row["base_fallback"],
                "qwen4b_generation_attempts": row["qwen4b_attempts"],
                "qwen4b_language_retry_used": row["qwen4b_language_retry_used"],
                "qwen4b_tts_backend": row["qwen4b_tts_backend"],
                "qwen4b_audio_samples": row["qwen4b_audio_samples"],
                "qwen4b_audio_rms": round(row["qwen4b_audio_rms"], 8),
            }
        )

    summary = summarize(samples)
    validity = {
        "source_manifest_hash_exact": sha256_file(source_path) == SOURCE_MANIFEST_SHA256,
        "validation_manifest_hash_exact": sha256_file(val_path) == VAL_EXPORT_SHA256,
        "base_revision_exact": qwen_revision(BASE_MODEL) == BASE_REVISION,
        "qwen4b_revision_exact": model_revision == MODEL_REVISION,
        "qwen4b_tree_hash_exact": model_tree_hash == MODEL_TREE_SHA256,
        "qwen4b_file_count_exact": len(model_files) == MODEL_FILE_COUNT,
        "qwen4b_bytes_exact": model_bytes == MODEL_BYTES,
        "qwen4b_license_exact": sha256_file(model_dir / "LICENSE")
        == MODEL_LICENSE_SHA256,
        "panel_indices_exact": tuple(sample["manifest_row_index"] for sample in samples)
        == PANEL_INDICES,
        "prior_rows_excluded": not bool(set(PANEL_INDICES) & set(PRIOR_INDICES)),
        "exactly_forty_rows_measured": len(samples) == EXPECTED_ROWS,
        "all_240_judge_calls_valid": summary["judge_calls_valid"]
        == EXPECTED_JUDGE_CALLS
        and summary["judge_calls_failed"] == 0,
        "all_judge_responses_finished": all(
            metadata["finish_reason"] == "stop"
            and metadata["response_model"] == JUDGE_SERVED_MODEL
            for sample in samples
            for arm in sample["arms"].values()
            for metadata in arm["judge_metadata"]
        ),
        "no_base_fallback": all(sample["base_fallback_used"] is False for sample in samples),
        "all_base_replies_persian_script": all(
            sample["arms"]["base"]["script_statistics"]["persian_letter_fraction"] >= 0.8
            for sample in samples
        ),
        "all_qwen4b_replies_persian_script": all(
            sample["arms"]["qwen4b"]["script_statistics"]["persian_letter_fraction"]
            >= 0.8
            for sample in samples
        ),
        "piper_backend_exact_for_all_qwen4b_replies": all(
            sample["qwen4b_tts_backend"] == "piper" for sample in samples
        ),
        "all_qwen4b_audio_nonempty_finite": all(
            sample["qwen4b_audio_samples"] >= 1600
            and math.isfinite(sample["qwen4b_audio_rms"])
            and sample["qwen4b_audio_rms"] >= 1e-3
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
        "evidence_class": "automatic_same_family_llm_as_judge_qwen4b_cascade_proxy",
        "claim": {
            "measurement_valid": mechanically_valid,
            "predeclared_automatic_engineering_gate_passed": passed,
            "qwen4b_automatic_semantic_improvement": passed,
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
            "indices": list(PANEL_INDICES),
            "sample_count": len(samples),
            "prior_indices_excluded": list(PRIOR_INDICES),
            "residual_unused_indices": list(RESIDUAL_INDICES),
            "test_used": False,
        },
        "responders": {
            "base": {"backend": BASE_MODEL, "revision": qwen_revision(BASE_MODEL)},
            "qwen4b": {
                "backend": f"{MODEL_NAME}@{MODEL_REVISION}",
                "revision": model_revision,
                "tree_sha256": model_tree_hash,
                "files": len(model_files),
                "bytes": model_bytes,
                "license": MODEL_LICENSE,
                "cuda_memory_allocated_bytes_after_load": candidate_cuda_bytes,
                "max_new_tokens": 64,
                "primary_prompt_sha256": sha256_text(PRIMARY_PROMPT),
                "retry_prompt_sha256": sha256_text(RETRY_PROMPT),
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
            "qwen4b_relevance_mean_min": MIN_RELEVANCE_MEAN,
            "qwen4b_coherence_mean_min": MIN_COHERENCE_MEAN,
            "qwen4b_minus_base_relevance_mean_min": MIN_RELEVANCE_GAIN,
            "qwen4b_minus_base_coherence_mean_min": MIN_COHERENCE_GAIN,
            "qwen4b_relevance_win_rate_min": MIN_RELEVANCE_WIN_RATE,
            "qwen4b_relevance_at_least_two_rate_min": MIN_RELEVANCE_AT_LEAST_TWO_RATE,
        },
        "aggregate": summary,
        "automatic_engineering_gate": gate,
        "samples": samples,
        "validity_requirements": validity,
        "hardware": gpu_inventory(),
        "artifacts": {
            "source_manifest_sha256": sha256_file(source_path),
            "validation_manifest_sha256": sha256_file(val_path),
            "qwen4b_tree_sha256": model_tree_hash,
            "protocol_sha256": sha256_file(protocol_path),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "piper_model_sha256": sha256_file(piper_path),
        },
        "limitations": [
            "This is an automatic LLM-as-judge proxy, not human semantic evaluation.",
            "Judge and candidate use different weights but are from the Qwen family.",
            "The rubric has not been calibrated against independent Persian ratings.",
            "Reference turns are automatically aligned and not human-verified gold answers.",
            "The panel contains only three source sessions.",
            "Scores do not establish factuality, safety, naturalness, or population usefulness.",
            "Physical RTX 4090 coexistence and browser latency remain unmeasured.",
        ],
        "privacy": {
            "plaintext_input_transcripts_stored": False,
            "plaintext_base_replies_stored": False,
            "plaintext_qwen4b_replies_stored": False,
            "plaintext_reference_replies_stored": False,
            "judge_rationales_stored": False,
            "audio_stored": False,
            "host_model_path_stored": False,
            "project_text_sent_beyond_loopback": False,
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
                "qwen4b_relevance": summary["qwen4b"]["relevance"]["mean"],
                "qwen4b_coherence": summary["qwen4b"]["coherence"]["mean"],
                "relevance_gain": summary["paired"]["qwen4b_minus_base_relevance_mean"],
                "gate_passed": passed,
                "output": output_path.relative_to(ROOT).as_posix(),
            }
        )
    )
    return 0 if mechanically_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
