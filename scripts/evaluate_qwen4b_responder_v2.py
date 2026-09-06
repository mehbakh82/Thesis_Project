#!/usr/bin/env python3
"""Run frozen Qwen3-4B prompt-v2 eligibility or unlocked final-test evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict
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
from scripts.evaluate_qwen4b_responder import (  # noqa: E402
    MODEL_BYTES,
    MODEL_FILE_COUNT,
    MODEL_LICENSE,
    MODEL_LICENSE_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    MODEL_TREE_SHA256,
    PIPER_MODEL_SHA256,
)
from scripts.evaluate_responder_lora import judge_pair, load_jsonl  # noqa: E402
from scripts.train_responder_lora import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    SOURCE_MANIFEST_SHA256,
    VAL_EXPORT_SHA256,
    adapter_tree,
)
from thesis_s2s.data.verbatim import verbatim_normalize  # noqa: E402
from thesis_s2s.metrics import gpu_inventory  # noqa: E402
from thesis_s2s.runtime.cascade import TextResponder, _wav_bytes, asr_http  # noqa: E402
from thesis_s2s.runtime.tts import synthesize  # noqa: E402

SEED = 20260906
FINAL_SELECTION_SEED = "20260906-qwen4b-v2-final-test"
DEV_INDICES = (23, 24, 33, 40, 47, 54, 122)
FINAL_INDICES = (
    1,
    4,
    7,
    9,
    12,
    13,
    17,
    30,
    31,
    32,
    33,
    34,
    35,
    37,
    40,
    42,
    74,
    79,
    90,
    92,
    98,
    106,
    109,
    122,
    129,
    133,
    141,
    142,
    148,
    160,
    167,
    169,
    173,
    183,
    185,
    191,
    192,
    194,
    200,
    202,
)
EXPECTED_VAL_ROWS = 131
EXPECTED_TEST_ROWS = 204
TEST_EXPORT_SHA256 = "44d5912201ed359dabe3c026b6ae605b3bf946538e83116f57514448ed0794fe"
TEST_ID_SEQUENCE_SHA256 = "b0de78cf27a112dbe3d7775940e6901e017f8d6296b9d537d75380a71fcb73dc"
TEST_SESSION_HASH_COUNTS = {
    "150d88bddf0bf3c9135536c8ef45630499d880af7c60333b1ef1ce2718e2fe61": 84,
    "393cb4fa713cb1c954d2c180bbb3ec9f2003b3ed95d91aacdb1e864a66ed4af3": 31,
    "544313ff018cf2cfc4d43f86d7236e2021105bc9c029b9362b1234d6b5b817fe": 34,
    "4aa1bfbd43cd94793c9ef4682a202b6f11813593e412c1cc51f63e4e34131c59": 13,
    "21bf9dbf9e64cf12ae0ed1a2a3b89d185348cd41d5b67e18fe5671e420cfa44c": 42,
}
ARMS = ("base", "qwen4b_v2", "reference")
REPEATS = 2
PRIMARY_PROMPT = (
    "متن کاربر ممکن است خروجی ناقص گفتاربه‌متن باشد. منظور اصلی قابل‌فهم را تشخیص بده و "
    "دقیقاً یک جمله کامل و روان فارسی با حداکثر بیست‌وپنج واژه بنویس که مستقیماً به همان "
    "منظور پاسخ دهد. متن کاربر را تکرار نکن، جمله را نیمه‌تمام نگذار، حرف لاتین، فهرست یا "
    "توضیح حاشیه‌ای ننویس. اگر هیچ منظوری قابل‌فهم نیست، در یک جمله درخواست تکرار کن."
)
RETRY_PROMPT = (
    "فقط یک جمله کامل، کوتاه، روان و مرتبط با منظور اصلی کاربر به خط فارسی بنویس. "
    "هیچ حرف لاتین، کد، فهرست، تکرار ورودی یا جمله نیمه‌تمام ننویس."
)
MIN_RELEVANCE_MEAN = 2.0
MIN_COHERENCE_MEAN = 2.5
MIN_RELEVANCE_GAIN = 0.5
MIN_COHERENCE_GAIN = 0.0
MIN_RELEVANCE_WIN_RATE = 0.60
MIN_RELEVANCE_AT_LEAST_TWO_RATE = 0.70


def load_source_split(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == split:
                rows.append(row)
    return rows


def test_panel_indices(rows: list[dict[str, Any]]) -> tuple[int, ...]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["session_id"])].append(index)
    observed_counts = {
        sha256_text(session): len(indices) for session, indices in grouped.items()
    }
    if observed_counts != TEST_SESSION_HASH_COUNTS:
        raise ValueError("test-session hash/count drift")
    selected: list[int] = []
    for session in sorted(
        grouped,
        key=lambda value: sha256_text(FINAL_SELECTION_SEED + "\0" + value),
    ):
        candidates = sorted(
            grouped[session],
            key=lambda index: sha256_text(
                FINAL_SELECTION_SEED + "\0" + str(rows[index]["utt_id"])
            ),
        )
        selected.extend(candidates[:8])
    return tuple(sorted(selected))


def arm_order(stage: str, row_index: int) -> tuple[str, ...]:
    return tuple(
        sorted(
            ARMS,
            key=lambda arm: sha256_text(f"qwen4b-v2\0{stage}\0{SEED}\0{row_index}\0{arm}"),
        )
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


def summarize(samples: list[dict[str, Any]], expected_calls: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": len(samples),
        "judge_calls_expected": expected_calls,
        "judge_calls_valid": sum(
            len(arm["scores"])
            for sample in samples
            for arm in sample["arms"].values()
        ),
    }
    summary["judge_calls_failed"] = expected_calls - summary["judge_calls_valid"]
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
            score["relevance"] for score in sample["arms"]["qwen4b_v2"]["scores"]
        ]
        wins += int(np.median(candidate) > np.median(base))
        at_least_two += int(np.median(candidate) >= 2)
    summary["paired"] = {
        "candidate_minus_base_relevance_mean": round(
            summary["qwen4b_v2"]["relevance"]["mean"]
            - summary["base"]["relevance"]["mean"],
            6,
        ),
        "candidate_minus_base_coherence_mean": round(
            summary["qwen4b_v2"]["coherence"]["mean"]
            - summary["base"]["coherence"]["mean"],
            6,
        ),
        "candidate_relevance_win_rows": wins,
        "candidate_relevance_win_rate": round(wins / max(1, len(samples)), 6),
        "candidate_relevance_at_least_two_rows": at_least_two,
        "candidate_relevance_at_least_two_rate": round(
            at_least_two / max(1, len(samples)), 6
        ),
    }
    return summary


def engineering_gate(summary: dict[str, Any], valid: bool) -> dict[str, bool]:
    return {
        "mechanically_valid": valid,
        "candidate_relevance_mean_at_least_2_0": (
            summary["qwen4b_v2"]["relevance"]["mean"] >= MIN_RELEVANCE_MEAN
        ),
        "candidate_coherence_mean_at_least_2_5": (
            summary["qwen4b_v2"]["coherence"]["mean"] >= MIN_COHERENCE_MEAN
        ),
        "relevance_gain_at_least_0_50": (
            summary["paired"]["candidate_minus_base_relevance_mean"]
            >= MIN_RELEVANCE_GAIN
        ),
        "coherence_gain_nonnegative": (
            summary["paired"]["candidate_minus_base_coherence_mean"]
            >= MIN_COHERENCE_GAIN
        ),
        "relevance_win_rate_at_least_0_60": (
            summary["paired"]["candidate_relevance_win_rate"]
            >= MIN_RELEVANCE_WIN_RATE
        ),
        "relevance_at_least_two_rate_at_least_0_70": (
            summary["paired"]["candidate_relevance_at_least_two_rate"]
            >= MIN_RELEVANCE_AT_LEAST_TWO_RATE
        ),
    }


class PromptV2Responder:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16
        ).to("cuda").eval()
        self.backend = f"{MODEL_NAME}@{MODEL_REVISION}:prompt-v2"
        self.last_attempts = 0
        self.last_language_retry_used = False
        self.cuda_allocated_bytes = int(torch.cuda.memory_allocated())

    def reply(self, user_text: str) -> str:
        import torch

        self.last_attempts = 0
        self.last_language_retry_used = False
        for attempt, prompt in enumerate((PRIMARY_PROMPT, RETRY_PROMPT), start=1):
            self.last_attempts = attempt
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
            self.last_language_retry_used = attempt == 1
        raise RuntimeError("prompt-v2 response failed Persian-script constraint twice")

    def unload(self) -> None:
        import torch

        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()


def verify_final_unlocked(dev_path: Path, protocol_path: Path) -> dict[str, Any]:
    if not dev_path.is_file():
        raise FileNotFoundError("final test locked: Stage-A report is absent")
    report = json.loads(dev_path.read_text(encoding="utf-8"))
    if (
        report.get("stage") != "development_eligibility"
        or report.get("status") != "passed"
        or not all((report.get("automatic_engineering_gate") or {}).values())
        or not all((report.get("validity_requirements") or {}).values())
        or (report.get("artifacts") or {}).get("protocol_sha256")
        != sha256_file(protocol_path)
        or (report.get("artifacts") or {}).get("evaluator_sha256")
        != sha256_file(Path(__file__).resolve())
    ):
        raise RuntimeError("final test locked: Stage-A evidence is not an exact pass")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "final"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/processed/manifests/conversations.jsonl"),
    )
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path("docs/QWEN4B_CASCADE_V2_PROTOCOL.md")
    )
    parser.add_argument(
        "--development-report",
        type=Path,
        default=Path("results/eval/qwen4b_responder_v2_development_proxy.json"),
    )
    parser.add_argument(
        "--piper-model", type=Path, default=Path("models/piper/fa_IR-mana-medium.onnx")
    )
    parser.add_argument("--asr-url", default="http://127.0.0.1:8090")
    parser.add_argument("--asr-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8003")
    parser.add_argument("--judge-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--judge-container", default=JUDGE_CONTAINER)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    stage = args.stage
    is_final = stage == "final"
    stage_name = "final_test" if is_final else "development_eligibility"
    split = "test" if is_final else "val"
    indices = FINAL_INDICES if is_final else DEV_INDICES
    expected_rows = EXPECTED_TEST_ROWS if is_final else EXPECTED_VAL_ROWS
    expected_export_hash = TEST_EXPORT_SHA256 if is_final else VAL_EXPORT_SHA256
    expected_calls = len(indices) * len(ARMS) * REPEATS
    export_argument = args.export_manifest or Path(
        f"data/processed/moshi_finetune/{split}.jsonl"
    )
    output_argument = args.out or Path(
        "results/eval/qwen4b_responder_v2_final_test_proxy.json"
        if is_final
        else "results/eval/qwen4b_responder_v2_development_proxy.json"
    )
    model_dir = args.model_dir.resolve()
    source_path = (ROOT / args.source_manifest).resolve()
    export_path = (ROOT / export_argument).resolve()
    protocol_path = (ROOT / args.protocol).resolve()
    dev_report_path = (ROOT / args.development_report).resolve()
    piper_path = (ROOT / args.piper_model).resolve()
    piper_config_path = piper_path.with_suffix(piper_path.suffix + ".json")
    output_path = (ROOT / output_argument).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    for required in (model_dir, source_path, protocol_path, piper_path, piper_config_path):
        if not required.exists():
            raise FileNotFoundError(required)
    unlock_report = verify_final_unlocked(dev_report_path, protocol_path) if is_final else None
    if not export_path.is_file():
        raise FileNotFoundError(export_path)
    if export_path.name != f"{split}.jsonl":
        raise ValueError("stage/export split mismatch")

    model_files, model_tree_hash = adapter_tree(model_dir)
    model_bytes = sum(int(row["bytes"]) for row in model_files)
    metadata_path = model_dir / ".cache/huggingface/download/config.json.metadata"
    model_revision = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
    model_identity_valid = (
        model_tree_hash == MODEL_TREE_SHA256
        and len(model_files) == MODEL_FILE_COUNT
        and model_bytes == MODEL_BYTES
        and model_revision == MODEL_REVISION
        and sha256_file(model_dir / "LICENSE") == MODEL_LICENSE_SHA256
    )
    if not model_identity_valid:
        raise RuntimeError("Qwen3-4B model identity drift")
    if sha256_file(source_path) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("source manifest hash drift")
    if sha256_file(export_path) != expected_export_hash:
        raise RuntimeError("stage export-manifest hash drift")
    if sha256_file(piper_path) != PIPER_MODEL_SHA256:
        raise RuntimeError("Piper model hash drift")

    source_rows = load_source_split(source_path, split)
    exported_rows = load_jsonl(export_path)
    if len(source_rows) != expected_rows or len(exported_rows) != expected_rows:
        raise RuntimeError("stage row-count drift")
    source_pair_ids: list[str] = []
    for source, exported in zip(source_rows, exported_rows, strict=True):
        sidecar = json.loads(Path(str(exported["path"])).with_suffix(".json").read_text())
        pair_id = str(sidecar.get("source_pair_id") or "")
        source_pair_ids.append(pair_id)
        if pair_id != source.get("utt_id"):
            raise RuntimeError("source/exported row order drift")
    id_sequence_hash = sha256_text("\n".join(source_pair_ids))
    if is_final:
        if id_sequence_hash != TEST_ID_SEQUENCE_SHA256:
            raise RuntimeError("test ID-sequence drift")
        if test_panel_indices(source_rows) != FINAL_INDICES:
            raise RuntimeError("test-panel selection drift")

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
    for row_index in indices:
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

    candidate = PromptV2Responder(model_dir)
    candidate_cuda_bytes = candidate.cuda_allocated_bytes
    for row in runtime_rows:
        row["qwen4b_v2"] = candidate.reply(row["transcript"])
        row["candidate_attempts"] = candidate.last_attempts
        row["candidate_retry"] = candidate.last_language_retry_used
        audio, backend = synthesize(row["qwen4b_v2"])
        row["candidate_tts_backend"] = backend
        row["candidate_audio_samples"] = int(audio.size)
        row["candidate_audio_rms"] = float(np.sqrt(np.mean(np.square(audio))))
    candidate.unload()

    samples: list[dict[str, Any]] = []
    for row in runtime_rows:
        payloads = {name: row[name] for name in ARMS}
        order = arm_order(stage, int(row["row_index"]))
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
                "candidate_generation_attempts": row["candidate_attempts"],
                "candidate_language_retry_used": row["candidate_retry"],
                "candidate_tts_backend": row["candidate_tts_backend"],
                "candidate_audio_samples": row["candidate_audio_samples"],
                "candidate_audio_rms": round(row["candidate_audio_rms"], 8),
            }
        )

    summary = summarize(samples, expected_calls)
    validity = {
        "stage_prerequisite_passed": not is_final or unlock_report is not None,
        "source_manifest_hash_exact": sha256_file(source_path) == SOURCE_MANIFEST_SHA256,
        "stage_export_hash_exact": sha256_file(export_path) == expected_export_hash,
        "source_export_order_exact": all(
            pair_id == source["utt_id"]
            for pair_id, source in zip(source_pair_ids, source_rows, strict=True)
        ),
        "base_revision_exact": qwen_revision(BASE_MODEL) == BASE_REVISION,
        "qwen4b_model_identity_exact": model_identity_valid,
        "panel_indices_exact": tuple(sample["manifest_row_index"] for sample in samples)
        == indices,
        "expected_rows_measured": len(samples) == len(indices),
        "all_expected_judge_calls_valid": summary["judge_calls_valid"] == expected_calls
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
        "all_candidate_replies_persian_script": all(
            sample["arms"]["qwen4b_v2"]["script_statistics"]["persian_letter_fraction"]
            >= 0.8
            for sample in samples
        ),
        "piper_backend_exact_for_all_candidate_replies": all(
            sample["candidate_tts_backend"] == "piper" for sample in samples
        ),
        "all_candidate_audio_nonempty_finite": all(
            sample["candidate_audio_samples"] >= 1600
            and math.isfinite(sample["candidate_audio_rms"])
            and sample["candidate_audio_rms"] >= 1e-3
            for sample in samples
        ),
        "audited_user_channel_used": all(
            sample["selected_user_channel"] == 1 for sample in samples
        ),
        **judge_identity_validity,
    }
    mechanically_valid = all(validity.values())
    gate = engineering_gate(summary, mechanically_valid)
    passed = all(gate.values())
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage_name,
        "status": "passed" if passed else ("valid_negative" if mechanically_valid else "invalid"),
        "evidence_class": (
            "automatic_same_family_llm_as_judge_qwen4b_prompt_v2_final_test"
            if is_final
            else "automatic_same_family_llm_as_judge_qwen4b_prompt_v2_eligibility"
        ),
        "claim": {
            "measurement_valid": mechanically_valid,
            "predeclared_automatic_engineering_gate_passed": passed,
            "final_test_unlocked": passed if not is_final else True,
            "automatic_final_test_result": is_final and mechanically_valid,
            "automatic_final_test_gate_passed": is_final and passed,
            "human_semantic_result": False,
            "independent_dialogue_benchmark_result": False,
            "factuality_or_safety_result": False,
            "physical_4090_result": False,
        },
        "panel": {
            "split": split,
            "split_unit": "source_session_id",
            "manifest_rows": len(exported_rows),
            "indices": list(indices),
            "sample_count": len(samples),
            "source_pair_id_sequence_sha256": id_sequence_hash,
        },
        "responders": {
            "base": {"backend": BASE_MODEL, "revision": qwen_revision(BASE_MODEL)},
            "qwen4b_v2": {
                "backend": f"{MODEL_NAME}@{MODEL_REVISION}:prompt-v2",
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
            "candidate_relevance_mean_min": MIN_RELEVANCE_MEAN,
            "candidate_coherence_mean_min": MIN_COHERENCE_MEAN,
            "candidate_minus_base_relevance_mean_min": MIN_RELEVANCE_GAIN,
            "candidate_minus_base_coherence_mean_min": MIN_COHERENCE_GAIN,
            "candidate_relevance_win_rate_min": MIN_RELEVANCE_WIN_RATE,
            "candidate_relevance_at_least_two_rate_min": MIN_RELEVANCE_AT_LEAST_TWO_RATE,
        },
        "aggregate": summary,
        "automatic_engineering_gate": gate,
        "samples": samples,
        "validity_requirements": validity,
        "hardware": gpu_inventory(),
        "artifacts": {
            "source_manifest_sha256": sha256_file(source_path),
            "stage_export_sha256": sha256_file(export_path),
            "qwen4b_tree_sha256": model_tree_hash,
            "development_report_sha256": (
                sha256_file(dev_report_path) if is_final else None
            ),
            "protocol_sha256": sha256_file(protocol_path),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "piper_model_sha256": sha256_file(piper_path),
        },
        "limitations": [
            "This is an automatic LLM-as-judge proxy, not human semantic evaluation.",
            "Judge and candidate use different weights but are from the Qwen family.",
            "The rubric has not been calibrated against independent Persian ratings.",
            "Reference turns are automatically aligned and not human-verified gold answers.",
            "Scores do not establish factuality, safety, naturalness, or population usefulness.",
            "Physical RTX 4090 coexistence and browser latency remain unmeasured.",
            (
                "Test metadata/text lengths were aggregate-audited before freeze, but no test "
                "model outcome was used before Stage-A eligibility passed."
                if is_final
                else "Seven residual validation rows are a small engineering eligibility panel."
            ),
        ],
        "privacy": {
            "plaintext_input_transcripts_stored": False,
            "plaintext_base_replies_stored": False,
            "plaintext_candidate_replies_stored": False,
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
                "stage": stage_name,
                "status": report["status"],
                "rows": summary["rows"],
                "judge_calls_valid": summary["judge_calls_valid"],
                "base_relevance": summary["base"]["relevance"]["mean"],
                "candidate_relevance": summary["qwen4b_v2"]["relevance"]["mean"],
                "candidate_coherence": summary["qwen4b_v2"]["coherence"]["mean"],
                "relevance_gain": summary["paired"][
                    "candidate_minus_base_relevance_mean"
                ],
                "gate_passed": passed,
                "output": output_path.relative_to(ROOT).as_posix(),
            }
        )
    )
    return 0 if mechanically_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
