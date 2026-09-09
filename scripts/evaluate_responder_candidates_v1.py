#!/usr/bin/env python3
"""Privacy-safe, group-disjoint Qwen responder comparison.

The plan stage freezes both panels without running a model.  Development may
select a challenger; final is hash-locked and stays closed when the incumbent
is retained.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_cascade_real_service_v4 import (  # noqa: E402
    script_statistics,
    sha256_file,
    sha256_text,
)
from scripts.evaluate_responder_lora import SCORE_KEYS, judge_pair  # noqa: E402

SOURCE_SHA256 = "aabf12268cce2854ef8b16ffd9339c6d703339c6b4c50a23a105447047a13a86"
SEED = "20260908-responder-candidates-v1"
CHANNELS = ("Digiato", "Mehran Rowshan Persian", "Tabaghe16", "Zoomit")
SESSIONS_PER_CHANNEL = 2
ROWS_PER_SESSION = 5
EXPECTED_ROWS = len(CHANNELS) * SESSIONS_PER_CHANNEL * ROWS_PER_SESSION
REPEATS = 2
CURRENT_ARM = "qwen3-4b-instruct-2507-prompt-v2"
JUDGE_MODEL = "qwen3.8-27b"
PROTOCOL = ROOT / "docs" / "RESPONDER_CANDIDATE_COMPARISON_V1.md"
MODEL_ROOT = ROOT / "models"

PROMPT = (
    "متن کاربر ممکن است خروجی ناقص گفتاربه‌متن باشد. منظور اصلی قابل‌فهم را تشخیص بده و "
    "دقیقاً یک جمله کامل و روان فارسی با حداکثر بیست‌وپنج واژه بنویس که مستقیماً به همان "
    "منظور پاسخ دهد. متن کاربر را تکرار نکن، جمله را نیمه‌تمام نگذار، حرف لاتین، فهرست یا "
    "توضیح حاشیه‌ای ننویس. اگر هیچ منظوری قابل‌فهم نیست، در یک جمله درخواست تکرار کن."
)

MODEL_SPECS = {
    CURRENT_ARM: {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "default_dir": MODEL_ROOT / "Qwen3-4B-Instruct",
    },
    "qwen3.5-4b": {
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "default_dir": MODEL_ROOT / "Qwen3.5-4B",
    },
    "qwen3.5-0.8b": {
        "model": "Qwen/Qwen3.5-0.8B",
        "revision": "23c69c53358a07516b5827588b3fdb12ae78fd65",
        "default_dir": MODEL_ROOT / "Qwen3.5-0.8b",
    },
}


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_source(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("source conversation manifest SHA-256 drift")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(rows) != 6754:
        raise ValueError("source conversation row-count drift")
    return rows


def channel_of(row: dict[str, Any]) -> str:
    return str(row.get("session_id") or "").split("/", 1)[0]


def persian_fraction(text: str) -> float:
    return float(script_statistics(text)["persian_letter_fraction"])


def row_eligible(row: dict[str, Any]) -> bool:
    if row.get("split") != "train":
        return False
    user = str(row.get("text") or "").strip()
    response = str(row.get("response_text") or "").strip()
    user_words = len(user.split())
    response_words = len(response.split())
    return bool(
        channel_of(row) in CHANNELS
        and 2 <= user_words <= 80
        and 2 <= response_words <= 80
        and persian_fraction(user) >= 0.80
        and persian_fraction(response) >= 0.80
    )


def select_panels(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row_eligible(row):
            grouped[channel_of(row)][str(row["session_id"])].append(row)

    panels: dict[str, list[dict[str, Any]]] = {"development": [], "final": []}
    for channel in CHANNELS:
        sessions = [
            session
            for session, session_rows in grouped[channel].items()
            if len(session_rows) >= ROWS_PER_SESSION
        ]
        sessions.sort(key=lambda value: sha256_text(f"{SEED}\0{channel}\0{value}"))
        required = 2 * SESSIONS_PER_CHANNEL
        if len(sessions) < required:
            raise ValueError(f"insufficient eligible sessions for {channel}")
        for stage_index, stage in enumerate(("development", "final")):
            start = stage_index * SESSIONS_PER_CHANNEL
            for session in sessions[start : start + SESSIONS_PER_CHANNEL]:
                candidates = sorted(
                    grouped[channel][session],
                    key=lambda row: sha256_text(f"{SEED}\0{stage}\0{row['utt_id']}"),
                )
                panels[stage].extend(candidates[:ROWS_PER_SESSION])
    for stage, panel in panels.items():
        if len(panel) != EXPECTED_ROWS:
            raise ValueError(f"{stage} panel size drift")
    development_sessions = {str(row["session_id"]) for row in panels["development"]}
    final_sessions = {str(row["session_id"]) for row in panels["final"]}
    if development_sessions & final_sessions:
        raise ValueError("development/final session leakage")
    return panels


def panel_receipt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    identities = [str(row["utt_id"]) for row in rows]
    sessions = [str(row["session_id"]) for row in rows]
    return {
        "rows": len(rows),
        "sessions": len(set(sessions)),
        "channels": {
            channel: sum(channel_of(row) == channel for row in rows) for channel in CHANNELS
        },
        "row_id_sequence_sha256": sha256_text("\n".join(identities)),
        "session_set_sha256": sha256_text("\n".join(sorted(set(sessions)))),
        "row_hashes": [sha256_text(identity) for identity in identities],
    }


def model_identity(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        item for item in path.rglob("*") if item.is_file() and ".cache" not in item.parts
    )
    critical_names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "LICENSE",
    }
    critical = {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in files
        if item.name in critical_names
    }
    weights = [
        {
            "name": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
        }
        for item in files
        if item.suffix == ".safetensors"
    ]
    return {
        "model": spec["model"],
        "revision": spec["revision"],
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "critical_file_sha256": critical,
        "weight_files": weights,
    }


class LocalResponder:
    def __init__(self, model_dir: Path, seed: int = 20260908):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir,
                local_files_only=True,
                dtype=torch.bfloat16,
            )
            .to("cuda")
            .eval()
        )

    def reply(self, user_text: str) -> tuple[str, float]:
        import torch

        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": user_text},
        ]
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        encoded = {name: value.to("cuda") for name, value in encoded.items()}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        reply = self.tokenizer.decode(
            output[0, encoded["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        return reply, elapsed_ms

    def peak_allocated_bytes(self) -> int:
        import torch

        return int(torch.cuda.max_memory_allocated())

    def close(self) -> None:
        import torch

        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()


def judge_identity(base_url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    model_ids = sorted(str(item.get("id")) for item in payload.get("data") or [])
    return {"model_ids": model_ids, "expected_present": JUDGE_MODEL in model_ids}


def dimension_summary(values: list[int]) -> dict[str, Any]:
    return {
        "mean": round(float(np.mean(values)), 6) if values else None,
        "p50": round(float(np.percentile(values, 50)), 3) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "histogram": {str(score): values.count(score) for score in range(5)},
    }


def summarize_arm(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": len(samples),
        "generation_failures": sum(bool(row.get("failure")) for row in samples),
        "judge_calls_expected": len(samples) * REPEATS,
        "judge_calls_valid": sum(len(row.get("scores") or []) for row in samples),
    }
    summary["judge_calls_failed"] = summary["judge_calls_expected"] - summary["judge_calls_valid"]
    for dimension in SCORE_KEYS:
        values = [int(score[dimension]) for row in samples for score in row.get("scores") or []]
        summary[dimension] = dimension_summary(values)
    timings = [float(row["generation_ms"]) for row in samples if row.get("generation_ms")]
    summary["generation_ms"] = {
        "p50": round(float(np.percentile(timings, 50)), 3) if timings else None,
        "p95": round(float(np.percentile(timings, 95)), 3) if timings else None,
        "max": round(max(timings), 3) if timings else None,
    }
    summary["repeat_exact_agreement_rows"] = sum(
        len(row.get("scores") or []) == REPEATS and row["scores"][0] == row["scores"][1]
        for row in samples
    )
    summary["mechanically_valid"] = bool(
        len(samples) == EXPECTED_ROWS
        and summary["generation_failures"] == 0
        and summary["judge_calls_failed"] == 0
    )
    return summary


def paired_result(
    incumbent: list[dict[str, Any]], challenger: list[dict[str, Any]]
) -> dict[str, Any]:
    incumbent_by_row = {row["row_hash"]: row for row in incumbent}
    challenger_by_row = {row["row_hash"]: row for row in challenger}
    shared = sorted(set(incumbent_by_row) & set(challenger_by_row))
    wins = 0
    for row_hash in shared:
        old = [score["relevance"] for score in incumbent_by_row[row_hash]["scores"]]
        new = [score["relevance"] for score in challenger_by_row[row_hash]["scores"]]
        wins += int(np.median(new) > np.median(old))
    old_rel = [score["relevance"] for row in incumbent for score in row.get("scores") or []]
    new_rel = [score["relevance"] for row in challenger for score in row.get("scores") or []]
    old_coh = [score["coherence"] for row in incumbent for score in row.get("scores") or []]
    new_coh = [score["coherence"] for row in challenger for score in row.get("scores") or []]
    return {
        "shared_rows": len(shared),
        "relevance_gain": round(float(np.mean(new_rel) - np.mean(old_rel)), 6),
        "coherence_gain": round(float(np.mean(new_coh) - np.mean(old_coh)), 6),
        "relevance_win_rows": wins,
        "relevance_win_rate": round(wins / max(1, len(shared)), 6),
    }


def select_development_winner(
    arm_samples: dict[str, list[dict[str, Any]]], summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    eligible = [name for name, summary in summaries.items() if summary["mechanically_valid"]]
    if CURRENT_ARM not in eligible:
        return {
            "selected_for_final": None,
            "status": "failed_closed_incumbent_invalid",
            "eligible_arms": eligible,
        }
    ranked = sorted(
        eligible,
        key=lambda name: (
            -float(summaries[name]["relevance"]["mean"]),
            -float(summaries[name]["coherence"]["mean"]),
            float(summaries[name]["generation_ms"]["p50"]),
            name,
        ),
    )
    best = ranked[0]
    comparisons = {
        name: paired_result(arm_samples[CURRENT_ARM], arm_samples[name])
        for name in eligible
        if name != CURRENT_ARM
    }
    promoted = False
    if best != CURRENT_ARM:
        comparison = comparisons[best]
        promoted = bool(
            comparison["relevance_gain"] >= 0.25
            and comparison["coherence_gain"] >= 0.0
            and comparison["relevance_win_rate"] >= 0.55
        )
    selected = best if promoted else CURRENT_ARM
    return {
        "selected_for_final": selected,
        "status": "challenger_promoted" if promoted else "incumbent_retained",
        "eligible_arms": eligible,
        "ranking": ranked,
        "pairwise_against_incumbent": comparisons,
        "promotion_thresholds": {
            "minimum_relevance_gain": 0.25,
            "minimum_coherence_gain": 0.0,
            "minimum_relevance_win_rate": 0.55,
        },
    }


def evaluate_arms(
    panel: list[dict[str, Any]],
    arms: list[str],
    model_dirs: dict[str, Path],
    judge_url: str,
    timeout: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    all_samples: dict[str, list[dict[str, Any]]] = {}
    identities: dict[str, Any] = {}
    for arm in arms:
        spec = MODEL_SPECS[arm]
        model_dir = model_dirs[arm]
        identities[arm] = model_identity(model_dir, spec)
        responder = LocalResponder(model_dir)
        samples: list[dict[str, Any]] = []
        try:
            for row in panel:
                user_text = str(row["text"])
                sample: dict[str, Any] = {
                    "row_hash": sha256_text(str(row["utt_id"])),
                    "session_hash": sha256_text(str(row["session_id"])),
                    "channel": channel_of(row),
                    "user_text_sha256": sha256_text(user_text),
                    "reference_text_sha256": sha256_text(str(row["response_text"])),
                    "scores": [],
                    "judge_receipts": [],
                }
                try:
                    reply, generation_ms = responder.reply(user_text)
                    statistics = script_statistics(reply)
                    sample.update(
                        {
                            "reply_sha256": sha256_text(reply),
                            "reply_statistics": statistics,
                            "generation_ms": round(generation_ms, 3),
                        }
                    )
                    if not reply or float(statistics["persian_letter_fraction"]) < 0.80:
                        raise ValueError("reply failed Persian-script requirement")
                    for _ in range(REPEATS):
                        score, receipt = judge_pair(
                            judge_url,
                            transcript=user_text,
                            reply=reply,
                            timeout=timeout,
                        )
                        sample["scores"].append(score)
                        sample["judge_receipts"].append(receipt)
                except Exception as exc:
                    sample["failure"] = f"{type(exc).__name__}: {exc}"
                samples.append(sample)
            identities[arm]["peak_torch_allocated_bytes"] = responder.peak_allocated_bytes()
        finally:
            responder.close()
        all_samples[arm] = samples
    return all_samples, identities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("plan", "development", "final"), required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "processed" / "manifests" / "conversations.jsonl",
    )
    parser.add_argument("--judge-url", default="http://127.0.0.1:8003")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--qwen3-4b-dir", type=Path, default=MODEL_SPECS[CURRENT_ARM]["default_dir"]
    )
    parser.add_argument(
        "--qwen35-4b-dir", type=Path, default=MODEL_SPECS["qwen3.5-4b"]["default_dir"]
    )
    parser.add_argument(
        "--qwen35-08b-dir", type=Path, default=MODEL_SPECS["qwen3.5-0.8b"]["default_dir"]
    )
    parser.add_argument(
        "--plan-out",
        type=Path,
        default=ROOT / "results" / "eval" / "responder_candidates_v1_plan.json",
    )
    parser.add_argument(
        "--development-out",
        type=Path,
        default=ROOT / "results" / "eval" / "responder_candidates_v1_development.json",
    )
    parser.add_argument(
        "--final-out",
        type=Path,
        default=ROOT / "results" / "eval" / "responder_candidates_v1_final.json",
    )
    args = parser.parse_args()

    rows = load_source(args.source)
    panels = select_panels(rows)
    plan = {
        "schema_version": 1,
        "stage": "preinference_plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": SOURCE_SHA256,
        "protocol_sha256": sha256_file(PROTOCOL),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "selection_seed": SEED,
        "development": panel_receipt(panels["development"]),
        "final": panel_receipt(panels["final"]),
        "session_disjoint": not bool(
            {str(row["session_id"]) for row in panels["development"]}
            & {str(row["session_id"]) for row in panels["final"]}
        ),
        "plaintext_retained": False,
    }
    if args.stage == "plan":
        atomic_write(args.plan_out, plan)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    if not args.plan_out.is_file():
        raise FileNotFoundError("run --stage plan before inference")
    recorded_plan = json.loads(args.plan_out.read_text(encoding="utf-8"))
    if any(
        recorded_plan.get(key) != plan.get(key)
        for key in (
            "source_sha256",
            "protocol_sha256",
            "evaluator_sha256",
            "selection_seed",
            "development",
            "final",
            "session_disjoint",
        )
    ):
        raise RuntimeError("preinference plan or code/protocol/source drift")

    model_dirs = {
        CURRENT_ARM: args.qwen3_4b_dir,
        "qwen3.5-4b": args.qwen35_4b_dir,
        "qwen3.5-0.8b": args.qwen35_08b_dir,
    }
    judge = judge_identity(args.judge_url, args.timeout)
    if not judge["expected_present"]:
        raise RuntimeError("judge service identity mismatch")

    if args.stage == "development":
        arms = list(MODEL_SPECS)
        samples, identities = evaluate_arms(
            panels["development"], arms, model_dirs, args.judge_url, args.timeout
        )
        summaries = {name: summarize_arm(rows_) for name, rows_ in samples.items()}
        selection = select_development_winner(samples, summaries)
        report = {
            "schema_version": 1,
            "stage": "development_selection",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if selection["selected_for_final"] else "failed",
            "plan_sha256": sha256_file(args.plan_out),
            "protocol_sha256": sha256_file(PROTOCOL),
            "evaluator_sha256": sha256_file(Path(__file__)),
            "source_sha256": SOURCE_SHA256,
            "panel": recorded_plan["development"],
            "judge": judge,
            "models": identities,
            "summary": summaries,
            "selection": selection,
            "samples": samples,
            "claim_boundary": {
                "automatic_same_family_judge_proxy": True,
                "human_evaluation": False,
                "final_panel_opened": False,
                "plaintext_retained": False,
            },
        }
        atomic_write(args.development_out, report)
        print(json.dumps({"status": report["status"], "selection": selection}, indent=2))
        return 0 if report["status"] == "passed" else 1

    if not args.development_out.is_file():
        raise FileNotFoundError("final locked: development report absent")
    development = json.loads(args.development_out.read_text(encoding="utf-8"))
    selection = development.get("selection") or {}
    selected = selection.get("selected_for_final")
    if development.get("status") != "passed" or not selected:
        raise RuntimeError("final locked: development did not pass")
    if selected == CURRENT_ARM:
        raise RuntimeError("final remains sealed: development retained the incumbent")
    if any(
        development.get(key) != plan.get(key)
        for key in ("protocol_sha256", "evaluator_sha256", "source_sha256")
    ):
        raise RuntimeError("final locked: development hash drift")
    arms = [CURRENT_ARM, str(selected)]
    samples, identities = evaluate_arms(
        panels["final"], arms, model_dirs, args.judge_url, args.timeout
    )
    summaries = {name: summarize_arm(rows_) for name, rows_ in samples.items()}
    pairwise = paired_result(samples[CURRENT_ARM], samples[str(selected)])
    report = {
        "schema_version": 1,
        "stage": "locked_final_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed"
        if all(item["mechanically_valid"] for item in summaries.values())
        else "failed",
        "development_sha256": sha256_file(args.development_out),
        "protocol_sha256": sha256_file(PROTOCOL),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "source_sha256": SOURCE_SHA256,
        "panel": recorded_plan["final"],
        "judge": judge,
        "models": identities,
        "summary": summaries,
        "pairwise_selected_minus_incumbent": pairwise,
        "samples": samples,
        "claim_boundary": {
            "automatic_same_family_judge_proxy": True,
            "human_evaluation": False,
            "plaintext_retained": False,
        },
    }
    atomic_write(args.final_out, report)
    print(json.dumps({"status": report["status"], "pairwise": pairwise}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
