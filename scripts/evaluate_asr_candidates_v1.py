#!/usr/bin/env python3
"""Hash-locked, privacy-safe NeMo versus Qwen3-ASR comparison."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_cascade_intelligibility import (  # noqa: E402
    summarize,
    text_error_counts,
)
from scripts.evaluate_cascade_real_service_v4 import (  # noqa: E402
    script_statistics,
    sha256_file,
    sha256_text,
)
from thesis_s2s.data.verbatim import verbatim_normalize  # noqa: E402
from thesis_s2s.runtime.cascade import _wav_bytes, asr_http  # noqa: E402

SOURCE_SHA256 = "aabf12268cce2854ef8b16ffd9339c6d703339c6b4c50a23a105447047a13a86"
PROVENANCE_SHA256 = "307c90f98dc72e4f0ac56a5c9cdf1f5963bc06a2d4c3446c09d00ee4436d7deb"
NEMO_API_IMAGE = "sha256:1c681f0d7cc6f192aa753efd98cedb0ef2db2c47fb8159d52912f9c03d7ad1b7"
NEMO_TRITON_IMAGE = "sha256:96b53a69eeff0b9783baf36eaa07d54701c7bbe652fcbe6af140f8d5569d5606"
SEED = "20260908-asr-candidates-v1"
CHANNELS = ("Digiato", "Zoomit")
SESSIONS_PER_CHANNEL = 4
ROWS_PER_SESSION = 5
EXPECTED_ROWS = len(CHANNELS) * SESSIONS_PER_CHANNEL * ROWS_PER_SESSION
CURRENT_ARM = "nemo-persian-finetune"
QWEN_ARMS = ("qwen3-asr-1.7b", "qwen3-asr-0.6b")
ALL_ARMS = (CURRENT_ARM, *QWEN_ARMS)
PROTOCOL = ROOT / "docs" / "ASR_CANDIDATE_COMPARISON_V1.md"
PROVENANCE = Path("/mnt/md0/mehbakh/asr_nemo_soroush/ROADMAP_3_MONTHS_FA.md")
BOOTSTRAP_DRAWS = 10_000


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_source(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("source conversation manifest SHA-256 drift")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 6754:
        raise ValueError("source conversation row-count drift")
    return rows


def channel_of(row: dict[str, Any]) -> str:
    return str(row.get("session_id") or "").split("/", 1)[0]


def audio_path(row: dict[str, Any]) -> Path:
    path = Path(str(row.get("audio_filepath") or ""))
    return path if path.is_absolute() else ROOT / path


def row_eligible(row: dict[str, Any]) -> bool:
    reference = str(row.get("text") or "").strip()
    duration = float(row.get("user_duration") or row.get("duration") or 0.0)
    return bool(
        row.get("split") == "train"
        and channel_of(row) in CHANNELS
        and 1.0 <= duration <= 20.0
        and 2 <= len(reference.split()) <= 80
        and float(script_statistics(reference)["persian_letter_fraction"]) >= 0.80
        and audio_path(row).is_file()
    )


def select_panels(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row_eligible(row):
            grouped[channel_of(row)][str(row["session_id"])].append(row)
    panels: dict[str, list[dict[str, Any]]] = {"development": [], "final": []}
    for channel in CHANNELS:
        sessions = sorted(
            (
                session
                for session, session_rows in grouped[channel].items()
                if len(session_rows) >= ROWS_PER_SESSION
            ),
            key=lambda value: sha256_text(f"{SEED}\0{channel}\0{value}"),
        )
        if len(sessions) < 2 * SESSIONS_PER_CHANNEL:
            raise ValueError(f"insufficient eligible sessions for {channel}")
        for stage_index, stage in enumerate(("development", "final")):
            start = stage_index * SESSIONS_PER_CHANNEL
            for session in sessions[start : start + SESSIONS_PER_CHANNEL]:
                candidates = sorted(
                    grouped[channel][session],
                    key=lambda row: sha256_text(f"{SEED}\0{stage}\0{row['utt_id']}"),
                )
                panels[stage].extend(candidates[:ROWS_PER_SESSION])
    development_sessions = {str(row["session_id"]) for row in panels["development"]}
    final_sessions = {str(row["session_id"]) for row in panels["final"]}
    if any(len(panel) != EXPECTED_ROWS for panel in panels.values()):
        raise ValueError("panel size drift")
    if development_sessions & final_sessions:
        raise ValueError("development/final session leakage")
    return panels


def panel_receipt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_ids = [str(row["utt_id"]) for row in rows]
    sessions = [str(row["session_id"]) for row in rows]
    return {
        "rows": len(rows),
        "sessions": len(set(sessions)),
        "channels": {
            channel: sum(channel_of(row) == channel for row in rows) for channel in CHANNELS
        },
        "seconds": round(
            sum(float(row.get("user_duration") or row.get("duration") or 0.0) for row in rows), 6
        ),
        "row_id_sequence_sha256": sha256_text("\n".join(row_ids)),
        "session_set_sha256": sha256_text("\n".join(sorted(set(sessions)))),
        "row_hashes": [sha256_text(value) for value in row_ids],
    }


def model_identity(path: Path, repository: str) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        item for item in path.rglob("*") if item.is_file() and ".cache" not in item.parts
    )
    locked = {
        item.relative_to(path).as_posix(): {
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in files
    }
    return {
        "repository": repository,
        "files": len(locked),
        "bytes": sum(item["bytes"] for item in locked.values()),
        "file_manifest": locked,
    }


def nemo_identity(api_url: str, timeout: float) -> dict[str, Any]:
    import urllib.request

    images = {}
    for container in ("asr_nemo_soroush_api", "asr_nemo_soroush_triton"):
        images[container] = subprocess.run(
            ["docker", "inspect", container, "--format", "{{.Image}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    if images["asr_nemo_soroush_api"] != NEMO_API_IMAGE:
        raise RuntimeError("NeMo API container image drift")
    if images["asr_nemo_soroush_triton"] != NEMO_TRITON_IMAGE:
        raise RuntimeError("NeMo Triton container image drift")
    with urllib.request.urlopen(f"{api_url.rstrip('/')}/health", timeout=timeout) as response:
        health = json.loads(response.read().decode("utf-8"))
    return {
        "api_image": NEMO_API_IMAGE,
        "triton_image": NEMO_TRITON_IMAGE,
        "observed_images": images,
        "health": health,
        "provenance_sha256": sha256_file(PROVENANCE),
    }


class QwenAsr:
    def __init__(self, model_dir: Path):
        import torch
        from qwen_asr import Qwen3ASRModel

        torch.cuda.reset_peak_memory_stats()
        self.model: Any = Qwen3ASRModel.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
            device_map="cuda:0",
            local_files_only=True,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def transcribe(self, path: Path) -> tuple[str, str | None, float]:
        import torch

        torch.cuda.synchronize()
        started = time.perf_counter()
        result = self.model.transcribe(audio=str(path), language="Persian")[0]
        torch.cuda.synchronize()
        return (
            str(result.text),
            str(result.language) if result.language else None,
            time.perf_counter() - started,
        )

    def peak_allocated_bytes(self) -> int:
        import torch

        return int(torch.cuda.max_memory_allocated())

    def close(self) -> None:
        import torch

        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


def nemo_transcribe(path: Path, url: str, timeout: float) -> tuple[str, str | None, float]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16_000 or np.asarray(audio).ndim != 1:
        raise ValueError("expected mono 16 kHz user audio")
    started = time.perf_counter()
    result = asr_http(_wav_bytes(np.asarray(audio, dtype=np.float32)), url)
    elapsed = time.perf_counter() - started
    if result.get("error"):
        raise RuntimeError("NeMo HTTP transcription failed")
    return str(result.get("persian") or result.get("text") or ""), "Persian", elapsed


def evaluate_arm(
    arm: str,
    panel: list[dict[str, Any]],
    model_dir: Path | None,
    nemo_url: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    backend = QwenAsr(model_dir) if model_dir is not None else None
    samples: list[dict[str, Any]] = []
    try:
        for row in panel:
            path = audio_path(row)
            duration = float(row.get("user_duration") or row.get("duration") or 0.0)
            sample: dict[str, Any] = {
                "row_hash": sha256_text(str(row["utt_id"])),
                "session_hash": sha256_text(str(row["session_id"])),
                "channel": channel_of(row),
                "audio_sha256": sha256_file(path),
                "reference_sha256": sha256_text(verbatim_normalize(str(row["text"]))),
                "duration_seconds": round(duration, 6),
            }
            try:
                if backend is None:
                    raw, language, elapsed = nemo_transcribe(path, nemo_url, timeout)
                else:
                    raw, language, elapsed = backend.transcribe(path)
                hypothesis = verbatim_normalize(raw)
                counts = text_error_counts(verbatim_normalize(str(row["text"])), hypothesis)
                sample.update(counts)
                sample.update(
                    {
                        "hypothesis_sha256": sha256_text(hypothesis),
                        "empty_hypothesis": not bool(hypothesis),
                        "language": language,
                        "elapsed_seconds": round(elapsed, 6),
                        "real_time_factor": round(elapsed / duration, 6),
                    }
                )
            except Exception as exc:
                reference = verbatim_normalize(str(row["text"]))
                sample.update(text_error_counts(reference, ""))
                sample.update(
                    {
                        "hypothesis_sha256": sha256_text(""),
                        "empty_hypothesis": True,
                    }
                )
                sample["failure_type"] = type(exc).__name__
            samples.append(sample)
        identity = {
            "arm": arm,
            "peak_torch_allocated_bytes": backend.peak_allocated_bytes() if backend else None,
        }
        return samples, identity
    finally:
        if backend is not None:
            backend.close()


def arm_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if "failure_type" not in sample]
    metrics = summarize(samples) if samples else {}
    rtfs = [float(sample["real_time_factor"]) for sample in valid]
    metrics.update(
        {
            "rows_expected": EXPECTED_ROWS,
            "rows_valid": len(valid),
            "failures": len(samples) - len(valid),
            "empty_hypotheses": sum(bool(sample.get("empty_hypothesis")) for sample in samples),
            "mechanically_valid": len(valid) == EXPECTED_ROWS,
            "real_time_factor": {
                "p50": round(float(np.percentile(rtfs, 50)), 6) if rtfs else None,
                "p95": round(float(np.percentile(rtfs, 95)), 6) if rtfs else None,
                "max": round(max(rtfs), 6) if rtfs else None,
            },
        }
    )
    return metrics


def _session_rate(samples: list[dict[str, Any]], selected: list[str]) -> float:
    by_session: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for sample in samples:
        edits, chars = by_session[sample["session_hash"]]
        by_session[sample["session_hash"]] = (
            edits + int(sample["character_edits"]),
            chars + int(sample["reference_characters"]),
        )
    edits = sum(by_session[key][0] for key in selected)
    chars = sum(by_session[key][1] for key in selected)
    return edits / max(1, chars)


def paired_comparison(
    incumbent: list[dict[str, Any]], challenger: list[dict[str, Any]], arm: str
) -> dict[str, Any]:
    old = list(incumbent)
    new = list(challenger)
    old_summary = summarize(old)
    new_summary = summarize(new)
    sessions = sorted(
        {sample["session_hash"] for sample in old} & {sample["session_hash"] for sample in new}
    )
    rng = np.random.default_rng(int(sha256_text(f"{SEED}\0{arm}")[:16], 16))
    differences = []
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = [sessions[index] for index in rng.integers(0, len(sessions), len(sessions))]
        differences.append(_session_rate(new, chosen) - _session_rate(old, chosen))
    old_cer = float(old_summary["character_error_rate"]["micro"])
    new_cer = float(new_summary["character_error_rate"]["micro"])
    old_wer = float(old_summary["word_error_rate"]["micro"])
    new_wer = float(new_summary["word_error_rate"]["micro"])
    return {
        "shared_sessions": len(sessions),
        "cer_absolute_reduction": round(old_cer - new_cer, 6),
        "wer_absolute_reduction": round(old_wer - new_wer, 6),
        "challenger_minus_incumbent_cer_bootstrap_95": [
            round(float(np.percentile(differences, 2.5)), 6),
            round(float(np.percentile(differences, 97.5)), 6),
        ],
    }


def select_winner(
    samples: dict[str, list[dict[str, Any]]], summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    eligible = [arm for arm in ALL_ARMS if summaries[arm]["mechanically_valid"]]
    if CURRENT_ARM not in eligible:
        return {
            "status": "failed_closed_incumbent_invalid",
            "selected_for_final": None,
            "eligible_arms": eligible,
        }
    ranked = sorted(
        eligible,
        key=lambda arm: (
            float(summaries[arm]["character_error_rate"]["micro"]),
            float(summaries[arm]["word_error_rate"]["micro"]),
            float(summaries[arm]["real_time_factor"]["p50"]),
            arm,
        ),
    )
    comparisons = {
        arm: paired_comparison(samples[CURRENT_ARM], samples[arm], arm)
        for arm in eligible
        if arm != CURRENT_ARM
    }
    best = ranked[0]
    promoted = False
    if best != CURRENT_ARM:
        comparison = comparisons[best]
        promoted = bool(
            comparison["cer_absolute_reduction"] >= 0.02
            and comparison["wer_absolute_reduction"] >= 0.0
            and comparison["challenger_minus_incumbent_cer_bootstrap_95"][1] < 0.0
        )
    return {
        "status": "challenger_promoted" if promoted else "incumbent_retained",
        "selected_for_final": best if promoted else CURRENT_ARM,
        "eligible_arms": eligible,
        "ranking": ranked,
        "pairwise_against_incumbent": comparisons,
        "promotion_thresholds": {
            "minimum_cer_absolute_reduction": 0.02,
            "minimum_wer_absolute_reduction": 0.0,
            "maximum_cer_difference_ci95_upper": 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("plan", "development", "final"), required=True)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/processed/manifests/conversations.jsonl"
    )
    parser.add_argument("--nemo-url", default="http://127.0.0.1:8090")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--qwen17-dir", type=Path, default=Path("/mnt/md0/models/Qwen3-ASR-1.7B"))
    parser.add_argument("--qwen06-dir", type=Path, default=Path("/mnt/md0/models/Qwen3-ASR-0.6B"))
    parser.add_argument(
        "--plan-out", type=Path, default=ROOT / "results/eval/asr_candidates_v1_plan.json"
    )
    parser.add_argument(
        "--development-out",
        type=Path,
        default=ROOT / "results/eval/asr_candidates_v1_development.json",
    )
    parser.add_argument(
        "--final-out", type=Path, default=ROOT / "results/eval/asr_candidates_v1_final.json"
    )
    args = parser.parse_args()

    os.environ["ASR_TIMEOUT_SECONDS"] = str(args.timeout)
    if sha256_file(PROVENANCE) != PROVENANCE_SHA256:
        raise RuntimeError("NeMo training-provenance document drift")
    rows = load_source(args.source)
    panels = select_panels(rows)
    model_dirs = {"qwen3-asr-1.7b": args.qwen17_dir, "qwen3-asr-0.6b": args.qwen06_dir}
    identities = {
        CURRENT_ARM: nemo_identity(args.nemo_url, args.timeout),
        "qwen3-asr-1.7b": model_identity(args.qwen17_dir, "Qwen/Qwen3-ASR-1.7B"),
        "qwen3-asr-0.6b": model_identity(args.qwen06_dir, "Qwen/Qwen3-ASR-0.6B"),
    }
    plan = {
        "schema_version": 1,
        "stage": "preinference_plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": SOURCE_SHA256,
        "provenance_sha256": PROVENANCE_SHA256,
        "protocol_sha256": sha256_file(PROTOCOL),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "selection_seed": SEED,
        "development": panel_receipt(panels["development"]),
        "final": panel_receipt(panels["final"]),
        "model_identities": identities,
        "runtime_packages": {
            name: importlib.metadata.version(name)
            for name in ("qwen-asr", "torch", "transformers", "numpy")
        },
        "session_disjoint": not bool(
            {str(row["session_id"]) for row in panels["development"]}
            & {str(row["session_id"]) for row in panels["final"]}
        ),
        "plaintext_retained": False,
    }
    if args.stage == "plan":
        atomic_write(args.plan_out, plan)
        print(
            json.dumps(
                {"status": "planned", "development": plan["development"], "final": plan["final"]},
                indent=2,
            )
        )
        return 0
    if not args.plan_out.is_file():
        raise FileNotFoundError("run --stage plan before inference")
    recorded_plan = json.loads(args.plan_out.read_text(encoding="utf-8"))
    locked_keys = (
        "source_sha256",
        "provenance_sha256",
        "protocol_sha256",
        "evaluator_sha256",
        "selection_seed",
        "development",
        "final",
        "model_identities",
        "runtime_packages",
        "session_disjoint",
    )
    if any(recorded_plan.get(key) != plan.get(key) for key in locked_keys):
        raise RuntimeError("preinference plan or code/protocol/model/source drift")

    if args.stage == "development":
        arm_samples: dict[str, list[dict[str, Any]]] = {}
        runtime_identities: dict[str, Any] = {}
        for arm in ALL_ARMS:
            samples, identity = evaluate_arm(
                arm, panels["development"], model_dirs.get(arm), args.nemo_url, args.timeout
            )
            arm_samples[arm] = samples
            runtime_identities[arm] = identity
        summaries = {arm: arm_summary(samples) for arm, samples in arm_samples.items()}
        selection = select_winner(arm_samples, summaries)
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
            "runtime_identities": runtime_identities,
            "summary": summaries,
            "selection": selection,
            "samples": arm_samples,
            "claim_boundary": {
                "automatic_caption_references": True,
                "human_corrected_references": False,
                "offline_not_streaming_latency": True,
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
    selected = (development.get("selection") or {}).get("selected_for_final")
    if development.get("status") != "passed" or not selected:
        raise RuntimeError("final locked: development did not pass")
    if selected == CURRENT_ARM:
        raise RuntimeError("final remains sealed: development retained the incumbent")
    if any(
        development.get(key) != plan.get(key)
        for key in ("protocol_sha256", "evaluator_sha256", "source_sha256")
    ):
        raise RuntimeError("final locked: development hash drift")
    final_samples: dict[str, list[dict[str, Any]]] = {}
    runtime_identities = {}
    for arm in (CURRENT_ARM, str(selected)):
        samples, identity = evaluate_arm(
            arm, panels["final"], model_dirs.get(arm), args.nemo_url, args.timeout
        )
        final_samples[arm] = samples
        runtime_identities[arm] = identity
    summaries = {arm: arm_summary(samples) for arm, samples in final_samples.items()}
    comparison = paired_comparison(
        final_samples[CURRENT_ARM], final_samples[str(selected)], str(selected)
    )
    confirmed = bool(
        all(summary["mechanically_valid"] for summary in summaries.values())
        and comparison["cer_absolute_reduction"] > 0.0
        and comparison["wer_absolute_reduction"] >= 0.0
        and comparison["challenger_minus_incumbent_cer_bootstrap_95"][1] < 0.0
    )
    report = {
        "schema_version": 1,
        "stage": "locked_final_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if confirmed else "failed",
        "development_sha256": sha256_file(args.development_out),
        "protocol_sha256": sha256_file(PROTOCOL),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "source_sha256": SOURCE_SHA256,
        "panel": recorded_plan["final"],
        "runtime_identities": runtime_identities,
        "summary": summaries,
        "pairwise_selected_minus_incumbent": comparison,
        "challenger_confirmed": confirmed,
        "samples": final_samples,
        "claim_boundary": {
            "automatic_caption_references": True,
            "human_corrected_references": False,
            "offline_not_streaming_latency": True,
            "plaintext_retained": False,
        },
    }
    atomic_write(args.final_out, report)
    print(json.dumps({"status": report["status"], "comparison": comparison}, indent=2))
    return 0 if confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
