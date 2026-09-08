#!/usr/bin/env python3
"""Hash-locked, privacy-safe Mana-Piper versus MMS Persian TTS comparison."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_asr_candidates_v1 import nemo_identity  # noqa: E402
from scripts.evaluate_cascade_intelligibility import (  # noqa: E402
    summarize,
    text_error_counts,
)
from scripts.evaluate_cascade_real_service_v4 import (  # noqa: E402
    script_statistics,
    sha256_file,
    sha256_text,
)
from scripts.evaluate_responder_candidates_v1 import (  # noqa: E402
    select_panels as select_responder_panels,
)
from thesis_s2s import SAMPLE_RATE  # noqa: E402
from thesis_s2s.audio import resample  # noqa: E402
from thesis_s2s.data.verbatim import verbatim_normalize  # noqa: E402
from thesis_s2s.runtime.cascade import _wav_bytes, asr_http  # noqa: E402
from thesis_s2s.runtime.tts import piper_synthesize  # noqa: E402

SOURCE_SHA256 = "aabf12268cce2854ef8b16ffd9339c6d703339c6b4c50a23a105447047a13a86"
MMS_REVISION = "8818d36618d125a0b40b5d2b2713a852877e9b68"
SEED = "20260908-tts-candidates-v1"
CHANNELS = ("Digiato", "Mehran Rowshan Persian", "Tabaghe16", "Zoomit")
SESSIONS_PER_CHANNEL = 2
ROWS_PER_SESSION = 5
EXPECTED_ROWS = len(CHANNELS) * SESSIONS_PER_CHANNEL * ROWS_PER_SESSION
CURRENT_ARM = "mana-persian-piper"
CHALLENGER_ARM = "facebook-mms-tts-fas"
ALL_ARMS = (CURRENT_ARM, CHALLENGER_ARM)
BOOTSTRAP_DRAWS = 10_000
PROTOCOL = ROOT / "docs" / "TTS_CANDIDATE_COMPARISON_V1.md"
ASR_EVALUATOR = ROOT / "scripts" / "evaluate_asr_candidates_v1.py"
RESPONDER_EVALUATOR = ROOT / "scripts" / "evaluate_responder_candidates_v1.py"
EXCLUDED_WARMUP_TEXT = "این یک جملهٔ گرم‌کردن خارج از پنل ارزیابی است."


class Talker(Protocol):
    load_seconds: float

    def synthesize(self, text: str, row_hash: str) -> tuple[np.ndarray, float]: ...

    def close(self) -> None: ...


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
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 6754:
        raise ValueError("source conversation row-count drift")
    return rows


def channel_of(row: dict[str, Any]) -> str:
    return str(row.get("session_id") or "").split("/", 1)[0]


def row_eligible(row: dict[str, Any], excluded_sessions: set[str]) -> bool:
    text = str(row.get("response_text") or "").strip()
    words = len(text.split())
    return bool(
        row.get("split") == "train"
        and channel_of(row) in CHANNELS
        and str(row.get("session_id")) not in excluded_sessions
        and 2 <= words <= 25
        and float(script_statistics(text)["persian_letter_fraction"]) >= 0.80
    )


def responder_panel_sessions(rows: list[dict[str, Any]]) -> set[str]:
    panels = select_responder_panels(rows)
    return {
        str(row["session_id"])
        for panel in panels.values()
        for row in panel
    }


def select_panels(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    excluded_sessions = responder_panel_sessions(rows)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row_eligible(row, excluded_sessions):
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
    if (development_sessions | final_sessions) & excluded_sessions:
        raise ValueError("responder/TTS session leakage")
    return panels


def panel_receipt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_ids = [str(row["utt_id"]) for row in rows]
    sessions = [str(row["session_id"]) for row in rows]
    texts = [verbatim_normalize(str(row["response_text"])) for row in rows]
    return {
        "rows": len(rows),
        "sessions": len(set(sessions)),
        "channels": {
            channel: sum(channel_of(row) == channel for row in rows) for channel in CHANNELS
        },
        "reference_words": sum(len(text.split()) for text in texts),
        "reference_characters": sum(len(text.replace(" ", "")) for text in texts),
        "row_id_sequence_sha256": sha256_text("\n".join(row_ids)),
        "session_set_sha256": sha256_text("\n".join(sorted(set(sessions)))),
        "reference_sequence_sha256": sha256_text("\n".join(texts)),
        "row_hashes": [sha256_text(value) for value in row_ids],
    }


def file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def model_identities(piper_model: Path, mms_dir: Path) -> dict[str, Any]:
    piper_config = piper_model.with_suffix(piper_model.suffix + ".json")
    if not mms_dir.is_dir():
        raise FileNotFoundError(mms_dir)
    mms_files = sorted(
        item for item in mms_dir.rglob("*") if item.is_file() and ".cache" not in item.parts
    )
    mms_manifest = {
        item.relative_to(mms_dir).as_posix(): file_identity(item) for item in mms_files
    }
    return {
        CURRENT_ARM: {
            "repository": "MahtaFetrat/Mana-Persian-Piper",
            "model": file_identity(piper_model),
            "config": file_identity(piper_config),
            "deterministic_noise_scale": 0.0,
            "deterministic_noise_w_scale": 0.0,
        },
        CHALLENGER_ARM: {
            "repository": "facebook/mms-tts-fas",
            "revision": MMS_REVISION,
            "license": "CC-BY-NC-4.0",
            "files": len(mms_manifest),
            "bytes": sum(item["bytes"] for item in mms_manifest.values()),
            "file_manifest": mms_manifest,
        },
    }


class PiperTalker:
    def __init__(self) -> None:
        started = time.perf_counter()
        warmup = piper_synthesize(EXCLUDED_WARMUP_TEXT, deterministic=True)
        self.load_seconds = time.perf_counter() - started
        if warmup is None or not len(warmup):
            raise RuntimeError("Piper warmup failed")

    def synthesize(self, text: str, row_hash: str) -> tuple[np.ndarray, float]:
        del row_hash
        started = time.perf_counter()
        audio = piper_synthesize(text, deterministic=True)
        elapsed = time.perf_counter() - started
        if audio is None:
            raise RuntimeError("Piper synthesis failed")
        return np.asarray(audio, dtype=np.float32), elapsed

    def close(self) -> None:
        return None


class MmsTalker:
    def __init__(self, model_dir: Path) -> None:
        from transformers import AutoTokenizer, VitsModel

        started = time.perf_counter()
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model: Any = VitsModel.from_pretrained(model_dir, local_files_only=True).eval()
        self.sample_rate = int(self.model.config.sampling_rate)
        self.load_seconds = time.perf_counter() - started
        self.synthesize(EXCLUDED_WARMUP_TEXT, sha256_text(EXCLUDED_WARMUP_TEXT))

    def synthesize(self, text: str, row_hash: str) -> tuple[np.ndarray, float]:
        import torch

        seed = int(sha256_text(f"{SEED}\0{row_hash}")[:16], 16) % (2**31)
        torch.manual_seed(seed)
        inputs = self.tokenizer(text, return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(**inputs).waveform.squeeze().float().cpu().numpy()
        elapsed = time.perf_counter() - started
        audio = np.asarray(output, dtype=np.float32)
        if self.sample_rate != SAMPLE_RATE:
            audio = resample(audio, self.sample_rate, SAMPLE_RATE)
        return np.asarray(audio, dtype=np.float32), elapsed

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()


def audio_sha256(audio: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes()).hexdigest()


def evaluate_arm(
    arm: str,
    panel: list[dict[str, Any]],
    mms_dir: Path,
    nemo_url: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    os.environ["ASR_TIMEOUT_SECONDS"] = str(timeout)
    backend: Talker = MmsTalker(mms_dir) if arm == CHALLENGER_ARM else PiperTalker()
    samples: list[dict[str, Any]] = []
    try:
        for row in panel:
            row_hash = sha256_text(str(row["utt_id"]))
            reference = verbatim_normalize(str(row["response_text"]))
            sample: dict[str, Any] = {
                "row_hash": row_hash,
                "session_hash": sha256_text(str(row["session_id"])),
                "channel": channel_of(row),
                "reference_sha256": sha256_text(reference),
                "reference_words": len(reference.split()),
                "reference_characters": len(reference.replace(" ", "")),
            }
            try:
                audio, synthesis_seconds = backend.synthesize(reference, row_hash)
                finite = bool(np.isfinite(audio).all())
                rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
                peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
                duration = len(audio) / SAMPLE_RATE
                if not finite or len(audio) < SAMPLE_RATE // 10 or rms <= 1e-4:
                    raise ValueError("invalid or silent synthesized audio")
                sample.update(
                    {
                        "audio_sha256": audio_sha256(audio),
                        "audio_samples": len(audio),
                        "audio_duration_seconds": round(duration, 6),
                        "audio_rms": round(rms, 6),
                        "audio_peak": round(peak, 6),
                        "audio_finite": finite,
                        "synthesis_seconds": round(synthesis_seconds, 6),
                        "real_time_factor": round(synthesis_seconds / duration, 6),
                    }
                )
                started = time.perf_counter()
                result = asr_http(_wav_bytes(audio), nemo_url)
                asr_seconds = time.perf_counter() - started
                if result.get("error"):
                    raise RuntimeError("NeMo HTTP transcription failed")
                hypothesis = verbatim_normalize(
                    str(result.get("persian") or result.get("text") or "")
                )
                sample.update(text_error_counts(reference, hypothesis))
                sample.update(
                    {
                        "hypothesis_sha256": sha256_text(hypothesis),
                        "empty_hypothesis": not bool(hypothesis),
                        "asr_seconds": round(asr_seconds, 6),
                    }
                )
            except Exception as exc:
                sample.update(text_error_counts(reference, ""))
                sample.update(
                    {
                        "hypothesis_sha256": sha256_text(""),
                        "empty_hypothesis": True,
                        "failure_type": type(exc).__name__,
                        "failure_stage": (
                            "asr" if "audio_sha256" in sample else "synthesis"
                        ),
                    }
                )
            samples.append(sample)
        return samples, {
            "arm": arm,
            "load_and_warmup_seconds": round(backend.load_seconds, 6),
            "sample_rate": SAMPLE_RATE,
        }
    finally:
        backend.close()


def arm_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [sample for sample in samples if "failure_type" not in sample]
    summary = summarize(samples) if samples else {}
    rtfs = [float(sample["real_time_factor"]) for sample in successful]
    latencies = [float(sample["synthesis_seconds"]) for sample in successful]
    durations = [float(sample["audio_duration_seconds"]) for sample in successful]
    summary.update(
        {
            "rows_expected": EXPECTED_ROWS,
            "rows_valid": len(successful),
            "failures": len(samples) - len(successful),
            "empty_hypotheses": sum(bool(sample.get("empty_hypothesis")) for sample in samples),
            "mechanically_valid": len(successful) == EXPECTED_ROWS,
            "synthesis_seconds": distribution(latencies),
            "real_time_factor": distribution(rtfs),
            "audio_duration_seconds": distribution(durations),
            "per_channel": {
                channel: summarize(
                    [sample for sample in samples if sample["channel"] == channel]
                )
                for channel in CHANNELS
            },
        }
    )
    return summary


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": round(float(np.percentile(values, 50)), 6) if values else None,
        "p95": round(float(np.percentile(values, 95)), 6) if values else None,
        "max": round(max(values), 6) if values else None,
    }


def _session_rate(samples: list[dict[str, Any]], selected: list[str]) -> float:
    by_session: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for sample in samples:
        edits, characters = by_session[sample["session_hash"]]
        by_session[sample["session_hash"]] = (
            edits + int(sample["character_edits"]),
            characters + int(sample["reference_characters"]),
        )
    edits = sum(by_session[key][0] for key in selected)
    characters = sum(by_session[key][1] for key in selected)
    return edits / max(1, characters)


def paired_comparison(
    incumbent: list[dict[str, Any]], challenger: list[dict[str, Any]]
) -> dict[str, Any]:
    old_summary = summarize(incumbent)
    new_summary = summarize(challenger)
    sessions = sorted(
        {sample["session_hash"] for sample in incumbent}
        & {sample["session_hash"] for sample in challenger}
    )
    rng = np.random.default_rng(int(sha256_text(f"{SEED}\0bootstrap")[:16], 16))
    differences = []
    for _ in range(BOOTSTRAP_DRAWS):
        selected = [sessions[index] for index in rng.integers(0, len(sessions), len(sessions))]
        differences.append(
            _session_rate(challenger, selected) - _session_rate(incumbent, selected)
        )
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
    comparison = paired_comparison(samples[CURRENT_ARM], samples[CHALLENGER_ARM])
    both_valid = all(summaries[arm]["mechanically_valid"] for arm in ALL_ARMS)
    challenger_rtf = summaries[CHALLENGER_ARM]["real_time_factor"]["p50"]
    promoted = bool(
        both_valid
        and comparison["cer_absolute_reduction"] >= 0.02
        and comparison["wer_absolute_reduction"] >= 0.0
        and comparison["challenger_minus_incumbent_cer_bootstrap_95"][1] < 0.0
        and challenger_rtf is not None
        and challenger_rtf <= 1.0
    )
    return {
        "status": "challenger_advanced" if promoted else "incumbent_retained",
        "selected_for_final": CHALLENGER_ARM if promoted else CURRENT_ARM,
        "final_panel_unlocked": promoted,
        "automatic_proxy_only": True,
        "production_promotion_allowed": False,
        "pairwise_against_incumbent": comparison,
        "promotion_thresholds": {
            "minimum_cer_absolute_reduction": 0.02,
            "minimum_wer_absolute_reduction": 0.0,
            "maximum_cer_difference_ci95_upper": 0.0,
            "maximum_challenger_p50_render_rtf": 1.0,
        },
    }


def runtime_packages() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("numpy", "piper-tts", "torch", "transformers", "uroman")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("plan", "development", "final"), required=True)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/processed/manifests/conversations.jsonl"
    )
    parser.add_argument(
        "--piper-model", type=Path, default=ROOT / "models/piper/fa_IR-mana-medium.onnx"
    )
    parser.add_argument(
        "--mms-dir", type=Path, default=Path("/mnt/md0/models/MMS-TTS-fas")
    )
    parser.add_argument("--nemo-url", default="http://127.0.0.1:8090")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--plan-out", type=Path, default=ROOT / "results/eval/tts_candidates_v1_plan.json"
    )
    parser.add_argument(
        "--development-out",
        type=Path,
        default=ROOT / "results/eval/tts_candidates_v1_development.json",
    )
    parser.add_argument(
        "--final-out", type=Path, default=ROOT / "results/eval/tts_candidates_v1_final.json"
    )
    args = parser.parse_args()

    rows = load_source(args.source)
    panels = select_panels(rows)
    identities = model_identities(args.piper_model, args.mms_dir)
    dependencies = {
        "asr_evaluator_sha256": sha256_file(ASR_EVALUATOR),
        "responder_evaluator_sha256": sha256_file(RESPONDER_EVALUATOR),
    }
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
        "model_identities": identities,
        "runtime_packages": runtime_packages(),
        "nemo_identity": nemo_identity(args.nemo_url, args.timeout),
        "dependencies": dependencies,
        "session_disjoint": not bool(
            {str(row["session_id"]) for row in panels["development"]}
            & {str(row["session_id"]) for row in panels["final"]}
        ),
        "responder_sessions_excluded": True,
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
        "protocol_sha256",
        "evaluator_sha256",
        "selection_seed",
        "development",
        "final",
        "model_identities",
        "runtime_packages",
        "nemo_identity",
        "dependencies",
        "session_disjoint",
        "responder_sessions_excluded",
    )
    if any(recorded_plan.get(key) != plan.get(key) for key in locked_keys):
        raise RuntimeError("preinference plan or code/protocol/model/source drift")

    if args.stage == "development":
        arm_samples: dict[str, list[dict[str, Any]]] = {}
        runtime_identities: dict[str, Any] = {}
        for arm in ALL_ARMS:
            samples, identity = evaluate_arm(
                arm, panels["development"], args.mms_dir, args.nemo_url, args.timeout
            )
            arm_samples[arm] = samples
            runtime_identities[arm] = identity
        summaries = {arm: arm_summary(samples) for arm, samples in arm_samples.items()}
        selection = select_winner(arm_samples, summaries)
        report = {
            "schema_version": 1,
            "stage": "development_selection",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if summaries[CURRENT_ARM]["mechanically_valid"] else "failed",
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
                "human_listening_result": False,
                "full_render_not_streaming_latency": True,
                "mms_weights_redistributed": False,
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
    if development.get("status") != "passed" or not selection.get("final_panel_unlocked"):
        raise RuntimeError("final remains sealed: development retained the incumbent")
    if any(
        development.get(key) != plan.get(key)
        for key in ("protocol_sha256", "evaluator_sha256", "source_sha256")
    ):
        raise RuntimeError("final locked: development hash drift")

    final_samples: dict[str, list[dict[str, Any]]] = {}
    runtime_identities = {}
    for arm in ALL_ARMS:
        samples, identity = evaluate_arm(
            arm, panels["final"], args.mms_dir, args.nemo_url, args.timeout
        )
        final_samples[arm] = samples
        runtime_identities[arm] = identity
    summaries = {arm: arm_summary(samples) for arm, samples in final_samples.items()}
    comparison = paired_comparison(
        final_samples[CURRENT_ARM], final_samples[CHALLENGER_ARM]
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
        "challenger_automatically_confirmed": confirmed,
        "production_promotion_allowed": False,
        "samples": final_samples,
        "claim_boundary": {
            "automatic_caption_references": True,
            "human_listening_result": False,
            "full_render_not_streaming_latency": True,
            "mms_weights_redistributed": False,
            "plaintext_retained": False,
        },
    }
    atomic_write(args.final_out, report)
    print(json.dumps({"status": report["status"], "comparison": comparison}, indent=2))
    return 0 if confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
