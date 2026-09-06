#!/usr/bin/env python3
"""Post-hoc, privacy-safe descriptive analysis of the frozen cascade panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_s2s.metrics import write_json
from thesis_s2s.repro import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "results" / "eval" / "cascade_real_service_validation_panel.json"
DEFAULT_OUT = ROOT / "results" / "eval" / "cascade_validation_descriptive_analysis.json"


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "minimum": None, "p50": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "minimum": round(float(np.min(array)), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "maximum": round(float(np.max(array)), 6),
    }


def _numeric(samples: list[dict[str, Any]], key: str, nested: str | None = None) -> list[float]:
    values: list[float] = []
    for sample in samples:
        source = sample.get(nested) if nested else sample
        value = source.get(key) if isinstance(source, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
            values.append(float(value))
    return values


def _extreme(
    samples: list[dict[str, Any]], key: str, *, largest: bool, nested: str | None = None
) -> dict[str, float | int | None]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for sample in samples:
        source = sample.get(nested) if nested else sample
        value = source.get(key) if isinstance(source, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
            candidates.append((float(value), sample))
    if not candidates:
        return {"manifest_row_index": None, "value": None}
    value, sample = (max if largest else min)(candidates, key=lambda item: item[0])
    manifest_index = sample.get("manifest_row_index")
    return {
        "manifest_row_index": int(manifest_index)
        if isinstance(manifest_index, int)
        else None,
        "value": round(value, 6),
    }


def analyze(report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("cascade report must be a JSON object")
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("cascade report must contain a non-empty samples list")
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError("every cascade sample must be a JSON object")

    sample_rows: list[dict[str, Any]] = samples
    transcript_chars = _numeric(sample_rows, "characters", "transcript_statistics")
    reply_chars = _numeric(sample_rows, "characters", "reply_statistics")
    reply_seconds = _numeric(sample_rows, "reply_audio_duration_seconds")
    reply_rms = _numeric(sample_rows, "reply_audio_rms")
    full_turn_ms = _numeric(sample_rows, "full_turn_generation_ms")
    requirement_failures: dict[str, int] = {}
    for sample in sample_rows:
        requirements = sample.get("requirements") or {}
        for name, passed in requirements.items():
            if passed is not True:
                requirement_failures[str(name)] = requirement_failures.get(str(name), 0) + 1

    transcript_hashes = [
        str(sample["transcript_sha256"])
        for sample in sample_rows
        if sample.get("transcript_sha256")
    ]
    reply_hashes = [
        str(sample["reply_text_sha256"])
        for sample in sample_rows
        if sample.get("reply_text_sha256")
    ]
    measurement_missing_rows = {
        "transcript_characters": len(sample_rows) - len(transcript_chars),
        "reply_characters": len(sample_rows) - len(reply_chars),
        "reply_audio_seconds": len(sample_rows) - len(reply_seconds),
        "reply_audio_rms": len(sample_rows) - len(reply_rms),
        "full_turn_generation_ms": len(sample_rows) - len(full_turn_ms),
    }
    hash_missing_rows = {
        "transcript_sha256": len(sample_rows) - len(transcript_hashes),
        "reply_text_sha256": len(sample_rows) - len(reply_hashes),
    }
    correlation = None
    if (
        len(transcript_chars) == len(full_turn_ms)
        and len(full_turn_ms) > 1
        and np.std(transcript_chars) > 0
        and np.std(full_turn_ms) > 0
    ):
        correlation = round(float(np.corrcoef(transcript_chars, full_turn_ms)[0, 1]), 6)

    passed_rows = sum(sample.get("passes") is True for sample in sample_rows)
    fallback_rows = sum(sample.get("responder_fallback_used") is True for sample in sample_rows)
    retry_rows = sum(
        sample.get("responder_language_retry_used") is True for sample in sample_rows
    )
    asr_error_rows = sum(bool(sample.get("asr_error")) for sample in sample_rows)
    report_requirements = report.get("requirements") or {}
    report_gate_failures = sorted(
        str(name) for name, passed in report_requirements.items() if passed is not True
    )

    try:
        portable_source = report_path.relative_to(ROOT).as_posix()
    except ValueError:
        portable_source = str(report_path)
    complete = bool(
        report.get("status") == "passed"
        and passed_rows == len(sample_rows)
        and not report_gate_failures
        and not requirement_failures
        and not any(measurement_missing_rows.values())
        and not any(hash_missing_rows.values())
    )

    return {
        "schema_version": 1,
        "analysis_scope": "post_hoc_privacy_safe_descriptive_error_analysis",
        "source_report": portable_source,
        "source_report_sha256": sha256_file(report_path),
        "source_evidence_class": report.get("evidence_class"),
        "source_status": report.get("status"),
        "panel_rows": len(sample_rows),
        "panel_passed_rows": passed_rows,
        "panel_failed_rows": len(sample_rows) - passed_rows,
        "report_gate_failures": report_gate_failures,
        "sample_requirement_failure_counts": requirement_failures,
        "analysis_integrity": {
            "measurement_missing_rows": measurement_missing_rows,
            "hash_missing_rows": hash_missing_rows,
            "all_required_measurements_and_hashes_present": not any(
                (*measurement_missing_rows.values(), *hash_missing_rows.values())
            ),
        },
        "execution_outcomes": {
            "asr_error_rows": asr_error_rows,
            "responder_fallback_rows": fallback_rows,
            "language_retry_rows": retry_rows,
            "unique_transcript_hashes": len(set(transcript_hashes)),
            "unique_reply_hashes": len(set(reply_hashes)),
            "duplicate_transcript_rows": len(transcript_hashes) - len(set(transcript_hashes)),
            "duplicate_reply_rows": len(reply_hashes) - len(set(reply_hashes)),
        },
        "distributions": {
            "transcript_characters": _summary(transcript_chars),
            "reply_characters": _summary(reply_chars),
            "reply_audio_seconds": _summary(reply_seconds),
            "reply_audio_rms": _summary(reply_rms),
            "full_turn_generation_ms": _summary(full_turn_ms),
        },
        "descriptive_risk_counts": {
            "transcript_over_1000_characters": sum(value > 1000 for value in transcript_chars),
            "reply_audio_over_8_seconds": sum(value > 8 for value in reply_seconds),
            "full_turn_over_10_seconds": sum(value > 10_000 for value in full_turn_ms),
            "full_turn_over_500_ms_not_official_latency": sum(
                value > 500 for value in full_turn_ms
            ),
        },
        "row_extremes": {
            "longest_transcript_characters": _extreme(
                sample_rows, "characters", largest=True, nested="transcript_statistics"
            ),
            "slowest_full_turn_ms": _extreme(
                sample_rows, "full_turn_generation_ms", largest=True
            ),
            "longest_reply_audio_seconds": _extreme(
                sample_rows, "reply_audio_duration_seconds", largest=True
            ),
            "lowest_reply_audio_rms": _extreme(sample_rows, "reply_audio_rms", largest=False),
        },
        "transcript_length_full_turn_pearson_r": correlation,
        "interpretation": {
            "mechanics": (
                f"{passed_rows}/{len(sample_rows)} rows passed every predeclared automatic gate."
            ),
            "primary_engineering_risk": (
                "Long inputs and complete-response synthesis produce multi-second full-turn time; "
                "streaming first-audio latency remains unmeasured."
            ),
            "semantic_relevance": "not_measured",
            "human_pronunciation_and_naturalness": "not_measured",
        },
        "claim_boundary": {
            "post_hoc": True,
            "plaintext_transcripts_or_replies_read_or_emitted": False,
            "final_test_accessed": False,
            "scientific_generalization_claim_allowed": False,
            "semantic_quality_claim_allowed": False,
            "human_quality_claim_allowed": False,
            "official_latency_claim_allowed": False,
            "note": (
                "This analysis describes the already-open nine-row validation report. "
                "It is not a new held-out evaluation and cannot replace semantic, human, "
                "browser, or physical-target evidence."
            ),
        },
        "status": "complete" if complete else "source_report_incomplete_or_has_failures",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = analyze(args.report)
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
