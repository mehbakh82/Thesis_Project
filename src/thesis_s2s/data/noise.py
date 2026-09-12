"""Evidence-based background-noise labels for diarized conversation windows."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from thesis_s2s.audio import read_wav, to_float32_mono
from thesis_s2s.metrics import write_json


def _merged_speech_intervals(turns: list[dict], duration: float) -> list[tuple[float, float]]:
    valid = []
    for turn in turns:
        try:
            start = max(0.0, float(turn["start"]))
            end = min(duration, float(turn["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            valid.append((start, end))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(valid):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _complement(intervals: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    gaps = []
    cursor = 0.0
    for start, end in intervals:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


def _window_rms(
    audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
    *,
    window_seconds: float = 0.5,
    max_windows: int = 32,
) -> list[float]:
    width = max(1, round(window_seconds * sample_rate))
    candidates = []
    for start, end in intervals:
        first = round(start * sample_rate)
        last = round(end * sample_rate) - width
        if last < first:
            continue
        candidates.append((first + last) // 2)
    if len(candidates) > max_windows:
        indices = np.linspace(0, len(candidates) - 1, max_windows, dtype=int)
        candidates = [candidates[index] for index in indices]
    values = []
    for start in candidates:
        frame = audio[start : start + width]
        if len(frame) == width:
            values.append(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))))
    return values


def estimate_noise_condition(
    audio: np.ndarray,
    sample_rate: int,
    speaker_turns: list[dict],
) -> dict:
    """Estimate background SNR from deterministic speech/nonspeech windows."""

    mono = to_float32_mono(audio)
    duration = len(mono) / sample_rate
    speech_intervals = _merged_speech_intervals(speaker_turns, duration)
    background_intervals = _complement(speech_intervals, duration)
    speech_rms = _window_rms(mono, sample_rate, speech_intervals)
    background_rms = _window_rms(mono, sample_rate, background_intervals)
    if not speech_rms or not background_rms:
        return {
            "noise_condition": "unestimated",
            "snr": None,
            "noise_evidence": {
                "method": "speech_nonspeech_rms_v1",
                "speech_windows": len(speech_rms),
                "background_windows": len(background_rms),
                "status": "insufficient_speech_or_nonspeech",
            },
        }
    speech_level = float(np.median(speech_rms))
    background_level = float(np.median(background_rms))
    snr_db = 20.0 * np.log10(max(speech_level, 1e-8) / max(background_level, 1e-8))
    if snr_db <= 20.0:
        condition = "background-noisy"
    elif snr_db <= 30.0:
        condition = "background-moderate"
    else:
        condition = "background-clean"
    return {
        "noise_condition": condition,
        "snr": round(float(snr_db), 3),
        "noise_evidence": {
            "method": "speech_nonspeech_rms_v1",
            "speech_windows": len(speech_rms),
            "background_windows": len(background_rms),
            "speech_rms": round(speech_level, 7),
            "background_rms": round(background_level, 7),
            "status": "estimated",
            "thresholds_db": {"noisy_max": 20.0, "moderate_max": 30.0},
        },
    }


def annotate_noise_conditions(
    in_jsonl: Path,
    out_jsonl: Path,
    *,
    report_path: Path | None = None,
) -> dict:
    """Add noise evidence to a new manifest without mutating diarization output."""

    in_jsonl = Path(in_jsonl)
    out_jsonl = Path(out_jsonl)
    if in_jsonl.resolve() == out_jsonl.resolve():
        raise ValueError("noise-label output must differ from its source manifest")
    rows = [
        json.loads(line)
        for line in in_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts: Counter[str] = Counter()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out_jsonl.parent,
            prefix=f".{out_jsonl.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                path = Path(str(row.get("audio_filepath") or row.get("audio_path") or ""))
                turns = list(row.get("exclusive_speaker_turns") or row.get("speaker_turns") or [])
                if row.get("diarization_status") == "complete" and path.is_file() and turns:
                    audio, sample_rate = read_wav(path)
                    row.update(estimate_noise_condition(audio, sample_rate, turns))
                else:
                    row.update(
                        {
                            "noise_condition": "unestimated",
                            "snr": None,
                            "noise_evidence": {
                                "method": "speech_nonspeech_rms_v1",
                                "status": "missing_complete_diarization_audio_or_turns",
                            },
                        }
                    )
                counts[str(row["noise_condition"])] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if temporary is None:  # defensive: NamedTemporaryFile sets it before replacement
            raise RuntimeError("noise annotation temporary file was not created")
        temporary.replace(out_jsonl)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    report = {
        "source_manifest": str(in_jsonl),
        "out_manifest": str(out_jsonl),
        "windows": len(rows),
        "condition_counts": dict(counts),
        "estimated_windows": len(rows) - counts["unestimated"],
        "method": "speech_nonspeech_rms_v1",
        "warning": "Automatic acoustic condition labels require stratified listening QA.",
    }
    if report_path is not None:
        write_json(report_path, report)
    return report
