"""Optional Community-1 overlap labels via the existing diarization HTTP service.

Off by default. Only run on a podcast/long-episode subset, never Shorts.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from thesis_s2s.audio import read_wav
from thesis_s2s.data.rights import rights_record_verified
from thesis_s2s.metrics import write_json


def _overlap_intervals(turns: list[dict]) -> list[list[float]]:
    intervals: list[list[float]] = []
    ordered = sorted(turns, key=lambda item: (float(item["start"]), float(item["end"])))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if float(right["start"]) >= float(left["end"]):
                break
            if left.get("speaker") == right.get("speaker"):
                continue
            start = max(float(left["start"]), float(right["start"]))
            end = min(float(left["end"]), float(right["end"]))
            if end > start:
                intervals.append([round(start, 3), round(end, 3)])
    return intervals


def diarize_file(wav_path: Path, base: str | None = None) -> dict:
    base = (base or os.environ.get("DIARIZATION_SERVICE_URL") or "").rstrip("/")
    if not base:
        return {
            "available": False,
            "overlap_intervals": [],
            "note": "DIARIZATION_SERVICE_URL unset",
        }
    audio, sample_rate = read_wav(wav_path)
    req = Request(
        f"{base}/diarize?sample_rate={sample_rate}",
        data=audio.astype("<f4").tobytes(),
        method="POST",
    )
    req.add_header("content-type", "application/octet-stream")
    try:
        timeout_seconds = max(
            30.0, min(3600.0, float(os.environ.get("DIARIZATION_TIMEOUT_SECONDS", "600")))
        )
    except ValueError:
        timeout_seconds = 600.0
    try:
        with urlopen(req, timeout=timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"available": False, "overlap_intervals": [], "error": str(exc)[:300]}
    turns = payload.get("turns") or []
    exclusive_turns = payload.get("exclusive_turns") or []
    overlap = _overlap_intervals(turns)
    return {
        "available": True,
        "overlap_intervals": overlap,
        "n_turns": len(turns),
        "speaker_turns": turns,
        "exclusive_speaker_turns": exclusive_turns,
        "raw_keys": sorted(payload.keys()),
    }


def annotate_manifest(
    in_jsonl: Path,
    out_jsonl: Path,
    *,
    channels: tuple[str, ...] = ("Zoomit", "Kooshiar"),
    limit: int = 32,
) -> dict:
    """Diarize a small podcast subset for overlap intervals."""

    n = 0
    ok = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with in_jsonl.open("r", encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            if n >= limit:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            if row.get("channel") not in channels:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            wav = Path(row.get("audio_filepath") or row.get("audio_path") or "")
            if not wav.is_file():
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            extra = diarize_file(wav)
            n += 1
            if extra.get("available"):
                ok += 1
                row["overlap_intervals"] = extra.get("overlap_intervals") or []
                row["speaker_turns"] = extra.get("speaker_turns") or []
                row["exclusive_speaker_turns"] = extra.get("exclusive_speaker_turns") or []
            else:
                row["diarization_note"] = extra.get("error") or extra.get("note")
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"attempted": n, "ok": ok, "limit": limit, "channels": list(channels)}


def align_reference_segments(
    reference_segments: list[dict],
    speaker_turns: list[dict],
    *,
    min_confidence: float = 0.50,
) -> tuple[list[dict], list[dict], dict]:
    """Assign each provided caption segment by maximum temporal overlap.

    Low-confidence captions remain in the aligned-reference audit trail but are
    excluded from response-pair supervision.
    """

    valid_turns: list[tuple[float, float, str]] = []
    for turn in speaker_turns:
        try:
            start = float(turn["start"])
            end = float(turn["end"])
        except (KeyError, TypeError, ValueError):
            continue
        speaker = str(turn.get("speaker") or turn.get("speaker_id") or "").strip()
        if speaker and 0 <= start < end:
            valid_turns.append((start, end, speaker))

    aligned: list[dict] = []
    usable: list[dict] = []
    confidences: list[float] = []
    for reference in reference_segments:
        try:
            start = float(reference["start"])
            end = float(reference["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= start < end:
            continue
        overlaps: dict[str, float] = defaultdict(float)
        for turn_start, turn_end, speaker in valid_turns:
            overlap = min(end, turn_end) - max(start, turn_start)
            if overlap > 0:
                overlaps[speaker] += overlap
        best_speaker = max(overlaps, key=lambda speaker: overlaps[speaker]) if overlaps else None
        confidence = overlaps.get(str(best_speaker), 0.0) / (end - start)
        item = dict(reference)
        item.update(
            {
                "speaker": best_speaker,
                "alignment_confidence": round(confidence, 4),
                "alignment_status": (
                    "usable" if best_speaker and confidence >= min_confidence else "low_confidence"
                ),
            }
        )
        aligned.append(item)
        confidences.append(confidence)
        if item["alignment_status"] == "usable":
            usable.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": best_speaker,
                    "text": str(reference.get("text") or reference.get("raw_text") or ""),
                    "alignment_confidence": round(confidence, 4),
                }
            )
    coverage = len(usable) / len(aligned) if aligned else 0.0
    stats = {
        "reference_segments": len(aligned),
        "usable_segments": len(usable),
        "usable_coverage": round(coverage, 4),
        "mean_alignment_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        ),
        "min_confidence": min_confidence,
    }
    return aligned, usable, stats


def _speaker_evidence(turns: list[dict]) -> dict:
    durations: dict[str, float] = defaultdict(float)
    for turn in turns:
        try:
            duration = max(0.0, float(turn["end"]) - float(turn["start"]))
        except (KeyError, TypeError, ValueError):
            continue
        speaker = str(turn.get("speaker") or turn.get("speaker_id") or "").strip()
        if speaker:
            durations[speaker] += duration
    ordered = sorted(durations.items(), key=lambda item: (-item[1], item[0]))
    total = sum(durations.values())
    secondary_seconds = ordered[1][1] if len(ordered) >= 2 else 0.0
    return {
        "speaker_seconds": {speaker: round(seconds, 3) for speaker, seconds in ordered},
        "n_speakers": len(ordered),
        "secondary_speaker_seconds": round(secondary_seconds, 3),
        "secondary_speaker_share": round(secondary_seconds / total, 4) if total else 0.0,
    }


def merge_conversation_window_manifests(
    inputs: list[Path],
    out_jsonl: Path,
) -> dict:
    """Merge disjoint primary/reserve window manifests without hiding duplicates."""

    if not inputs:
        raise ValueError("at least one input manifest is required")
    paths = [Path(path) for path in inputs]
    out_jsonl = Path(out_jsonl)
    if any(path.resolve() == out_jsonl.resolve() for path in paths):
        raise ValueError("output manifest must differ from every input manifest")

    rows: list[dict] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    by_input: dict[str, int] = {}
    for path in paths:
        input_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_input[str(path)] = len(input_rows)
        for row in input_rows:
            window_id = str(row.get("window_id") or "")
            if not window_id:
                raise ValueError(f"row without window_id in {path}")
            if window_id in seen:
                duplicates.append(window_id)
                continue
            seen.add(window_id)
            rows.append(row)
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"duplicate window_id values across inputs: {sample}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_jsonl.with_name(f".{out_jsonl.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(out_jsonl)
    verified_hours = sum(
        float(row.get("duration") or 0.0) / 3600.0
        for row in rows
        if row.get("automatic_multi_speaker_verified") is True
    )
    return {
        "inputs": by_input,
        "out_manifest": str(out_jsonl),
        "windows": len(rows),
        "episodes": len({str(row.get("episode_id")) for row in rows if row.get("episode_id")}),
        "channels": dict(Counter(str(row.get("channel") or "unknown") for row in rows)),
        "automatic_multi_speaker_hours": round(verified_hours, 3),
        "duplicate_window_ids": 0,
    }


def audit_diarized_windows(
    manifest: Path,
    out_json: Path | None = None,
    *,
    min_hours: float = 100.0,
    max_hours: float = 200.0,
) -> dict:
    rows = [
        json.loads(line)
        for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total_hours = sum(float(row.get("duration") or 0.0) for row in rows) / 3600.0
    verified_hours = (
        sum(
            float(row.get("duration") or 0.0)
            for row in rows
            if row.get("automatic_multi_speaker_verified") is True
        )
        / 3600.0
    )
    aligned_hours = (
        sum(
            float(row.get("duration") or 0.0)
            for row in rows
            if row.get("reference_alignment_status") == "complete"
        )
        / 3600.0
    )
    episodes = {str(row.get("episode_id")) for row in rows if row.get("episode_id")}
    channels = Counter(str(row.get("channel") or "unknown") for row in rows)
    reviewed = sum(bool(row.get("manual_qa_reviewed")) for row in rows)
    approved = sum(bool(row.get("human_verified")) for row in rows)
    eligible_qa_strata = {
        (
            str(row.get("channel") or "unknown"),
            (
                "automatic_pass"
                if row.get("automatic_multi_speaker_verified") is True
                else "automatic_reject"
            ),
        )
        for row in rows
        if row.get("diarization_status") == "complete"
    }
    reviewed_qa_strata = {
        (
            str(row.get("channel") or "unknown"),
            str((row.get("manual_qa") or {}).get("automatic_status") or ""),
        )
        for row in rows
        if row.get("manual_qa_reviewed")
    }
    rights_verified = sum(rights_record_verified(row) for row in rows)
    requirements = {
        "automatic_multi_speaker_hours_100_to_200": bool(rows)
        and verified_hours > 0
        and min_hours <= verified_hours <= max_hours,
        "reference_aligned_hours_100_to_200": bool(rows)
        and aligned_hours > 0
        and min_hours <= aligned_hours <= max_hours,
        "multiple_episodes": len(episodes) >= 2,
        "multiple_channels": len(channels) >= 2,
        "episode_group_splits_present": all(
            row.get("split") in {"train", "val", "test"} for row in rows
        ),
        "chunk_overlap_limitation_declared": bool(rows)
        and all(row.get("cross_chunk_overlap_recoverable") is False for row in rows),
        "licenses_verified": bool(rows) and rights_verified == len(rows),
        "manual_qa_sample_present": bool(eligible_qa_strata)
        and eligible_qa_strata <= reviewed_qa_strata,
    }
    report = {
        "windows": len(rows),
        "episodes": len(episodes),
        "channels": dict(channels),
        "prepared_hours": round(total_hours, 3),
        "automatic_multi_speaker_hours": round(verified_hours, 3),
        "reference_aligned_hours": round(aligned_hours, 3),
        "manual_qa_reviewed_windows": reviewed,
        "human_verified_windows": approved,
        "manual_qa_expected_strata": [
            f"{channel}/{status}" for channel, status in sorted(eligible_qa_strata)
        ],
        "manual_qa_reviewed_strata": [
            f"{channel}/{status}" for channel, status in sorted(reviewed_qa_strata) if status
        ],
        "license_verified_windows": rights_verified,
        "requirements": requirements,
        "thesis_evidence_gate_passes": all(requirements.values()),
        "warning": "Automatic diarization is pseudo-label evidence until a manual sample is reviewed.",
    }
    if out_json is not None:
        from thesis_s2s.metrics import write_json

        write_json(out_json, report)
    return report


def diarize_episode_manifest(
    in_jsonl: Path,
    out_jsonl: Path,
    *,
    limit: int | None = None,
    base: str | None = None,
    min_alignment_confidence: float = 0.50,
    min_alignment_coverage: float = 0.80,
    min_secondary_speaker_seconds: float = 20.0,
    min_secondary_speaker_share: float = 0.03,
    resume: bool = True,
    stats_path: Path | None = None,
) -> dict:
    """Diarize bounded windows, align CSV text, and verify multi-speaker evidence."""

    done: set[str] = set()
    existing_complete: list[dict] = []
    if resume and Path(out_jsonl).is_file():
        for line in Path(out_jsonl).read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing = json.loads(line)
                if existing.get("window_id") and existing.get("diarization_status") == "complete":
                    if existing.get("license") in {None, "", "youtube-internal"}:
                        existing["license"] = "pending-youtube-rights-review"
                    existing.setdefault("license_verified", False)
                    done.add(str(existing["window_id"]))
                    existing_complete.append(existing)
    in_jsonl = Path(in_jsonl)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    source_windows = sum(
        bool(line.strip()) for line in in_jsonl.read_text(encoding="utf-8").splitlines()
    )
    resumed_verified = [
        row for row in existing_complete if row.get("automatic_multi_speaker_verified") is True
    ]
    resumed_aligned = [
        row for row in existing_complete if row.get("reference_alignment_status") == "complete"
    ]
    stats: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_windows": source_windows,
        "attempted": 0,
        "complete": len(existing_complete),
        "completed_this_run": 0,
        "failed": 0,
        "automatic_multi_speaker_windows": len(resumed_verified),
        "automatic_multi_speaker_hours": sum(
            float(row.get("duration") or 0.0) / 3600.0 for row in resumed_verified
        ),
        "aligned_hours": sum(float(row.get("duration") or 0.0) / 3600.0 for row in resumed_aligned),
        "resumed_windows": len(done),
    }
    stats_path = Path(
        stats_path or out_jsonl.with_name("conversation_episode_diarization_stats.json")
    )

    def write_progress(*, in_progress: bool) -> None:
        snapshot = dict(stats)
        snapshot["automatic_multi_speaker_hours"] = round(
            float(snapshot["automatic_multi_speaker_hours"]), 3
        )
        snapshot["aligned_hours"] = round(float(snapshot["aligned_hours"]), 3)
        snapshot["in_progress"] = in_progress
        snapshot["output_windows"] = int(snapshot["complete"]) + int(snapshot["failed"])
        snapshot["limit"] = limit
        snapshot["source_manifest_complete"] = snapshot["output_windows"] == source_windows
        if not in_progress:
            snapshot["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(stats_path, snapshot)

    write_progress(in_progress=True)
    with (
        in_jsonl.open("r", encoding="utf-8") as src,
        out_jsonl.open("w", encoding="utf-8") as dst,
    ):
        for existing in existing_complete:
            dst.write(json.dumps(existing, ensure_ascii=False) + "\n")
        for line in src:
            if limit is not None and stats["attempted"] >= limit:
                break
            row = json.loads(line)
            if row.get("license") in {None, "", "youtube-internal"}:
                row["license"] = "pending-youtube-rights-review"
            row.setdefault("license_verified", False)
            if str(row.get("window_id") or "") in done:
                continue
            result = diarize_file(Path(str(row.get("audio_filepath") or "")), base)
            stats["attempted"] += 1
            if not result.get("available"):
                row["diarization_status"] = "failed"
                row["diarization_note"] = result.get("error") or result.get("note")
                stats["failed"] += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                dst.flush()
                write_progress(in_progress=True)
                continue
            raw_turns = list(result.get("speaker_turns") or [])
            exclusive = list(result.get("exclusive_speaker_turns") or [])
            alignment_turns = exclusive or raw_turns
            aligned, usable, alignment = align_reference_segments(
                list(row.get("reference_segments") or []),
                alignment_turns,
                min_confidence=min_alignment_confidence,
            )
            speakers = _speaker_evidence(alignment_turns)
            alignment_complete = alignment["usable_coverage"] >= min_alignment_coverage
            verified = (
                speakers["n_speakers"] >= 2
                and speakers["secondary_speaker_seconds"] >= min_secondary_speaker_seconds
                and speakers["secondary_speaker_share"] >= min_secondary_speaker_share
                and alignment_complete
            )
            row.update(
                {
                    "speaker_turns": raw_turns,
                    "exclusive_speaker_turns": exclusive,
                    "overlap_intervals": result.get("overlap_intervals") or [],
                    "reference_segments_aligned": aligned,
                    "segments": usable,
                    "reference_alignment": alignment,
                    "reference_alignment_status": (
                        "complete" if alignment_complete else "insufficient_coverage"
                    ),
                    "speaker_evidence": speakers,
                    "diarization_status": "complete",
                    "automatic_multi_speaker_verified": verified,
                    "conversation_verified": bool(row.get("human_verified")),
                    "annotation_source": "provided_csv_aligned_to_automatic_diarization",
                }
            )
            duration_hours = float(row.get("duration") or 0.0) / 3600.0
            stats["complete"] += 1
            stats["completed_this_run"] += 1
            if alignment_complete:
                stats["aligned_hours"] += duration_hours
            if verified:
                stats["automatic_multi_speaker_windows"] += 1
                stats["automatic_multi_speaker_hours"] += duration_hours
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            dst.flush()
            write_progress(in_progress=True)
    stats["automatic_multi_speaker_hours"] = round(float(stats["automatic_multi_speaker_hours"]), 3)
    write_progress(in_progress=False)
    stats["aligned_hours"] = round(float(stats["aligned_hours"]), 3)
    return stats
