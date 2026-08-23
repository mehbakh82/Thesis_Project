"""Reconstruct bounded episode windows from ordered YouTube caption chunks."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import to_float32_mono
from thesis_s2s.data.conversation_selection import load_jsonl
from thesis_s2s.data.ingest import _parse_clock, rclone_copy
from thesis_s2s.data.prepare_youtube import caption_ok, chunk_name, iter_csv_rows, load_wav_mono16k
from thesis_s2s.data.verbatim import verbatim_normalize
from thesis_s2s.metrics import write_json

FILTER_SPECIAL = re.compile(r"([\\*?\[\]{}])")
SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rights_metadata_is_explicit(row: dict) -> bool:
    license_name = str(row.get("license") or "").strip()
    verified = row.get("license_verified")
    redistribution_allowed = row.get("redistribution_allowed")
    research_authorized = row.get("internal_research_authorized")
    authorization_basis = row.get("authorization_basis")
    if license_name in {"", "unknown", "youtube-internal"}:
        return False
    if not isinstance(verified, bool):
        return False
    return license_name != "pending-youtube-rights-review" or (
        verified is False
        and research_authorized is False
        and authorization_basis == "pending"
        and redistribution_allowed is False
    )


def rclone_filter_literal(value: str) -> str:
    """Escape literal text embedded in an rclone include pattern."""

    return FILTER_SPECIAL.sub(r"\\\1", value)


def _safe_name(value: str) -> str:
    readable = SAFE_NAME.sub("-", value).strip("-")[:28]
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return f"{readable or 'item'}-{digest}"


def _pcm16(audio: np.ndarray) -> bytes:
    pcm = np.clip(to_float32_mono(audio) * 32767.0, -32768, 32767).astype("<i2")
    return pcm.tobytes()


def _chunk_candidates(chunk_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(chunk_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3"}:
            result.setdefault(path.name, path)
    return result


def _chunk_path(files: dict[str, Path], stem: str, index: int) -> Path | None:
    wav_name = chunk_name(stem, index)
    return files.get(wav_name) or files.get(wav_name.removesuffix(".wav") + ".mp3")


def reconstruct_episode_windows(
    episode: dict,
    chunk_root: Path,
    out_root: Path,
    *,
    window_seconds: float = 900.0,
    max_preserved_gap_s: float = 2.0,
    min_chunk_coverage: float = 0.95,
) -> tuple[list[dict], dict]:
    """Create <=15-minute analysis windows and preserve caption order.

    The source consists of pre-cut caption chunks, not the untouched episode.
    This limitation is carried into every output row and cross-chunk overlap is
    never fabricated.
    """

    if window_seconds < 60:
        raise ValueError("window_seconds must be at least 60")
    if not 0 < min_chunk_coverage <= 1:
        raise ValueError("min_chunk_coverage must be in (0, 1]")

    episode_id = str(episode.get("episode_id") or "")
    stem = str(episode.get("stem") or "")
    csv_path = Path(str(episode.get("csv_path") or ""))
    if not episode_id or not stem or not csv_path.is_file():
        return [], {"episode_id": episode_id, "status": "invalid_episode_record"}

    source_rows = iter_csv_rows(csv_path)
    eligible = [
        (index, row, caption_ok(str(row.get("text") or row.get("transcript") or "")))
        for index, row in enumerate(source_rows, start=1)
    ]
    eligible = [(index, row, text) for index, row, text in eligible if text is not None]
    files = _chunk_candidates(Path(chunk_root))
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".episode-partial-", dir=out_root))
    channel_dir = _safe_name(str(episode.get("channel") or "unknown"))
    episode_dir = _safe_name(episode_id)
    final_dir = out_root / str(episode.get("split") or "train") / channel_dir / episode_dir

    windows: list[dict] = []
    refs: list[dict] = []
    handle: wave.Wave_write | None = None
    cursor_samples = 0
    window_index = 0
    found = 0
    decode_errors = 0
    previous_source_end: float | None = None

    def open_window() -> None:
        nonlocal handle, cursor_samples
        path = staging / f"window_{window_index:04d}.wav"
        handle = wave.open(str(path), "wb")
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        cursor_samples = 0

    def close_window() -> None:
        nonlocal handle, refs, window_index
        if handle is None:
            return
        handle.close()
        handle = None
        if not refs:
            return
        final_path = final_dir / f"window_{window_index:04d}.wav"
        windows.append(
            {
                "window_id": f"{episode_id}#window-{window_index:04d}",
                "session_id": episode_id,
                "recording_id": episode_id,
                "episode_id": episode_id,
                "window_index": window_index,
                "audio_filepath": str(final_path),
                "audio_path": str(final_path),
                "duration": round(cursor_samples / SAMPLE_RATE, 3),
                "reference_segments": refs,
                "segments": [],
                "channel": episode.get("channel"),
                "source_kind": episode.get("source_kind"),
                "expected_multi_speaker": episode.get("expected_multi_speaker"),
                "conversation_candidate_reason": episode.get("conversation_candidate_reason"),
                "source_csv": str(csv_path),
                "reference_transcript_source": "provided_youtube_csv",
                "reference_alignment_status": "pending_diarization",
                "diarization_status": "not_run",
                "automatic_multi_speaker_verified": False,
                "conversation_verified": False,
                "human_verified": False,
                "split": episode.get("split"),
                "license": str(episode.get("license") or "pending-youtube-rights-review"),
                "license_verified": bool(episode.get("license_verified", False)),
                "internal_research_authorized": False,
                "authorization_basis": "pending",
                "redistribution_allowed": False,
                "annotation_source": "provided_csv_plus_pending_diarization",
                "audio_source_kind": "reconstructed_ordered_caption_chunks",
                "is_original_episode_audio": False,
                "cross_chunk_overlap_recoverable": False,
                "overlap_limitation": (
                    "pre-existing caption chunk boundaries may remove cross-boundary overlap"
                ),
            }
        )
        refs = []
        window_index += 1

    try:
        open_window()
        for row_index, row, text in eligible:
            chunk = _chunk_path(files, stem, row_index)
            if chunk is None:
                continue
            try:
                audio, sample_rate = load_wav_mono16k(chunk)
            except Exception:
                decode_errors += 1
                continue
            if sample_rate != SAMPLE_RATE or not len(audio):
                decode_errors += 1
                continue
            source_start = _parse_clock(str(row.get("start_time") or "0"))
            source_end = _parse_clock(str(row.get("end_time") or "0"))
            gap_s = (
                min(max_preserved_gap_s, max(0.0, source_start - previous_source_end))
                if previous_source_end is not None
                else 0.0
            )
            gap_samples = int(round(gap_s * SAMPLE_RATE))
            projected = cursor_samples + gap_samples + len(audio)
            if refs and projected / SAMPLE_RATE > window_seconds:
                close_window()
                open_window()
                gap_samples = 0
            assert handle is not None
            if gap_samples:
                handle.writeframesraw(b"\x00\x00" * gap_samples)
                cursor_samples += gap_samples
            start = cursor_samples / SAMPLE_RATE
            handle.writeframesraw(_pcm16(audio))
            cursor_samples += len(audio)
            end = cursor_samples / SAMPLE_RATE
            refs.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": verbatim_normalize(str(text)),
                    "raw_text": str(text),
                    "speaker": None,
                    "source_start": round(source_start, 3),
                    "source_end": round(source_end, 3),
                    "source_row": row_index,
                    "source_chunk": chunk.name,
                }
            )
            previous_source_end = source_end if source_end > source_start else source_start
            found += 1
        close_window()
    finally:
        if handle is not None:
            handle.close()

    expected = len(eligible)
    coverage = found / expected if expected else 0.0
    report = {
        "episode_id": episode_id,
        "expected_caption_chunks": expected,
        "decoded_chunks": found,
        "decode_errors": decode_errors,
        "chunk_coverage": round(coverage, 5),
        "windows": len(windows),
        "hours": round(sum(float(row["duration"]) for row in windows) / 3600.0, 4),
        "status": "complete" if windows and coverage >= min_chunk_coverage else "failed_coverage",
    }
    if not windows or coverage < min_chunk_coverage:
        shutil.rmtree(staging, ignore_errors=True)
        return [], report
    if final_dir.exists():
        shutil.rmtree(staging, ignore_errors=True)
        report["status"] = "failed_existing_untracked_output"
        return [], report
    final_dir.mkdir(parents=True)
    for row in windows:
        source = staging / Path(str(row["audio_filepath"])).name
        shutil.move(str(source), str(Path(str(row["audio_filepath"]))))
    shutil.rmtree(staging, ignore_errors=True)
    return windows, report


def preparation_stats_path(manifest: Path) -> Path:
    """Keep the historical primary name but isolate alternate selections."""

    manifest = Path(manifest)
    if manifest.name == "conversation_episode_windows.jsonl":
        return manifest.with_name("conversation_episode_prepare_stats.json")
    return manifest.with_name(f"{manifest.stem}_prepare_stats.json")


def _manifest_progress(path: Path) -> tuple[set[str], int, float]:
    done: set[str] = set()
    windows = 0
    hours = 0.0
    if not path.is_file():
        return done, windows, hours
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("episode_id"):
            done.add(str(row["episode_id"]))
        windows += 1
        hours += float(row.get("duration") or 0.0) / 3600.0
    return done, windows, hours


def audit_prepared_episode_windows(
    selection_jsonl: Path,
    manifest: Path,
    out_json: Path | None = None,
    *,
    stats_path: Path | None = None,
    min_hours: float = 100.0,
    max_hours: float = 240.0,
    max_window_seconds: float = 900.0,
    check_files: bool = True,
) -> dict:
    """Validate reconstruction completeness, provenance, and WAV integrity."""

    selection_jsonl = Path(selection_jsonl)
    manifest = Path(manifest)
    selected = load_jsonl(selection_jsonl)
    rows = load_jsonl(manifest) if manifest.is_file() else []
    selected_ids = [str(row.get("episode_id") or "") for row in selected]
    selected_id_set = {value for value in selected_ids if value}
    prepared_ids = {str(row.get("episode_id") or "") for row in rows if row.get("episode_id")}
    window_ids = [str(row.get("window_id") or "") for row in rows]
    total_hours = sum(float(row.get("duration") or 0.0) for row in rows) / 3600.0
    selected_candidate_hours = sum(float(row.get("csv_hours") or 0.0) for row in selected)

    expected_splits = {
        str(row.get("episode_id") or ""): str(row.get("split") or "")
        for row in selected
        if row.get("episode_id")
    }
    episode_splits: dict[str, set[str]] = {}
    episode_indices: dict[str, list[int]] = {}
    episode_source_rows: dict[str, list[int]] = {}
    schema_errors = 0
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        episode_splits.setdefault(episode_id, set()).add(str(row.get("split") or ""))
        try:
            episode_indices.setdefault(episode_id, []).append(int(row["window_index"]))
        except (KeyError, TypeError, ValueError):
            schema_errors += 1
        references = list(row.get("reference_segments") or [])
        if not references:
            schema_errors += 1
        for reference in references:
            try:
                episode_source_rows.setdefault(episode_id, []).append(int(reference["source_row"]))
                start = float(reference["start"])
                end = float(reference["end"])
            except (KeyError, TypeError, ValueError):
                schema_errors += 1
                continue
            if not 0 <= start < end <= float(row.get("duration") or 0.0) + 0.05:
                schema_errors += 1

    split_errors = sum(
        splits != {expected_splits.get(episode_id, "")}
        for episode_id, splits in episode_splits.items()
    )
    index_errors = sum(indices != list(range(len(indices))) for indices in episode_indices.values())
    source_row_errors = sum(
        values != sorted(set(values)) for values in episode_source_rows.values()
    )

    missing_files = 0
    wav_format_errors = 0
    duration_mismatches = 0
    if check_files:
        for row in rows:
            path = Path(str(row.get("audio_filepath") or ""))
            if not path.is_file():
                missing_files += 1
                continue
            try:
                with wave.open(str(path), "rb") as handle:
                    sample_rate = handle.getframerate()
                    channels = handle.getnchannels()
                    sample_width = handle.getsampwidth()
                    actual_duration = handle.getnframes() / sample_rate
            except (OSError, EOFError, wave.Error, ZeroDivisionError):
                wav_format_errors += 1
                continue
            if sample_rate != SAMPLE_RATE or channels != 1 or sample_width != 2:
                wav_format_errors += 1
            if abs(actual_duration - float(row.get("duration") or 0.0)) > 0.05:
                duration_mismatches += 1

    stats_path = Path(stats_path or preparation_stats_path(manifest))
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    provenance_complete = bool(rows) and all(
        row.get("audio_source_kind") == "reconstructed_ordered_caption_chunks"
        and row.get("is_original_episode_audio") is False
        and row.get("cross_chunk_overlap_recoverable") is False
        and row.get("reference_transcript_source") == "provided_youtube_csv"
        for row in rows
    )
    legacy_rights_rows = sum(not _rights_metadata_is_explicit(row) for row in rows)
    stats_current = (
        bool(stats.get("finished_at"))
        and int(stats.get("windows") or -1) == len(rows)
        and stats.get("in_progress") in (None, False)
        and int(stats.get("episodes_prepared") or 0) + int(stats.get("episodes_resumed") or 0)
        == len(prepared_ids)
    )
    requirements = {
        "selected_candidate_hours_100_to_200": bool(selected)
        and min_hours <= selected_candidate_hours <= 200.0,
        "prepared_audio_hours_at_least_100": bool(rows) and total_hours >= min_hours,
        "prepared_audio_hours_within_safety_cap": total_hours <= max_hours,
        "selection_complete": bool(selected_id_set) and prepared_ids == selected_id_set,
        "selection_episode_ids_unique": bool(selected_ids)
        and len(selected_ids) == len(selected_id_set)
        and all(selected_ids),
        "no_extra_episodes": prepared_ids <= selected_id_set,
        "window_ids_unique": bool(window_ids)
        and len(window_ids) == len(set(window_ids))
        and all(window_ids),
        "window_durations_bounded": bool(rows)
        and all(0 < float(row.get("duration") or 0.0) <= max_window_seconds + 0.05 for row in rows),
        "reference_schema_valid": schema_errors == 0,
        "source_rows_ordered_and_unique": source_row_errors == 0,
        "window_indices_contiguous": index_errors == 0,
        "episode_splits_preserved": split_errors == 0,
        "multiple_channels": len({str(row.get("channel") or "") for row in rows}) >= 2,
        "chunk_limitations_declared": provenance_complete,
        "rights_metadata_explicit": legacy_rights_rows == 0,
        "preparation_report_current_and_finished": stats_current,
        "no_reported_failures": stats_current and int(stats.get("episodes_failed") or 0) == 0,
        "audio_files_present": missing_files == 0 if check_files else None,
        "wav_format_valid": wav_format_errors == 0 if check_files else None,
        "manifest_duration_matches_wav": duration_mismatches == 0 if check_files else None,
    }
    report = {
        "selection_manifest": str(selection_jsonl),
        "window_manifest": str(manifest),
        "preparation_stats": str(stats_path),
        "artifact_sha256": {
            "selection_manifest": _sha256_file(selection_jsonl),
            "window_manifest": _sha256_file(manifest) if manifest.is_file() else None,
            "preparation_stats": _sha256_file(stats_path) if stats_path.is_file() else None,
        },
        "selected_episodes": len(selected_id_set),
        "prepared_episodes": len(prepared_ids),
        "missing_selected_episode_count": len(selected_id_set - prepared_ids),
        "missing_selected_episode_sample": sorted(selected_id_set - prepared_ids)[:50],
        "extra_episodes": sorted(prepared_ids - selected_id_set),
        "windows": len(rows),
        "prepared_audio_hours": round(total_hours, 3),
        "selected_candidate_hours": round(selected_candidate_hours, 3),
        "schema_errors": schema_errors,
        "split_errors": split_errors,
        "legacy_or_implicit_rights_rows": legacy_rights_rows,
        "window_index_errors": index_errors,
        "source_row_order_errors": source_row_errors,
        "duplicate_window_ids": len(window_ids) - len(set(window_ids)),
        "missing_audio_files": missing_files if check_files else None,
        "wav_format_errors": wav_format_errors if check_files else None,
        "duration_mismatches": duration_mismatches if check_files else None,
        "reported_failed_episodes": stats.get("episodes_failed"),
        "requirements": requirements,
        "reconstruction_gate_passes": all(value is True for value in requirements.values()),
    }
    if out_json is not None:
        write_json(out_json, report)
    return report


def prepare_selected_episodes(
    selection_jsonl: Path,
    out_root: Path,
    out_manifest: Path,
    *,
    max_episodes: int | None = None,
    max_source_hours: float | None = None,
    window_seconds: float = 900.0,
    min_chunk_coverage: float = 0.95,
    resume: bool = True,
) -> dict:
    selected = load_jsonl(Path(selection_jsonl))
    done, window_count, prepared_hours = (
        _manifest_progress(Path(out_manifest)) if resume else (set(), 0, 0.0)
    )
    selected_ids = {str(episode.get("episode_id") or "") for episode in selected}
    resumed_ids = done & selected_ids
    resumed_source_hours = sum(
        float(episode.get("csv_hours") or 0.0)
        for episode in selected
        if str(episode.get("episode_id") or "") in resumed_ids
    )
    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and out_manifest.is_file() else "w"
    stats: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "selected_episodes": len(selected),
        "episodes_prepared": 0,
        "episodes_failed": 0,
        "episodes_resumed": len(resumed_ids),
        "windows": window_count,
        "prepared_audio_hours": prepared_hours,
        "source_candidate_hours": resumed_source_hours,
        "failures": Counter(),
        "episode_reports": [],
    }
    stats_path = preparation_stats_path(out_manifest)

    def write_progress() -> None:
        snapshot = {key: value for key, value in stats.items() if key != "episode_reports"}
        snapshot.update(
            {
                "in_progress": True,
                "episode_report_count": len(stats["episode_reports"]),
                "prepared_audio_hours": round(float(stats["prepared_audio_hours"]), 3),
                "source_candidate_hours": round(float(stats["source_candidate_hours"]), 3),
                "failures": dict(stats["failures"]),
            }
        )
        write_json(stats_path, snapshot)

    write_progress()
    with out_manifest.open(mode, encoding="utf-8") as dst:
        for episode in selected:
            if max_episodes is not None and stats["episodes_prepared"] >= max_episodes:
                break
            episode_id = str(episode.get("episode_id") or "")
            if episode_id in done:
                continue
            source_hours = float(episode.get("csv_hours") or 0.0)
            if (
                max_source_hours is not None
                and stats["source_candidate_hours"] + source_hours > max_source_hours
                and stats["episodes_prepared"] > 0
            ):
                continue
            tmp = Path(tempfile.mkdtemp(prefix="conversation-episode-"))
            try:
                escaped_stem = rclone_filter_literal(str(episode.get("stem") or ""))
                rclone_copy(
                    str(episode["chunks_remote"]),
                    tmp,
                    include=f"{escaped_stem}_chunk_*.wav",
                )
                if not any(tmp.rglob("*.wav")):
                    rclone_copy(
                        str(episode["chunks_remote"]),
                        tmp,
                        include=f"{escaped_stem}_chunk_*.mp3",
                    )
                windows, report = reconstruct_episode_windows(
                    episode,
                    tmp,
                    Path(out_root),
                    window_seconds=window_seconds,
                    min_chunk_coverage=min_chunk_coverage,
                )
                stats["episode_reports"].append(report)
                if not windows:
                    stats["episodes_failed"] += 1
                    stats["failures"][str(report.get("status"))] += 1
                    continue
                for row in windows:
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                dst.flush()
                stats["episodes_prepared"] += 1
                stats["windows"] += len(windows)
                stats["source_candidate_hours"] += source_hours
                stats["prepared_audio_hours"] += (
                    sum(float(row.get("duration") or 0.0) for row in windows) / 3600.0
                )
            except subprocess.CalledProcessError:
                stats["episodes_failed"] += 1
                stats["failures"]["s3_download"] += 1
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                write_progress()
    stats["prepared_audio_hours"] = round(float(stats["prepared_audio_hours"]), 3)
    stats["source_candidate_hours"] = round(float(stats["source_candidate_hours"]), 3)
    stats["failures"] = dict(stats["failures"])
    stats["in_progress"] = False
    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(stats_path, stats)
    return stats
