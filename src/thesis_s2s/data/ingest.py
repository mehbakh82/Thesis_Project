"""Multi-channel S3 ingest: CSV inventory first, then per-episode chunk stream."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from thesis_s2s.audio import write_wav
from thesis_s2s.config import project_root
from thesis_s2s.data.channels import YOUTUBE_CHANNELS, ChannelSpec, remote_chunks, remote_csv
from thesis_s2s.data.prepare_youtube import (
    MAX_DURATION,
    MIN_DURATION,
    MIN_EPISODE_ROWS,
    caption_ok,
    chunk_name,
    episode_split,
    iter_csv_rows,
    load_wav_mono16k,
)
from thesis_s2s.data.quality import estimate_snr_db
from thesis_s2s.data.s3_inventory import rclone_prefix, rclone_process_env
from thesis_s2s.data.verbatim import verbatim_normalize
from thesis_s2s.metrics import write_json

CONVERSATION_TITLE = re.compile(
    r"(?:پادکست|گفت[‌ ]?وگو|مصاحبه|مهمان|podcast|interview|\bep(?:isode)?\s*\d+)",
    re.IGNORECASE,
)


def rclone_run(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = rclone_prefix() + args
    return subprocess.run(
        cmd, check=check, capture_output=True, text=True, env=rclone_process_env()
    )


def rclone_copy(remote: str, dest: Path, include: str | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    args = ["copy", remote, str(dest), "--transfers", "8"]
    if include:
        args.extend(["--include", include])
    rclone_run(args)


def rclone_lsf(remote: str, include: str = "*.csv") -> list[str]:
    result = rclone_run(["lsf", remote, "--include", include])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _csv_duration_hours(rows: list[dict]) -> float:
    seconds = 0.0
    for row in rows:
        start = _parse_clock(str(row.get("start_time") or "0"))
        end = _parse_clock(str(row.get("end_time") or "0"))
        if end > start:
            seconds += end - start
    return seconds / 3600.0


def _timeline_hours(rows: list[dict]) -> float:
    starts = [_parse_clock(str(row.get("start_time") or "0")) for row in rows]
    ends = [_parse_clock(str(row.get("end_time") or "0")) for row in rows]
    valid = [(start, end) for start, end in zip(starts, ends, strict=False) if end > start]
    if not valid:
        return 0.0
    return (max(end for _, end in valid) - min(start for start, _ in valid)) / 3600.0


def _conversation_candidate(spec: ChannelSpec, stem: str) -> tuple[bool, str]:
    if spec.expected_multi_speaker is True:
        return True, "podcast_prior_needs_diarization"
    if spec.expected_multi_speaker is None:
        return True, "mixed_channel_needs_diarization"
    if CONVERSATION_TITLE.search(stem):
        return True, "interview_title_needs_diarization"
    return False, "likely_monologue_excluded_by_default"


def _parse_clock(value: str) -> float:
    value = value.strip()
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        pass
    parts = value.replace(",", ".").split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[-1] if nums else 0.0


def inventory_csvs(
    csv_root: Path,
    channels: tuple[ChannelSpec, ...] = YOUTUBE_CHANNELS,
    min_episode_rows: int = MIN_EPISODE_ROWS,
) -> dict:
    """Download CSVs only and estimate hours before pulling audio."""

    episodes = []
    stats = {"channels": {}, "csv_hours": 0.0, "episodes_ok": 0, "shorts_skipped": 0}
    for spec in channels:
        dest = csv_root / spec.name / "csvs"
        already = dest.is_dir() and any(dest.rglob("*.csv"))
        try:
            if not already:
                rclone_copy(remote_csv(spec), dest, include="*.csv")
        except subprocess.CalledProcessError as exc:
            stats["channels"][spec.name] = {"error": (exc.stderr or str(exc))[:300]}
            continue
        hours = 0.0
        n_ok = 0
        n_short = 0
        for csv_path in sorted(dest.rglob("*.csv")):
            if csv_path.name.lower() == "no_captions_found.csv":
                continue
            rows = iter_csv_rows(csv_path)
            if len(rows) < min_episode_rows:
                n_short += 1
                continue
            dur_h = _csv_duration_hours(rows)
            timeline_h = _timeline_hours(rows)
            candidate, candidate_reason = _conversation_candidate(spec, csv_path.stem)
            hours += dur_h
            n_ok += 1
            episodes.append(
                {
                    "channel": spec.name,
                    "episode_id": f"{spec.name}/{csv_path.stem}",
                    "stem": csv_path.stem,
                    "csv_path": str(csv_path),
                    "n_rows": len(rows),
                    "csv_hours": round(dur_h, 4),
                    "timeline_hours": round(timeline_h, 4),
                    "chunks_remote": remote_chunks(spec),
                    "source_kind": spec.source_kind,
                    "conversation_priority": spec.conversation_priority,
                    "expected_multi_speaker": spec.expected_multi_speaker,
                    "selection_weight": spec.selection_weight,
                    "selection_eligible": candidate,
                    "conversation_candidate_reason": candidate_reason,
                    "conversation_verified": False,
                    "diarization_status": "not_run",
                    "reference_transcript_source": "provided_youtube_csv",
                    "audio_source_kind": "ordered_caption_chunks",
                    "license": "pending-youtube-rights-review",
                    "license_verified": False,
                    "internal_research_authorized": False,
                    "authorization_basis": "pending",
                    "redistribution_allowed": False,
                    "split": episode_split(f"{spec.name}/{csv_path.stem}"),
                }
            )
        stats["channels"][spec.name] = {
            "episodes_ok": n_ok,
            "shorts_skipped": n_short,
            "csv_hours": round(hours, 3),
        }
        stats["csv_hours"] += hours
        stats["episodes_ok"] += n_ok
        stats["shorts_skipped"] += n_short
    stats["csv_hours"] = round(stats["csv_hours"], 3)
    stats["n_episode_records"] = len(episodes)
    return {"stats": stats, "episodes": episodes}


def _manifest_progress(manifest_path: Path) -> tuple[set[str], set[tuple[str, str]], float, int]:
    seen: set[str] = set()
    stems: set[tuple[str, str]] = set()
    hours = 0.0
    written = 0
    if not manifest_path.is_file():
        return seen, stems, hours, written
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row.get("utt_id") or "")
            if uid:
                seen.add(uid)
            hours += float(row.get("duration") or 0) / 3600.0
            written += 1
            ch = str(row.get("channel") or "")
            stem = Path(str(row.get("source_csv") or "")).stem
            if ch and stem:
                stems.add((ch, stem))
    return seen, stems, hours, written


def ingest_episodes(
    episodes: list[dict],
    out_dir: Path,
    manifest_path: Path,
    *,
    max_hours: float = 10.0,
    max_episodes: int | None = None,
    min_episode_rows: int = MIN_EPISODE_ROWS,
    resume: bool = True,
) -> dict:
    """Stream one episode of chunks at a time. Does not dump the full 900 h locally."""

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids, done_stems, hours0, written0 = (
        _manifest_progress(manifest_path) if resume else (set(), set(), 0.0, 0)
    )
    stats = {
        "written": written0,
        "hours": hours0,
        "episodes": 0,
        "missing_wav": 0,
        "skipped_caption": 0,
        "errors": 0,
        "max_hours": max_hours,
        "resumed_hours": round(hours0, 3),
        "skipped_done_episodes": 0,
    }
    mode = "a" if resume and manifest_path.is_file() else "w"
    with manifest_path.open(mode, encoding="utf-8") as dst:
        for ep in episodes:
            if max_episodes is not None and stats["episodes"] >= max_episodes:
                break
            if stats["hours"] >= max_hours:
                break
            key = (ep["channel"], ep["stem"])
            if resume and key in done_stems:
                stats["skipped_done_episodes"] += 1
                continue
            tmp = Path(tempfile.mkdtemp(prefix="yt-ep-"))
            try:
                rclone_copy(ep["chunks_remote"], tmp, include=f"{ep['stem']}_chunk_*.wav")
                # some trees use mp3
                if not any(tmp.rglob(f"{ep['stem']}_chunk_*")):
                    rclone_copy(ep["chunks_remote"], tmp, include=f"{ep['stem']}_chunk_*.mp3")
                csv_path = Path(ep["csv_path"])
                rows = iter_csv_rows(csv_path)
                if len(rows) < min_episode_rows:
                    continue
                split = episode_split(ep["stem"])
                stats["episodes"] += 1
                for i, row in enumerate(rows, start=1):
                    if stats["hours"] >= max_hours:
                        break
                    caption = caption_ok(str(row.get("text") or row.get("transcript") or ""))
                    if caption is None:
                        stats["skipped_caption"] += 1
                        continue
                    wav_path = tmp / chunk_name(ep["stem"], i)
                    if not wav_path.is_file():
                        matches = list(tmp.rglob(chunk_name(ep["stem"], i)))
                        if not matches:
                            stem_mp3 = chunk_name(ep["stem"], i).replace(".wav", ".mp3")
                            matches = list(tmp.rglob(stem_mp3))
                        if not matches:
                            stats["missing_wav"] += 1
                            continue
                        wav_path = matches[0]
                    try:
                        audio, sr = load_wav_mono16k(wav_path)
                    except Exception:
                        stats["errors"] += 1
                        continue
                    duration = len(audio) / sr
                    if duration < MIN_DURATION or duration > MAX_DURATION:
                        stats["skipped_caption"] += 1
                        continue
                    dest = (
                        out_dir
                        / split
                        / ep["channel"]
                        / ep["stem"]
                        / f"{ep['stem']}_chunk_{i:04d}.wav"
                    )
                    utt_id = f"{ep['channel']}_{ep['stem']}_{i:04d}"
                    if resume and (utt_id in seen_ids or dest.is_file()):
                        continue
                    write_wav(dest, audio, sr)
                    record = {
                        "utt_id": utt_id,
                        "audio_filepath": str(dest),
                        "audio_path": str(dest),
                        "duration": round(duration, 3),
                        "transcript_caption": caption,
                        "transcript_nemo": None,
                        "text": verbatim_normalize(caption),
                        "speaker_id": None,
                        "overlap_intervals": [],
                        "interrupt_label": "none",
                        "snr": round(estimate_snr_db(audio), 2),
                        "license": "pending-youtube-rights-review",
                        "license_verified": False,
                        "internal_research_authorized": False,
                        "authorization_basis": "pending",
                        "redistribution_allowed": False,
                        "age_bin": None,
                        "split": split,
                        "channel": ep["channel"],
                        "source_csv": str(csv_path),
                    }
                    dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                    stats["hours"] += duration / 3600.0
                    seen_ids.add(utt_id)
                    if stats["written"] % 250 == 0:
                        dst.flush()
                        snap = dict(stats)
                        snap["hours"] = round(snap["hours"], 3)
                        snap["in_progress"] = True
                        write_json(manifest_path.with_name("ingest_stats.json"), snap)
            except subprocess.CalledProcessError:
                stats["errors"] += 1
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    stats["hours"] = round(stats["hours"], 3)
    stats["finished_at"] = datetime.utcnow().isoformat() + "Z"
    write_json(manifest_path.with_name("ingest_stats.json"), stats)
    return stats


def run_ingest(
    *,
    max_hours: float = 10.0,
    max_episodes: int | None = None,
    csv_root: Path | None = None,
    out_dir: Path | None = None,
    manifest_dir: Path | None = None,
    resume: bool = True,
) -> dict:
    root = project_root()
    csv_root = Path(csv_root or root / "data" / "raw" / "youtube_csvs")
    out_dir = Path(out_dir or root / "data" / "processed" / "youtube")
    manifest_dir = Path(manifest_dir or root / "data" / "processed" / "manifests")
    inv = inventory_csvs(csv_root)
    write_json(
        manifest_dir / "csv_inventory.json",
        {"stats": inv["stats"], "n_episodes": len(inv["episodes"]), "episodes": inv["episodes"]},
    )
    audio_stats = ingest_episodes(
        inv["episodes"],
        out_dir,
        manifest_dir / "youtube_all.jsonl",
        max_hours=max_hours,
        max_episodes=max_episodes,
        resume=resume,
    )
    return {"csv_inventory": inv["stats"], "audio": audio_stats}
