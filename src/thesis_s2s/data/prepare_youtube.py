"""Generalize Tabaghe16 S3/local chunk prep without copying secrets or ASR normalise()."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import write_wav
from thesis_s2s.data.verbatim import verbatim_normalize

MIN_EPISODE_ROWS = 50
MIN_DURATION = 1.0
MAX_DURATION = 20.0
MIN_WORDS = 2
BRACKET = re.compile(r"\[.*?\]|&nbsp;", re.DOTALL)


def episode_split(stem: str, val_pct: int = 3, test_pct: int = 5) -> str:
    bucket = int(hashlib.md5(stem.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


def chunk_name(stem: str, row_index_1: int) -> str:
    return f"{stem}_chunk_{row_index_1:04d}.wav"


def load_wav_mono16k(path: Path) -> tuple[np.ndarray, int]:
    from thesis_s2s.audio import read_wav

    return read_wav(path, SAMPLE_RATE)


def caption_ok(text: str) -> str | None:
    text = BRACKET.sub(" ", text).strip()
    if not text:
        return None
    words = text.split()
    if len(words) < MIN_WORDS:
        return None
    return text


def rclone_copy_episode(remote_chunks: str, stem: str, dest: Path) -> None:
    from thesis_s2s.data.s3_inventory import rclone_prefix, rclone_process_env

    dest.mkdir(parents=True, exist_ok=True)
    cmd = rclone_prefix() + [
        "copy",
        remote_chunks,
        str(dest),
        "--include",
        f"{stem}_chunk_*.wav",
        "--transfers",
        "8",
    ]
    subprocess.run(cmd, check=True, env=rclone_process_env())


def iter_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def prepare(
    csv_dir: Path,
    wav_dir: Path | None,
    out_dir: Path,
    manifest_dir: Path,
    *,
    remote_chunks: str | None = None,
    min_episode_rows: int = MIN_EPISODE_ROWS,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (manifest_dir / f"{split}_manifest_youtube.jsonl").open("w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    stats = {"written": 0, "skipped_short_episode": 0, "skipped_caption": 0, "missing_wav": 0, "hours": 0.0}
    csv_files = sorted(csv_dir.rglob("*.csv"))
    try:
        for csv_path in csv_files:
            if csv_path.name.lower() in {"no_captions_found.csv"}:
                continue
            stem = csv_path.stem
            rows = iter_csv_rows(csv_path)
            if len(rows) < min_episode_rows:
                stats["skipped_short_episode"] += 1
                continue
            split = episode_split(stem)
            local_wav_root = wav_dir
            tmp = None
            if wav_dir is None:
                if not remote_chunks:
                    raise ValueError("need --wav-dir or --remote-chunks")
                tmp = Path(tempfile.mkdtemp(prefix="yt-ep-"))
                rclone_copy_episode(remote_chunks, stem, tmp)
                local_wav_root = tmp
            assert local_wav_root is not None
            for i, row in enumerate(rows, start=1):
                caption = caption_ok(str(row.get("text") or row.get("transcript") or ""))
                if caption is None:
                    stats["skipped_caption"] += 1
                    continue
                wav_path = local_wav_root / chunk_name(stem, i)
                if not wav_path.is_file():
                    # some trees nest by channel
                    matches = list(local_wav_root.rglob(chunk_name(stem, i)))
                    if not matches:
                        stats["missing_wav"] += 1
                        continue
                    wav_path = matches[0]
                try:
                    audio, sr = load_wav_mono16k(wav_path)
                except Exception:
                    stats["missing_wav"] += 1
                    continue
                duration = len(audio) / sr
                if duration < MIN_DURATION or duration > MAX_DURATION:
                    stats["skipped_caption"] += 1
                    continue
                dest = out_dir / split / stem / f"{stem}_chunk_{i:04d}.wav"
                write_wav(dest, audio, sr)
                record = {
                    "utt_id": f"{stem}_{i:04d}",
                    "audio_filepath": str(dest),
                    "audio_path": str(dest),
                    "duration": round(duration, 3),
                    "transcript_caption": caption,
                    "transcript_nemo": None,
                    "text": verbatim_normalize(caption),
                    "speaker_id": None,
                    "overlap_intervals": [],
                    "interrupt_label": "none",
                    "snr": None,
                    "license": "pending-youtube-rights-review",
                    "license_verified": False,
                    "age_bin": None,
                    "split": split,
                    "source_csv": str(csv_path),
                }
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["written"] += 1
                stats["hours"] += duration / 3600.0
            if tmp is not None:
                import shutil

                shutil.rmtree(tmp, ignore_errors=True)
    finally:
        for handle in handles.values():
            handle.close()
    stats["hours"] = round(stats["hours"], 3)
    (manifest_dir / "prepare_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, default=None)
    parser.add_argument("--remote-chunks", type=str, default=None, help="rclone remote e.g. :s3:asr/youtube/chunks")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--min-episode-rows", type=int, default=MIN_EPISODE_ROWS)
    args = parser.parse_args(argv)
    stats = prepare(
        args.csv_dir,
        args.wav_dir,
        args.out_dir,
        args.manifest_dir,
        remote_chunks=args.remote_chunks,
        min_episode_rows=args.min_episode_rows,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
