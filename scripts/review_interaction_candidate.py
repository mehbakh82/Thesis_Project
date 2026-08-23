#!/usr/bin/env python3
"""Play one exact excerpt from the interruption-QA CSV without copying audio."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/processed/manifests/conversation_interruption_qa.csv"),
    )
    parser.add_argument("--row", type=int, required=True, help="1-based data-row number")
    parser.add_argument(
        "--show-only",
        action="store_true",
        help="print metadata and the safe argv without opening the player",
    )
    args = parser.parse_args()
    if args.row < 1:
        parser.error("--row must be at least 1")
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.row > len(rows):
        parser.error(f"--row must be at most {len(rows)}")
    row = rows[args.row - 1]
    audio_path = Path(str(row.get("audio_filepath") or ""))
    if not audio_path.is_file():
        parser.error(f"audio file does not exist: {audio_path}")
    start = float(row["listen_start_s"])
    duration = float(row["listen_duration_s"])
    if start < 0 or duration <= 0:
        parser.error("CSV contains an invalid listening interval")
    command = [
        "ffplay",
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-t",
        str(duration),
        str(audio_path),
    ]
    summary = {
        key: row.get(key)
        for key in (
            "candidate_id",
            "channel",
            "episode_id",
            "automatic_candidate",
            "overlap_seconds",
            "user_text",
            "response_text",
        )
    }
    summary["row"] = args.row
    summary["rows_total"] = len(rows)
    summary["command"] = shlex.join(command)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.show_only:
        return
    if shutil.which("ffplay") is None:
        parser.error("ffplay is not installed; use the printed start/duration in another player")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
