#!/usr/bin/env python3
"""Cut short event clips from conversation windows, locally or from a tar stream.

Windows that exist locally are read from disk. With ``--tar-stream`` the
script also reads an uncompressed tar stream on stdin (for example the
project's S3 backup piped through ``zstd -dc``) and cuts clips from matching
window members on the fly, so whole windows never need to be restored.
Clips are written as 16 kHz mono PCM16 WAV named ``<event_id>.wav``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PREFIX = "data/"


def _read_wav(handle: Any) -> tuple[np.ndarray, int]:
    with wave.open(handle, "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM windows")
        rate = wav.getframerate()
        channels = wav.getnchannels()
        frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    audio = frames.reshape(-1, channels).mean(axis=1) if channels > 1 else frames
    return audio.astype(np.int16), rate


def _write_clip(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(audio.astype("<i2").tobytes())


def cut(audio: np.ndarray, rate: int, events: list[dict[str, Any]], out_dir: Path) -> int:
    written = 0
    for event in events:
        start = int(round(float(event["clip_start"]) * rate))
        end = int(round(float(event["clip_end"]) * rate))
        if start < 0 or end > audio.size or end <= start:
            continue
        _write_clip(out_dir / f"{event['event_id']}.wav", audio[start:end], rate)
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tar-stream", action="store_true")
    args = parser.parse_args()

    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in args.events.read_text(encoding="utf-8").splitlines():
        if line.strip():
            event = json.loads(line)
            if not (args.out_dir / f"{event['event_id']}.wav").is_file():
                by_window[event["audio_filepath"]].append(event)
    written = 0
    remaining = dict(by_window)
    for audio_path in list(remaining):
        local = ROOT / audio_path
        if local.is_file():
            with local.open("rb") as handle:
                audio, rate = _read_wav(handle)
            written += cut(audio, rate, remaining.pop(audio_path), args.out_dir)
    if args.tar_stream and remaining:
        wanted = {
            path[len(DATA_PREFIX) :] if path.startswith(DATA_PREFIX) else path: path
            for path in remaining
        }
        with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or member.name not in wanted:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                audio, rate = _read_wav(io.BytesIO(handle.read()))
                key = wanted.pop(member.name)
                written += cut(audio, rate, remaining.pop(key), args.out_dir)
                print(f"cut {member.name} remaining={len(remaining)}", file=sys.stderr, flush=True)
                if not remaining:
                    break
    print(json.dumps({"clips_written": written, "windows_unresolved": len(remaining)}))
    return 0 if not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
