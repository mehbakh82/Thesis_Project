"""Optional Community-1 overlap labels via the existing diarization HTTP service.

Off by default. Only run on a podcast/long-episode subset, never Shorts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from thesis_s2s.audio import read_wav


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
        return {"available": False, "overlap_intervals": [], "note": "DIARIZATION_SERVICE_URL unset"}
    audio, sample_rate = read_wav(wav_path)
    req = Request(
        f"{base}/diarize?sample_rate={sample_rate}",
        data=audio.astype("<f4").tobytes(),
        method="POST",
    )
    req.add_header("content-type", "application/octet-stream")
    try:
        with urlopen(req, timeout=120) as resp:
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


def annotate_manifest(in_jsonl: Path, out_jsonl: Path, *, channels: tuple[str, ...] = ("Zoomit", "Kooshiar"), limit: int = 32) -> dict:
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
