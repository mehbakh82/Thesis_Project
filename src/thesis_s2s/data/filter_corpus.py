"""Filter conversational hours and mix synthetic barge-in into the thesis corpus."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

from thesis_s2s.audio import read_wav, write_wav
from thesis_s2s.bargein.synthetic import make_clip
from thesis_s2s.data.quality import conversational_ok, estimate_snr_db
from thesis_s2s.data.verbatim import s2s_text, verbatim_normalize
from thesis_s2s.metrics import write_json


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def filter_hours(
    in_jsonl: Path,
    out_jsonl: Path,
    min_hours: float = 100,
    max_hours: float = 200,
    min_snr_db: float = 3.0,
    compute_snr: bool = True,
    require_teacher: bool = False,
) -> dict:
    if not all(math.isfinite(value) for value in (min_hours, max_hours, min_snr_db)):
        raise ValueError("hour and SNR thresholds must be finite")
    if min_hours < 0 or max_hours < 0 or min_hours > max_hours:
        raise ValueError("require 0 <= min_hours <= max_hours")
    rows = []
    for line_number, line in enumerate(
        in_jsonl.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {in_jsonl}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {in_jsonl}:{line_number}")
        try:
            duration = float(row.get("duration") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid duration at {in_jsonl}:{line_number}") from exc
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(
                f"duration must be finite and non-negative at {in_jsonl}:{line_number}"
            )
        rows.append(row)
    rows.sort(key=lambda r: float(r.get("duration") or 0), reverse=True)
    kept = []
    hours = 0.0
    skipped = {
        "no_text": 0,
        "no_teacher": 0,
        "duration": 0,
        "lid": 0,
        "music": 0,
        "not_persian": 0,
        "short_text": 0,
        "low_snr": 0,
        "invalid_snr": 0,
    }
    for row in rows:
        if require_teacher and not str(row.get("transcript_nemo") or "").strip():
            skipped["no_teacher"] += 1
            continue
        text = verbatim_normalize(s2s_text(row))
        if not text.strip():
            skipped["no_text"] += 1
            continue
        duration = float(row.get("duration") or 0)
        snr = row.get("snr")
        if snr is None and compute_snr:
            path = row.get("audio_filepath") or row.get("audio_path")
            if path and Path(path).is_file():
                try:
                    audio, _ = read_wav(path)
                    snr = round(estimate_snr_db(audio), 2)
                    row["snr"] = snr
                except Exception:
                    snr = None
        ok, reason = conversational_ok(
            duration=duration,
            text=text,
            snr_db=float(snr) if snr is not None else None,
            language_status=row.get("language_status"),
            min_snr_db=min_snr_db,
        )
        if not ok:
            key = reason.split(":")[0]
            skipped[key] = skipped.get(key, 0) + 1
            continue
        row_hours = duration / 3600.0
        if hours + row_hours > max_hours:
            continue
        hours += row_hours
        kept_row = dict(row)
        kept_row["text"] = text
        kept.append(kept_row)
    _write_jsonl_atomic(out_jsonl, kept)
    report = {
        "n": len(kept),
        "hours": round(hours, 3),
        "min_hours_ok": hours >= min_hours,
        "max_hours": max_hours,
        "max_hours_ok": hours <= max_hours,
        "min_hours_target": min_hours,
        "require_teacher": require_teacher,
        "skipped": skipped,
        "note": (
            "Legacy NeMo-teacher filter; S2S/TTS should use CSV captions instead."
            if require_teacher
            else "S2S/TTS text is YouTube CSV caption after fa-verbatim-2. NeMo is not the teacher."
        ),
    }
    write_json(out_jsonl.with_name(out_jsonl.stem + "_stats.json"), report)
    return report


def write_synthetic_duplex(
    out_dir: Path,
    manifest: Path,
    n_per_class: int | None = None,
    target_hours: float = 2.0,
    clip_seconds: float = 8.0,
    seed: int = 1,
) -> dict:
    """Write labeled duplex mixtures. Scales toward 20–40 h via target_hours."""

    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = ("none", "interrupt", "backchannel", "noise")
    if n_per_class is None:
        per_clip = max(clip_seconds, 1.5)
        n_total = max(len(kinds), int(target_hours * 3600 / per_clip))
        n_per_class = max(1, n_total // len(kinds))
    hours = 0.0
    n = 0
    rng = np.random.default_rng(seed)
    with manifest.open("w", encoding="utf-8") as handle:
        for kind in kinds:
            for i in range(n_per_class):
                clip = make_clip(kind, rng=rng)
                audio = clip.audio
                if len(audio) / 16000 < clip_seconds:
                    reps = int(np.ceil(clip_seconds * 16000 / max(1, len(audio))))
                    audio = np.tile(audio, reps)[: int(clip_seconds * 16000)]
                path = out_dir / f"duplex_{kind}_{i:05d}.wav"
                write_wav(path, audio)
                dur = len(audio) / 16000
                hours += dur / 3600.0
                n += 1
                handle.write(
                    json.dumps(
                        {
                            "utt_id": path.stem,
                            "audio_filepath": str(path),
                            "duration": round(dur, 3),
                            "interrupt_label": kind,
                            "license": "synthetic",
                            "teacher_tts": "harmonic_mixer",
                            "transcript_caption": None,
                            "transcript_nemo": None,
                            "age_bin": None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    report = {
        "n": n,
        "hours": round(hours, 3),
        "n_per_class": n_per_class,
        "teacher_tts": "harmonic_mixer",
        "target_hours": target_hours,
        "recording_split": "not_collected",
    }
    write_json(manifest.with_name("synthetic_stats.json"), report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--min-hours", type=float, default=100)
    parser.add_argument("--max-hours", type=float, default=200)
    parser.add_argument("--require-teacher", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            filter_hours(
                args.in_jsonl,
                args.out_jsonl,
                args.min_hours,
                args.max_hours,
                require_teacher=args.require_teacher,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
