"""Export authorized response pairs for Kyutai's official Moshi LoRA trainer."""

from __future__ import annotations

import hashlib
import json
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import read_wav
from thesis_s2s.data.rights import training_use_authorized

MANA_PIPER_REVISION = "ad9dd8518bedf517bd7cbc9f63b8e5c844bf5bc0"
MANA_PIPER_SHA256 = "e390c0e74ba71fd97c49ba662ee0c6e1724b462ba2d4561698af4f564840f126"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe_pair_name(value: str) -> str:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=12).hexdigest()
    return f"pair-{digest}"


def _write_stereo_wav(path: Path, stereo: np.ndarray) -> None:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError(f"expected [samples, 2] stereo audio, got {stereo.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(stereo, -1.0, 1.0)
    pcm = np.clip(pcm * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _relative_starts(row: dict, user_samples: int) -> tuple[int, int]:
    try:
        span_start = float(row["source_span_start"])
        user_interval = row["source_user_interval"]
        response_interval = row["source_response_interval"]
        user_start = max(0.0, float(user_interval[0]) - span_start)
        response_start = max(0.0, float(response_interval[0]) - span_start)
    except (KeyError, TypeError, ValueError, IndexError):
        user_start = 0.0
        response_start = max(
            0.0,
            user_samples / SAMPLE_RATE + float(row.get("response_gap_s") or 0.0),
        )
    return round(user_start * SAMPLE_RATE), round(response_start * SAMPLE_RATE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_moshi_finetune_dataset(
    conversation_manifest: Path,
    out_dir: Path,
    *,
    report_path: Path | None = None,
    max_pairs: int | None = None,
    assistant_audio_mode: str = "piper",
) -> dict:
    """Create Moshi's stereo dialogue format from non-reused response pairs.

    Moshi expects channel 0 to be the model/assistant stream and channel 1 to
    be the user stream. Each WAV receives an adjacent JSON transcript with the
    assistant caption aligned to its original response interval.
    """

    conversation_manifest = Path(conversation_manifest)
    out_dir = Path(out_dir)
    if assistant_audio_mode not in {"piper", "source"}:
        raise ValueError("assistant_audio_mode must be 'piper' or 'source'")
    piper_model: Path | None = None
    piper_model_sha256: str | None = None
    if assistant_audio_mode == "piper":
        from thesis_s2s.runtime.tts import os_piper_model

        piper_model = os_piper_model()
        if piper_model is None or not piper_model.is_file():
            raise RuntimeError(
                "the primary Moshi export requires the pinned Persian Piper voice; "
                "set PIPER_MODEL or install models/piper/fa_IR-mana-medium.onnx"
            )
        piper_model_sha256 = _sha256(piper_model)
        if piper_model_sha256 != MANA_PIPER_SHA256:
            raise RuntimeError(
                "Persian Piper voice hash does not match the pinned training target: "
                f"{piper_model_sha256}"
            )
    rows = _read_jsonl(conversation_manifest)
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        rows = rows[:max_pairs]
    if not rows:
        raise ValueError(f"no conversation pairs in {conversation_manifest}")

    unauthorized = [
        str(row.get("utt_id") or "unknown") for row in rows if not training_use_authorized(row)
    ]
    if unauthorized:
        raise PermissionError(
            "Moshi export requires documented training authorization for every pair; "
            f"unauthorized: {', '.join(unauthorized[:5])}"
        )

    split_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    counts: Counter[str] = Counter()
    exported_seconds = 0.0
    seen_ids: set[str] = set()
    for row in rows:
        pair_id = str(row.get("utt_id") or "").strip()
        if not pair_id:
            counts["missing_pair_id"] += 1
            continue
        if pair_id in seen_ids:
            raise ValueError(f"duplicate conversation pair ID: {pair_id}")
        seen_ids.add(pair_id)
        split = str(row.get("split") or "").strip()
        if split not in split_records:
            counts["invalid_split"] += 1
            continue
        user_path = Path(str(row.get("audio_filepath") or ""))
        response_path = Path(str(row.get("response_audio_filepath") or ""))
        response_text = str(row.get("response_text") or row.get("assistant_text") or "").strip()
        if not user_path.is_file() or (
            assistant_audio_mode == "source" and not response_path.is_file()
        ):
            counts["missing_audio"] += 1
            continue
        if not response_text:
            counts["missing_response_text"] += 1
            continue

        user_audio, _ = read_wav(user_path)
        if assistant_audio_mode == "piper":
            from thesis_s2s.runtime.tts import piper_synthesize

            response_audio = piper_synthesize(
                response_text,
                SAMPLE_RATE,
                deterministic=True,
            )
            if response_audio is None:
                counts["piper_synthesis_failed"] += 1
                continue
        else:
            response_audio, _ = read_wav(response_path)
        if not len(user_audio) or not len(response_audio):
            counts["empty_audio"] += 1
            continue
        user_start, response_start = _relative_starts(row, len(user_audio))
        total_samples = max(user_start + len(user_audio), response_start + len(response_audio))
        stereo = np.zeros((total_samples, 2), dtype=np.float32)
        stereo[response_start : response_start + len(response_audio), 0] = response_audio
        stereo[user_start : user_start + len(user_audio), 1] = user_audio

        safe_name = _safe_pair_name(pair_id)
        wav_path = out_dir / "audio" / split / f"{safe_name}.wav"
        json_path = wav_path.with_suffix(".json")
        _write_stereo_wav(wav_path, stereo)
        response_start_s = response_start / SAMPLE_RATE
        response_end_s = (response_start + len(response_audio)) / SAMPLE_RATE
        metadata = {
            "alignments": [
                [
                    response_text,
                    [round(response_start_s, 4), round(response_end_s, 4)],
                    "SPEAKER_MAIN",
                ]
            ],
            "source_pair_id": pair_id,
            "source_session_id": row.get("session_id"),
            "annotation_source": row.get("annotation_source"),
            "assistant_audio_mode": assistant_audio_mode,
            "source_response_audio_filepath": str(response_path),
            "assistant_voice_model_sha256": piper_model_sha256,
        }
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        duration = total_samples / SAMPLE_RATE
        split_records[split].append({"path": str(wav_path.resolve()), "duration": duration})
        exported_seconds += duration
        counts["exported_pairs"] += 1
        counts[f"{split}_pairs"] += 1
        if row.get("human_verified") is True:
            counts["human_verified_pairs"] += 1

    manifest_hashes: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, records in split_records.items():
        manifest = out_dir / f"{split}.jsonl"
        temporary = manifest.with_name(f".{manifest.name}.partial")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(manifest)
        manifest_hashes[split] = _sha256(manifest)

    exported_pairs = counts["exported_pairs"]
    training_hours = exported_seconds / 3600.0
    manual_sample_present = counts["human_verified_pairs"] > 0
    requirements = {
        "response_pairs_exported": exported_pairs > 0,
        "all_selected_pairs_exported": exported_pairs == len(rows),
        "train_validation_test_present": all(split_records[split] for split in split_records),
        "training_hours_100_to_200": 100.0 <= training_hours <= 200.0,
        "manual_verification_sample_present": manual_sample_present,
        "training_authorization_complete": True,
        "consistent_assistant_voice": assistant_audio_mode == "piper",
        "assistant_voice_model_pinned": bool(piper_model_sha256)
        if assistant_audio_mode == "piper"
        else False,
        "raw_redistribution_disabled": True,
    }
    report = {
        "schema": "kyutai_moshi_finetune_stereo_v1",
        "source_manifest": str(conversation_manifest),
        "source_manifest_sha256": _sha256(conversation_manifest),
        "out_dir": str(out_dir),
        "assistant_channel": 0,
        "user_channel": 1,
        "sample_rate": SAMPLE_RATE,
        "assistant_audio_mode": assistant_audio_mode,
        "assistant_voice_model": (
            {
                "repository": "MahtaFetrat/Mana-Persian-Piper",
                "revision": MANA_PIPER_REVISION,
                "license": "MIT",
                "path": str(piper_model) if piper_model is not None else None,
                "sha256": piper_model_sha256,
            }
            if assistant_audio_mode == "piper"
            else None
        ),
        "source_pairs": len(rows),
        "exported_pairs": exported_pairs,
        "exported_hours": round(training_hours, 3),
        "counts": dict(counts),
        "source_rows_permitting_redistribution": sum(
            bool(row.get("redistribution_allowed")) for row in rows
        ),
        "manifest_sha256": manifest_hashes,
        "requirements": requirements,
        "final_training_ready": all(requirements.values()),
        "note": (
            "Internal research artifact; audio and transcripts are not redistributable. "
            "The official Moshi trainer learns text and assistant-audio token losses. "
            "The primary Piper mode follows Moshi's consistent-system-voice design; "
            "source response audio is retained only as a multi-voice ablation."
        ),
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report
