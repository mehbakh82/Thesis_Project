"""Build genuine response-pair supervision from diarized Persian conversations."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import read_wav, write_wav
from thesis_s2s.metrics import write_json

SAFE_PART = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def _safe_part(value: str) -> str:
    cleaned = SAFE_PART.sub("-", value).strip("-")[:36]
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=5).hexdigest()
    return f"{cleaned or 'session'}-{digest}"


def _stable_split(session_id: str) -> str:
    bucket = int.from_bytes(hashlib.blake2b(session_id.encode("utf-8"), digest_size=2).digest(), "big") % 100
    if bucket < 5:
        return "test"
    if bucket < 10:
        return "val"
    return "train"


def _segments(row: dict) -> list[SpeakerSegment]:
    raw_segments = row.get("segments") or row.get("speaker_turns") or []
    result: list[SpeakerSegment] = []
    for item in raw_segments:
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        speaker = str(item.get("speaker") or item.get("speaker_id") or "").strip()
        text = str(item.get("text") or item.get("raw_text") or item.get("transcript") or "").strip()
        if speaker and text and 0 <= start < end and 0.25 <= end - start <= 30.0:
            result.append(SpeakerSegment(start, end, speaker, text))
    return sorted(result, key=lambda item: (item.start, item.end, item.speaker))


def _merge_same_speaker(segments: list[SpeakerSegment], max_gap_s: float = 0.5) -> list[SpeakerSegment]:
    merged: list[SpeakerSegment] = []
    for segment in segments:
        if merged and merged[-1].speaker == segment.speaker and segment.start - merged[-1].end <= max_gap_s:
            previous = merged[-1]
            merged[-1] = SpeakerSegment(
                start=previous.start,
                end=max(previous.end, segment.end),
                speaker=previous.speaker,
                text=f"{previous.text} {segment.text}".strip(),
            )
        else:
            merged.append(segment)
    return merged


def _interaction_label(user: SpeakerSegment, assistant: SpeakerSegment) -> tuple[str, list[list[float]]]:
    overlap_end = min(user.end, assistant.end)
    if assistant.start >= overlap_end:
        return "none", []
    overlap = [[round(assistant.start, 3), round(overlap_end, 3)]]
    if assistant.duration <= 1.2 and assistant.end <= user.end + 0.2:
        return "backchannel", overlap
    return "interrupt", overlap


def build_conversation_manifest(
    diarized_jsonl: Path,
    out_jsonl: Path,
    clips_dir: Path,
    *,
    max_hours: float = 200.0,
    max_response_gap_s: float = 3.0,
) -> dict:
    """Extract non-overlapping user/response pairs from full diarized recordings.

    Input rows contain ``audio_filepath``, ``session_id``, ``license``, and a
    ``segments`` (or ``speaker_turns``) list with start/end/speaker/text.
    Consecutive source segments are consumed once, preventing duration inflation.
    """

    diarized_jsonl = Path(diarized_jsonl)
    out_jsonl = Path(out_jsonl)
    clips_dir = Path(clips_dir)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[dict] = []
    label_counts: Counter[str] = Counter()
    hours = 0.0
    skipped: Counter[str] = Counter()
    for line in diarized_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source = Path(str(row.get("audio_filepath") or row.get("audio_path") or ""))
        if not source.is_file():
            skipped["missing_audio"] += 1
            continue
        session_id = str(row.get("session_id") or row.get("recording_id") or source.stem)
        segments = _merge_same_speaker(_segments(row))
        if len({segment.speaker for segment in segments}) < 2:
            skipped["not_multi_speaker"] += 1
            continue
        audio, _ = read_wav(source)
        safe_session = _safe_part(session_id)
        i = 0
        consumed_until = -1.0
        while i + 1 < len(segments):
            user, assistant = segments[i], segments[i + 1]
            i += 2
            gap = assistant.start - user.end
            if user.speaker == assistant.speaker or gap > max_response_gap_s:
                skipped["not_adjacent_response"] += 1
                continue
            source_span_start = min(user.start, assistant.start)
            source_span_end = max(user.end, assistant.end)
            if source_span_start < consumed_until:
                skipped["source_interval_reuse"] += 1
                continue
            source_span_duration = source_span_end - source_span_start
            pair_hours = source_span_duration / 3600.0
            if hours + pair_hours > max_hours:
                break
            user_audio = audio[max(0, int(user.start * SAMPLE_RATE)) : min(len(audio), int(user.end * SAMPLE_RATE))]
            assistant_audio = audio[
                max(0, int(assistant.start * SAMPLE_RATE)) : min(len(audio), int(assistant.end * SAMPLE_RATE))
            ]
            if len(user_audio) < SAMPLE_RATE // 4 or len(assistant_audio) < SAMPLE_RATE // 4:
                skipped["empty_clip"] += 1
                continue
            pair_index = len(pairs)
            pair_id = f"{safe_session}_{pair_index:06d}"
            user_path = clips_dir / f"{pair_id}_user.wav"
            assistant_path = clips_dir / f"{pair_id}_assistant.wav"
            write_wav(user_path, user_audio)
            write_wav(assistant_path, assistant_audio)
            label, overlap = _interaction_label(user, assistant)
            label_counts[label] += 1
            consumed_until = source_span_end
            license_name = str(row.get("license") or "unknown")
            pairs.append(
                {
                    "utt_id": pair_id,
                    "session_id": session_id,
                    "speaker_id": f"{session_id}:{user.speaker}",
                    "assistant_speaker_id": f"{session_id}:{assistant.speaker}",
                    "audio_filepath": str(user_path),
                    "response_audio_filepath": str(assistant_path),
                    "duration": round(source_span_duration, 3),
                    "user_duration": round(user.duration, 3),
                    "source_span_start": round(source_span_start, 3),
                    "source_span_end": round(source_span_end, 3),
                    "source_user_interval": [round(user.start, 3), round(user.end, 3)],
                    "source_response_interval": [round(assistant.start, 3), round(assistant.end, 3)],
                    "assistant_duration": round(assistant.duration, 3),
                    "transcript_caption": user.text,
                    "text": user.text,
                    "assistant_text": assistant.text,
                    "response_text": assistant.text,
                    "interrupt_label": label,
                    "overlap_intervals": overlap,
                    "response_gap_s": round(gap, 3),
                    "split": _stable_split(session_id),
                    "license": license_name,
                    "source_audio_filepath": str(source),
                    "annotation_source": str(row.get("annotation_source") or "automatic_diarization_heuristic"),
                    "human_verified": bool(row.get("human_verified", False)),
                    "noise_condition": str(row.get("noise_condition") or row.get("room") or "unspecified"),
                    "snr": row.get("snr"),
                    "is_natural_dialogue": True,
                }
            )
            hours += pair_hours
        if hours >= max_hours:
            break
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return {
        "manifest": str(out_jsonl),
        "pairs": len(pairs),
        "hours": round(hours, 3),
        "label_counts": dict(label_counts),
        "skipped": dict(skipped),
        "evidence_scope": "natural human-human response pairs with automatic labels until manually verified",
    }


def export_llama_omni2_questions(manifest: Path, out_json: Path) -> dict:
    """Export the official LLaMA-Omni2 inference conversation schema."""

    conversations = []
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        user_audio = str(row.get("audio_filepath") or "")
        assistant_audio = str(row.get("response_audio_filepath") or "")
        user_text = str(row.get("text") or row.get("transcript_caption") or "")
        assistant_text = str(row.get("assistant_text") or row.get("response_text") or "")
        if not all((user_audio, assistant_audio, user_text, assistant_text)):
            continue
        conversations.append(
            {
                "id": str(row.get("utt_id")),
                "conversation": [
                    {"from": "human", "speech": user_audio, "text": user_text},
                    {"from": "gpt", "speech": assistant_audio, "text": assistant_text},
                ],
            }
        )
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(conversations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"out": str(out_json), "conversations": len(conversations), "schema": "llama_omni2_questions_v1"}


def audit_conversation_manifest(manifest: Path, out_json: Path | None = None, *, check_files: bool = True) -> dict:
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    hours = sum(float(row.get("duration") or 0.0) for row in rows) / 3600.0
    speakers = {
        str(value) for row in rows
        for value in (row.get("speaker_id"), row.get("assistant_speaker_id")) if value
    }
    sessions = {str(row.get("session_id")) for row in rows if row.get("session_id")}
    missing_files = 0
    if check_files:
        for row in rows:
            paths = (row.get("audio_filepath"), row.get("response_audio_filepath"))
            missing_files += sum(not value or not Path(str(value)).is_file() for value in paths)
    labels = Counter(str(row.get("interrupt_label") or "none") for row in rows)
    verified = sum(bool(row.get("human_verified")) for row in rows)
    licensed = sum(str(row.get("license") or "unknown") != "unknown" for row in rows)
    overlap_rows = sum(bool(row.get("overlap_intervals")) for row in rows)
    noise_rows = sum(
        str(row.get("noise_condition") or "unspecified").lower()
        not in {"", "unspecified", "none", "quiet", "clean"}
        or str(row.get("interrupt_label") or "") == "noise"
        for row in rows
    )
    session_speakers: dict[str, set[str]] = {}
    session_splits: dict[str, set[str]] = {}
    source_spans: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        session = str(row.get("session_id") or "")
        session_speakers.setdefault(session, set()).update(
            str(value)
            for value in (row.get("speaker_id"), row.get("assistant_speaker_id"))
            if value
        )
        session_splits.setdefault(session, set()).add(str(row.get("split") or ""))
        source = str(row.get("source_audio_filepath") or "")
        try:
            span = (float(row["source_span_start"]), float(row["source_span_end"]))
        except (KeyError, TypeError, ValueError):
            continue
        source_spans.setdefault(source, []).append(span)
    split_leaks = sum(len(splits) > 1 for splits in session_splits.values())
    reused_spans = 0
    for spans in source_spans.values():
        ordered = sorted(spans)
        reused_spans += sum(current[0] < previous[1] for previous, current in zip(ordered, ordered[1:], strict=False))
    requirements = {
        "hours_100_to_200": 100.0 <= hours <= 200.0,
        "response_pairs_present": bool(rows) and all(row.get("response_audio_filepath") and row.get("assistant_text") for row in rows),
        "natural_multi_speaker_sessions": bool(rows)
        and all(len(values) >= 2 for values in session_speakers.values()),
        "session_groups_at_least_2": len(sessions) >= 2,
        "session_group_split_clean": split_leaks == 0,
        "source_intervals_not_reused": reused_spans == 0,
        "interruptions_present": labels["interrupt"] > 0,
        "overlap_annotations_present": overlap_rows > 0,
        "noise_conditions_present": noise_rows > 0,
        "licenses_complete": licensed == len(rows) and bool(rows),
        "manual_verification_sample_present": verified > 0,
        "files_present": missing_files == 0 if check_files else None,
    }
    report = {
        "manifest": str(manifest),
        "pairs": len(rows),
        "hours": round(hours, 3),
        "speakers": len(speakers),
        "sessions": len(sessions),
        "label_counts": dict(labels),
        "overlap_rows": overlap_rows,
        "noise_condition_rows": noise_rows,
        "session_group_split_leaks": split_leaks,
        "reused_source_spans": reused_spans,
        "human_verified_rows": verified,
        "licensed_rows": licensed,
        "missing_files": missing_files if check_files else None,
        "requirements": requirements,
        "thesis_coverage_ok": all(value is True for value in requirements.values()),
        "warning": "Automatic diarization/overlap heuristics are pseudo-labels until a manual sample is reviewed.",
    }
    if out_json is not None:
        write_json(out_json, report)
    return report
