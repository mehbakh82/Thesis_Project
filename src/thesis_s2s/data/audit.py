"""Streaming manifest audit for leakage, schema, and thesis-data coverage."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean

from thesis_s2s.metrics import write_json

VALID_SPLITS = {"train", "val", "test"}
VALID_LABELS = {"none", "interrupt", "backchannel", "noise"}


def _group_id(row: dict) -> str:
    source = str(row.get("source_csv") or "")
    if source:
        return source
    audio = Path(str(row.get("audio_filepath") or row.get("audio_path") or ""))
    return str(audio.parent)


def audit_manifest(manifest: Path, out_json: Path | None = None, check_files: bool = True) -> dict:
    """Audit one JSONL without modifying it."""

    manifest = Path(manifest)
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_hours: dict[str, float] = defaultdict(float)
    seen_ids: set[str] = set()
    duplicate_ids = 0
    missing_audio = 0
    schema_errors = 0
    overlap_rows = 0
    response_pairs = 0
    groups: dict[str, set[str]] = defaultdict(set)
    text_splits: dict[str, set[str]] = defaultdict(set)
    n = 0
    hours = 0.0

    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                schema_errors += 1
                continue
            n += 1
            utt_id = str(row.get("utt_id") or "")
            split = str(row.get("split") or "unspecified")
            label = str(row.get("interrupt_label") or "none")
            try:
                duration = float(row.get("duration") or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
                schema_errors += 1
            if not utt_id or duration <= 0 or (split != "unspecified" and split not in VALID_SPLITS):
                schema_errors += 1
            if label not in VALID_LABELS:
                schema_errors += 1
            if utt_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(utt_id)
            split_counts[split] += 1
            label_counts[label] += 1
            split_hours[split] += duration / 3600.0
            hours += duration / 3600.0
            groups[_group_id(row)].add(split)
            text = str(row.get("transcript_caption") or row.get("text") or "").strip()
            if text:
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                text_splits[digest].add(split)
            if row.get("overlap_intervals"):
                overlap_rows += 1
            if row.get("response_text") or row.get("assistant_text") or row.get("response_audio_filepath"):
                response_pairs += 1
            audio = str(row.get("audio_filepath") or row.get("audio_path") or "")
            if check_files and (not audio or not Path(audio).is_file()):
                missing_audio += 1

    group_leaks = sum(1 for splits in groups.values() if len(splits - {"unspecified"}) > 1)
    text_cross_split = sum(1 for splits in text_splits.values() if len(splits - {"unspecified"}) > 1)
    requirements = {
        "hours_100_to_200": 100.0 <= hours <= 200.0,
        "interrupt_labels_present": label_counts["interrupt"] > 0,
        "overlap_annotations_present": overlap_rows > 0,
        "response_supervision_present": response_pairs > 0,
        "all_three_splits_present": all(split_counts[name] > 0 for name in VALID_SPLITS),
    }
    report = {
        "manifest": str(manifest),
        "n": n,
        "hours": round(hours, 3),
        "split_counts": dict(split_counts),
        "split_hours": {key: round(value, 3) for key, value in split_hours.items()},
        "label_counts": dict(label_counts),
        "duplicate_utt_ids": duplicate_ids,
        "files_checked": check_files,
        "missing_audio": missing_audio if check_files else None,
        "schema_errors": schema_errors,
        "group_split_leaks": group_leaks,
        "text_cross_split_duplicates": text_cross_split,
        "overlap_rows": overlap_rows,
        "response_supervision_rows": response_pairs,
        "integrity_ok": duplicate_ids == 0 and schema_errors == 0 and group_leaks == 0,
        "thesis_coverage": requirements,
        "thesis_coverage_ok": all(requirements.values()),
    }
    if out_json is not None:
        write_json(out_json, report)
    return report


def _alignment_tokens(text: str) -> list[str]:
    return [token for token in text.replace("\u200c", " ").split() if token]


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def audit_caption_alignment(manifest: Path, out_json: Path | None = None) -> dict:
    """Compare captions with an independent ASR transcript as a mismatch proxy."""

    manifest = Path(manifest)
    similarities: list[float] = []
    channel_ratios: dict[str, list[float]] = defaultdict(list)
    low_samples: list[dict[str, object]] = []
    teacher_counts: Counter[str] = Counter()
    rows_scanned = 0
    paired_hours = 0.0
    repetitive_teacher_rows = 0

    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows_scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            caption = str(row.get("transcript_caption") or "")
            teacher = str(row.get("transcript_nemo") or "")
            caption_tokens = _alignment_tokens(caption)
            teacher_tokens = _alignment_tokens(teacher)
            if not caption_tokens or not teacher_tokens:
                continue
            ratio = SequenceMatcher(None, caption_tokens, teacher_tokens, autojunk=False).ratio()
            similarities.append(ratio)
            channel_ratios[str(row.get("channel") or "unknown")].append(ratio)
            paired_hours += float(row.get("duration") or 0.0) / 3600.0
            teacher_counts[str(row.get("teacher") or "unknown")] += 1
            repetition = max(Counter(teacher_tokens).values()) / len(teacher_tokens)
            if len(teacher_tokens) >= 10 and repetition >= 0.5:
                repetitive_teacher_rows += 1
            if ratio < 0.4 and len(low_samples) < 20:
                low_samples.append(
                    {
                        "utt_id": str(row.get("utt_id") or ""),
                        "similarity": round(ratio, 4),
                        "caption": caption[:300],
                        "teacher": teacher[:300],
                    }
                )

    paired = len(similarities)
    low_fraction = sum(value < 0.4 for value in similarities) / paired if paired else 1.0
    repetitive_fraction = repetitive_teacher_rows / paired if paired else 0.0
    by_channel = {
        channel: {
            "n": len(values),
            "mean_similarity": round(fmean(values), 4),
            "low_similarity_fraction": round(sum(value < 0.4 for value in values) / len(values), 4),
        }
        for channel, values in sorted(channel_ratios.items())
    }
    report = {
        "manifest": str(manifest),
        "rows_scanned": rows_scanned,
        "paired_rows": paired,
        "pair_coverage": round(paired / rows_scanned, 4) if rows_scanned else 0.0,
        "paired_hours": round(paired_hours, 3),
        "teacher_counts": dict(teacher_counts),
        "token_sequence_similarity": {
            "mean": round(fmean(similarities), 4) if similarities else None,
            "median": round(_percentile(similarities, 0.5) or 0.0, 4) if similarities else None,
            "p10": round(_percentile(similarities, 0.1) or 0.0, 4) if similarities else None,
            "p25": round(_percentile(similarities, 0.25) or 0.0, 4) if similarities else None,
            "high_ge_0_7_fraction": round(sum(value >= 0.7 for value in similarities) / paired, 4) if paired else 0.0,
            "low_lt_0_4_fraction": round(low_fraction, 4),
            "exact_fraction": round(sum(value == 1.0 for value in similarities) / paired, 4) if paired else 0.0,
        },
        "repetitive_teacher_rows": repetitive_teacher_rows,
        "repetitive_teacher_fraction": round(repetitive_fraction, 4),
        "by_channel": by_channel,
        "low_similarity_samples": low_samples,
        "alignment_risk": low_fraction > 0.2 or repetitive_fraction > 0.05,
        "evidence_scope": (
            "Independent-ASR agreement is a screening proxy, not ground-truth alignment. "
            "Low-similarity rows require manual audio review; ASR hallucination can also lower agreement."
        ),
    }
    if out_json is not None:
        write_json(out_json, report)
    return report
