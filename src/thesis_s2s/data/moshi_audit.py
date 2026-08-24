"""Independent integrity audit and listening-sample preparation for Moshi exports."""

from __future__ import annotations

import csv
import hashlib
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.config import portable_project_values
from thesis_s2s.metrics import write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _descriptor(row: dict, record: dict) -> dict:
    session_id = str(row.get("session_id") or "")
    response_seconds = float(row.get("assistant_duration") or 0.0)
    if response_seconds <= 3.0:
        response_bucket = "short"
    elif response_seconds <= 10.0:
        response_bucket = "medium"
    else:
        response_bucket = "long"
    return {
        "pair_id": str(row.get("utt_id") or ""),
        "split": str(row.get("split") or ""),
        "channel": session_id.split("/", 1)[0] or "unknown",
        "noise_condition": str(row.get("noise_condition") or "unknown"),
        "interaction_label": str(row.get("interrupt_label") or "none"),
        "overlap": "yes" if row.get("overlap_intervals") else "no",
        "response_length": response_bucket,
        "wav_path": str(record["path"]),
        "metadata_path": str(Path(str(record["path"])).with_suffix(".json")),
        "review_status": "",
        "reviewer_id": "",
        "notes": "",
    }


def _stratified_sample(rows: list[dict], size: int) -> list[dict]:
    if size <= 0:
        return []
    remaining = list(rows)
    selected: list[dict] = []
    uncovered = {
        f"{field}:{row[field]}"
        for row in rows
        for field in (
            "split",
            "channel",
            "noise_condition",
            "interaction_label",
            "overlap",
            "response_length",
        )
    }
    while remaining and len(selected) < size:
        def score(row: dict) -> tuple[int, str]:
            tokens = {
                f"{field}:{row[field]}"
                for field in (
                    "split",
                    "channel",
                    "noise_condition",
                    "interaction_label",
                    "overlap",
                    "response_length",
                )
            }
            digest = hashlib.sha256(str(row["pair_id"]).encode("utf-8")).hexdigest()
            return len(tokens & uncovered), digest

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
        for field in (
            "split",
            "channel",
            "noise_condition",
            "interaction_label",
            "overlap",
            "response_length",
        ):
            uncovered.discard(f"{field}:{chosen[field]}")
    return selected


def audit_moshi_finetune_dataset(
    conversation_manifest: Path,
    export_dir: Path,
    export_report_path: Path,
    *,
    out_path: Path,
    sample_csv_path: Path,
    sample_size: int = 24,
) -> dict:
    """Recompute every exported hash/header/join and prepare an unreviewed sample."""

    conversation_manifest = Path(conversation_manifest)
    export_dir = Path(export_dir)
    export_report_path = Path(export_report_path)
    source_rows = _jsonl(conversation_manifest)
    source_by_id = {str(row.get("utt_id") or ""): row for row in source_rows}
    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
    failures: Counter[str] = Counter()
    records_by_id: dict[str, dict] = {}
    split_counts: Counter[str] = Counter()
    exported_seconds = 0.0
    manifest_hashes_current = True

    for split in ("train", "val", "test"):
        manifest_path = export_dir / f"{split}.jsonl"
        if not manifest_path.is_file():
            failures["missing_split_manifest"] += 1
            manifest_hashes_current = False
            continue
        if (export_report.get("manifest_sha256") or {}).get(split) != _sha256(manifest_path):
            failures["manifest_hash_mismatch"] += 1
            manifest_hashes_current = False
        for record in _jsonl(manifest_path):
            path = Path(str(record.get("path") or ""))
            metadata_path = path.with_suffix(".json")
            if not path.is_file() or not metadata_path.is_file():
                failures["missing_export_file"] += 1
                continue
            try:
                with wave.open(str(path), "rb") as handle:
                    channels = handle.getnchannels()
                    sample_rate = handle.getframerate()
                    sample_width = handle.getsampwidth()
                    frames = handle.getnframes()
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError, wave.Error):
                failures["unreadable_export_file"] += 1
                continue
            pair_id = str(metadata.get("source_pair_id") or "")
            source = source_by_id.get(pair_id)
            if not pair_id or source is None:
                failures["unknown_source_pair"] += 1
                continue
            if pair_id in records_by_id:
                failures["duplicate_export_pair"] += 1
                continue
            if str(source.get("split") or "") != split:
                failures["split_mismatch"] += 1
            alignments = metadata.get("alignments") or []
            response_text = str(source.get("response_text") or source.get("assistant_text") or "")
            if not alignments or not alignments[0] or alignments[0][0] != response_text:
                failures["transcript_mismatch"] += 1
            if (
                metadata.get("assistant_audio_mode") != export_report.get("assistant_audio_mode")
                or metadata.get("assistant_voice_model_sha256")
                != (export_report.get("assistant_voice_model") or {}).get("sha256")
                or metadata.get("qa_waiver_sha256") != source.get("qa_waiver_sha256")
            ):
                failures["metadata_policy_or_voice_mismatch"] += 1
            if channels != 2 or sample_rate != SAMPLE_RATE or sample_width != 2 or frames <= 0:
                failures["wav_format_mismatch"] += 1
            if record.get("sha256") != _sha256(path):
                failures["wav_hash_mismatch"] += 1
            if record.get("metadata_sha256") != _sha256(metadata_path):
                failures["metadata_hash_mismatch"] += 1
            if int(record.get("bytes") or -1) != path.stat().st_size:
                failures["byte_size_mismatch"] += 1
            if int(record.get("frames") or -1) != frames:
                failures["frame_count_mismatch"] += 1
            duration = frames / SAMPLE_RATE
            if abs(float(record.get("duration") or 0.0) - duration) > 1e-9:
                failures["duration_mismatch"] += 1
            records_by_id[pair_id] = record
            split_counts[split] += 1
            exported_seconds += duration

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        group_splits[str(row.get("session_id") or "")].add(str(row.get("split") or ""))
    group_split_leaks = sum(len(splits) > 1 for splits in group_splits.values())
    missing_pair_ids = set(source_by_id) - set(records_by_id)
    extra_pair_ids = set(records_by_id) - set(source_by_id)
    descriptors = [
        _descriptor(source_by_id[pair_id], record)
        for pair_id, record in records_by_id.items()
        if pair_id in source_by_id
    ]
    sample = _stratified_sample(descriptors, min(sample_size, len(descriptors)))
    sample_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(sample[0]) if sample else [
        "pair_id",
        "split",
        "channel",
        "noise_condition",
        "interaction_label",
        "overlap",
        "response_length",
        "wav_path",
        "metadata_path",
        "review_status",
        "reviewer_id",
        "notes",
    ]
    with sample_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sample)

    computed_hours = exported_seconds / 3600.0
    requirements = {
        "source_manifest_hash_current": export_report.get("source_manifest_sha256")
        == _sha256(conversation_manifest),
        "all_split_manifest_hashes_current": manifest_hashes_current,
        "all_source_pairs_exported_once": not missing_pair_ids
        and not extra_pair_ids
        and len(records_by_id) == len(source_rows),
        "all_file_header_metadata_and_hash_checks_pass": not failures,
        "session_group_splits_isolated": group_split_leaks == 0,
        "train_validation_test_present": all(split_counts[split] > 0 for split in ("train", "val", "test")),
        "reported_pair_count_matches": export_report.get("exported_pairs") == len(records_by_id),
        "reported_hours_match": abs(
            float(export_report.get("exported_hours") or 0.0) - round(computed_hours, 3)
        )
        <= 0.001,
        "channel_order_declared_assistant_0_user_1": export_report.get("assistant_channel") == 0
        and export_report.get("user_channel") == 1,
        "unreviewed_stratified_sample_prepared": len(sample)
        == min(sample_size, len(descriptors)),
    }
    report = {
        "schema_version": 1,
        "source_pairs": len(source_rows),
        "verified_export_pairs": len(records_by_id),
        "verified_export_hours": round(computed_hours, 3),
        "split_counts": dict(split_counts),
        "failure_counts": dict(failures),
        "missing_pair_ids": sorted(missing_pair_ids)[:20],
        "extra_pair_ids": sorted(extra_pair_ids)[:20],
        "group_split_leaks": group_split_leaks,
        "sample": {
            "path": str(sample_csv_path),
            "rows": len(sample),
            "human_review_complete": False,
            "coverage": {
                field: dict(Counter(str(row[field]) for row in sample))
                for field in (
                    "split",
                    "channel",
                    "noise_condition",
                    "interaction_label",
                    "overlap",
                    "response_length",
                )
            },
        },
        "requirements": requirements,
        "audit_passes": all(requirements.values()),
        "training_ready_under_qa_waiver": all(requirements.values())
        and export_report.get("training_ready_under_qa_waiver") is True,
        "strict_thesis_data_coverage": False,
        "note": (
            "The sample is prepared but deliberately unreviewed. Automated integrity "
            "checks do not replace listening or establish human-verification claims."
        ),
    }
    report = portable_project_values(report)
    write_json(out_path, report)
    return report
