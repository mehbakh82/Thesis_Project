"""Pair-level listening QA for automatic conversational overlap candidates."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from thesis_s2s.data.conversation import (
    SpeakerSegment,
    _automatic_interaction_candidate,
    _merge_same_speaker,
    _segments,
)
from thesis_s2s.metrics import write_json

REVIEW_CHECK_FIELDS = (
    "speakers_distinct_correct",
    "user_turn_boundary_correct",
    "response_turn_boundary_correct",
    "audible_overlap_correct",
)
REVIEW_FIELDS = (
    "review_status",
    *REVIEW_CHECK_FIELDS,
    "corrected_label",
    "reviewer_id",
    "notes",
)
YES = {"1", "true", "yes", "y", "بله"}
VALID_LABELS = {"interrupt", "backchannel"}


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _candidate_key(row: dict) -> str:
    return hashlib.sha256(str(row["candidate_id"]).encode("utf-8")).hexdigest()


def _window_candidates(
    row: dict,
    max_response_gap_s: float = 3.0,
    *,
    include_manual_rejected: bool = False,
) -> list[dict]:
    if (
        "automatic_multi_speaker_verified" in row
        and row.get("automatic_multi_speaker_verified") is not True
    ):
        return []
    if "reference_alignment_status" in row and row.get("reference_alignment_status") != "complete":
        return []
    if (
        not include_manual_rejected
        and row.get("manual_qa_reviewed") is True
        and row.get("human_verified") is not True
    ):
        return []
    segments = _merge_same_speaker(_segments(row))
    if len({segment.speaker for segment in segments}) < 2:
        return []

    result: list[dict] = []
    consumed_until = -1.0
    index = 0
    while index + 1 < len(segments):
        user: SpeakerSegment = segments[index]
        response: SpeakerSegment = segments[index + 1]
        index += 2
        if (
            user.speaker == response.speaker
            or response.start - user.end > max_response_gap_s
            or user.duration < 0.25
            or response.duration < 0.25
        ):
            continue
        span_start = min(user.start, response.start)
        span_end = max(user.end, response.end)
        if span_start < consumed_until:
            continue
        consumed_until = span_end
        candidate = _automatic_interaction_candidate(row, user, response)
        if candidate is None:
            continue
        listen_start, listen_end = candidate["listen_interval"]
        result.append(
            {
                "candidate_id": candidate["candidate_id"],
                "window_id": row.get("window_id"),
                "episode_id": row.get("episode_id"),
                "channel": row.get("channel") or "unknown",
                "audio_filepath": row.get("audio_filepath") or row.get("audio_path"),
                "listen_start_s": listen_start,
                "listen_end_s": listen_end,
                "listen_duration_s": round(listen_end - listen_start, 3),
                "automatic_candidate": candidate["automatic_label"],
                "overlap_seconds": candidate["overlap_seconds"],
                "aligned_user_start_s": candidate["aligned_user_interval"][0],
                "aligned_user_end_s": candidate["aligned_user_interval"][1],
                "aligned_response_start_s": candidate["aligned_response_interval"][0],
                "aligned_response_end_s": candidate["aligned_response_interval"][1],
                "raw_user_start_s": candidate["raw_user_interval"][0],
                "raw_user_end_s": candidate["raw_user_interval"][1],
                "raw_response_start_s": candidate["raw_response_interval"][0],
                "raw_response_end_s": candidate["raw_response_interval"][1],
                "user_match_ratio": candidate["user_match_ratio"],
                "response_match_ratio": candidate["response_match_ratio"],
                "user_speaker": user.speaker,
                "response_speaker": response.speaker,
                "user_text": user.text,
                "response_text": response.text,
                **{field: "" for field in REVIEW_FIELDS},
            }
        )
    return result


def _quantile_sample(rows: list[dict], count: int) -> list[dict]:
    if count <= 0 or not rows:
        return []
    ordered = sorted(rows, key=lambda row: (float(row["overlap_seconds"]), _candidate_key(row)))
    count = min(count, len(ordered))
    selected: list[dict] = []
    used_episodes: set[str] = set()
    for index in range(count):
        lower = index * len(ordered) // count
        upper = (index + 1) * len(ordered) // count
        bucket = ordered[lower : max(lower + 1, upper)]
        novel = [row for row in bucket if str(row.get("episode_id")) not in used_episodes]
        choice = min(novel or bucket, key=_candidate_key)
        selected.append(choice)
        used_episodes.add(str(choice.get("episode_id")))
    return selected


def sample_interaction_qa(
    diarized_jsonl: Path,
    out_csv: Path,
    *,
    per_channel: int = 6,
    report_path: Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Create a deterministic, channel-balanced short listening sheet."""

    if per_channel < 1:
        raise ValueError("per_channel must be positive")
    rows = _read_jsonl(Path(diarized_jsonl))
    candidates = [candidate for row in rows for candidate in _window_candidates(row)]
    ids = [str(row["candidate_id"]) for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("automatic interaction candidates contain duplicate IDs")

    by_channel: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_channel[str(candidate["channel"])].append(candidate)
    sampled: list[dict] = []
    sampled_channels: dict[str, int] = {}
    for channel, channel_rows in sorted(by_channel.items()):
        backchannels = [row for row in channel_rows if row["automatic_candidate"] == "backchannel"]
        interruptions = [row for row in channel_rows if row["automatic_candidate"] == "interrupt"]
        backchannel_slots = 1 if per_channel > 1 and backchannels else 0
        chosen = _quantile_sample(interruptions, per_channel - backchannel_slots)
        chosen.extend(_quantile_sample(backchannels, backchannel_slots))
        if len(chosen) < min(per_channel, len(channel_rows)):
            chosen_ids = {str(row["candidate_id"]) for row in chosen}
            remainder = [row for row in channel_rows if str(row["candidate_id"]) not in chosen_ids]
            chosen.extend(_quantile_sample(remainder, per_channel - len(chosen)))
        sampled.extend(chosen)
        sampled_channels[channel] = len(chosen)

    sampled.sort(
        key=lambda row: (
            str(row["channel"]),
            str(row["automatic_candidate"]),
            str(row["candidate_id"]),
        )
    )
    out_csv = Path(out_csv)
    if out_csv.is_file() and not overwrite:
        with out_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
        if any(str(row.get(field) or "").strip() for row in existing for field in REVIEW_FIELDS):
            raise FileExistsError(
                f"interaction QA contains reviewer data and will not be overwritten: {out_csv}"
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(sampled[0])
        if sampled
        else [
            "candidate_id",
            "window_id",
            "episode_id",
            "channel",
            "audio_filepath",
            "listen_start_s",
            "listen_end_s",
            "listen_duration_s",
            "automatic_candidate",
            "overlap_seconds",
            *REVIEW_FIELDS,
        ]
    )
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    candidate_counts = Counter(str(row["automatic_candidate"]) for row in candidates)
    channel_counts = Counter(str(row["channel"]) for row in candidates)
    report = {
        "source_manifest": str(diarized_jsonl),
        "out_csv": str(out_csv),
        "source_windows": len(rows),
        "automatic_candidates": len(candidates),
        "automatic_candidate_counts": dict(candidate_counts),
        "candidate_channels": dict(sorted(channel_counts.items())),
        "sampled_candidates": len(sampled),
        "sampled_per_channel": sampled_channels,
        "new_recordings_required": False,
        "instruction": (
            "Listen only between listen_start_s and listen_end_s. Set review_status="
            "pass or fail. A pass requires all four checks=yes, corrected_label="
            "interrupt or backchannel, and reviewer_id. Automatic labels are not claims."
        ),
    }
    if report_path is not None:
        write_json(report_path, report)
    return report


def _yes(value: object) -> bool:
    return str(value or "").strip().lower() in YES


def apply_interaction_qa(
    diarized_jsonl: Path,
    qa_csv: Path,
    out_jsonl: Path,
    *,
    report_path: Path | None = None,
) -> dict:
    """Attach reviewed pair labels to windows without trusting CSV evidence fields."""

    decisions: dict[str, dict] = {}
    duplicates = 0
    with Path(qa_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for parsed_decision in csv.DictReader(handle):
            candidate_id = str(parsed_decision.get("candidate_id") or "").strip()
            if not candidate_id:
                continue
            if candidate_id in decisions:
                duplicates += 1
            decisions[candidate_id] = parsed_decision
    if duplicates:
        raise ValueError(f"duplicate candidate_id values in interaction QA CSV: {duplicates}")

    rows = _read_jsonl(Path(diarized_jsonl))
    row_candidates = [_window_candidates(row, include_manual_rejected=True) for row in rows]
    known_ids = {str(candidate["candidate_id"]) for items in row_candidates for candidate in items}
    eligible_ids = {
        str(candidate["candidate_id"])
        for row, items in zip(rows, row_candidates, strict=True)
        if not (row.get("manual_qa_reviewed") is True and row.get("human_verified") is not True)
        for candidate in items
    }
    counts: Counter[str] = Counter()
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row, candidates in zip(rows, row_candidates, strict=True):
            candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
            existing = {
                str(review.get("candidate_id")): review
                for review in row.get("interaction_reviews") or []
                if isinstance(review, dict) and review.get("candidate_id")
            }
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                decision = decisions.get(candidate_id)
                if decision is None:
                    continue
                status = str(decision.get("review_status") or "").strip().lower()
                reviewer = str(decision.get("reviewer_id") or "").strip()
                corrected_label = str(decision.get("corrected_label") or "").strip().lower()
                checks = {field: _yes(decision.get(field)) for field in REVIEW_CHECK_FIELDS}
                checks_supplied = all(
                    str(decision.get(field) or "").strip() for field in REVIEW_CHECK_FIELDS
                )
                complete = bool(reviewer) and (
                    status == "fail"
                    or (status == "pass" and checks_supplied and corrected_label in VALID_LABELS)
                )
                window_eligible = not (
                    row.get("manual_qa_reviewed") is True and row.get("human_verified") is not True
                )
                approved = (
                    complete and status == "pass" and all(checks.values()) and window_eligible
                )
                review = {
                    "candidate_id": candidate_id,
                    "review_status": status or "pending",
                    "reviewer_id": reviewer,
                    "checks": checks,
                    "automatic_label": candidate["automatic_candidate"],
                    "label": corrected_label if approved else "",
                    "human_verified": approved,
                    "notes": str(decision.get("notes") or ""),
                    "decision_source": str(qa_csv),
                }
                existing[candidate_id] = review
                if not complete:
                    counts["incomplete"] += 1
                elif not window_eligible:
                    counts["excluded_by_window_qa"] += 1
                elif approved:
                    counts["approved"] += 1
                    counts[f"approved_{corrected_label}"] += 1
                else:
                    counts["rejected"] += 1
            if candidate_ids or existing:
                row["interaction_reviews"] = [existing[key] for key in sorted(existing)]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts["source_windows"] = len(rows)
    counts["source_candidates"] = len(known_ids)
    counts["eligible_candidates"] = len(eligible_ids)
    counts["qa_decisions"] = len(decisions)
    counts["unknown_candidate_ids"] = len(set(decisions) - known_ids)
    report = {
        "source_manifest": str(diarized_jsonl),
        "qa_csv": str(qa_csv),
        "out_manifest": str(out_jsonl),
        "counts": dict(counts),
        "fail_closed": True,
        "qa_complete": bool(decisions)
        and counts["incomplete"] == 0
        and counts["unknown_candidate_ids"] == 0,
        "verified_interruption_present": counts["approved_interrupt"] > 0,
        "note": (
            "Only complete passing rows become human-verified labels. Rejected, "
            "incomplete, unsampled, and unknown candidates remain automatic evidence only."
        ),
    }
    if report_path is not None:
        write_json(report_path, report)
    return report
