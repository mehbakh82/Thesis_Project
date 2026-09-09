"""Build genuine response-pair supervision from diarized Persian conversations."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import read_wav, write_wav
from thesis_s2s.data.qa_policy import load_qa_waiver, row_matches_waiver
from thesis_s2s.data.rights import rights_record_verified, training_use_authorized
from thesis_s2s.metrics import write_json

SAFE_PART = re.compile(r"[^A-Za-z0-9_-]+")
RAW_TURN_MATCH_RATIO = 0.5
RAW_BOUNDARY_TOLERANCE_S = 0.5
MIN_RAW_OVERLAP_S = 0.2


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class PairCandidate:
    user: SpeakerSegment
    assistant: SpeakerSegment
    automatic_candidate: dict | None
    review: dict | None
    user_start: float
    user_end: float
    response_start: float
    response_end: float
    label: str
    overlap: list[list[float]]

    @property
    def span_start(self) -> float:
        return min(self.user_start, self.response_start)

    @property
    def span_end(self) -> float:
        return max(self.user_end, self.response_end)

    @property
    def span_duration(self) -> float:
        return self.span_end - self.span_start


def _safe_part(value: str) -> str:
    cleaned = SAFE_PART.sub("-", value).strip("-")[:36]
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=5).hexdigest()
    return f"{cleaned or 'session'}-{digest}"


def _require_training_authorized(rows: list[dict], operation: str) -> None:
    """Refuse artifacts from rows lacking documented training authorization."""

    unverified = [
        str(row.get("window_id") or row.get("utt_id") or row.get("session_id") or "unknown")
        for row in rows
        if not training_use_authorized(row)
    ]
    if unverified:
        sample = ", ".join(unverified[:5])
        raise PermissionError(
            f"{operation} requires documented training authorization for every input "
            f"row; unauthorized: {sample}"
        )


def _stable_split(session_id: str) -> str:
    bucket = (
        int.from_bytes(hashlib.blake2b(session_id.encode("utf-8"), digest_size=2).digest(), "big")
        % 100
    )
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


def _merge_same_speaker(
    segments: list[SpeakerSegment], max_gap_s: float = 0.5
) -> list[SpeakerSegment]:
    merged: list[SpeakerSegment] = []
    for segment in segments:
        if (
            merged
            and merged[-1].speaker == segment.speaker
            and segment.start - merged[-1].end <= max_gap_s
        ):
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


def _interaction_label(
    user: SpeakerSegment,
    assistant: SpeakerSegment,
    known_overlap: list[list[float]] | None = None,
) -> tuple[str, list[list[float]]]:
    overlap_end = min(user.end, assistant.end)
    response_starts_during_user = assistant.start < overlap_end
    overlap = (
        [[round(assistant.start, 3), round(overlap_end, 3)]] if response_starts_during_user else []
    )
    if not overlap:
        pair_start, pair_end = min(user.start, assistant.start), max(user.end, assistant.end)
        overlap = [
            [
                round(max(pair_start, float(interval[0])), 3),
                round(min(pair_end, float(interval[1])), 3),
            ]
            for interval in (known_overlap or [])
            if len(interval) == 2
            and min(pair_end, float(interval[1])) > max(pair_start, float(interval[0]))
        ]
    if not overlap:
        return "none", []
    if not response_starts_during_user:
        return "overlap_unattributed", overlap
    if assistant.duration <= 1.2 and assistant.end <= user.end + 0.2:
        return "backchannel", overlap
    return "interrupt", overlap


def _interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _automatic_interaction_candidate(
    row: dict,
    user: SpeakerSegment,
    assistant: SpeakerSegment,
) -> dict | None:
    """Recover conservative cross-speaker boundary candidates from raw diarization.

    Caption-aligned segments are intentionally sequential, so their boundaries can
    hide overlap retained in ``speaker_turns``. This helper only proposes a
    candidate when both raw turns strongly match their caption speaker, occur at
    the shared caption boundary, and overlap for at least 200 ms. Human review is
    still required before the candidate becomes a training/evaluation label.
    """

    raw_turns: list[tuple[float, float, str]] = []
    for item in row.get("speaker_turns") or []:
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (AttributeError, TypeError, ValueError):
            continue
        speaker = str(item.get("speaker") or item.get("speaker_id") or "").strip()
        if speaker and 0 <= start < end:
            raw_turns.append((start, end, speaker))

    def matches(
        segment: SpeakerSegment,
        raw_start: float,
        raw_end: float,
    ) -> float:
        overlap = _interval_overlap(segment.start, segment.end, raw_start, raw_end)
        denominator = min(segment.duration, raw_end - raw_start)
        return overlap / denominator if denominator > 0 else 0.0

    user_turns = [
        (start, end, matches(user, start, end))
        for start, end, speaker in raw_turns
        if speaker == user.speaker
        and matches(user, start, end) >= RAW_TURN_MATCH_RATIO
        and end >= user.end - RAW_BOUNDARY_TOLERANCE_S
    ]
    assistant_turns = [
        (start, end, matches(assistant, start, end))
        for start, end, speaker in raw_turns
        if speaker == assistant.speaker
        and matches(assistant, start, end) >= RAW_TURN_MATCH_RATIO
        and start <= assistant.start + RAW_BOUNDARY_TOLERANCE_S
    ]
    candidates: list[tuple[float, float, float, float, float, float, float]] = []
    for user_start, user_end, user_match in user_turns:
        for response_start, response_end, response_match in assistant_turns:
            overlap = min(user_end, response_end) - max(user_start, response_start)
            if response_start > user_start and overlap >= MIN_RAW_OVERLAP_S:
                candidates.append(
                    (
                        overlap,
                        min(user_match, response_match),
                        user_start,
                        user_end,
                        response_start,
                        response_end,
                        response_match,
                    )
                )
    if not candidates:
        return None

    (
        overlap_seconds,
        _minimum_match,
        raw_user_start,
        raw_user_end,
        raw_response_start,
        raw_response_end,
        response_match,
    ) = max(candidates)
    user_match = matches(user, raw_user_start, raw_user_end)
    automatic_label = (
        "backchannel"
        if raw_response_end - raw_response_start <= 1.2 and raw_response_end <= raw_user_end + 0.2
        else "interrupt"
    )
    identity = json.dumps(
        {
            "window_id": row.get("window_id") or row.get("session_id"),
            "user": [user.start, user.end, user.speaker],
            "response": [assistant.start, assistant.end, assistant.speaker],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    candidate_id = (
        "interaction-" + hashlib.blake2b(identity.encode("utf-8"), digest_size=12).hexdigest()
    )
    overlap_start = max(raw_user_start, raw_response_start)
    overlap_end = min(raw_user_end, raw_response_end)
    duration = float(row.get("duration") or max(user.end, assistant.end))
    return {
        "candidate_id": candidate_id,
        "method": "caption_speaker_plus_raw_boundary_v1",
        "automatic_label": automatic_label,
        "human_verified": False,
        "aligned_user_interval": [round(user.start, 3), round(user.end, 3)],
        "aligned_response_interval": [round(assistant.start, 3), round(assistant.end, 3)],
        "raw_user_interval": [round(raw_user_start, 3), round(raw_user_end, 3)],
        "raw_response_interval": [round(raw_response_start, 3), round(raw_response_end, 3)],
        "overlap_interval": [round(overlap_start, 3), round(overlap_end, 3)],
        "overlap_seconds": round(overlap_seconds, 3),
        "user_match_ratio": round(user_match, 4),
        "response_match_ratio": round(response_match, 4),
        "listen_interval": [
            round(max(0.0, overlap_start - 3.0), 3),
            round(min(duration, overlap_end + 3.0), 3),
        ],
        "thresholds": {
            "minimum_raw_turn_match_ratio": RAW_TURN_MATCH_RATIO,
            "caption_boundary_tolerance_s": RAW_BOUNDARY_TOLERANCE_S,
            "minimum_raw_overlap_s": MIN_RAW_OVERLAP_S,
        },
    }


def _verified_interaction_review(row: dict, candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    candidate_id = str(candidate["candidate_id"])
    for review in row.get("interaction_reviews") or []:
        if (
            isinstance(review, dict)
            and str(review.get("candidate_id") or "") == candidate_id
            and review.get("human_verified") is True
            and str(review.get("label") or "") in {"interrupt", "backchannel"}
        ):
            return review
    return None


def _pair_candidate(
    row: dict,
    user: SpeakerSegment,
    assistant: SpeakerSegment,
    *,
    max_response_gap_s: float,
    include_verified_reviews: bool,
) -> PairCandidate | None:
    if user.speaker == assistant.speaker or assistant.start - user.end > max_response_gap_s:
        return None
    if user.duration < 0.25 or assistant.duration < 0.25:
        return None
    candidate = _automatic_interaction_candidate(row, user, assistant)
    review = _verified_interaction_review(row, candidate) if include_verified_reviews else None
    user_start, user_end = user.start, user.end
    response_start, response_end = assistant.start, assistant.end
    label, overlap = _interaction_label(
        user,
        assistant,
        list(row.get("overlap_intervals") or []),
    )
    if review is not None and candidate is not None:
        user_start, user_end = map(float, candidate["raw_user_interval"])
        response_start, response_end = map(float, candidate["raw_response_interval"])
        label = str(review["label"])
        overlap = [list(candidate["overlap_interval"])]
    return PairCandidate(
        user=user,
        assistant=assistant,
        automatic_candidate=candidate,
        review=review,
        user_start=user_start,
        user_end=user_end,
        response_start=response_start,
        response_end=response_end,
        label=label,
        overlap=overlap,
    )


def _select_pair_candidates(
    row: dict,
    segments: list[SpeakerSegment],
    *,
    max_response_gap_s: float,
    pair_selection_method: str,
    include_verified_reviews: bool = True,
) -> tuple[list[PairCandidate], Counter[str]]:
    """Select adjacent turns without segment or source-interval reuse."""

    if pair_selection_method not in {"fixed_stride_v1", "max_duration_interval_v2"}:
        raise ValueError(f"unsupported pair_selection_method: {pair_selection_method}")
    step = 2 if pair_selection_method == "fixed_stride_v1" else 1
    candidates: list[PairCandidate] = []
    skipped: Counter[str] = Counter()
    for index in range(0, len(segments) - 1, step):
        candidate = _pair_candidate(
            row,
            segments[index],
            segments[index + 1],
            max_response_gap_s=max_response_gap_s,
            include_verified_reviews=include_verified_reviews,
        )
        if candidate is None:
            skipped["not_adjacent_response"] += 1
        else:
            candidates.append(candidate)

    if pair_selection_method == "fixed_stride_v1":
        selected: list[PairCandidate] = []
        consumed_until = -1.0
        for candidate in candidates:
            if candidate.span_start < consumed_until:
                skipped["source_interval_reuse"] += 1
                continue
            selected.append(candidate)
            consumed_until = candidate.span_end
        return selected, skipped

    ordered = sorted(
        candidates,
        key=lambda item: (item.span_end, item.span_start, item.user.start, item.assistant.end),
    )
    ends = [item.span_end for item in ordered]
    scores: list[tuple[float, int]] = [(0.0, 0)]
    take_flags: list[bool] = [False]
    predecessors: list[int] = [0]
    for index, candidate in enumerate(ordered, start=1):
        predecessor = bisect_right(ends, candidate.span_start, 0, index - 1)
        take_score = (
            scores[predecessor][0] + candidate.span_duration,
            scores[predecessor][1] + 1,
        )
        skip_score = scores[index - 1]
        take = take_score > skip_score
        scores.append(take_score if take else skip_score)
        take_flags.append(take)
        predecessors.append(predecessor)

    chosen: list[PairCandidate] = []
    index = len(ordered)
    while index > 0:
        if take_flags[index]:
            chosen.append(ordered[index - 1])
            index = predecessors[index]
        else:
            index -= 1
    chosen.reverse()
    skipped["valid_adjacent_candidate_not_selected"] += len(candidates) - len(chosen)
    return chosen, skipped


def estimate_conversation_pair_yield(
    diarized_jsonl: Path,
    out_json: Path | None = None,
    *,
    max_response_gap_s: float = 3.0,
    pair_selection_method: str = "fixed_stride_v1",
) -> dict:
    """Estimate builder yield without exporting clips or bypassing later gates."""

    rows = [
        json.loads(line)
        for line in Path(diarized_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = 0
    seconds = 0.0
    labels: Counter[str] = Counter()
    automatic_candidates: Counter[str] = Counter()
    verified_interactions: Counter[str] = Counter()
    by_channel_seconds: defaultdict[str, float] = defaultdict(float)
    skipped: Counter[str] = Counter()
    for row in rows:
        if (
            "automatic_multi_speaker_verified" in row
            and row.get("automatic_multi_speaker_verified") is not True
        ):
            skipped["not_verified_multi_speaker"] += 1
            continue
        if (
            "reference_alignment_status" in row
            and row.get("reference_alignment_status") != "complete"
        ):
            skipped["reference_alignment_incomplete"] += 1
            continue
        if row.get("manual_qa_reviewed") is True and row.get("human_verified") is not True:
            skipped["manual_qa_rejected"] += 1
            continue
        segments = _merge_same_speaker(_segments(row))
        if len({segment.speaker for segment in segments}) < 2:
            skipped["not_multi_speaker"] += 1
            continue
        selected, row_skipped = _select_pair_candidates(
            row,
            segments,
            max_response_gap_s=max_response_gap_s,
            pair_selection_method=pair_selection_method,
        )
        skipped.update(row_skipped)
        for selected_pair in selected:
            candidate = selected_pair.automatic_candidate
            if candidate is not None:
                automatic_candidates[str(candidate["automatic_label"])] += 1
            if selected_pair.review is not None:
                verified_interactions[selected_pair.label] += 1
            labels[selected_pair.label] += 1
            seconds += selected_pair.span_duration
            by_channel_seconds[str(row.get("channel") or "unknown")] += selected_pair.span_duration
            pairs += 1
    hours = seconds / 3600.0
    report = {
        "source_manifest": str(diarized_jsonl),
        "source_manifest_sha256": _sha256_path(Path(diarized_jsonl)),
        "source_windows": len(rows),
        "estimated_pairs": pairs,
        "estimated_pair_hours": round(hours, 3),
        "hours_100_to_200": 100.0 <= hours <= 200.0,
        "label_counts": dict(labels),
        "automatic_interaction_candidate_counts": dict(automatic_candidates),
        "automatic_interaction_candidates": sum(automatic_candidates.values()),
        "automatic_direct_interruption_candidates": automatic_candidates["interrupt"],
        "automatic_backchannel_candidates": automatic_candidates["backchannel"],
        "human_verified_interaction_counts": dict(verified_interactions),
        "human_verified_direct_interruption_pairs": verified_interactions["interrupt"],
        "direct_interruption_pairs": verified_interactions["interrupt"],
        "unattributed_overlap_pairs": labels["overlap_unattributed"],
        "channel_hours": {
            channel: round(value / 3600.0, 3)
            for channel, value in sorted(by_channel_seconds.items())
        },
        "skipped": dict(skipped),
        "candidate_rule": {
            "method": "caption_speaker_plus_raw_boundary_v1",
            "pair_selection_method": pair_selection_method,
            "minimum_raw_turn_match_ratio": RAW_TURN_MATCH_RATIO,
            "caption_boundary_tolerance_s": RAW_BOUNDARY_TOLERANCE_S,
            "minimum_raw_overlap_s": MIN_RAW_OVERLAP_S,
        },
        "non_mutating_estimate": True,
        "training_ready": False,
        "warning": (
            "This estimate writes no training clips and does not replace manual QA, "
            "rights application, file checks, or the final conversation audit. "
            "Automatic interaction candidates and overlap_unattributed are not "
            "human-verified interruption claims."
        ),
    }
    if out_json is not None:
        write_json(out_json, report)
    return report


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_conversation_reserve_by_yield(
    primary_manifest: Path,
    reserve_manifest: Path,
    primary_selection: Path,
    reserve_selection: Path,
    out_jsonl: Path,
    *,
    report_path: Path | None = None,
    target_pair_hours: float = 105.0,
    max_candidate_hours: float = 200.0,
) -> dict:
    """Select the minimum whole-episode reserve prefix needed for safe yield."""

    if target_pair_hours <= 0:
        raise ValueError("target_pair_hours must be positive")
    if max_candidate_hours <= 0:
        raise ValueError("max_candidate_hours must be positive")
    primary_manifest = Path(primary_manifest)
    reserve_manifest = Path(reserve_manifest)
    primary_selection = Path(primary_selection)
    reserve_selection = Path(reserve_selection)
    out_jsonl = Path(out_jsonl)
    selection_rows = [
        json.loads(line)
        for line in reserve_selection.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selection_ids = [str(row.get("episode_id") or "") for row in selection_rows]
    if not selection_ids or any(not value for value in selection_ids):
        raise ValueError("reserve selection must contain non-empty episode_id values")
    if len(selection_ids) != len(set(selection_ids)):
        raise ValueError("reserve selection contains duplicate episode_id values")
    primary_selection_rows = [
        json.loads(line)
        for line in primary_selection.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    primary_candidate_hours = sum(
        float(row.get("csv_hours") or 0.0) for row in primary_selection_rows
    )
    if primary_candidate_hours > max_candidate_hours:
        raise ValueError(
            "primary candidate hours already exceed max_candidate_hours: "
            f"{primary_candidate_hours:.3f} > {max_candidate_hours:.3f}"
        )
    reserve_rows = [
        json.loads(line)
        for line in reserve_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in reserve_rows:
        grouped[str(row.get("episode_id") or "")].append(row)
    primary_report = estimate_conversation_pair_yield(primary_manifest)
    running_pair_hours = float(primary_report["estimated_pair_hours"])
    running_candidate_hours = primary_candidate_hours
    selected_rows: list[dict] = []
    selected_details: list[dict] = []
    skipped: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="conversation-yield-selection-") as directory:
        for index, episode in enumerate(selection_rows):
            if running_pair_hours >= target_pair_hours:
                break
            episode_id = str(episode["episode_id"])
            episode_rows = grouped.get(episode_id, [])
            if not episode_rows:
                skipped["missing_diarized_episode"] += 1
                continue
            candidate_hours = float(episode.get("csv_hours") or 0.0)
            if candidate_hours <= 0:
                skipped["invalid_candidate_hours"] += 1
                continue
            if running_candidate_hours + candidate_hours > max_candidate_hours:
                skipped["candidate_hour_cap"] += 1
                continue
            temporary_manifest = Path(directory) / f"episode-{index:04d}.jsonl"
            temporary_manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episode_rows),
                encoding="utf-8",
            )
            episode_report = estimate_conversation_pair_yield(temporary_manifest)
            pair_hours = float(episode_report["estimated_pair_hours"])
            if pair_hours <= 0:
                skipped["zero_pair_yield"] += 1
                continue
            selected_rows.extend(episode_rows)
            running_candidate_hours += candidate_hours
            running_pair_hours += pair_hours
            selected_details.append(
                {
                    "episode_id": episode_id,
                    "candidate_hours": round(candidate_hours, 4),
                    "estimated_pair_hours": round(pair_hours, 3),
                    "windows": len(episode_rows),
                    "cumulative_candidate_hours": round(running_candidate_hours, 3),
                    "cumulative_pair_hours": round(running_pair_hours, 3),
                }
            )
    window_ids = [str(row.get("window_id") or "") for row in selected_rows]
    if any(not value for value in window_ids) or len(window_ids) != len(set(window_ids)):
        raise ValueError("selected reserve rows must have unique non-empty window_id values")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out_jsonl.parent,
            prefix=f".{out_jsonl.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in selected_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        assert temporary is not None
        temporary.replace(out_jsonl)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    selected_report = (
        estimate_conversation_pair_yield(out_jsonl)
        if selected_rows
        else {"estimated_pair_hours": 0.0, "estimated_pairs": 0}
    )
    combined_pair_hours = float(primary_report["estimated_pair_hours"]) + float(
        selected_report["estimated_pair_hours"]
    )
    selected_candidate_hours = running_candidate_hours - primary_candidate_hours
    allowed_ids = set(selection_ids)
    report = {
        "primary_manifest": str(primary_manifest),
        "reserve_manifest": str(reserve_manifest),
        "primary_selection": str(primary_selection),
        "reserve_selection": str(reserve_selection),
        "out_manifest": str(out_jsonl),
        "target_pair_hours": target_pair_hours,
        "max_candidate_hours": max_candidate_hours,
        "primary_candidate_hours": round(primary_candidate_hours, 3),
        "primary_estimated_pair_hours": primary_report["estimated_pair_hours"],
        "selected_reserve_episodes": len(selected_details),
        "selected_reserve_windows": len(selected_rows),
        "selected_reserve_candidate_hours": round(selected_candidate_hours, 3),
        "selected_reserve_estimated_pair_hours": selected_report["estimated_pair_hours"],
        "combined_candidate_hours": round(running_candidate_hours, 3),
        "combined_estimated_pair_hours": round(combined_pair_hours, 3),
        "target_reached": combined_pair_hours >= target_pair_hours,
        "candidate_cap_passes": running_candidate_hours <= max_candidate_hours,
        "selected_episodes": selected_details,
        "skipped": dict(skipped),
        "reserve_rows_outside_selection": sum(
            str(row.get("episode_id") or "") not in allowed_ids for row in reserve_rows
        ),
        "artifact_sha256": {
            "primary_manifest": _sha256_path(primary_manifest),
            "reserve_manifest": _sha256_path(reserve_manifest),
            "primary_selection": _sha256_path(primary_selection),
            "reserve_selection": _sha256_path(reserve_selection),
            "out_manifest": _sha256_path(out_jsonl),
        },
        "non_mutating_audio_selection": True,
        "training_ready": False,
        "warning": (
            "This deterministic selector chooses whole episodes using automatic estimated "
            "yield. Manual QA, rights application, final pair export, and audits still apply."
        ),
    }
    if report_path is not None:
        write_json(report_path, report)
    return report


def build_conversation_manifest(
    diarized_jsonl: Path,
    out_jsonl: Path,
    clips_dir: Path,
    *,
    max_hours: float = 200.0,
    max_response_gap_s: float = 3.0,
    pair_selection_method: str = "fixed_stride_v1",
    qa_waiver_path: Path | None = None,
) -> dict:
    """Extract non-overlapping user/response pairs from full diarized recordings.

    Consecutive caption-aligned source segments are consumed once. A reviewed
    interaction candidate uses its raw diarizer boundaries so the exported clips
    and stereo timing preserve the verified overlap; unreviewed candidates remain
    conservative metadata and never become interruption claims.
    """

    diarized_jsonl = Path(diarized_jsonl)
    out_jsonl = Path(out_jsonl)
    clips_dir = Path(clips_dir)
    qa_waiver = load_qa_waiver(qa_waiver_path)
    source_rows = [
        json.loads(line)
        for line in diarized_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require_training_authorized(source_rows, "conversation pair building")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[dict] = []
    label_counts: Counter[str] = Counter()
    automatic_candidate_counts: Counter[str] = Counter()
    verified_interaction_counts: Counter[str] = Counter()
    hours = 0.0
    skipped: Counter[str] = Counter()
    for row in source_rows:
        if (
            "automatic_multi_speaker_verified" in row
            and row.get("automatic_multi_speaker_verified") is not True
        ):
            skipped["not_verified_multi_speaker"] += 1
            continue
        if (
            "reference_alignment_status" in row
            and row.get("reference_alignment_status") != "complete"
        ):
            skipped["reference_alignment_incomplete"] += 1
            continue
        if row.get("manual_qa_reviewed") is True and row.get("human_verified") is not True:
            skipped["manual_qa_rejected"] += 1
            continue
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
        selected, row_skipped = _select_pair_candidates(
            row,
            segments,
            max_response_gap_s=max_response_gap_s,
            pair_selection_method=pair_selection_method,
            include_verified_reviews=qa_waiver is None,
        )
        skipped.update(row_skipped)
        for selected_pair in selected:
            user = selected_pair.user
            assistant = selected_pair.assistant
            candidate = selected_pair.automatic_candidate
            review = selected_pair.review
            user_start, user_end = selected_pair.user_start, selected_pair.user_end
            response_start = selected_pair.response_start
            response_end = selected_pair.response_end
            label = selected_pair.label
            overlap = selected_pair.overlap
            source_span_start = selected_pair.span_start
            source_span_end = selected_pair.span_end
            source_span_duration = selected_pair.span_duration
            pair_hours = source_span_duration / 3600.0
            if hours + pair_hours > max_hours:
                break
            user_audio = audio[
                max(0, int(user_start * SAMPLE_RATE)) : min(len(audio), int(user_end * SAMPLE_RATE))
            ]
            assistant_audio = audio[
                max(0, int(response_start * SAMPLE_RATE)) : min(
                    len(audio), int(response_end * SAMPLE_RATE)
                )
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
            if candidate is not None:
                automatic_candidate_counts[str(candidate["automatic_label"])] += 1
            label_counts[label] += 1
            if review is not None:
                verified_interaction_counts[label] += 1
            license_name = str(row.get("license") or "unknown")
            license_verified = rights_record_verified(row)
            training_authorized = training_use_authorized(row)
            pairs.append(
                {
                    "utt_id": pair_id,
                    "session_id": session_id,
                    "speaker_id": f"{session_id}:{user.speaker}",
                    "assistant_speaker_id": f"{session_id}:{assistant.speaker}",
                    "audio_filepath": str(user_path),
                    "response_audio_filepath": str(assistant_path),
                    "duration": round(source_span_duration, 3),
                    "user_duration": round(user_end - user_start, 3),
                    "source_span_start": round(source_span_start, 3),
                    "source_span_end": round(source_span_end, 3),
                    "source_user_interval": [round(user_start, 3), round(user_end, 3)],
                    "source_response_interval": [
                        round(response_start, 3),
                        round(response_end, 3),
                    ],
                    "assistant_duration": round(response_end - response_start, 3),
                    "transcript_caption": user.text,
                    "text": user.text,
                    "assistant_text": assistant.text,
                    "response_text": assistant.text,
                    "interrupt_label": label,
                    "overlap_intervals": overlap,
                    "response_gap_s": round(response_start - user_end, 3),
                    "automatic_interaction_candidate": candidate,
                    "interaction_review": review,
                    "human_verified_interaction_label": review is not None,
                    "interaction_annotation_source": (
                        "human_reviewed_raw_diarizer_boundary"
                        if review is not None
                        else "automatic_candidate_or_caption_boundary"
                    ),
                    "split": _stable_split(session_id),
                    "license": license_name,
                    "license_verified": license_verified,
                    "internal_research_authorized": bool(
                        row.get("internal_research_authorized", False)
                    ),
                    "training_use_authorized": training_authorized,
                    "authorization_basis": row.get("authorization_basis"),
                    "redistribution_allowed": bool(row.get("redistribution_allowed", False)),
                    "rights_review": row.get("rights_review"),
                    "source_audio_filepath": str(source),
                    "source_window_id": row.get("window_id"),
                    "automatic_multi_speaker_verified": row.get("automatic_multi_speaker_verified"),
                    "annotation_source": str(
                        row.get("annotation_source") or "automatic_diarization_heuristic"
                    ),
                    "human_verified": (
                        False if qa_waiver is not None else bool(row.get("human_verified", False))
                    ),
                    "manual_qa_waived": qa_waiver is not None,
                    "qa_policy": (qa_waiver["policy"] if qa_waiver is not None else "strict"),
                    "qa_waiver_sha256": (qa_waiver["sha256"] if qa_waiver is not None else None),
                    "noise_condition": str(
                        row.get("noise_condition") or row.get("room") or "unspecified"
                    ),
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
        "automatic_interaction_candidate_counts": dict(automatic_candidate_counts),
        "human_verified_interaction_counts": dict(verified_interaction_counts),
        "skipped": dict(skipped),
        "pair_selection_method": pair_selection_method,
        "qa_policy": qa_waiver or {"policy": "strict", "valid": True},
        "evidence_scope": (
            "automatic-only natural response pairs under a documented QA waiver; no "
            "human-verification or verified-interruption claim"
            if qa_waiver is not None
            else "natural human-human response pairs; interruption/backchannel claims "
            "require pair-level human verification"
        ),
    }


def export_llama_omni2_questions(manifest: Path, out_json: Path) -> dict:
    """Export the official LLaMA-Omni2 inference conversation schema."""

    source_rows = [
        json.loads(line)
        for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require_training_authorized(source_rows, "LLaMA-Omni2 export")
    conversations = []
    for row in source_rows:
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
    out_json.write_text(
        json.dumps(conversations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "out": str(out_json),
        "conversations": len(conversations),
        "schema": "llama_omni2_questions_v1",
    }


def audit_conversation_manifest(
    manifest: Path,
    out_json: Path | None = None,
    *,
    check_files: bool = True,
    qa_waiver_path: Path | None = None,
) -> dict:
    qa_waiver = load_qa_waiver(qa_waiver_path)
    rows = [
        json.loads(line)
        for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hours = sum(float(row.get("duration") or 0.0) for row in rows) / 3600.0
    speakers = {
        str(value)
        for row in rows
        for value in (row.get("speaker_id"), row.get("assistant_speaker_id"))
        if value
    }
    sessions = {str(row.get("session_id")) for row in rows if row.get("session_id")}
    missing_files = 0
    if check_files:
        for row in rows:
            paths = (row.get("audio_filepath"), row.get("response_audio_filepath"))
            missing_files += sum(not value or not Path(str(value)).is_file() for value in paths)
    labels = Counter(str(row.get("interrupt_label") or "none") for row in rows)
    verified_interaction_labels = Counter(
        str(row.get("interrupt_label") or "none")
        for row in rows
        if row.get("human_verified_interaction_label") is True
    )
    automatic_interaction_candidates = sum(
        isinstance(row.get("automatic_interaction_candidate"), dict) for row in rows
    )
    verified = sum(bool(row.get("human_verified")) for row in rows)
    licensed = sum(rights_record_verified(row) for row in rows)
    training_authorized = sum(training_use_authorized(row) for row in rows)
    overlap_rows = sum(bool(row.get("overlap_intervals")) for row in rows)
    noise_rows = sum(
        (
            (condition := str(row.get("noise_condition") or "unspecified").lower())
            not in {"", "unestimated", "unspecified", "none", "quiet", "clean"}
            and not condition.endswith("-clean")
        )
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
        reused_spans += sum(
            current[0] < previous[1]
            for previous, current in zip(ordered, ordered[1:], strict=False)
        )
    requirements = {
        "hours_100_to_200": 100.0 <= hours <= 200.0,
        "response_pairs_present": bool(rows)
        and all(row.get("response_audio_filepath") and row.get("assistant_text") for row in rows),
        "natural_multi_speaker_sessions": bool(rows)
        and all(len(values) >= 2 for values in session_speakers.values()),
        "session_groups_at_least_2": len(sessions) >= 2,
        "session_group_split_clean": split_leaks == 0,
        "source_intervals_not_reused": reused_spans == 0,
        "interruptions_present": verified_interaction_labels["interrupt"] > 0,
        "interaction_verification_sample_present": sum(verified_interaction_labels.values()) > 0,
        "overlap_annotations_present": overlap_rows > 0,
        "noise_conditions_present": noise_rows > 0,
        "training_authorization_complete": training_authorized == len(rows) and bool(rows),
        "manual_verification_sample_present": verified > 0,
        "files_present": missing_files == 0 if check_files else None,
    }
    automatic_requirements = {
        key: value
        for key, value in requirements.items()
        if key
        not in {
            "interruptions_present",
            "interaction_verification_sample_present",
            "manual_verification_sample_present",
        }
    }
    waiver_requirements = {
        **automatic_requirements,
        "documented_qa_waiver_valid": qa_waiver is not None,
        "all_pairs_declare_same_waiver": bool(rows)
        and qa_waiver is not None
        and all(row_matches_waiver(row, qa_waiver) for row in rows),
        "automatic_interaction_candidates_present": automatic_interaction_candidates > 0,
        "human_verified_claims_disabled": qa_waiver is not None,
    }
    training_ready_under_qa_waiver = qa_waiver is not None and all(
        value is True for value in waiver_requirements.values()
    )
    report = {
        "manifest": str(manifest),
        "pairs": len(rows),
        "hours": round(hours, 3),
        "speakers": len(speakers),
        "sessions": len(sessions),
        "label_counts": dict(labels),
        "human_verified_interaction_label_counts": dict(verified_interaction_labels),
        "automatic_interaction_candidates": automatic_interaction_candidates,
        "manifest_sha256": _sha256_path(Path(manifest)),
        "qa_waiver_sha256": (qa_waiver["sha256"] if qa_waiver is not None else None),
        "overlap_rows": overlap_rows,
        "noise_condition_rows": noise_rows,
        "session_group_split_leaks": split_leaks,
        "reused_source_spans": reused_spans,
        "human_verified_rows": verified,
        "licensed_rows": licensed,
        "training_authorized_rows": training_authorized,
        "missing_files": missing_files if check_files else None,
        "requirements": requirements,
        "thesis_coverage_ok": all(value is True for value in requirements.values()),
        "qa_policy": qa_waiver or {"policy": "strict", "valid": True},
        "waiver_requirements": waiver_requirements,
        "training_ready_under_qa_waiver": training_ready_under_qa_waiver,
        "claims": {
            "human_verified_data": verified > 0,
            "human_verified_interruptions": verified_interaction_labels["interrupt"] > 0,
            "strict_thesis_data_coverage": all(value is True for value in requirements.values()),
        },
        "warning": (
            "Manual QA was deliberately waived under a student time-constraint "
            "decision. Automatic labels remain pseudo-labels; this report permits "
            "limited internal training but does not claim strict thesis data coverage "
            "or human-verified interruptions."
            if qa_waiver is not None
            else "Automatic diarization/overlap candidates are pseudo-labels. The "
            "interruptions_present gate counts only pair-level human-verified labels."
        ),
    }
    if out_json is not None:
        write_json(out_json, report)
    return report
