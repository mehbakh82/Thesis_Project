"""Quantitative source-balance audit for a conversation-pair manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_s2s.config import portable_path, project_root
from thesis_s2s.data.conversation_selection import (
    load_jsonl,
    select_conversation_episodes,
    write_jsonl,
)
from thesis_s2s.data.prepare_youtube import caption_ok, iter_csv_rows
from thesis_s2s.metrics import write_json
from thesis_s2s.repro import sha256_file


def _channel(row: dict[str, Any]) -> str:
    explicit = str(row.get("channel") or "").strip()
    if explicit:
        return explicit
    session = str(row.get("session_id") or "").strip()
    return session.split("/", 1)[0] if "/" in session else "unknown"


def audit_conversation_balance(
    manifest_path: str | Path,
    out_path: str | Path,
    *,
    max_channel_hour_share: float = 0.55,
    min_channels: int = 4,
    root: str | Path | None = None,
) -> dict[str, Any]:
    if not 0.0 < max_channel_hour_share <= 1.0:
        raise ValueError("max_channel_hour_share must be in (0, 1]")
    if min_channels < 1:
        raise ValueError("min_channels must be positive")

    project = Path(root or project_root()).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = project / manifest
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("conversation manifest is empty")

    aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "pairs": 0,
            "seconds": 0.0,
            "user_seconds": 0.0,
            "response_seconds": 0.0,
            "sessions": set(),
            "human_verified_rows": 0,
            "interaction_candidates": 0,
        }
    )
    invalid_rows: list[int] = []
    for index, row in enumerate(rows, start=1):
        channel = _channel(row)
        duration = float(row.get("duration") or 0.0)
        user_duration = float(row.get("user_duration") or 0.0)
        response_duration = float(row.get("assistant_duration") or 0.0)
        session = str(row.get("session_id") or "")
        if duration <= 0.0 or not session or channel == "unknown":
            invalid_rows.append(index)
        item = aggregates[channel]
        item["pairs"] += 1
        item["seconds"] += max(0.0, duration)
        item["user_seconds"] += max(0.0, user_duration)
        item["response_seconds"] += max(0.0, response_duration)
        item["sessions"].add(session)
        item["human_verified_rows"] += int(bool(row.get("human_verified")))
        item["interaction_candidates"] += int(bool(row.get("automatic_interaction_candidate")))

    total_seconds = sum(float(item["seconds"]) for item in aggregates.values())
    total_hours = total_seconds / 3600.0
    total_pairs = len(rows)
    channel_rows: dict[str, Any] = {}
    for channel in sorted(aggregates):
        item = aggregates[channel]
        hours = float(item["seconds"]) / 3600.0
        channel_rows[channel] = {
            "pairs": int(item["pairs"]),
            "pair_share": round(int(item["pairs"]) / total_pairs, 6),
            "sessions": len(item["sessions"]),
            "source_pair_hours": round(hours, 6),
            "hour_share": round(hours / total_hours, 6),
            "user_hours": round(float(item["user_seconds"]) / 3600.0, 6),
            "source_response_hours": round(float(item["response_seconds"]) / 3600.0, 6),
            "human_verified_rows": int(item["human_verified_rows"]),
            "interaction_candidates": int(item["interaction_candidates"]),
        }

    dominant_channel = max(
        channel_rows,
        key=lambda name: float(channel_rows[name]["source_pair_hours"]),
    )
    dominant_hours = float(channel_rows[dominant_channel]["source_pair_hours"])
    dominant_share = dominant_hours / total_hours
    excess_at_fixed_total = max(0.0, dominant_hours - max_channel_hour_share * total_hours)
    target_total_if_only_adding = dominant_hours / max_channel_hour_share
    additional_if_only_adding = max(0.0, target_total_if_only_adding - total_hours)
    human_verified_rows = sum(int(item["human_verified_rows"]) for item in aggregates.values())

    balance_passes = bool(
        len(channel_rows) >= min_channels
        and dominant_share <= max_channel_hour_share + 1e-12
        and not invalid_rows
    )
    recommendations: list[str] = []
    if dominant_share > max_channel_hour_share:
        recommendations.extend(
            [
                (
                    f"At a fixed {total_hours:.3f} h total, replace at least "
                    f"{excess_at_fixed_total:.3f} h of {dominant_channel} with other sources."
                ),
                (
                    f"If retaining every {dominant_channel} pair, add at least "
                    f"{additional_if_only_adding:.3f} h from other sources."
                ),
            ]
        )
    if human_verified_rows == 0:
        recommendations.append(
            "Complete the preserved listening QA before making human-quality claims."
        )

    report = {
        "schema_version": 1,
        "evidence_class": "deterministic_manifest_balance_audit",
        "manifest": portable_path(manifest, root=project),
        "manifest_sha256": sha256_file(manifest),
        "requirements": {
            "max_channel_hour_share": max_channel_hour_share,
            "minimum_channels": min_channels,
        },
        "totals": {
            "pairs": total_pairs,
            "source_pair_hours": round(total_hours, 6),
            "channels": len(channel_rows),
            "sessions": len({str(row.get("session_id") or "") for row in rows}),
            "human_verified_rows": human_verified_rows,
        },
        "channels": channel_rows,
        "dominance": {
            "channel": dominant_channel,
            "hours": round(dominant_hours, 6),
            "hour_share": round(dominant_share, 6),
            "balance_gate_passes": balance_passes,
            "hours_to_replace_at_fixed_total": round(excess_at_fixed_total, 6),
            "non_dominant_hours_to_add_if_retaining_all": round(additional_if_only_adding, 6),
        },
        "quality_boundaries": {
            "invalid_rows": invalid_rows,
            "all_rows_human_verified": human_verified_rows == total_pairs,
            "automatic_balance_is_human_quality_evidence": False,
        },
        "recommendations": recommendations,
        "strict_representative_balance_passes": balance_passes,
    }
    write_json(out_path, report)
    return report


def select_balanced_conversation_pairs(
    manifest_path: str | Path,
    out_path: str | Path,
    report_path: str | Path,
    *,
    target_max_channel_share: float = 0.54,
    min_hours: float = 100.0,
    max_hours: float = 200.0,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Keep every minority-source pair and session-fairly cap the dominant source."""

    if not 0.0 < target_max_channel_share < 1.0:
        raise ValueError("target_max_channel_share must be in (0, 1)")
    if not 0.0 < min_hours <= max_hours:
        raise ValueError("require 0 < min_hours <= max_hours")
    project = Path(root or project_root()).resolve()

    def resolve(path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else project / value

    source_file = resolve(manifest_path)
    output_file = resolve(out_path)
    report_file = resolve(report_path)
    rows = load_jsonl(source_file)
    if not rows:
        raise ValueError("conversation manifest is empty")
    indexed = list(enumerate(rows))
    by_channel_seconds: dict[str, float] = defaultdict(float)
    for _, row in indexed:
        by_channel_seconds[_channel(row)] += max(0.0, float(row.get("duration") or 0.0))
    dominant_channel = max(by_channel_seconds, key=lambda channel: by_channel_seconds[channel])
    minority = [(index, row) for index, row in indexed if _channel(row) != dominant_channel]
    dominant = [(index, row) for index, row in indexed if _channel(row) == dominant_channel]
    minority_seconds = sum(max(0.0, float(row.get("duration") or 0.0)) for _, row in minority)
    if minority_seconds <= 0.0:
        raise ValueError("at least one positive-duration minority channel is required")
    if minority_seconds / 3600.0 > max_hours:
        raise ValueError("minority-channel pairs alone exceed max_hours")
    dominant_share_budget = (
        target_max_channel_share / (1.0 - target_max_channel_share) * minority_seconds
    )
    dominant_total_budget = max_hours * 3600.0 - minority_seconds
    dominant_budget = max(0.0, min(dominant_share_budget, dominant_total_budget))

    queues: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for item in dominant:
        session = str(item[1].get("session_id") or "unknown")
        queues[session].append(item)

    def stable(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    for session_rows in queues.values():
        session_rows.sort(key=lambda item: stable(str(item[1].get("utt_id") or item[0])))
    sessions = sorted(queues, key=stable)
    selected_dominant: list[tuple[int, dict]] = []
    selected_dominant_seconds = 0.0
    remaining = True
    while remaining:
        remaining = False
        for session in sessions:
            queue = queues[session]
            if not queue:
                continue
            remaining = True
            item = queue.pop(0)
            duration = max(0.0, float(item[1].get("duration") or 0.0))
            if selected_dominant_seconds + duration <= dominant_budget + 1e-9:
                selected_dominant.append(item)
                selected_dominant_seconds += duration

    selected = sorted(minority + selected_dominant, key=lambda item: item[0])
    selected_rows = [row for _, row in selected]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_file, selected_rows)
    total_seconds = minority_seconds + selected_dominant_seconds
    selected_by_channel: dict[str, float] = defaultdict(float)
    for row in selected_rows:
        selected_by_channel[_channel(row)] += max(0.0, float(row.get("duration") or 0.0))
    largest_share = max(selected_by_channel.values(), default=0.0) / total_seconds
    utt_ids = [str(row.get("utt_id") or "") for row in selected_rows]
    gate = bool(
        min_hours <= total_seconds / 3600.0 <= max_hours
        and largest_share <= target_max_channel_share + 1e-12
        and len(selected_by_channel) >= 2
        and all(utt_ids)
        and len(utt_ids) == len(set(utt_ids))
    )
    report = {
        "schema_version": 1,
        "evidence_class": "deterministic_session_fair_balance_selection",
        "source_manifest": portable_path(source_file, root=project),
        "source_manifest_sha256": sha256_file(source_file),
        "out_manifest": portable_path(output_file, root=project),
        "out_manifest_sha256": sha256_file(output_file),
        "policy": {
            "target_max_channel_share": target_max_channel_share,
            "min_hours": min_hours,
            "max_hours": max_hours,
            "minority_policy": "retain_all",
            "dominant_policy": "deterministic_session_round_robin",
        },
        "dominant_channel": dominant_channel,
        "input_pairs": len(rows),
        "selected_pairs": len(selected_rows),
        "selected_hours": round(total_seconds / 3600.0, 6),
        "selected_hours_by_channel": {
            channel: round(seconds / 3600.0, 6)
            for channel, seconds in sorted(selected_by_channel.items())
        },
        "largest_channel_share": round(largest_share, 6),
        "retained_all_minority_pairs": len(minority)
        == sum(_channel(row) != dominant_channel for row in selected_rows),
        "selected_dominant_sessions": len(
            {str(row.get("session_id") or "") for _, row in selected_dominant}
        ),
        "selection_gate_passes": gate,
        "training_ready": False,
    }
    write_json(report_file, report)
    return report


def plan_balanced_supplement(
    inventory_path: str | Path,
    current_pairs_path: str | Path,
    used_windows_path: str | Path,
    out_selection_path: str | Path,
    out_report_path: str | Path,
    *,
    allowed_channels: tuple[str, ...] = (
        "Digiato",
        "Mehran Rowshan Persian",
        "Zoomit",
    ),
    target_candidate_hours: float = 55.0,
    max_candidate_hours: float = 65.0,
    max_supplement_channel_share: float = 0.40,
    min_final_pair_hours: float = 100.0,
    max_final_channel_share: float = 0.55,
    expected_yield_safety_factor: float = 1.25,
    current_yield_report_path: str | Path | None = None,
    local_chunk_manifest_path: str | Path | None = None,
    min_local_chunk_coverage: float = 0.95,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Plan a disjoint non-dominant supplement without changing frozen v1.

    Expected pair yield is calibrated per channel from the current final pair
    manifest divided by all candidate hours already prepared for that channel.
    It is planning evidence only: selected episodes still require preparation,
    diarization, alignment, authorization, export, and independent audit.
    """

    if not allowed_channels or any(not channel.strip() for channel in allowed_channels):
        raise ValueError("allowed_channels must contain non-empty names")
    if not 0.0 < target_candidate_hours <= max_candidate_hours:
        raise ValueError("require 0 < target_candidate_hours <= max_candidate_hours")
    if not 0.0 < max_supplement_channel_share <= 1.0:
        raise ValueError("max_supplement_channel_share must be in (0, 1]")
    if min_final_pair_hours <= 0.0:
        raise ValueError("min_final_pair_hours must be positive")
    if not 0.0 < max_final_channel_share < 1.0:
        raise ValueError("max_final_channel_share must be in (0, 1)")
    if expected_yield_safety_factor < 1.0:
        raise ValueError("expected_yield_safety_factor must be at least 1")
    if not 0.0 < min_local_chunk_coverage <= 1.0:
        raise ValueError("min_local_chunk_coverage must be in (0, 1]")

    project = Path(root or project_root()).resolve()

    def resolve(path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else project / value

    inventory_file = resolve(inventory_path)
    current_pairs_file = resolve(current_pairs_path)
    used_windows_file = resolve(used_windows_path)
    selection_file = resolve(out_selection_path)
    report_file = resolve(out_report_path)
    yield_report_file = (
        resolve(current_yield_report_path) if current_yield_report_path is not None else None
    )
    local_chunk_manifest_file = (
        resolve(local_chunk_manifest_path) if local_chunk_manifest_path is not None else None
    )
    inventory = load_jsonl(inventory_file)
    current_pairs = load_jsonl(current_pairs_file)
    used_windows = load_jsonl(used_windows_file)
    if not inventory or not current_pairs or not used_windows:
        raise ValueError("inventory, current pairs, and used windows must be non-empty")

    allowed = set(allowed_channels)
    inventory_by_id = {
        str(row.get("episode_id") or ""): row for row in inventory if row.get("episode_id")
    }
    used_ids = {
        str(row.get("episode_id") or row.get("session_id") or "")
        for row in used_windows
        if row.get("episode_id") or row.get("session_id")
    }

    pair_hours: dict[str, float] = defaultdict(float)
    yield_report: dict[str, Any] | None = None
    if yield_report_file is not None:
        if not yield_report_file.is_file():
            raise FileNotFoundError(f"current yield report not found: {yield_report_file}")
        yield_report = json.loads(yield_report_file.read_text(encoding="utf-8"))
        if yield_report.get("source_manifest_sha256") != sha256_file(used_windows_file):
            raise ValueError("current yield report is not bound to used_windows")
        for channel, hours in dict(yield_report.get("channel_hours") or {}).items():
            pair_hours[str(channel)] = max(0.0, float(hours))
    else:
        for row in current_pairs:
            pair_hours[_channel(row)] += max(0.0, float(row.get("duration") or 0.0)) / 3600.0
    if not pair_hours:
        raise ValueError("current pair-hour evidence is empty")
    current_total = sum(pair_hours.values())
    dominant_channel = max(pair_hours, key=lambda channel: pair_hours[channel])
    dominant_hours = pair_hours[dominant_channel]
    current_non_dominant = current_total - dominant_hours

    used_candidate_hours: dict[str, float] = defaultdict(float)
    for episode_id in used_ids:
        episode = inventory_by_id.get(episode_id)
        if episode is not None:
            used_candidate_hours[str(episode.get("channel") or "unknown")] += max(
                0.0, float(episode.get("csv_hours") or 0.0)
            )
    yield_rates = {
        channel: pair_hours.get(channel, 0.0) / candidate_hours
        for channel, candidate_hours in used_candidate_hours.items()
        if channel in allowed and candidate_hours > 0.0
    }
    local_source_csvs: set[str] | None = None
    local_chunk_names: dict[str, set[str]] = defaultdict(set)
    local_coverage_by_source: dict[str, float] = {}
    local_manifest_rows = 0
    if local_chunk_manifest_file is not None:
        if not local_chunk_manifest_file.is_file():
            raise FileNotFoundError(f"local chunk manifest not found: {local_chunk_manifest_file}")
        local_source_csvs = set()
        with local_chunk_manifest_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                source_csv = str(row.get("source_csv") or "")
                audio_path = Path(str(row.get("audio_filepath") or row.get("audio_path") or ""))
                if source_csv and audio_path.is_file():
                    local_source_csvs.add(source_csv)
                    local_chunk_names[source_csv].add(audio_path.name)
                    local_manifest_rows += 1

        for source_csv in local_source_csvs:
            csv_path = Path(source_csv)
            if not csv_path.is_file():
                continue
            expected = sum(
                caption_ok(str(row.get("text") or row.get("transcript") or "")) is not None
                for row in iter_csv_rows(csv_path)
            )
            local_coverage_by_source[source_csv] = (
                len(local_chunk_names[source_csv]) / expected if expected else 0.0
            )

    eligible = [
        row
        for row in inventory
        if str(row.get("channel") or "") in allowed
        and (
            local_source_csvs is None
            or local_coverage_by_source.get(str(row.get("csv_path") or ""), 0.0)
            >= min_local_chunk_coverage
        )
    ]
    calibrated_channels_needed = {
        str(row.get("channel") or "") for row in eligible if row.get("channel")
    }
    missing_calibration = sorted(calibrated_channels_needed - set(yield_rates))
    if missing_calibration:
        raise ValueError(
            "no observed yield calibration for allowed channels: " + ", ".join(missing_calibration)
        )

    selected = select_conversation_episodes(
        eligible,
        target_hours=target_candidate_hours,
        min_hours=target_candidate_hours,
        max_hours=max_candidate_hours,
        max_channel_share=max_supplement_channel_share,
        exclude_episode_ids=used_ids,
    )
    if not selected:
        raise ValueError("no unused eligible episodes satisfy the supplement policy")
    write_jsonl(selection_file, selected)

    selected_candidate_hours: dict[str, float] = defaultdict(float)
    selected_episodes: Counter[str] = Counter()
    for row in selected:
        channel = str(row.get("channel") or "unknown")
        selected_candidate_hours[channel] += max(0.0, float(row.get("csv_hours") or 0.0))
        selected_episodes[channel] += 1
    selected_total = sum(selected_candidate_hours.values())
    expected_pair_hours = {
        channel: selected_candidate_hours[channel] * yield_rates[channel]
        for channel in sorted(selected_candidate_hours)
    }
    expected_total = sum(expected_pair_hours.values())

    required_non_dominant_at_minimum = min_final_pair_hours * (1.0 - max_final_channel_share)
    minimum_new_pair_hours = max(0.0, required_non_dominant_at_minimum - current_non_dominant)
    safety_target = minimum_new_pair_hours * expected_yield_safety_factor
    dominant_to_retain = min(dominant_hours, min_final_pair_hours * max_final_channel_share)
    dominant_to_drop = max(0.0, dominant_hours - dominant_to_retain)
    estimated_non_dominant_available = current_non_dominant + expected_total
    estimated_balance_feasible = bool(
        current_total + expected_total >= min_final_pair_hours
        and estimated_non_dominant_available >= required_non_dominant_at_minimum
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "evidence_class": "deterministic_balance_supplement_plan",
        "status": "planned",
        "frozen_v1_mutated": False,
        "inputs": {
            "inventory": portable_path(inventory_file, root=project),
            "inventory_sha256": sha256_file(inventory_file),
            "current_pairs": portable_path(current_pairs_file, root=project),
            "current_pairs_sha256": sha256_file(current_pairs_file),
            "used_windows": portable_path(used_windows_file, root=project),
            "used_windows_sha256": sha256_file(used_windows_file),
            "current_yield_report": (
                portable_path(yield_report_file, root=project) if yield_report_file else None
            ),
            "current_yield_report_sha256": (
                sha256_file(yield_report_file) if yield_report_file else None
            ),
            "current_pair_hours_source": (
                "bound_non_mutating_yield_report" if yield_report else "exported_pair_manifest"
            ),
            "local_chunk_manifest": (
                portable_path(local_chunk_manifest_file, root=project)
                if local_chunk_manifest_file
                else None
            ),
            "local_chunk_manifest_sha256": (
                sha256_file(local_chunk_manifest_file) if local_chunk_manifest_file else None
            ),
            "local_manifest_existing_audio_rows": local_manifest_rows,
            "local_manifest_covered_source_csvs": (
                len(local_source_csvs) if local_source_csvs is not None else None
            ),
            "local_manifest_source_csvs_passing_coverage": (
                sum(
                    coverage >= min_local_chunk_coverage
                    for coverage in local_coverage_by_source.values()
                )
                if local_source_csvs is not None
                else None
            ),
        },
        "policy": {
            "allowed_channels": list(allowed_channels),
            "target_candidate_hours": target_candidate_hours,
            "max_candidate_hours": max_candidate_hours,
            "max_supplement_channel_share": max_supplement_channel_share,
            "min_final_pair_hours": min_final_pair_hours,
            "max_final_channel_share": max_final_channel_share,
            "expected_yield_safety_factor": expected_yield_safety_factor,
            "min_local_chunk_coverage": min_local_chunk_coverage,
        },
        "current": {
            "pair_hours": round(current_total, 6),
            "dominant_channel": dominant_channel,
            "dominant_pair_hours": round(dominant_hours, 6),
            "non_dominant_pair_hours": round(current_non_dominant, 6),
            "pair_hours_by_channel": {
                channel: round(hours, 6) for channel, hours in sorted(pair_hours.items())
            },
        },
        "calibration": {
            channel: {
                "used_candidate_hours": round(used_candidate_hours[channel], 6),
                "observed_pair_hours": round(pair_hours.get(channel, 0.0), 6),
                "observed_pair_yield_rate": round(rate, 6),
            }
            for channel, rate in sorted(yield_rates.items())
        },
        "selection": {
            "manifest": portable_path(selection_file, root=project),
            "manifest_sha256": sha256_file(selection_file),
            "episodes": len(selected),
            "candidate_hours": round(selected_total, 6),
            "by_channel": {
                channel: {
                    "episodes": selected_episodes[channel],
                    "candidate_hours": round(selected_candidate_hours[channel], 6),
                    "expected_pair_hours": round(expected_pair_hours[channel], 6),
                }
                for channel in sorted(selected_candidate_hours)
            },
            "disjoint_from_used_episodes": all(
                str(row.get("episode_id") or "") not in used_ids for row in selected
            ),
            "all_episodes_have_local_chunks": (
                all(
                    local_coverage_by_source.get(str(row.get("csv_path") or ""), 0.0)
                    >= min_local_chunk_coverage
                    for row in selected
                )
                if local_source_csvs is not None
                else None
            ),
        },
        "projection": {
            "minimum_new_non_dominant_pair_hours": round(minimum_new_pair_hours, 6),
            "safety_target_pair_hours": round(safety_target, 6),
            "expected_new_pair_hours": round(expected_total, 6),
            "expected_yield_safety_gate_passes": expected_total >= safety_target,
            "dominant_pair_hours_to_retain_at_minimum": round(dominant_to_retain, 6),
            "dominant_pair_hours_to_drop_at_minimum": round(dominant_to_drop, 6),
            "estimated_non_dominant_pair_hours_available": round(
                estimated_non_dominant_available, 6
            ),
            "estimated_balanced_minimum_feasible": estimated_balance_feasible,
        },
        "planning_gate_passes": bool(
            selected_total >= target_candidate_hours
            and expected_total >= safety_target
            and estimated_balance_feasible
            and all(str(row.get("episode_id") or "") not in used_ids for row in selected)
            and (
                local_source_csvs is None
                or all(
                    local_coverage_by_source.get(str(row.get("csv_path") or ""), 0.0)
                    >= min_local_chunk_coverage
                    for row in selected
                )
            )
        ),
        "training_ready": False,
        "next_stage": (
            "prepare and diarize this disjoint selection into versioned paths; then apply "
            "the existing alignment, authorization, pair-building, and balance audits"
        ),
    }
    write_json(report_file, report)
    return report
