"""Build and audit a deterministic, conversation-first episode selection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from thesis_s2s.config import portable_path, project_root
from thesis_s2s.data.ingest import inventory_csvs
from thesis_s2s.data.rights import training_use_authorized
from thesis_s2s.metrics import write_json


def _episode_hours(row: dict) -> float:
    return max(0.0, float(row.get("csv_hours") or 0.0))


def _stable_key(row: dict) -> str:
    episode_id = str(row.get("episode_id") or f"{row.get('channel')}/{row.get('stem')}")
    return hashlib.sha256(episode_id.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_conversation_episodes(
    episodes: list[dict],
    *,
    target_hours: float = 180.0,
    min_hours: float = 100.0,
    max_hours: float = 200.0,
    max_channel_share: float = 0.55,
    exclude_episode_ids: set[str] | None = None,
) -> list[dict]:
    """Weighted-fair whole-episode selection over the complete eligible pool.

    Selection never asserts that an episode is conversational. It only chooses
    candidates for episode reconstruction, diarization, alignment, and review.
    """

    if not 0 < min_hours <= target_hours <= max_hours:
        raise ValueError("require 0 < min_hours <= target_hours <= max_hours")
    if not 0 < max_channel_share <= 1:
        raise ValueError("max_channel_share must be in (0, 1]")

    queues: dict[str, list[dict]] = defaultdict(list)
    weights: dict[str, float] = {}
    excluded = exclude_episode_ids or set()
    for episode in episodes:
        channel = str(episode.get("channel") or "")
        episode_id = str(episode.get("episode_id") or "")
        weight = max(0.0, float(episode.get("selection_weight") or 0.0))
        if (
            channel
            and episode_id not in excluded
            and bool(episode.get("selection_eligible"))
            and _episode_hours(episode) > 0
            and weight > 0
        ):
            queues[channel].append(dict(episode))
            weights[channel] = max(weights.get(channel, 0.0), weight)
    for queue in queues.values():
        queue.sort(key=_stable_key)
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        return []
    normalized = {channel: weight / weight_sum for channel, weight in weights.items()}

    selected: list[dict] = []
    selected_hours: dict[str, float] = defaultdict(float)
    total_hours = 0.0
    channel_cap = target_hours * max_channel_share
    while total_hours < target_hours:
        options: list[tuple[float, int, str, str, int, dict]] = []
        for channel, queue in queues.items():
            for index, episode in enumerate(queue):
                hours = _episode_hours(episode)
                if total_hours + hours > max_hours + 1e-9:
                    continue
                if selected_hours[channel] + hours > channel_cap + 1e-9:
                    continue
                fair_share = selected_hours[channel] / normalized[channel]
                options.append(
                    (
                        fair_share,
                        int(episode.get("conversation_priority") or 99),
                        channel,
                        _stable_key(episode),
                        index,
                        episode,
                    )
                )
                break
        if not options:
            break
        _, _, channel, _, index, episode = min(options, key=lambda item: item[:4])
        queues[channel].pop(index)
        hours = _episode_hours(episode)
        total_hours += hours
        selected_hours[channel] += hours
        episode.update(
            {
                "selected_for_conversation_prep": True,
                "selection_rank": len(selected) + 1,
                "selection_status": "candidate_pending_diarization",
                "selection_target_hours": target_hours,
                "selection_max_hours": max_hours,
                "selection_max_channel_share": max_channel_share,
            }
        )
        selected.append(episode)
    return selected


def audit_episode_selection(
    selected: list[dict],
    *,
    min_hours: float = 100.0,
    max_hours: float = 200.0,
    target_hours: float = 180.0,
    max_channel_share: float = 0.55,
) -> dict:
    hours = sum(_episode_hours(row) for row in selected)
    by_channel_hours: dict[str, float] = defaultdict(float)
    by_channel_episodes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in selected:
        channel = str(row.get("channel") or "unknown")
        by_channel_hours[channel] += _episode_hours(row)
        by_channel_episodes[channel] += 1
        reasons[str(row.get("conversation_candidate_reason") or "unknown")] += 1

    episode_ids = [str(row.get("episode_id") or "") for row in selected]
    podcast_channels = {
        str(row.get("channel")) for row in selected if row.get("source_kind") == "interview_podcast"
    }
    largest_share = max(by_channel_hours.values(), default=0.0) / hours if hours else 0.0
    automatic_hours = sum(
        _episode_hours(row) for row in selected if bool(row.get("automatic_multi_speaker_verified"))
    )
    aligned_hours = sum(
        _episode_hours(row)
        for row in selected
        if row.get("reference_alignment_status") == "complete"
    )
    manual_episodes = sum(bool(row.get("human_verified")) for row in selected)
    diarized_episodes = sum(row.get("diarization_status") == "complete" for row in selected)

    planning_requirements = {
        "candidate_hours_100_to_200": min_hours <= hours <= max_hours,
        "selection_target_reached": hours >= target_hours,
        "whole_episode_selection": all(
            row.get("selected_for_conversation_prep") is True and row.get("episode_id")
            for row in selected
        ),
        "episode_ids_unique": bool(episode_ids)
        and len(episode_ids) == len(set(episode_ids))
        and all(episode_ids),
        "at_least_two_channels": len(by_channel_hours) >= 2,
        "both_podcast_sources_present": len(podcast_channels) >= 2,
        "channel_share_within_cap": largest_share <= max_channel_share + 0.01,
        "likely_monologues_excluded": all(
            row.get("conversation_candidate_reason") != "likely_monologue_excluded_by_default"
            for row in selected
        ),
        "reference_csv_available": all(
            row.get("reference_transcript_source") == "provided_youtube_csv"
            and Path(str(row.get("csv_path") or "")).is_file()
            for row in selected
        ),
        "episode_group_split_present": all(
            row.get("split") in {"train", "val", "test"} for row in selected
        ),
    }
    evidence_requirements = {
        "diarization_complete_for_selected": bool(selected) and diarized_episodes == len(selected),
        "automatic_multi_speaker_hours_100_to_200": min_hours <= automatic_hours <= max_hours,
        "reference_alignment_hours_100_to_200": min_hours <= aligned_hours <= max_hours,
        "manual_qa_sample_present": manual_episodes > 0,
        "training_authorized_for_selected": bool(selected)
        and all(map(training_use_authorized, selected)),
    }
    return {
        "selection_is_candidate_data_not_final_evidence": True,
        "selected_episodes": len(selected),
        "selected_candidate_hours": round(hours, 3),
        "target_hours": target_hours,
        "min_hours": min_hours,
        "max_hours": max_hours,
        "channels": {
            channel: {
                "episodes": by_channel_episodes[channel],
                "hours": round(channel_hours, 3),
                "share": round(channel_hours / hours, 4) if hours else 0.0,
            }
            for channel, channel_hours in sorted(by_channel_hours.items())
        },
        "candidate_reasons": dict(reasons),
        "largest_channel_share": round(largest_share, 4),
        "diarized_episodes": diarized_episodes,
        "automatic_multi_speaker_hours": round(automatic_hours, 3),
        "reference_aligned_hours": round(aligned_hours, 3),
        "human_verified_episodes": manual_episodes,
        "planning_requirements": planning_requirements,
        "planning_gate_passes": all(planning_requirements.values()),
        "evidence_requirements": evidence_requirements,
        "thesis_evidence_gate_passes": all(evidence_requirements.values()),
        "next_stage": "prepare episode windows, diarize, align provided CSV text, then review a sample",
    }


def run_conversation_selection(
    *,
    target_hours: float = 180.0,
    min_hours: float = 100.0,
    max_hours: float = 200.0,
    max_channel_share: float = 0.55,
    reserve_hours: float = 120.0,
    reserve_max_hours: float = 150.0,
    csv_root: Path | None = None,
    manifest_dir: Path | None = None,
    audit_path: Path | None = None,
) -> dict:
    root = project_root()
    csv_root = Path(csv_root or root / "data" / "raw" / "youtube_csvs")
    manifest_dir = Path(manifest_dir or root / "data" / "processed" / "manifests")
    audit_path = Path(audit_path or root / "results" / "conversation_selection_audit.json")
    inventory = inventory_csvs(csv_root)
    episodes = inventory["episodes"]
    selected = select_conversation_episodes(
        episodes,
        target_hours=target_hours,
        min_hours=min_hours,
        max_hours=max_hours,
        max_channel_share=max_channel_share,
    )
    selected_ids = {str(row["episode_id"]) for row in selected}
    reserve = select_conversation_episodes(
        episodes,
        target_hours=reserve_hours,
        min_hours=min(100.0, reserve_hours),
        max_hours=reserve_max_hours,
        max_channel_share=max_channel_share,
        exclude_episode_ids=selected_ids,
    )
    inventory_path = manifest_dir / "youtube_episode_inventory.jsonl"
    selection_path = manifest_dir / "youtube_conversation_selection.jsonl"
    reserve_path = manifest_dir / "youtube_conversation_reserve.jsonl"
    write_jsonl(inventory_path, episodes)
    write_jsonl(selection_path, selected)
    write_jsonl(reserve_path, reserve)
    audit = audit_episode_selection(
        selected,
        min_hours=min_hours,
        max_hours=max_hours,
        target_hours=target_hours,
        max_channel_share=max_channel_share,
    )
    audit.update(
        {
            "inventory_stats": inventory["stats"],
            "inventory_manifest": portable_path(inventory_path, root=root),
            "selection_manifest": portable_path(selection_path, root=root),
            "reserve_manifest": portable_path(reserve_path, root=root),
            "reserve_episodes": len(reserve),
            "reserve_candidate_hours": round(
                sum(_episode_hours(row) for row in reserve),
                3,
            ),
            "reserve_note": (
                "prepare in small batches only if verified primary yield is below 100 h"
            ),
        }
    )
    write_json(audit_path, audit)
    return audit
