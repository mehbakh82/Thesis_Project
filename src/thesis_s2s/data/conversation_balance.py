"""Quantitative source-balance audit for a conversation-pair manifest."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis_s2s.config import portable_path, project_root
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
        item["interaction_candidates"] += int(
            bool(row.get("automatic_interaction_candidate"))
        )

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
            "source_response_hours": round(
                float(item["response_seconds"]) / 3600.0, 6
            ),
            "human_verified_rows": int(item["human_verified_rows"]),
            "interaction_candidates": int(item["interaction_candidates"]),
        }

    dominant_channel = max(
        channel_rows,
        key=lambda name: float(channel_rows[name]["source_pair_hours"]),
    )
    dominant_hours = float(channel_rows[dominant_channel]["source_pair_hours"])
    dominant_share = dominant_hours / total_hours
    excess_at_fixed_total = max(
        0.0, dominant_hours - max_channel_hour_share * total_hours
    )
    target_total_if_only_adding = dominant_hours / max_channel_hour_share
    additional_if_only_adding = max(0.0, target_total_if_only_adding - total_hours)
    human_verified_rows = sum(
        int(item["human_verified_rows"]) for item in aggregates.values()
    )

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
            "non_dominant_hours_to_add_if_retaining_all": round(
                additional_if_only_adding, 6
            ),
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
