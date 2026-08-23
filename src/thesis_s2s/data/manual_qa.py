"""Deterministic human-QA handoff for already-recorded conversation windows."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from thesis_s2s.metrics import write_json

QA_FIELDS = (
    "review_status",
    "speaker_count_correct",
    "speaker_assignment_correct",
    "caption_acceptable",
    "overlap_annotation_correct",
    "reviewer_id",
    "notes",
)
YES = {"1", "true", "yes", "y", "بله"}


def _stable_key(row: dict) -> str:
    return hashlib.sha256(str(row.get("window_id") or "").encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sample_manual_qa(
    diarized_jsonl: Path,
    out_csv: Path,
    *,
    per_stratum: int = 5,
) -> dict:
    """Sample by channel and automatic pass/reject status.

    The CSV references existing internal audio. It does not collect or copy any
    new voice recording.
    """

    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    rows = [
        row
        for row in _read_jsonl(Path(diarized_jsonl))
        if row.get("diarization_status") == "complete" and row.get("window_id")
    ]
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        status = (
            "automatic_pass" if row.get("automatic_multi_speaker_verified") else "automatic_reject"
        )
        strata[(str(row.get("channel") or "unknown"), status)].append(row)

    sampled: list[dict] = []
    for (channel, status), candidates in sorted(strata.items()):
        for row in sorted(candidates, key=_stable_key)[:per_stratum]:
            alignment = row.get("reference_alignment") or {}
            speakers = row.get("speaker_evidence") or {}
            sampled.append(
                {
                    "window_id": row.get("window_id"),
                    "episode_id": row.get("episode_id"),
                    "channel": channel,
                    "automatic_status": status,
                    "audio_filepath": row.get("audio_filepath"),
                    "duration": row.get("duration"),
                    "n_speakers": speakers.get("n_speakers"),
                    "secondary_speaker_share": speakers.get("secondary_speaker_share"),
                    "alignment_coverage": alignment.get("usable_coverage"),
                    "overlap_intervals": len(row.get("overlap_intervals") or []),
                    "license": row.get("license"),
                    "license_verified": row.get("license_verified"),
                    "internal_research_authorized": row.get("internal_research_authorized"),
                    "authorization_basis": row.get("authorization_basis"),
                    **{field: "" for field in QA_FIELDS},
                }
            )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(sampled[0])
        if sampled
        else [
            "window_id",
            "episode_id",
            "channel",
            "automatic_status",
            "audio_filepath",
            "duration",
            "n_speakers",
            "secondary_speaker_share",
            "alignment_coverage",
            "overlap_intervals",
            "license",
            "license_verified",
            "internal_research_authorized",
            "authorization_basis",
            *QA_FIELDS,
        ]
    )
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)
    return {
        "out_csv": str(out_csv),
        "eligible_windows": len(rows),
        "sampled_windows": len(sampled),
        "strata": {
            f"{channel}/{status}": min(per_stratum, len(candidates))
            for (channel, status), candidates in sorted(strata.items())
        },
        "instruction": (
            "Set review_status=pass or fail; pass also requires all four correctness "
            "fields=yes and a non-empty reviewer_id."
        ),
    }


def _yes(value: object) -> bool:
    return str(value or "").strip().lower() in YES


def apply_manual_qa(
    diarized_jsonl: Path,
    qa_csv: Path,
    out_jsonl: Path,
    *,
    report_path: Path | None = None,
) -> dict:
    """Apply completed decisions without mutating the automatic source manifest."""

    decisions: dict[str, dict] = {}
    duplicates = 0
    with Path(qa_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for parsed_decision in csv.DictReader(handle):
            window_id = str(parsed_decision.get("window_id") or "").strip()
            if not window_id:
                continue
            if window_id in decisions:
                duplicates += 1
            decisions[window_id] = parsed_decision
    if duplicates:
        raise ValueError(f"duplicate window_id values in QA CSV: {duplicates}")

    rows = _read_jsonl(Path(diarized_jsonl))
    counts: Counter[str] = Counter()
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as dst:
        for row in rows:
            window_id = str(row.get("window_id") or "")
            decision = decisions.get(window_id)
            if decision is None:
                counts["not_sampled"] += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            status = str(decision.get("review_status") or "").strip().lower()
            reviewer = str(decision.get("reviewer_id") or "").strip()
            checks = {
                field: _yes(decision.get(field))
                for field in (
                    "speaker_count_correct",
                    "speaker_assignment_correct",
                    "caption_acceptable",
                    "overlap_annotation_correct",
                )
            }
            complete = status in {"pass", "fail"} and bool(reviewer)
            approved = complete and status == "pass" and all(checks.values())
            row.update(
                {
                    "manual_qa_reviewed": complete,
                    "human_verified": approved,
                    "manual_qa": {
                        "review_status": status or "pending",
                        "reviewer_id": reviewer,
                        "checks": checks,
                        "notes": str(decision.get("notes") or ""),
                        "automatic_status": (
                            "automatic_pass"
                            if row.get("automatic_multi_speaker_verified") is True
                            else "automatic_reject"
                        ),
                    },
                }
            )
            if not complete:
                counts["incomplete"] += 1
            elif approved:
                counts["approved"] += 1
            else:
                counts["rejected"] += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts["source_windows"] = len(rows)
    counts["qa_decisions"] = len(decisions)
    counts["unknown_window_ids"] = len(
        set(decisions) - {str(row.get("window_id") or "") for row in rows}
    )
    report = {
        "source_manifest": str(diarized_jsonl),
        "qa_csv": str(qa_csv),
        "out_manifest": str(out_jsonl),
        "counts": dict(counts),
        "fail_closed": True,
        "note": "Manual QA verifies labels only; licensing approval remains a separate gate.",
    }
    if report_path is not None:
        write_json(report_path, report)
    return report
