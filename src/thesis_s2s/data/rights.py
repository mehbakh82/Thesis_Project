"""Fail-closed rights-review handoff for internal conversation sources."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from thesis_s2s.metrics import write_json

DECISION_FIELDS = (
    "approval_status",
    "internal_training_allowed",
    "thesis_reporting_allowed",
    "derived_artifacts_allowed",
    "redistribution_allowed",
    "evidence_reference",
    "approved_by",
    "approval_date",
    "notes",
)
YES = {"1", "true", "yes", "y", "بله"}


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _yes(value: object) -> bool:
    return str(value or "").strip().lower() in YES


def _valid_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def rights_record_verified(row: dict) -> bool:
    """Require auditable evidence for this workflow's internal approval label."""

    if row.get("license_verified") is not True:
        return False
    if row.get("license") != "approved-internal-research":
        return True
    review = row.get("rights_review") or {}
    permissions = review.get("permissions") or {}
    return bool(
        review.get("decision_complete") is True
        and review.get("status") == "approved"
        and permissions.get("internal_training") is True
        and permissions.get("thesis_reporting") is True
        and permissions.get("derived_artifacts") is True
        and str(review.get("evidence_reference") or "").strip()
        and str(review.get("approved_by") or "").strip()
        and _valid_iso_date(str(review.get("approval_date") or ""))
    )


def create_conversation_rights_review(
    in_jsonl: Path,
    out_csv: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Create one blank decision row per source channel.

    Approval evidence must cover the complete channel/source scope represented
    by that row. The command never infers permission from data possession.
    """

    rows = _read_jsonl(Path(in_jsonl))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("channel") or "unknown")].append(row)

    out_csv = Path(out_csv)
    if out_csv.exists() and not overwrite:
        raise FileExistsError(
            f"rights review already exists: {out_csv}; use --overwrite only for an intentional reset"
        )
    review_rows: list[dict] = []
    for channel, source_rows in sorted(grouped.items()):
        licenses = sorted({str(row.get("license") or "unknown") for row in source_rows})
        review_rows.append(
            {
                "scope_type": "channel",
                "scope_id": channel,
                "episodes": len(
                    {str(row.get("episode_id")) for row in source_rows if row.get("episode_id")}
                ),
                "window_hours": round(
                    sum(float(row.get("duration") or 0.0) for row in source_rows) / 3600.0,
                    3,
                ),
                "current_licenses": ";".join(licenses),
                **{field: "" for field in DECISION_FIELDS},
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(review_rows[0])
        if review_rows
        else [
            "scope_type",
            "scope_id",
            "episodes",
            "window_hours",
            "current_licenses",
            *DECISION_FIELDS,
        ]
    )
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)
    return {
        "source_manifest": str(in_jsonl),
        "out_csv": str(out_csv),
        "source_windows": len(rows),
        "review_scopes": len(review_rows),
        "scope": "channel/source",
        "instruction": (
            "Approval requires status=approved, yes for internal training, thesis "
            "reporting, and derived artifacts, plus evidence_reference, approved_by, "
            "and an ISO approval_date. Evidence must cover the full scope_id."
        ),
    }


def _load_decisions(path: Path) -> dict[str, dict]:
    decisions: dict[str, dict] = {}
    duplicates: list[str] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scope_type = str(row.get("scope_type") or "").strip()
            scope_id = str(row.get("scope_id") or "").strip()
            if scope_type != "channel" or not scope_id:
                continue
            if scope_id in decisions:
                duplicates.append(scope_id)
            decisions[scope_id] = row
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"duplicate channel rights decisions: {sample}")
    return decisions


def apply_conversation_rights_review(
    in_jsonl: Path,
    rights_csv: Path,
    out_jsonl: Path,
    *,
    report_path: Path | None = None,
) -> dict:
    """Apply explicit source decisions to a new manifest, never in place."""

    in_jsonl = Path(in_jsonl)
    out_jsonl = Path(out_jsonl)
    if in_jsonl.resolve() == out_jsonl.resolve():
        raise ValueError("rights application output must differ from its source manifest")
    rows = _read_jsonl(in_jsonl)
    decisions = _load_decisions(Path(rights_csv))
    source_channels = {str(row.get("channel") or "unknown") for row in rows}
    unknown_scopes = sorted(set(decisions) - source_channels)
    if unknown_scopes:
        raise ValueError(f"rights CSV contains unknown channel scopes: {unknown_scopes[:5]}")

    counts: Counter[str] = Counter()
    output: list[dict] = []
    approved_channels: set[str] = set()
    for row in rows:
        channel = str(row.get("channel") or "unknown")
        decision = decisions.get(channel)
        if decision is None:
            complete = False
            approved = False
            status = "pending"
            permissions = {
                "internal_training": False,
                "thesis_reporting": False,
                "derived_artifacts": False,
                "redistribution": False,
            }
            evidence = approver = approval_date = notes = ""
        else:
            status = str(decision.get("approval_status") or "").strip().lower()
            permissions = {
                "internal_training": _yes(decision.get("internal_training_allowed")),
                "thesis_reporting": _yes(decision.get("thesis_reporting_allowed")),
                "derived_artifacts": _yes(decision.get("derived_artifacts_allowed")),
                "redistribution": _yes(decision.get("redistribution_allowed")),
            }
            evidence = str(decision.get("evidence_reference") or "").strip()
            approver = str(decision.get("approved_by") or "").strip()
            approval_date = str(decision.get("approval_date") or "").strip()
            notes = str(decision.get("notes") or "")
            complete = (
                status in {"approved", "rejected"}
                and bool(evidence)
                and bool(approver)
                and _valid_iso_date(approval_date)
            )
            approved = (
                complete
                and status == "approved"
                and permissions["internal_training"]
                and permissions["thesis_reporting"]
                and permissions["derived_artifacts"]
            )
        row.update(
            {
                "license": (
                    "approved-internal-research" if approved else "pending-youtube-rights-review"
                ),
                "license_verified": approved,
                "redistribution_allowed": approved and permissions["redistribution"],
                "rights_review": {
                    "scope_type": "channel",
                    "scope_id": channel,
                    "status": status,
                    "decision_complete": complete,
                    "permissions": permissions,
                    "evidence_reference": evidence,
                    "approved_by": approver,
                    "approval_date": approval_date,
                    "notes": notes,
                },
            }
        )
        if approved:
            counts["approved_windows"] += 1
            approved_channels.add(channel)
        elif complete and status == "rejected":
            counts["rejected_windows"] += 1
        else:
            counts["pending_windows"] += 1
        output.append(row)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_jsonl.with_name(f".{out_jsonl.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(out_jsonl)
    approved_hours = sum(
        float(row.get("duration") or 0.0) / 3600.0
        for row in output
        if row.get("license_verified") is True
    )
    report = {
        "source_manifest": str(in_jsonl),
        "rights_csv": str(rights_csv),
        "out_manifest": str(out_jsonl),
        "source_windows": len(rows),
        "source_channels": sorted(source_channels),
        "decision_scopes": len(decisions),
        "approved_channels": sorted(approved_channels),
        "approved_hours": round(approved_hours, 3),
        "counts": dict(counts),
        "rights_gate_passes": bool(output) and all(rights_record_verified(row) for row in output),
        "redistribution_is_separate": True,
    }
    if report_path is not None:
        write_json(report_path, report)
    return report
