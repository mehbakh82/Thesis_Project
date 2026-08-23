"""Fail-closed rights-review handoff for internal conversation sources."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from thesis_s2s.metrics import write_json

UNVERIFIED_LICENSES = {
    "",
    "unknown",
    "youtube-internal",
    "pending-youtube-rights-review",
    "approved-internal-research",
}

DECISION_FIELDS = (
    "approval_status",
    "authorization_basis",
    "license_name",
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
EXPLICIT_LICENSE_BASIS = "explicit-source-license"
SUPERVISOR_RESEARCH_BASIS = "supervisor-approved-internal-research"
PENDING_AUTHORIZATION_BASIS = "pending"
AUTHORIZATION_BASES = {EXPLICIT_LICENSE_BASIS, SUPERVISOR_RESEARCH_BASIS}


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


def _documented_training_review(row: dict) -> tuple[bool, str]:
    """Validate a documented decision without claiming it is a copyright license."""

    review = row.get("rights_review")
    if not isinstance(review, dict):
        return False, ""
    permissions = review.get("permissions")
    if not isinstance(permissions, dict):
        return False, ""
    basis = str(review.get("authorization_basis") or "").strip()
    valid = bool(
        basis in AUTHORIZATION_BASES
        and row.get("authorization_basis") == basis
        and review.get("decision_complete") is True
        and review.get("status") == "approved"
        and permissions.get("internal_training") is True
        and permissions.get("thesis_reporting") is True
        and permissions.get("derived_artifacts") is True
        and str(review.get("evidence_reference") or "").strip()
        and str(review.get("approved_by") or "").strip()
        and _valid_iso_date(str(review.get("approval_date") or ""))
    )
    return valid, basis


def rights_record_verified(row: dict) -> bool:
    """Return whether an actual reusable source license is verified."""

    if row.get("license_verified") is not True:
        return False
    license_name = str(row.get("license") or "").strip()
    if license_name in UNVERIFIED_LICENSES:
        return False
    review = row.get("rights_review")
    if review is None:
        return True
    valid_review, basis = _documented_training_review(row)
    return valid_review and basis == EXPLICIT_LICENSE_BASIS


def training_use_authorized(row: dict) -> bool:
    """Require a verified license or documented supervisor research approval."""

    if rights_record_verified(row):
        return True
    if row.get("internal_research_authorized") is not True:
        return False
    valid_review, basis = _documented_training_review(row)
    if not valid_review or basis != SUPERVISOR_RESEARCH_BASIS:
        return False
    return bool(row.get("license_verified") is False and row.get("redistribution_allowed") is False)


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
            "Approval requires authorization_basis, status=approved, yes for internal "
            "training, thesis reporting, and derived artifacts, plus evidence_reference, "
            "approved_by, and an ISO approval_date. An explicit-source-license basis "
            "also requires license_name. Evidence must cover the full scope_id."
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
    """Apply documented source-use decisions to a new manifest, never in place."""

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
    authorized_channels: set[str] = set()
    license_verified_channels: set[str] = set()
    for row in rows:
        channel = str(row.get("channel") or "unknown")
        decision = decisions.get(channel)
        if decision is None:
            complete = False
            training_approved = False
            license_approved = False
            status = "pending"
            basis = PENDING_AUTHORIZATION_BASIS
            requested_license = ""
            permissions = {
                "internal_training": False,
                "thesis_reporting": False,
                "derived_artifacts": False,
                "redistribution": False,
            }
            evidence = approver = approval_date = notes = ""
        else:
            status = str(decision.get("approval_status") or "").strip().lower()
            basis = str(decision.get("authorization_basis") or "").strip()
            requested_license = str(decision.get("license_name") or "").strip()
            permissions = {
                "internal_training": _yes(decision.get("internal_training_allowed")),
                "thesis_reporting": _yes(decision.get("thesis_reporting_allowed")),
                "derived_artifacts": _yes(decision.get("derived_artifacts_allowed")),
                "redistribution": _yes(decision.get("redistribution_allowed")),
            }
            if basis != EXPLICIT_LICENSE_BASIS:
                permissions["redistribution"] = False
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
            license_basis_valid = basis != EXPLICIT_LICENSE_BASIS or (
                bool(requested_license) and requested_license not in UNVERIFIED_LICENSES
            )
            training_approved = bool(
                complete
                and status == "approved"
                and basis in AUTHORIZATION_BASES
                and license_basis_valid
                and permissions["internal_training"]
                and permissions["thesis_reporting"]
                and permissions["derived_artifacts"]
            )
            license_approved = training_approved and basis == EXPLICIT_LICENSE_BASIS

        source_license = str(row.get("license") or "pending-youtube-rights-review")
        if source_license == "approved-internal-research":
            source_license = "pending-youtube-rights-review"
        row.update(
            {
                "license": requested_license if license_approved else source_license,
                "license_verified": license_approved,
                "internal_research_authorized": bool(
                    training_approved and basis == SUPERVISOR_RESEARCH_BASIS
                ),
                "authorization_basis": (
                    basis if training_approved else PENDING_AUTHORIZATION_BASIS
                ),
                "redistribution_allowed": bool(license_approved and permissions["redistribution"]),
                "rights_review": {
                    "scope_type": "channel",
                    "scope_id": channel,
                    "status": status,
                    "authorization_basis": basis,
                    "license_name": requested_license,
                    "decision_complete": complete,
                    "permissions": permissions,
                    "evidence_reference": evidence,
                    "approved_by": approver,
                    "approval_date": approval_date,
                    "notes": notes,
                },
            }
        )
        if training_approved:
            counts["authorized_windows"] += 1
            authorized_channels.add(channel)
            if license_approved:
                counts["license_verified_windows"] += 1
                license_verified_channels.add(channel)
            else:
                counts["supervisor_authorized_windows"] += 1
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
    authorized_hours = sum(
        float(row.get("duration") or 0.0) / 3600.0 for row in output if training_use_authorized(row)
    )
    license_verified_hours = sum(
        float(row.get("duration") or 0.0) / 3600.0 for row in output if rights_record_verified(row)
    )
    authorization_passes = bool(output) and all(training_use_authorized(row) for row in output)
    report = {
        "source_manifest": str(in_jsonl),
        "rights_csv": str(rights_csv),
        "out_manifest": str(out_jsonl),
        "source_windows": len(rows),
        "source_channels": sorted(source_channels),
        "decision_scopes": len(decisions),
        "authorized_channels": sorted(authorized_channels),
        "license_verified_channels": sorted(license_verified_channels),
        "authorized_hours": round(authorized_hours, 3),
        "license_verified_hours": round(license_verified_hours, 3),
        "counts": dict(counts),
        "training_authorization_gate_passes": authorization_passes,
        "rights_gate_passes": authorization_passes,
        "license_gate_passes": bool(output) and all(rights_record_verified(row) for row in output),
        "license_and_research_authorization_are_distinct": True,
        "redistribution_is_separate": True,
    }
    if report_path is not None:
        write_json(report_path, report)
    return report


def normalize_pending_rights_metadata(
    in_jsonl: Path,
    out_jsonl: Path | None = None,
) -> dict:
    """Atomically make pending license and training-authorization metadata explicit."""

    in_jsonl = Path(in_jsonl)
    out_jsonl = Path(out_jsonl or in_jsonl)
    rows = _read_jsonl(in_jsonl)
    changed_rows = 0
    normalized_legacy_rows = 0
    normalized_invalid_flags = 0
    normalized_authorization_flags = 0
    output: list[dict] = []
    for source_row in rows:
        row = dict(source_row)
        changed = False
        license_name = str(row.get("license") or "").strip()
        if license_name in {"", "unknown", "youtube-internal", "approved-internal-research"}:
            row["license"] = "pending-youtube-rights-review"
            row["license_verified"] = False
            row["redistribution_allowed"] = False
            normalized_legacy_rows += 1
            changed = True
        elif not isinstance(row.get("license_verified"), bool) or (
            license_name == "pending-youtube-rights-review"
            and (
                row.get("license_verified") is not False
                or row.get("redistribution_allowed") is not False
            )
        ):
            row["license_verified"] = False
            row["redistribution_allowed"] = False
            normalized_invalid_flags += 1
            changed = True

        if row.get("license") == "pending-youtube-rights-review" and (
            row.get("internal_research_authorized") is not False
            or row.get("authorization_basis") != PENDING_AUTHORIZATION_BASIS
        ):
            row["internal_research_authorized"] = False
            row["authorization_basis"] = PENDING_AUTHORIZATION_BASIS
            normalized_authorization_flags += 1
            changed = True
        if changed:
            changed_rows += 1
        output.append(row)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_jsonl.with_name(f".{out_jsonl.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(out_jsonl)
    return {
        "source_manifest": str(in_jsonl),
        "out_manifest": str(out_jsonl),
        "rows": len(output),
        "changed_rows": changed_rows,
        "normalized_legacy_rows": normalized_legacy_rows,
        "normalized_invalid_flags": normalized_invalid_flags,
        "normalized_authorization_flags": normalized_authorization_flags,
        "license_approval_inferred": False,
        "training_authorization_inferred": False,
    }
