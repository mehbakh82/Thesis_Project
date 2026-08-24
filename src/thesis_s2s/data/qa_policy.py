"""Explicit, non-fabricating policy for proceeding without manual conversation QA."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

from thesis_s2s.config import load_yaml

WAIVER_POLICY = "automatic_only_documented_waiver"


def _valid_date(value: object) -> bool:
    try:
        date.fromisoformat(str(value or ""))
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_qa_waiver(path: Path | None) -> dict | None:
    """Validate a deliberate QA waiver; absence always means strict policy."""

    if path is None:
        return None
    waiver_path = Path(path)
    if not waiver_path.is_file():
        raise FileNotFoundError(f"QA waiver does not exist: {waiver_path}")
    payload = load_yaml(waiver_path)
    scope = payload.get("scope") or {}
    claims = payload.get("claims") or {}
    preservation = payload.get("preservation") or {}
    requirements = {
        "schema_supported": payload.get("schema_version") == 1,
        "status_acknowledged": payload.get("status") == "acknowledged",
        "policy_explicit": payload.get("policy") == WAIVER_POLICY,
        "student_authority_explicit": payload.get("decision_authority") == "student_project_owner",
        "supervisor_approval_not_claimed": payload.get("supervisor_approval_claimed") is False,
        "reason_recorded": bool(str(payload.get("reason") or "").strip()),
        "decision_date_recorded": _valid_date(payload.get("decision_date")),
        "limited_internal_training_explicit": scope.get("internal_training")
        == "allowed_under_waiver",
        "both_reviews_waived": scope.get("window_manual_qa") == "waived"
        and scope.get("interaction_manual_qa") == "waived",
        "human_claims_disabled": claims.get("human_verified_data") is False
        and claims.get("human_verified_interruptions") is False
        and claims.get("strict_thesis_data_coverage") is False,
        "best_practice_assets_preserved": all(
            preservation.get(key) is True
            for key in (
                "qa_sheets",
                "reviewer_guides",
                "playback_helper",
                "future_review_supported",
            )
        ),
    }
    if not all(requirements.values()):
        failed = [name for name, passed in requirements.items() if not passed]
        raise ValueError(f"invalid QA waiver; failed requirements: {', '.join(failed)}")
    return {
        "policy": WAIVER_POLICY,
        "path": str(waiver_path),
        "sha256": _sha256(waiver_path),
        "decision_date": payload.get("decision_date"),
        "decision_authority": payload.get("decision_authority"),
        "reason": payload.get("reason"),
        "supervisor_approval_claimed": False,
        "human_verified_data_claim_allowed": False,
        "human_verified_interruption_claim_allowed": False,
        "strict_thesis_data_coverage_claim_allowed": False,
        "requirements": requirements,
        "valid": True,
    }


def row_matches_waiver(row: dict, waiver: dict) -> bool:
    """Require pair metadata to carry the exact waiver used by the builder."""

    return (
        row.get("qa_policy") == waiver["policy"]
        and row.get("qa_waiver_sha256") == waiver["sha256"]
        and row.get("manual_qa_waived") is True
        and row.get("human_verified") is not True
        and row.get("human_verified_interaction_label") is not True
    )
