"""Fail-closed audit of requirements-constrained model selection evidence.

The audit deliberately separates official documentation from measurements made
inside this project.  A model cannot become an empirical winner merely because
its model card claims a capability, and an unknown field never passes a gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thesis_s2s.config import load_yaml, portable_path, project_root
from thesis_s2s.metrics import write_json
from thesis_s2s.repro import sha256_file

VALID_STATUSES = {"verified", "documented", "failed", "unknown"}
EVIDENCE_RANK = {"failed": -1, "unknown": 0, "documented": 1, "verified": 2}


def _validate_source(source: str, root: Path) -> dict[str, Any]:
    if source.startswith("local:"):
        path = root / source.removeprefix("local:")
        return {
            "kind": "local",
            "path": portable_path(path, root=root),
            "present": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    if source.startswith(("https://", "http://")):
        return {"kind": "remote", "url": source, "declared": True}
    return {"kind": "invalid", "value": source, "declared": False}


def _criterion_result(
    criterion: str,
    requirement: dict[str, Any],
    evidence: dict[str, Any] | None,
    root: Path,
) -> dict[str, Any]:
    item = evidence or {"status": "unknown", "source": ""}
    status = str(item.get("status") or "unknown")
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid evidence status for {criterion}: {status}")
    minimum = str(requirement.get("minimum") or "verified")
    if minimum not in {"documented", "verified"}:
        raise ValueError(f"invalid minimum evidence for {criterion}: {minimum}")
    source = str(item.get("source") or "")
    source_receipt = _validate_source(source, root) if source else {
        "kind": "missing",
        "declared": False,
    }
    source_ok = bool(
        source_receipt.get("present")
        if source_receipt.get("kind") == "local"
        else source_receipt.get("declared")
    )
    meets = bool(
        source_ok
        and status != "failed"
        and EVIDENCE_RANK[status] >= EVIDENCE_RANK[minimum]
    )
    return {
        "criterion": criterion,
        "minimum": minimum,
        "status": status,
        "meets": meets,
        "source": source_receipt,
        "notes": str(item.get("notes") or ""),
    }


def audit_model_selection(
    catalog_path: str | Path,
    out_path: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit a declarative candidate catalog without running any models."""

    project = Path(root or project_root()).resolve()
    catalog_file = Path(catalog_path)
    if not catalog_file.is_absolute():
        catalog_file = project / catalog_file
    payload = load_yaml(catalog_file)
    tracks = payload.get("tracks")
    if not isinstance(tracks, dict) or not tracks:
        raise ValueError("catalog must define at least one model-selection track")

    track_reports: dict[str, Any] = {}
    strict_blockers: list[str] = []
    for track_name, track in tracks.items():
        if not isinstance(track, dict):
            raise ValueError(f"track must be a mapping: {track_name}")
        requirements = track.get("requirements")
        candidates = track.get("candidates")
        if not isinstance(requirements, dict) or not requirements:
            raise ValueError(f"track has no requirements: {track_name}")
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError(f"track has no candidates: {track_name}")

        candidate_reports: dict[str, Any] = {}
        eligible: list[str] = []
        for candidate_name, candidate in candidates.items():
            if not isinstance(candidate, dict):
                raise ValueError(f"candidate must be a mapping: {candidate_name}")
            evidence = candidate.get("evidence")
            if not isinstance(evidence, dict):
                evidence = {}
            criteria = {
                criterion: _criterion_result(
                    criterion,
                    requirement if isinstance(requirement, dict) else {},
                    evidence.get(criterion) if isinstance(evidence.get(criterion), dict) else None,
                    project,
                )
                for criterion, requirement in requirements.items()
            }
            missing = [name for name, result in criteria.items() if not result["meets"]]
            is_eligible = not missing
            if is_eligible:
                eligible.append(str(candidate_name))
            candidate_reports[str(candidate_name)] = {
                "eligible": is_eligible,
                "missing_or_failed_criteria": missing,
                "criteria": criteria,
                "notes": str(candidate.get("notes") or ""),
            }

        selected = track.get("current_selection")
        selected_name = str(selected) if selected else None
        selected_eligible = bool(selected_name and selected_name in eligible)
        controlled = bool(track.get("controlled_same_panel_comparison"))
        catalog_comparison_complete = bool(
            controlled and selected_eligible and len(eligible) >= 2
        )
        if not selected_eligible:
            strict_blockers.append(f"{track_name}:current_selection_not_fully_eligible")
        if not controlled:
            strict_blockers.append(f"{track_name}:no_controlled_same_panel_comparison")
        if len(eligible) < 2:
            strict_blockers.append(f"{track_name}:fewer_than_two_fully_eligible_candidates")
        track_reports[str(track_name)] = {
            "purpose": str(track.get("purpose") or ""),
            "current_selection": selected_name,
            "current_selection_fully_eligible": selected_eligible,
            "controlled_same_panel_comparison": controlled,
            "eligible_candidates": eligible,
            "catalog_comparison_complete": catalog_comparison_complete,
            "candidates": candidate_reports,
        }

    report = {
        "schema_version": 2,
        "evidence_class": "requirements_constrained_static_model_audit",
        "as_of": str(payload.get("as_of") or ""),
        "catalog": portable_path(catalog_file, root=project),
        "catalog_sha256": sha256_file(catalog_file),
        "claim_boundary": (
            "This audit validates declared evidence and fail-closed eligibility. It does not "
            "run models, establish unreported Persian quality, or prove global optimality."
        ),
        "tracks": track_reports,
        "model_selection_audit": {
            "passes": not strict_blockers,
            "blockers": sorted(set(strict_blockers)),
            "catalog_bounded_selection_claim_allowed": bool(
                track_reports
                and all(
                    track["catalog_comparison_complete"]
                    for track in track_reports.values()
                )
            ),
            "global_best_model_claim_allowed": False,
        },
    }
    write_json(out_path, report)
    return report
