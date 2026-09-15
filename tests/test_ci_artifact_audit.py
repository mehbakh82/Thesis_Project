from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import ci_artifact_audit
from scripts.audit_proposal_alignment import (
    EXPECTED_PROPOSAL_NAME,
    EXPECTED_PROPOSAL_SHA256,
)
from scripts.audit_proposal_alignment import (
    build_report as build_proposal_alignment_report,
)


def test_ci_runs_once_per_pull_request_and_on_release_refs() -> None:
    workflow_path = Path(__file__).parents[1] / ".github/workflows/ci.yml"
    workflow = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert workflow["on"] == {
        "push": {
            "branches": ["main"],
            "tags": ["submission-*"],
        },
        "pull_request": {"branches": ["main"]},
    }


def test_tracked_file_git_query_has_validated_timeout(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(ci_artifact_audit, "ROOT", tmp_path)
    monkeypatch.setenv("THESIS_GIT_TIMEOUT_SECONDS", "4.5")
    monkeypatch.setattr(
        ci_artifact_audit.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(stdout=b""),
    )
    assert ci_artifact_audit.tracked_files() == []
    assert calls[0][1]["timeout"] == 4.5
    monkeypatch.setenv("THESIS_GIT_TIMEOUT_SECONDS", "nan")
    with pytest.raises(ValueError, match="positive and finite"):
        ci_artifact_audit.tracked_files()


def test_generated_evidence_summary_is_bound_to_json(tmp_path: Path, monkeypatch) -> None:
    evidence = tmp_path / "EVIDENCE_STATUS.json"
    summary = tmp_path / "SUMMARY.md"
    payload = {"generated_at": "now", "gates": {}, "required_to_complete": []}
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    from thesis_s2s.eval.evidence import render_evidence_summary

    summary.write_text(render_evidence_summary(payload), encoding="utf-8")
    monkeypatch.setattr(ci_artifact_audit, "AGGREGATE_EVIDENCE", evidence)
    monkeypatch.setattr(ci_artifact_audit, "EVIDENCE_SUMMARY", summary)
    assert ci_artifact_audit.evidence_summary_audit() == []
    summary.write_text("stale", encoding="utf-8")
    assert ci_artifact_audit.evidence_summary_audit() == ["generated evidence summary drift"]


def test_aggregate_artifact_provenance_audit_recomputes_bindings(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "receipt.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    evidence = tmp_path / "EVIDENCE_STATUS.json"
    evidence.write_text(
        json.dumps(
            {
                "artifact_provenance": {
                    "receipt": {
                        "path": "receipt.json",
                        "present": True,
                        "bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ci_artifact_audit, "ROOT", tmp_path)
    monkeypatch.setattr(ci_artifact_audit, "AGGREGATE_EVIDENCE", evidence)
    monkeypatch.setattr(ci_artifact_audit, "ARTIFACTS", {"receipt": "receipt.json"})

    assert ci_artifact_audit.aggregate_artifact_provenance_audit() == []
    artifact.write_text("changed\n", encoding="utf-8")
    assert ci_artifact_audit.aggregate_artifact_provenance_audit() == [
        "aggregate artifact provenance mismatch: receipt -> receipt.json"
    ]


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_proposal_template_is_forbidden_in_current_tree_and_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    proposal = repository / EXPECTED_PROPOSAL_NAME
    proposal.write_bytes(b"private proposal fixture")
    _git(repository, "add", EXPECTED_PROPOSAL_NAME)
    _git(
        repository,
        "-c",
        "user.name=Privacy Test",
        "-c",
        "user.email=privacy@example.invalid",
        "commit",
        "-q",
        "-m",
        "private fixture",
    )
    monkeypatch.setattr(ci_artifact_audit, "ROOT", repository)
    monkeypatch.setattr(ci_artifact_audit, "tracked_files", lambda: [proposal])

    assert ci_artifact_audit.audit() == [
        f"forbidden tracked path: {EXPECTED_PROPOSAL_NAME}"
    ]
    assert ci_artifact_audit.history_audit() == [
        f"forbidden path exists in Git history: {EXPECTED_PROPOSAL_NAME}"
    ]


def _tagged_repository(root: Path) -> tuple[str, str]:
    root.mkdir()
    _git(root, "init", "-q")
    (root / "tracked.txt").write_text("release fixture\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(
        root,
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release@example.invalid",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    commit = _git(root, "rev-parse", "HEAD")
    _git(
        root,
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release@example.invalid",
        "tag",
        "-a",
        "submission-test",
        "-m",
        "fixture tag",
    )
    return commit, _git(root, "rev-parse", "submission-test")


def _write_attestation(path: Path, *, commit: str, tag_object: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "tag": "submission-test",
                "commit": commit,
                "tag_object": tag_object,
            }
        ),
        encoding="utf-8",
    )


def test_release_attestation_audit_resolves_annotated_tag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    commit, tag_object = _tagged_repository(repository)
    receipt = repository / "results/release/submission_tag_attestation.json"
    _write_attestation(receipt, commit=commit, tag_object=tag_object)
    monkeypatch.setattr(ci_artifact_audit, "ROOT", repository)
    monkeypatch.setattr(ci_artifact_audit, "RELEASE_ATTESTATION", receipt)

    assert ci_artifact_audit.release_attestation_audit() == []


def test_release_attestation_audit_rejects_object_hash_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    commit, _ = _tagged_repository(repository)
    receipt = repository / "results/release/submission_tag_attestation.json"
    _write_attestation(receipt, commit=commit, tag_object="0" * 40)
    monkeypatch.setattr(ci_artifact_audit, "ROOT", repository)
    monkeypatch.setattr(ci_artifact_audit, "RELEASE_ATTESTATION", receipt)

    assert "release tag-object hash mismatch" in (ci_artifact_audit.release_attestation_audit())


def test_release_attestation_audit_rejects_malformed_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    receipt = tmp_path / "submission_tag_attestation.json"
    receipt.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(ci_artifact_audit, "RELEASE_ATTESTATION", receipt)

    errors = ci_artifact_audit.release_attestation_audit()

    assert len(errors) == 1
    assert errors[0].startswith("invalid release tag attestation:")


def _write_final_audit_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    bound = root / "bound.txt"
    bound.write_text("bound release bytes\n", encoding="utf-8")
    digest = hashlib.sha256(bound.read_bytes()).hexdigest()
    attestation = root / "attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "tag": "submission-test",
                "tag_object": "a" * 40,
                "commit": "b" * 40,
                "branch_ci": {"run_id": 10},
                "tag_ci": {"run_id": 11},
            }
        ),
        encoding="utf-8",
    )
    snapshot = root / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "git": {"commit": "c" * 40, "dirty": False},
                "file_count": 1,
            }
        ),
        encoding="utf-8",
    )
    audit = root / "final_audit.json"
    audit.write_text(
        json.dumps(
            {
                "audited_commit": "b" * 40,
                "fixture": {"sha256": digest},
                "release_verification": {
                    "tag": "submission-test",
                    "tag_object": "a" * 40,
                    "branch_ci_run": 10,
                    "tag_ci_run": 11,
                    "snapshot_file_count": 1,
                    "snapshot_source_commit": "c" * 40,
                },
            }
        ),
        encoding="utf-8",
    )
    return audit, snapshot, attestation, bound


def test_final_audit_receipt_binds_files_snapshot_and_tag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audit, snapshot, attestation, _ = _write_final_audit_fixture(tmp_path)
    monkeypatch.setattr(ci_artifact_audit, "ROOT", tmp_path)
    monkeypatch.setattr(ci_artifact_audit, "FINAL_AUDIT", audit)
    monkeypatch.setattr(ci_artifact_audit, "FINAL_SNAPSHOT", snapshot)
    monkeypatch.setattr(ci_artifact_audit, "RELEASE_ATTESTATION", attestation)
    monkeypatch.setattr(
        ci_artifact_audit,
        "FINAL_AUDIT_HASH_BINDINGS",
        {("fixture", "sha256"): "bound.txt"},
    )

    assert ci_artifact_audit.final_audit_receipt_audit() == []


def test_final_audit_receipt_rejects_hash_and_release_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audit, snapshot, attestation, bound = _write_final_audit_fixture(tmp_path)
    bound.write_text("changed\n", encoding="utf-8")
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["release_verification"]["tag_ci_run"] = 99
    audit.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(ci_artifact_audit, "ROOT", tmp_path)
    monkeypatch.setattr(ci_artifact_audit, "FINAL_AUDIT", audit)
    monkeypatch.setattr(ci_artifact_audit, "FINAL_SNAPSHOT", snapshot)
    monkeypatch.setattr(ci_artifact_audit, "RELEASE_ATTESTATION", attestation)
    monkeypatch.setattr(
        ci_artifact_audit,
        "FINAL_AUDIT_HASH_BINDINGS",
        {("fixture", "sha256"): "bound.txt"},
    )

    errors = ci_artifact_audit.final_audit_receipt_audit()

    assert "final audit hash mismatch: fixture.sha256 -> bound.txt" in errors
    assert "final audit release binding mismatch: tag_ci_run" in errors


def _write_proposal_alignment_fixture(root: Path) -> tuple[Path, Path]:
    evidence = root / "results/eval/EVIDENCE_STATUS.json"
    evidence.parent.mkdir(parents=True)
    evidence_payload = {"schema_version": 12}
    evidence.write_text(json.dumps(evidence_payload), encoding="utf-8")
    evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    receipt = root / "results/proposal_alignment_audit.json"
    receipt.write_text(
        json.dumps(
            build_proposal_alignment_report(
                Path(EXPECTED_PROPOSAL_NAME),
                evidence_payload,
                evidence_sha256=evidence_sha256,
                verified_proposal_sha256=EXPECTED_PROPOSAL_SHA256,
            )
        ),
        encoding="utf-8",
    )
    return receipt, evidence


def test_proposal_alignment_receipt_audit_accepts_exact_evidence_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    receipt, evidence = _write_proposal_alignment_fixture(tmp_path)
    monkeypatch.setattr(ci_artifact_audit, "PROPOSAL_ALIGNMENT_RECEIPT", receipt)
    monkeypatch.setattr(ci_artifact_audit, "AGGREGATE_EVIDENCE", evidence)

    assert ci_artifact_audit.proposal_alignment_receipt_audit() == []


def test_proposal_alignment_receipt_audit_rejects_evidence_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    receipt, evidence = _write_proposal_alignment_fixture(tmp_path)
    evidence.write_text(json.dumps({"schema_version": 13}), encoding="utf-8")
    monkeypatch.setattr(ci_artifact_audit, "PROPOSAL_ALIGNMENT_RECEIPT", receipt)
    monkeypatch.setattr(ci_artifact_audit, "AGGREGATE_EVIDENCE", evidence)

    errors = ci_artifact_audit.proposal_alignment_receipt_audit()

    assert "proposal alignment receipt evidence hash mismatch" in errors
    assert "proposal alignment receipt evidence schema mismatch" in errors


def test_proposal_alignment_receipt_audit_rejects_malformed_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    receipt, evidence = _write_proposal_alignment_fixture(tmp_path)
    receipt.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(ci_artifact_audit, "PROPOSAL_ALIGNMENT_RECEIPT", receipt)
    monkeypatch.setattr(ci_artifact_audit, "AGGREGATE_EVIDENCE", evidence)

    errors = ci_artifact_audit.proposal_alignment_receipt_audit()

    assert len(errors) == 1
    assert errors[0].startswith("invalid proposal alignment receipt input:")


def test_proposal_alignment_receipt_audit_rejects_tampered_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    receipt, evidence = _write_proposal_alignment_fixture(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["scoring_contract"]["strict_acceptance_percent"] = 100.0
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(ci_artifact_audit, "PROPOSAL_ALIGNMENT_RECEIPT", receipt)
    monkeypatch.setattr(ci_artifact_audit, "AGGREGATE_EVIDENCE", evidence)

    assert ci_artifact_audit.proposal_alignment_receipt_audit() == [
        "proposal alignment receipt content mismatch"
    ]


def test_proposal_alignment_receipt_audit_rejects_nonobject_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    receipt, evidence = _write_proposal_alignment_fixture(tmp_path)
    receipt.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(ci_artifact_audit, "PROPOSAL_ALIGNMENT_RECEIPT", receipt)
    monkeypatch.setattr(ci_artifact_audit, "AGGREGATE_EVIDENCE", evidence)

    assert ci_artifact_audit.proposal_alignment_receipt_audit() == [
        "invalid proposal alignment receipt input: root values must be objects"
    ]
