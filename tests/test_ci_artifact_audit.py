from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts import ci_artifact_audit
from scripts.audit_proposal_alignment import (
    EXPECTED_PROPOSAL_NAME,
    EXPECTED_PROPOSAL_SHA256,
)
from scripts.audit_proposal_alignment import (
    build_report as build_proposal_alignment_report,
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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
    tmp_path: Path, monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    commit, tag_object = _tagged_repository(repository)
    receipt = repository / "results/release/submission_tag_attestation.json"
    _write_attestation(receipt, commit=commit, tag_object=tag_object)
    monkeypatch.setattr(ci_artifact_audit, "ROOT", repository)
    monkeypatch.setattr(ci_artifact_audit, "RELEASE_ATTESTATION", receipt)

    assert ci_artifact_audit.release_attestation_audit() == []


def test_release_attestation_audit_rejects_object_hash_mismatch(
    tmp_path: Path, monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    commit, _ = _tagged_repository(repository)
    receipt = repository / "results/release/submission_tag_attestation.json"
    _write_attestation(receipt, commit=commit, tag_object="0" * 40)
    monkeypatch.setattr(ci_artifact_audit, "ROOT", repository)
    monkeypatch.setattr(ci_artifact_audit, "RELEASE_ATTESTATION", receipt)

    assert "release tag-object hash mismatch" in (
        ci_artifact_audit.release_attestation_audit()
    )


def test_release_attestation_audit_rejects_malformed_json(
    tmp_path: Path, monkeypatch,
) -> None:
    receipt = tmp_path / "submission_tag_attestation.json"
    receipt.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(ci_artifact_audit, "RELEASE_ATTESTATION", receipt)

    errors = ci_artifact_audit.release_attestation_audit()

    assert len(errors) == 1
    assert errors[0].startswith("invalid release tag attestation:")


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
    tmp_path: Path, monkeypatch,
) -> None:
    receipt, evidence = _write_proposal_alignment_fixture(tmp_path)
    monkeypatch.setattr(ci_artifact_audit, "PROPOSAL_ALIGNMENT_RECEIPT", receipt)
    monkeypatch.setattr(ci_artifact_audit, "AGGREGATE_EVIDENCE", evidence)

    assert ci_artifact_audit.proposal_alignment_receipt_audit() == []


def test_proposal_alignment_receipt_audit_rejects_evidence_drift(
    tmp_path: Path, monkeypatch,
) -> None:
    receipt, evidence = _write_proposal_alignment_fixture(tmp_path)
    evidence.write_text(json.dumps({"schema_version": 13}), encoding="utf-8")
    monkeypatch.setattr(ci_artifact_audit, "PROPOSAL_ALIGNMENT_RECEIPT", receipt)
    monkeypatch.setattr(ci_artifact_audit, "AGGREGATE_EVIDENCE", evidence)

    errors = ci_artifact_audit.proposal_alignment_receipt_audit()

    assert "proposal alignment receipt evidence hash mismatch" in errors
    assert "proposal alignment receipt evidence schema mismatch" in errors


def test_proposal_alignment_receipt_audit_rejects_malformed_receipt(
    tmp_path: Path, monkeypatch,
) -> None:
    receipt, evidence = _write_proposal_alignment_fixture(tmp_path)
    receipt.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(ci_artifact_audit, "PROPOSAL_ALIGNMENT_RECEIPT", receipt)
    monkeypatch.setattr(ci_artifact_audit, "AGGREGATE_EVIDENCE", evidence)

    errors = ci_artifact_audit.proposal_alignment_receipt_audit()

    assert len(errors) == 1
    assert errors[0].startswith("invalid proposal alignment receipt input:")


def test_proposal_alignment_receipt_audit_rejects_tampered_score(
    tmp_path: Path, monkeypatch,
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
    tmp_path: Path, monkeypatch,
) -> None:
    receipt, evidence = _write_proposal_alignment_fixture(tmp_path)
    receipt.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(ci_artifact_audit, "PROPOSAL_ALIGNMENT_RECEIPT", receipt)
    monkeypatch.setattr(ci_artifact_audit, "AGGREGATE_EVIDENCE", evidence)

    assert ci_artifact_audit.proposal_alignment_receipt_audit() == [
        "invalid proposal alignment receipt input: root values must be objects"
    ]
