from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import ci_artifact_audit


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
