#!/usr/bin/env python3
"""Fail CI when tracked artifacts are malformed, private, secret, or oversized."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the minimum-version CI runner
    import tomli as tomllib

import yaml

from thesis_s2s.eval.evidence import render_evidence_summary

try:
    from scripts.audit_proposal_alignment import (
        EXPECTED_PROPOSAL_NAME,
        EXPECTED_PROPOSAL_SHA256,
    )
    from scripts.audit_proposal_alignment import (
        build_report as build_proposal_alignment_report,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from audit_proposal_alignment import (  # type: ignore[no-redef]
        EXPECTED_PROPOSAL_NAME,
        EXPECTED_PROPOSAL_SHA256,
    )
    from audit_proposal_alignment import (
        build_report as build_proposal_alignment_report,
    )

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 5 * 1024 * 1024
FORBIDDEN_PREFIXES = (
    ".venv/",
    ".venv-moshi/",
    "checkpoints/",
    "data/processed/",
    "data/raw/",
    "data/recordings/",
    "data/s3-cache/",
    "hf_cache/",
    "third_party/checkouts/",
)
FORBIDDEN_EXACT = {
    ".env",
    "codex_review_and_improve_thesis_project.md",
    "cursor_bsc_thesis_project_planning.md",
    "cursor_bsc_thesis_project_report.md",
    "تعریف پروژه.docx",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".nemo",
    ".onnx",
    ".pem",
    ".pkl",
    ".pt",
    ".safetensors",
}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".html",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(
        r"\b(?:gh[opusr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
}
PRIVATE_HOST_PATH = re.compile(r"/(?:mnt/md0|home/srv)(?:/|\b)")
HISTORY_SECRET_ERE = (
    r"(-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
    r"|gh[opusr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|(AKIA|ASIA)[A-Z0-9]{16}|hf_[A-Za-z0-9]{30,}"
    r"|xox[baprs]-[A-Za-z0-9-]{20,})"
)
RELEASE_ATTESTATION = ROOT / "results" / "release" / "submission_tag_attestation.json"
PROPOSAL_ALIGNMENT_RECEIPT = ROOT / "results" / "proposal_alignment_audit.json"
AGGREGATE_EVIDENCE = ROOT / "results" / "eval" / "EVIDENCE_STATUS.json"
EVIDENCE_SUMMARY = ROOT / "results" / "eval" / "SUMMARY.md"


def git_timeout_seconds() -> float:
    raw = os.environ.get("THESIS_GIT_TIMEOUT_SECONDS", "120")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("THESIS_GIT_TIMEOUT_SECONDS must be positive and finite") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("THESIS_GIT_TIMEOUT_SECONDS must be positive and finite")
    return timeout


def tracked_files() -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
        timeout=git_timeout_seconds(),
    ).stdout
    return [ROOT / item.decode("utf-8") for item in output.split(b"\0") if item]


def parse_structured(path: Path, text: str) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        json.loads(text)
    elif suffix == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_number}: {exc}") from exc
    elif suffix in {".yaml", ".yml"}:
        list(yaml.safe_load_all(text))
    elif path.name == "pyproject.toml":
        tomllib.loads(text)


def history_audit() -> list[str]:
    """Scan release refs without handing repository contents to a third party.

    Local tool-managed refs (for example ``refs/codex/turn-diffs``) can contain
    worktree snapshots of intentionally ignored files. They are never pushed,
    so the release boundary is branches, tags, and remote-tracking refs.
    """

    errors: list[str] = []
    objects = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "rev-list",
            "--objects",
            "--branches",
            "--tags",
            "--remotes",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=git_timeout_seconds(),
    ).stdout.splitlines()
    for row in objects:
        parts = row.split(" ", 1)
        if len(parts) != 2:
            continue
        relative = parts[1]
        path = Path(relative)
        if relative in FORBIDDEN_EXACT or relative.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"forbidden path exists in Git history: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/secret suffix exists in Git history: {relative}")

    commits = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "rev-list",
            "--branches",
            "--tags",
            "--remotes",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=git_timeout_seconds(),
    ).stdout.splitlines()
    if not commits:
        return errors
    scan = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-I", "-n", "-E", HISTORY_SECRET_ERE, *commits],
        check=False,
        capture_output=True,
        text=True,
        timeout=git_timeout_seconds(),
    )
    if scan.returncode not in {0, 1}:
        detail = scan.stderr.strip()[:200]
        errors.append(f"Git history secret scan failed: {detail}")
    elif scan.returncode == 0:
        for finding in scan.stdout.splitlines():
            location = ":".join(finding.split(":", 3)[:3])
            errors.append(f"possible secret in Git history: {location}")
    return errors


def release_attestation_audit() -> list[str]:
    """Verify the recorded immutable release tag against the local Git objects."""

    if not RELEASE_ATTESTATION.is_file():
        return ["missing release tag attestation"]
    try:
        payload = json.loads(RELEASE_ATTESTATION.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid release tag attestation: {exc}"]
    tag = payload.get("tag")
    commit = payload.get("commit")
    tag_object = payload.get("tag_object")
    if not all(isinstance(value, str) and value for value in (tag, commit, tag_object)):
        return ["release tag attestation is missing tag, tag_object, or commit"]

    errors: list[str] = []
    checks = (
        (["cat-file", "-t", tag], "tag", "release ref is not an annotated tag"),
        (["rev-parse", tag], tag_object, "release tag-object hash mismatch"),
        (["rev-parse", f"{tag}^{{}}"], commit, "release tag target mismatch"),
    )
    for arguments, expected, message in checks:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=git_timeout_seconds(),
        )
        if result.returncode != 0 or result.stdout.strip() != expected:
            errors.append(message)
    ancestry = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=git_timeout_seconds(),
    )
    if ancestry.returncode != 0:
        errors.append("attested release commit is not an ancestor of HEAD")
    return errors


def proposal_alignment_receipt_audit() -> list[str]:
    """Bind the public proposal summary to the exact aggregate evidence bytes."""

    if not PROPOSAL_ALIGNMENT_RECEIPT.is_file():
        return ["missing proposal alignment receipt"]
    if not AGGREGATE_EVIDENCE.is_file():
        return ["missing aggregate evidence for proposal alignment receipt"]
    try:
        receipt = json.loads(PROPOSAL_ALIGNMENT_RECEIPT.read_text(encoding="utf-8"))
        evidence = json.loads(AGGREGATE_EVIDENCE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid proposal alignment receipt input: {exc}"]
    if not isinstance(receipt, dict) or not isinstance(evidence, dict):
        return ["invalid proposal alignment receipt input: root values must be objects"]

    binding = receipt.get("aggregate_evidence")
    proposal = receipt.get("proposal")
    if not isinstance(binding, dict) or not isinstance(proposal, dict):
        return ["invalid proposal alignment receipt structure"]
    errors: list[str] = []
    if receipt.get("schema_version") != 2:
        errors.append("proposal alignment receipt schema is not 2")
    if binding.get("path") != "results/eval/EVIDENCE_STATUS.json":
        errors.append("proposal alignment receipt evidence path mismatch")
    observed_hash = hashlib.sha256(AGGREGATE_EVIDENCE.read_bytes()).hexdigest()
    if binding.get("sha256") != observed_hash:
        errors.append("proposal alignment receipt evidence hash mismatch")
    if binding.get("schema_version") != evidence.get("schema_version"):
        errors.append("proposal alignment receipt evidence schema mismatch")
    if proposal.get("path") != EXPECTED_PROPOSAL_NAME:
        errors.append("proposal alignment receipt proposal path mismatch")
    if proposal.get("sha256") != EXPECTED_PROPOSAL_SHA256:
        errors.append("proposal alignment receipt proposal hash mismatch")
    if not errors:
        expected = build_proposal_alignment_report(
            Path(EXPECTED_PROPOSAL_NAME),
            evidence,
            evidence_sha256=observed_hash,
            evidence_path="results/eval/EVIDENCE_STATUS.json",
            verified_proposal_sha256=EXPECTED_PROPOSAL_SHA256,
        )
        if receipt != expected:
            errors.append("proposal alignment receipt content mismatch")
    return errors


def evidence_summary_audit() -> list[str]:
    """Require the human-readable evidence view to match authoritative JSON exactly."""

    if not AGGREGATE_EVIDENCE.is_file() or not EVIDENCE_SUMMARY.is_file():
        return ["missing aggregate evidence or generated evidence summary"]
    try:
        payload = json.loads(AGGREGATE_EVIDENCE.read_text(encoding="utf-8"))
        observed = EVIDENCE_SUMMARY.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid generated evidence summary input: {exc}"]
    if not isinstance(payload, dict):
        return ["invalid generated evidence summary input: JSON root must be an object"]
    return [] if observed == render_evidence_summary(payload) else ["generated evidence summary drift"]


def audit() -> list[str]:
    errors: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative in FORBIDDEN_EXACT or relative.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"forbidden tracked path: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden tracked binary/secret suffix: {relative}")
        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds 5 MiB: {relative} ({size} bytes)")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            parse_structured(path, text)
        except (
            json.JSONDecodeError,
            tomllib.TOMLDecodeError,
            yaml.YAMLError,
            ValueError,
        ) as exc:
            errors.append(f"malformed structured file: {relative}: {exc}")
        if PRIVATE_HOST_PATH.search(text):
            errors.append(f"private absolute host path: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label}: {relative}")
    return errors


def main() -> int:
    errors = audit()
    errors.extend(history_audit())
    errors.extend(release_attestation_audit())
    errors.extend(proposal_alignment_receipt_audit())
    errors.extend(evidence_summary_audit())
    if errors:
        print("Tracked-artifact audit failed:", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Tracked-artifact audit passed ({len(tracked_files())} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
