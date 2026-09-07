#!/usr/bin/env python3
"""Fail CI when tracked artifacts are malformed, private, secret, or oversized."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the minimum-version CI runner
    import tomli as tomllib

import yaml

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


def tracked_files() -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
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
    ).stdout.splitlines()
    if not commits:
        return errors
    scan = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-I", "-n", "-E", HISTORY_SECRET_ERE, *commits],
        check=False,
        capture_output=True,
        text=True,
    )
    if scan.returncode not in {0, 1}:
        detail = scan.stderr.strip()[:200]
        errors.append(f"Git history secret scan failed: {detail}")
    elif scan.returncode == 0:
        for finding in scan.stdout.splitlines():
            location = ":".join(finding.split(":", 3)[:3])
            errors.append(f"possible secret in Git history: {location}")
    return errors


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
    if errors:
        print("Tracked-artifact audit failed:", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Tracked-artifact audit passed ({len(tracked_files())} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
