#!/usr/bin/env python3
"""Verify portable evaluator files and their exact frozen historical sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def git_timeout_seconds() -> float:
    raw = os.environ.get("THESIS_GIT_TIMEOUT_SECONDS", "120")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("THESIS_GIT_TIMEOUT_SECONDS must be positive and finite") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("THESIS_GIT_TIMEOUT_SECONDS must be positive and finite")
    return timeout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError(f"unsafe repository-relative path: {value!r}")
    return Path(*posix.parts)


def _nested(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing receipt field: {dotted}")
        value = value[part]
    return value


def verify_manifest(
    manifest: dict[str, Any],
    root: Path,
    load_historical: Callable[[str, str], bytes],
) -> dict[str, Any]:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported frozen-source manifest schema")
    commit = str(manifest.get("historical_commit") or "")
    if not COMMIT_RE.fullmatch(commit):
        raise ValueError("historical_commit must be a full SHA-1")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("frozen-source entries must be a non-empty list")

    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("each frozen-source entry must be an object")
        relative_text = str(raw.get("path") or "")
        relative = _safe_relative(relative_text)
        if relative_text in seen:
            raise ValueError(f"duplicate frozen-source path: {relative_text}")
        seen.add(relative_text)
        current = (root / relative).read_bytes()
        historical = load_historical(commit, relative.as_posix())
        current_sha = _sha256(current)
        historical_sha = _sha256(historical)
        if current_sha != raw.get("current_sha256"):
            raise ValueError(f"current source drift: {relative_text}")
        if historical_sha != raw.get("historical_sha256"):
            raise ValueError(f"historical source drift: {relative_text}")

        bindings = raw.get("bindings")
        if not isinstance(bindings, list) or not bindings:
            raise ValueError(f"missing receipt bindings: {relative_text}")
        checked_bindings = 0
        for binding in bindings:
            if not isinstance(binding, list) or len(binding) != 2:
                raise ValueError(f"invalid binding for {relative_text}")
            receipt_relative = _safe_relative(str(binding[0]))
            receipt = json.loads((root / receipt_relative).read_text(encoding="utf-8"))
            if _nested(receipt, str(binding[1])) != historical_sha:
                raise ValueError(
                    f"receipt does not bind historical source: {receipt_relative}:{binding[1]}"
                )
            checked_bindings += 1
        verified.append(
            {
                "path": relative.as_posix(),
                "current_sha256": current_sha,
                "historical_sha256": historical_sha,
                "receipt_bindings_verified": checked_bindings,
            }
        )

    return {
        "schema_version": 1,
        "status": "passed",
        "historical_commit": commit,
        "entries_verified": len(verified),
        "receipt_bindings_verified": sum(
            int(item["receipt_bindings_verified"]) for item in verified
        ),
        "entries": verified,
    }


def _git_loader(root: Path) -> Callable[[str, str], bytes]:
    def load(commit: str, path: str) -> bytes:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=git_timeout_seconds(),
        ).stdout

    return load


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "configs/frozen_evaluator_sources.json"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = verify_manifest(manifest, ROOT, _git_loader(ROOT))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
