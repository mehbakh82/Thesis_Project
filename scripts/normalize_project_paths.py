#!/usr/bin/env python3
"""Normalize project-local absolute paths in tracked JSON evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from thesis_s2s.config import portable_project_values
from thesis_s2s.metrics import write_json

ROOT = Path(__file__).resolve().parents[1]


def tracked_json_files() -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "*.json"],
        check=True,
        capture_output=True,
    ).stdout
    return [ROOT / item.decode("utf-8") for item in output.split(b"\0") if item]


def normalize(*, write: bool) -> list[str]:
    changed: list[str] = []
    for path in tracked_json_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        normalized = portable_project_values(payload, root=ROOT)
        if normalized == payload:
            continue
        changed.append(path.relative_to(ROOT).as_posix())
        if write:
            if not isinstance(normalized, dict):
                raise TypeError(f"expected a top-level JSON object: {path}")
            write_json(path, normalized)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite affected tracked JSON files; default is a read-only check",
    )
    args = parser.parse_args()
    changed = normalize(write=args.write)
    if changed:
        action = "normalized" if args.write else "require normalization"
        for path in changed:
            print(f"{action}: {path}")
        return 0 if args.write else 1
    print("All tracked JSON paths are portable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
