#!/usr/bin/env python3
"""Download and verify the exact Moshika blobs recorded in the upstream lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "third_party" / "UPSTREAMS.lock.json"
TARGET_NAME = "Moshika-PyTorch-BF16"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_entry() -> dict:
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return next(row for row in payload["upstreams"] if row["name"] == TARGET_NAME)


def target_dir(entry: dict) -> Path:
    return PROJECT_ROOT / "hf_cache" / "pinned" / f"moshika-pytorch-bf16-{entry['revision'][:7]}"


def verify(entry: dict, destination: Path) -> dict:
    files = {}
    for filename, expected in entry["weights"].items():
        path = destination / filename
        actual_size = path.stat().st_size if path.is_file() else None
        actual_sha256 = sha256_file(path) if path.is_file() else None
        files[filename] = {
            "path": str(path),
            "expected_bytes": expected["bytes"],
            "actual_bytes": actual_size,
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual_sha256,
            "valid": actual_size == expected["bytes"] and actual_sha256 == expected["sha256"],
        }
    return {
        "repository": entry["repository"],
        "revision": entry["revision"],
        "destination": str(destination),
        "files": files,
        "valid": all(row["valid"] for row in files.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check local files without network access",
    )
    args = parser.parse_args()
    entry = locked_entry()
    destination = target_dir(entry)
    destination.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        from huggingface_hub import hf_hub_download

        repo_id = entry["repository"].removeprefix("https://huggingface.co/")
        for filename, expected in entry["weights"].items():
            local_path = destination / filename
            already_valid = (
                local_path.is_file()
                and local_path.stat().st_size == expected["bytes"]
                and sha256_file(local_path) == expected["sha256"]
            )
            if already_valid:
                continue
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=entry["revision"],
                local_dir=destination,
            )
    report = verify(entry, destination)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
