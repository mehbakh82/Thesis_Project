#!/usr/bin/env python3
"""Freeze an untouched group-held-out split for the corrected Moshi v2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def heldout_session(session_id: str, *, modulus: int = 10, bucket: int = 0) -> bool:
    if not session_id or modulus <= 1 or not 0 <= bucket < modulus:
        raise ValueError("valid nonempty session and holdout bucket are required")
    value = int.from_bytes(
        hashlib.blake2b(session_id.encode("utf-8"), digest_size=8).digest(),
        "big",
    )
    return value % modulus == bucket


def split_rows(
    rows: list[dict],
    session_for_row: Callable[[dict], str],
) -> tuple[list[dict], list[dict], dict[str, set[str]]]:
    train: list[dict] = []
    heldout: list[dict] = []
    sessions: dict[str, set[str]] = {"train": set(), "heldout": set()}
    for row in rows:
        session_id = session_for_row(row)
        destination = "heldout" if heldout_session(session_id) else "train"
        sessions[destination].add(session_id)
        (heldout if destination == "heldout" else train).append(row)
    if not train or not heldout:
        raise RuntimeError("deterministic v2 split produced an empty partition")
    if sessions["train"] & sessions["heldout"]:
        raise RuntimeError("session leakage in deterministic v2 split")
    return train, heldout, sessions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-train",
        type=Path,
        default=Path("data/processed/moshi_finetune/train.jsonl"),
    )
    parser.add_argument(
        "--source-validation",
        type=Path,
        default=Path("data/processed/moshi_finetune/val.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/moshi_finetune_v2"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/moshi_v2_split.json"),
    )
    args = parser.parse_args()

    source_train = (ROOT / args.source_train).resolve()
    source_validation = (ROOT / args.source_validation).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    source_rows = read_jsonl(source_train)
    validation_rows = read_jsonl(source_validation)

    session_cache: dict[str, str] = {}

    def session_for_row(row: dict) -> str:
        path = Path(str(row.get("path") or "")).resolve()
        key = str(path)
        if key not in session_cache:
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            session_id = str(metadata.get("source_session_id") or "").strip()
            if not session_id:
                raise ValueError(f"missing source_session_id: {metadata_path}")
            session_cache[key] = session_id
        return session_cache[key]

    train_rows, heldout_rows, sessions = split_rows(source_rows, session_for_row)
    validation_sessions = {session_for_row(row) for row in validation_rows}
    if validation_sessions & sessions["train"] or validation_sessions & sessions["heldout"]:
        raise RuntimeError("v2 train/final-test sessions overlap the validation sessions")

    outputs = {
        "train": out_dir / "train.jsonl",
        "validation": out_dir / "val.jsonl",
        "final_test": out_dir / "test.jsonl",
    }
    write_jsonl(outputs["train"], train_rows)
    write_jsonl(outputs["validation"], validation_rows)
    write_jsonl(outputs["final_test"], heldout_rows)

    counts = Counter()
    for name, partition in (
        ("train", train_rows),
        ("validation", validation_rows),
        ("final_test", heldout_rows),
    ):
        counts[f"{name}_rows"] = len(partition)
        counts[f"{name}_milliseconds"] = round(
            1000 * sum(float(row["duration"]) for row in partition)
        )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Untouched group-held-out final test for the corrected 100-second Moshi v2 run. "
            "The old test remains v1-only and is not an input to v2 training, selection, or testing."
        ),
        "rule": {
            "group": "source_session_id",
            "hash": "blake2b-64",
            "modulus": 10,
            "heldout_bucket": 0,
            "frozen_before_v2_training": True,
        },
        "inputs": {
            "source_train": source_train.relative_to(ROOT).as_posix(),
            "source_train_sha256": sha256_file(source_train),
            "source_validation": source_validation.relative_to(ROOT).as_posix(),
            "source_validation_sha256": sha256_file(source_validation),
            "old_test_used": False,
        },
        "outputs": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "rows": counts[f"{name}_rows"],
                "hours": round(counts[f"{name}_milliseconds"] / 3_600_000, 6),
            }
            for name, path in outputs.items()
        },
        "sessions": {
            "train": len(sessions["train"]),
            "validation": len(validation_sessions),
            "final_test": len(sessions["heldout"]),
            "all_pairwise_disjoint": True,
        },
        "requirements": {
            "source_hashes_current": True,
            "deterministic_rule_frozen": True,
            "train_nonempty": bool(train_rows),
            "validation_nonempty": bool(validation_rows),
            "final_test_nonempty": bool(heldout_rows),
            "session_groups_pairwise_disjoint": True,
            "old_v1_test_not_used": True,
        },
    }
    report["split_passes"] = all(report["requirements"].values())
    report_path = (ROOT / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
