#!/usr/bin/env python3
"""Freeze a deterministic train-only corpus for the Moshi v6 capacity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SELECTION_SIZE = 32
MIN_USER_SECONDS = 1.0
MAX_USER_SECONDS = 4.5
MIN_ASSISTANT_SECONDS = 1.0
MAX_ASSISTANT_SECONDS = 5.5
MAX_EXPORT_SECONDS = 12.0
MIN_WORDS = 2
MAX_WORDS = 14
MIN_PERSIAN_LETTER_FRACTION = 0.95
MAX_COMPLETE_USER_END_SECONDS = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSON-object lines: {path}")
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def persian_letter_fraction(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    persian = sum("\u0600" <= character <= "\u06ff" for character in letters)
    return persian / len(letters)


def source_is_eligible(source: dict[str, Any], export: dict[str, Any]) -> bool:
    text = str(source.get("response_text") or source.get("assistant_text") or "").strip()
    try:
        source_span_start = float(source["source_span_start"])
        user_end = float(source["source_user_interval"][1]) - source_span_start
        user_seconds = float(source["user_duration"])
        assistant_seconds = float(source["assistant_duration"])
        export_seconds = float(export["duration"])
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return (
        source.get("split") == "train"
        and source.get("training_use_authorized") is True
        and source.get("noise_condition") == "background-clean"
        and not source.get("overlap_intervals")
        and source.get("interrupt_label") == "none"
        and MIN_USER_SECONDS <= user_seconds <= MAX_USER_SECONDS
        and MIN_ASSISTANT_SECONDS <= assistant_seconds <= MAX_ASSISTANT_SECONDS
        and export_seconds <= MAX_EXPORT_SECONDS
        and MIN_WORDS <= len(text.split()) <= MAX_WORDS
        and persian_letter_fraction(text) >= MIN_PERSIAN_LETTER_FRACTION
        and user_end <= MAX_COMPLETE_USER_END_SECONDS
    )


def select_rows(
    export_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    *,
    metadata_loader: Callable[[Path], dict[str, Any]],
    size: int = SELECTION_SIZE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if size < 1:
        raise ValueError("selection size must be positive")
    source_by_id = {str(row.get("utt_id") or ""): row for row in source_rows}
    if "" in source_by_id or len(source_by_id) != len(source_rows):
        raise ValueError("source rows require unique nonempty utt_id values")

    candidates: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for manifest_index, export in enumerate(export_rows):
        export_path = Path(str(export.get("path") or "")).resolve()
        if not export_path.is_file() or str(export_path) in seen_paths:
            raise ValueError(f"missing or duplicate export path: {export_path}")
        seen_paths.add(str(export_path))
        metadata = metadata_loader(export_path.with_suffix(".json"))
        pair_id = str(metadata.get("source_pair_id") or "")
        source = source_by_id.get(pair_id)
        if source is None or not source_is_eligible(source, export):
            continue
        session_id = str(source.get("session_id") or "")
        if not session_id:
            raise ValueError(f"eligible pair has no session: {pair_id}")
        candidates.append(
            {
                "manifest_index": manifest_index,
                "pair_id": pair_id,
                "session_id": session_id,
                "channel": session_id.split("/", 1)[0],
                "response_text_sha256": hashlib.sha256(
                    str(source["response_text"]).encode("utf-8")
                ).hexdigest(),
                "word_count": len(str(source["response_text"]).split()),
                "user_seconds": float(source["user_duration"]),
                "assistant_seconds": float(source["assistant_duration"]),
                "export_seconds": float(export["duration"]),
                "snr_db": float(source.get("snr") or 0.0),
                "export": export,
            }
        )
    if len(candidates) < size:
        raise RuntimeError(f"only {len(candidates)} eligible rows; {size} required")

    ordered = sorted(
        candidates,
        key=lambda row: hashlib.sha256(str(row["pair_id"]).encode("utf-8")).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_sessions: set[str] = set()
    for row in ordered:
        if row["session_id"] in selected_sessions:
            continue
        selected.append(row)
        selected_ids.add(str(row["pair_id"]))
        selected_sessions.add(str(row["session_id"]))
        if len(selected) >= size:
            break
    for row in ordered:
        if len(selected) >= size:
            break
        if row["pair_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(str(row["pair_id"]))
    selected = selected[:size]
    if len(selected) != size or len(selected_ids) != size:
        raise RuntimeError("deterministic selection did not produce unique requested rows")
    return [dict(row["export"]) for row in selected], selected, len(candidates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-train",
        type=Path,
        default=Path("data/processed/moshi_finetune_v2/train.jsonl"),
    )
    parser.add_argument(
        "--source-conversations",
        type=Path,
        default=Path("data/processed/manifests/conversations.jsonl"),
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=Path("data/processed/moshi_finetune_v6_overfit/train.jsonl"),
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=Path("results/moshi_v6_overfit_data.json"),
    )
    parser.add_argument("--size", type=int, default=SELECTION_SIZE)
    args = parser.parse_args()

    source_train = (ROOT / args.source_train).resolve()
    source_conversations = (ROOT / args.source_conversations).resolve()
    out_manifest = (ROOT / args.out_manifest).resolve()
    out_report = (ROOT / args.out_report).resolve()
    if out_report.exists():
        raise FileExistsError(f"refusing to overwrite frozen v6 data report: {out_report}")
    export_rows = read_jsonl(source_train)
    source_rows = read_jsonl(source_conversations)
    selected_exports, descriptors, eligible_count = select_rows(
        export_rows,
        source_rows,
        metadata_loader=lambda path: json.loads(path.read_text(encoding="utf-8")),
        size=args.size,
    )
    write_jsonl(out_manifest, selected_exports)

    channels = Counter(str(row["channel"]) for row in descriptors)
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "train-only in-sample Moshi v6 Persian capacity diagnostic",
        "scientific_scope": "diagnostic_not_validation_or_final_test",
        "human_verified": False,
        "mechanically_filtered_not_human_clean": True,
        "final_test_accessed": False,
        "selection_rule_frozen_before_optimizer_step": True,
        "inputs": {
            "source_train": source_train.relative_to(ROOT).as_posix(),
            "source_train_sha256": sha256_file(source_train),
            "source_conversations": source_conversations.relative_to(ROOT).as_posix(),
            "source_conversations_sha256": sha256_file(source_conversations),
        },
        "output": {
            "manifest": out_manifest.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(out_manifest),
            "rows": len(selected_exports),
        },
        "criteria": {
            "source_split": "train",
            "training_use_authorized": True,
            "noise_condition": "background-clean",
            "overlap_intervals": "empty",
            "interrupt_label": "none",
            "user_seconds": [MIN_USER_SECONDS, MAX_USER_SECONDS],
            "assistant_seconds": [MIN_ASSISTANT_SECONDS, MAX_ASSISTANT_SECONDS],
            "maximum_export_seconds": MAX_EXPORT_SECONDS,
            "word_count": [MIN_WORDS, MAX_WORDS],
            "minimum_persian_letter_fraction": MIN_PERSIAN_LETTER_FRACTION,
            "maximum_complete_user_end_seconds": MAX_COMPLETE_USER_END_SECONDS,
            "selection": "sha256(pair_id), one-per-session first, then deterministic fill",
        },
        "eligible_before_limit": eligible_count,
        "selected": {
            "rows": len(descriptors),
            "sessions": len({str(row["session_id"]) for row in descriptors}),
            "channels": dict(sorted(channels.items())),
            "duration_seconds": round(sum(float(row["export_seconds"]) for row in descriptors), 6),
            "median_snr_db": round(
                statistics.median(float(row["snr_db"]) for row in descriptors), 3
            ),
            "pair_ids": [str(row["pair_id"]) for row in descriptors],
            "response_text_sha256": [str(row["response_text_sha256"]) for row in descriptors],
        },
        "requirements": {
            "exact_requested_size": len(descriptors) == args.size == SELECTION_SIZE,
            "all_rows_from_train_manifest": all(
                row["export"] in export_rows for row in descriptors
            ),
            "all_selected_rows_meet_mechanical_filter": all(
                source_is_eligible(
                    next(row for row in source_rows if row["utt_id"] == descriptor["pair_id"]),
                    descriptor["export"],
                )
                for descriptor in descriptors
            ),
            "unique_pairs": len({row["pair_id"] for row in descriptors}) == len(descriptors),
            "multiple_sessions": len({row["session_id"] for row in descriptors}) >= 8,
            "manifest_hash_recorded": bool(sha256_file(out_manifest)),
            "final_test_not_read_or_used": True,
        },
        "limitations": [
            "The captions and speaker boundaries remain automatic and are not human-reviewed.",
            "Passing this in-sample diagnostic proves optimization capacity only, not generalization.",
            "No held-out or final-test row is an input to selection, training, or diagnosis.",
        ],
    }
    report["passes"] = all(report["requirements"].values())
    if report["passes"] is not True:
        raise RuntimeError(f"v6 overfit data freeze failed: {report['requirements']}")
    write_json(out_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
