#!/usr/bin/env python3
"""Verify corpus turn-boundary, overlap and caption labels with Gemini.

Protocol: ``docs/CORPUS_LABEL_VERIFICATION_PROTOCOL.md``. Subcommands:

``sample``  draw the stratified sample from the pairs whose source window audio
            is on local disk (deterministic);
``label``   label one shard of the sample with two boundary passes and one
            verbatim transcript per pair (resumable, cached Gemini calls);
``score``   write the write-once receipt (no audio, transcript or caption text).

Gemini output is an independent automatic label, never human verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from label_interaction_events import LABEL_KEYS, PROMPT_A, PROMPT_B, _validate  # noqa: E402

from thesis_s2s.eval.asr_normalize import NORMALIZER_VERSION, error_counts  # noqa: E402
from thesis_s2s.labeling.gemini import VERBATIM_PROMPT, GeminiClient  # noqa: E402

SEED = 20261018
PER_STRATUM = 150
BOUNDARY_HALF_S = 1.0  # overlap stratum: an overlap interval within 1 s of the boundary
CLIP_HALF_S = 4.0
MAX_TRANSCRIBED_TURN_S = 60.0
NEW_VOICE = {"interrupt", "backchannel", "turn_change"}


def boundary_overlap(row: dict[str, Any]) -> bool:
    b = float(row["source_user_interval"][1])
    return any(
        min(e, b + BOUNDARY_HALF_S) > max(s, b - BOUNDARY_HALF_S) for s, e in (row.get("overlap_intervals") or [])
    )


def read_segment(path: Path, start: float, end: float) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or rate != 16_000:
            raise SystemExit(f"unexpected audio format in {path}")
        first = max(0, int(start * rate))
        last = min(handle.getnframes(), int(end * rate))
        handle.setpos(first)
        raw = handle.readframes(max(0, last - first))
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z = 1.959963984540054
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def sample(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    local = sorted((r for r in rows if (ROOT / r["source_audio_filepath"]).is_file()), key=lambda r: r["utt_id"])
    rng = random.Random(SEED)
    picked = []
    for stratum in (True, False):
        pool = [r for r in local if boundary_overlap(r) is stratum]
        picked += [{"stratum": "boundary_overlap" if stratum else "boundary_clean", **r} for r in rng.sample(pool, min(PER_STRATUM, len(pool)))]
    frame = {
        "manifest_pairs": len(rows),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "local_frame_pairs": len(local),
        "strata_full_manifest": {
            "boundary_overlap": sum(boundary_overlap(r) for r in rows),
            "boundary_clean": sum(not boundary_overlap(r) for r in rows),
        },
        "strata_local_frame": {
            "boundary_overlap": sum(boundary_overlap(r) for r in local),
            "boundary_clean": sum(not boundary_overlap(r) for r in local),
        },
        "channels_local_frame": _channels(local),
        "channels_sample": _channels(picked),
    }
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked), encoding="utf-8")
    args.out.with_suffix(".frame.json").write_text(json.dumps(frame, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(frame))
    return 0


def _channels(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        name = r["session_id"].split("/")[0]
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def label_pair(client: GeminiClient, row: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / row["source_audio_filepath"]
    b = float(row["source_user_interval"][1])
    start = max(0.0, b - CLIP_HALF_S)
    clip = read_segment(path, start, b + CLIP_HALF_S)
    passes = {}
    for name, template in (("a", PROMPT_A), ("b", PROMPT_B)):
        reply = client.label_audio(template.format(onset=b - start), clip, 16_000, required_keys=LABEL_KEYS[:2])
        passes[name] = _validate(reply)
    user_start, user_end = (float(x) for x in row["source_user_interval"])
    transcript = None
    if user_end - user_start <= MAX_TRANSCRIBED_TURN_S:
        reply = client.label_audio(
            VERBATIM_PROMPT, read_segment(path, user_start, user_end), 16_000, required_keys=("transcript",)
        )
        transcript = str(reply.get("transcript") or "")
    return {
        "utt_id": row["utt_id"],
        "stratum": row["stratum"],
        "channel": row["session_id"].split("/")[0],
        "user_seconds": round(user_end - user_start, 3),
        "pass_a": passes["a"],
        "pass_b": passes["b"],
        "agreed": passes["a"]["label"] == passes["b"]["label"],
        "overlap_agreed": passes["a"]["overlap"] == passes["b"]["overlap"],
        "gemini_transcript": transcript,
        "caption": row["text"],
        "independent_automatic_label": True,
        "human_verified": False,
    }


def label(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = [json.loads(line) for line in args.sample.read_text(encoding="utf-8").splitlines() if line]
    rows = [r for i, r in enumerate(rows) if i % args.shards == args.shard]
    done = set()
    if args.out.is_file():
        done = {json.loads(line)["utt_id"] for line in args.out.read_text(encoding="utf-8").splitlines() if line}
    todo = [r for r in rows if r["utt_id"] not in done]
    client = GeminiClient(cache_dir=args.cache, requests_per_minute=args.rpm)
    failures = 0
    with args.out.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(label_pair, client, r): r for r in todo}
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as exc:  # counted; a rerun resumes the missing pairs
                failures += 1
                print(f"failed {futures[future]['utt_id']}: {type(exc).__name__}", file=sys.stderr, flush=True)
                continue
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            if count % 25 == 0:
                print(f"shard {args.shard}: {count}/{len(todo)} failures={failures}", file=sys.stderr, flush=True)
    print(json.dumps({"shard": args.shard, "labeled": len(todo) - failures, "failures": failures}))
    return 0


def score(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite write-once receipt {args.out}")
    frame = json.loads(args.sample.with_suffix(".frame.json").read_text(encoding="utf-8"))
    sample_rows = [json.loads(line) for line in args.sample.read_text(encoding="utf-8").splitlines() if line]
    labels: dict[str, dict[str, Any]] = {}
    for path in args.labels:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                labels[row["utt_id"]] = row
    missing = [r["utt_id"] for r in sample_rows if r["utt_id"] not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} sampled pairs are not labeled yet")
    rows = [labels[r["utt_id"]] for r in sample_rows]
    total = sum(frame["strata_full_manifest"].values())
    weights = {k: v / total for k, v in frame["strata_full_manifest"].items()}
    strata: dict[str, Any] = {}
    for name in ("boundary_overlap", "boundary_clean"):
        mine = [r for r in rows if r["stratum"] == name]
        agreed = [r for r in mine if r["agreed"]]
        new_voice = sum(r["pass_a"]["label"] in NEW_VOICE for r in agreed)
        both_overlap = [r for r in mine if r["overlap_agreed"]]
        heard = sum(r["pass_a"]["overlap"] for r in both_overlap)
        expected = heard if name == "boundary_overlap" else len(both_overlap) - heard
        dist: dict[str, int] = {}
        for r in agreed:
            dist[r["pass_a"]["label"]] = dist.get(r["pass_a"]["label"], 0) + 1
        strata[name] = {
            "pairs": len(mine),
            "label_agreement": len(agreed) / len(mine),
            "agreed_label_distribution": dict(sorted(dist.items(), key=lambda kv: -kv[1])),
            "speaker_change_rate": new_voice / len(agreed) if agreed else None,
            "speaker_change_ci95": wilson(new_voice, len(agreed)),
            "overlap_flag_agreement": len(both_overlap) / len(mine),
            "overlap_label_confirmed_rate": expected / len(both_overlap) if both_overlap else None,
            "overlap_label_confirmed_ci95": wilson(expected, len(both_overlap)),
        }
    weighted = sum(weights[k] * strata[k]["speaker_change_rate"] for k in strata)
    counts = [
        (r, error_counts(r["caption"], r["gemini_transcript"]))
        for r in rows
        if r["gemini_transcript"] is not None and r["gemini_transcript"].strip()
    ]
    per_turn = [c["char_errors"] / max(1, c["reference_chars"]) for _, c in counts]
    receipt = {
        "schema_version": 1,
        "experiment": "corpus_label_verification",
        "protocol": "docs/CORPUS_LABEL_VERIFICATION_PROTOCOL.md",
        "labeler": "google/gemini-3.8-flash, two boundary passes + one verbatim transcript, temperature 0",
        "independent_automatic_labels": True,
        "human_verified": False,
        "frame": frame,
        "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        "strata_weights_full_manifest": weights,
        "strata": strata,
        "speaker_change_rate_weighted": weighted,
        "caption_vs_gemini": {
            "normalizer": NORMALIZER_VERSION,
            "reference": "caption (the corpus text); hypothesis: Gemini verbatim transcript",
            "turns_scored": len(counts),
            "turns_longer_than_60_s": sum(r["gemini_transcript"] is None for r in rows),
            "turns_without_gemini_speech": sum(
                r["gemini_transcript"] is not None and not r["gemini_transcript"].strip() for r in rows
            ),
            "corpus_cer": sum(c["char_errors"] for _, c in counts) / sum(c["reference_chars"] for _, c in counts),
            "median_turn_cer": statistics.median(per_turn),
            "share_turns_cer_le_20pct": sum(x <= 0.2 for x in per_turn) / len(per_turn),
        },
        "text_retained": False,
        "claim_boundary": (
            "Automatic corpus labels checked by an independent automatic listener on a convenience "
            "frame (pairs whose audio is still local); not human annotation."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("strata", "speaker_change_rate_weighted", "caption_vs_gemini")}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifests/conversations_balanced_v2.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("label")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--rpm", type=float, default=15.0)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("score")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--labels", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return {"sample": sample, "label": label, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
