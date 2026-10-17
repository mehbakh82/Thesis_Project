#!/usr/bin/env python3
"""Score ASR v2 predictions under docs/ASR_CANDIDATE_COMPARISON_V2_PROTOCOL.md.

Reads a frozen panel and one prediction JSONL per arm (from
``scripts/asr_predict.py``), computes micro CER/WER with the punctuation-free
Persian scoring normalizer, strata, clip-bootstrap intervals, paired
differences against the incumbent, latency statistics, and applies the frozen
latency-constrained selection rule. The receipt stores counts and hashes only;
no reference or hypothesis text is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.eval.asr_normalize import NORMALIZER_VERSION, error_counts  # noqa: E402

INCUMBENT = "nemo_v4_ctc"
LATENCY_BUDGET_MS = 150.0
SHORT_CLIP_SECONDS = 8.0
BOOTSTRAP_DRAWS = 10_000
SEED = 20261016


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def micro(rows: list[dict[str, int]], kind: str) -> float:
    errors = sum(r[f"{kind}_errors"] for r in rows)
    total = sum(r[f"reference_{kind}s"] for r in rows)
    return errors / total if total else float("nan")


def bootstrap(rows: list[dict[str, int]], kind: str, rng: random.Random) -> list[float]:
    n = len(rows)
    values = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        values.append(micro(sample, kind))
    values.sort()
    return [values[int(0.025 * BOOTSTRAP_DRAWS)], values[int(0.975 * BOOTSTRAP_DRAWS) - 1]]


def paired_difference(
    arm: list[dict[str, int]], base: list[dict[str, int]], kind: str, rng: random.Random
) -> dict[str, float]:
    n = len(arm)
    diffs = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(micro([arm[i] for i in idx], kind) - micro([base[i] for i in idx], kind))
    diffs.sort()
    return {
        "point": micro(arm, kind) - micro(base, kind),
        "ci95_low": diffs[int(0.025 * BOOTSTRAP_DRAWS)],
        "ci95_high": diffs[int(0.975 * BOOTSTRAP_DRAWS) - 1],
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-name", required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    panel = read_jsonl(args.panel)
    by_id = {row["id"]: row for row in panel}
    results: dict[str, Any] = {}
    per_arm_counts: dict[str, list[dict[str, int]]] = {}
    for arm in args.arms:
        pred_path = args.pred_dir / f"{arm}.jsonl"
        meta_path = pred_path.with_suffix(".meta.json")
        predictions = {row["id"]: row for row in read_jsonl(pred_path)}
        missing = [row_id for row_id in by_id if row_id not in predictions]
        counts: list[dict[str, int]] = []
        latencies: list[float] = []
        short_latencies: list[float] = []
        errors = 0
        for row in panel:
            pred = predictions.get(row["id"], {"hypothesis": "", "error": "missing"})
            if pred.get("error"):
                errors += 1
            counts.append(error_counts(row["text"], str(pred.get("hypothesis") or "")))
            latency = pred.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
                if float(row["duration"]) <= SHORT_CLIP_SECONDS:
                    short_latencies.append(float(latency))
        per_arm_counts[arm] = counts
        strata: dict[str, Any] = {}
        for name, predicate in {
            "senior": lambda r: r.get("age") == "senior",
            "non_senior": lambda r: r.get("age") != "senior",
            "noisy_or_phone_or_reverberant": lambda r: r.get("environment") != "clean",
            "clean": lambda r: r.get("environment") == "clean",
            "spontaneous": lambda r: r.get("spontaneous") is True,
            "read_or_scripted": lambda r: r.get("spontaneous") is False,
        }.items():
            subset = [c for c, r in zip(counts, panel, strict=True) if predicate(r)]
            strata[name] = {
                "clips": len(subset),
                "micro_cer": micro(subset, "char"),
                "micro_wer": micro(subset, "word"),
            }
        rng = random.Random(f"{SEED}:{arm}")
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        results[arm] = {
            "clips": len(counts),
            "missing_predictions": len(missing),
            "runtime_errors": errors,
            "micro_cer": micro(counts, "char"),
            "micro_wer": micro(counts, "word"),
            "cer_ci95": bootstrap(counts, "char", rng),
            "wer_ci95": bootstrap(counts, "word", rng),
            "strata": strata,
            "latency_ms": {
                "median": statistics.median(latencies) if latencies else None,
                "p90": percentile(latencies, 0.9) if latencies else None,
                "short_clip_median": statistics.median(short_latencies) if short_latencies else None,
                "short_clip_p90": percentile(short_latencies, 0.9) if short_latencies else None,
                "short_clips": len(short_latencies),
            },
            "load_seconds": meta.get("load_seconds"),
            "peak_torch_gib": meta.get("peak_torch_gib"),
            "checkpoint": Path(str(meta.get("checkpoint", ""))).name or None,
            "predictions_sha256": sha256_file(pred_path),
        }
    if INCUMBENT in per_arm_counts:
        for arm in args.arms:
            rng = random.Random(f"{SEED}:paired:{arm}")
            results[arm]["paired_vs_incumbent"] = {
                "cer": paired_difference(per_arm_counts[arm], per_arm_counts[INCUMBENT], "char", rng),
                "wer": paired_difference(per_arm_counts[arm], per_arm_counts[INCUMBENT], "word", rng),
            }
    eligible = [
        arm
        for arm in args.arms
        if results[arm]["latency_ms"]["short_clip_median"] is not None
        and results[arm]["latency_ms"]["short_clip_median"] <= LATENCY_BUDGET_MS
        and results[arm]["missing_predictions"] == 0
    ]
    deployment = min(eligible, key=lambda a: results[a]["micro_cer"]) if eligible else None
    reference = min(args.arms, key=lambda a: results[a]["micro_cer"])
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": "docs/ASR_CANDIDATE_COMPARISON_V2_PROTOCOL.md",
        "protocol_sha256": sha256_file(ROOT / "docs/ASR_CANDIDATE_COMPARISON_V2_PROTOCOL.md"),
        "panel_name": args.panel_name,
        "panel_sha256": sha256_file(args.panel),
        "panel_clips": len(panel),
        "normalizer": NORMALIZER_VERSION,
        "scorer_sha256": sha256_file(Path(__file__)),
        "incumbent": INCUMBENT,
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "arms": results,
        "selection": {
            "latency_eligible_arms": eligible,
            "deployment_asr": deployment,
            "accuracy_reference_asr": reference,
        },
        "text_retained": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = {
        a: (
            round(results[a]["micro_cer"], 4),
            round(results[a]["micro_wer"], 4),
            round(results[a]["strata"]["senior"]["micro_cer"], 4),
            results[a]["latency_ms"]["short_clip_median"],
        )
        for a in args.arms
    }
    print(json.dumps({"cer, wer, senior_cer, short_median_ms": table, **receipt["selection"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
