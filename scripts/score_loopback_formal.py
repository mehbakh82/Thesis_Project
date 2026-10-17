#!/usr/bin/env python3
"""Score the formal browser-loopback runs into one write-once receipt.

Inputs are the per-condition receipts of ``scripts/loopback_latency.py``
(plain, elderly, bargein_interrupt, bargein_backchannel), the sampled GPU
memory of the stack processes, and the server ``/health`` identity.

Every *planned* user turn is scored, as the protocol requires: server turn
records are mapped to planned turns by their end-of-speech sample (the server
timeline starts with the first captured frame, i.e. at the start of the fake
microphone file), and a planned turn with no first audio counts as a failure.
For a planned turn split into several server turns by a pause, the first reply
is the one the user would hear first and is used. Latency acceptance uses the
proposal criteria (mean <= 500 ms and p90 <= 500 ms of T_first_audio on the
plain panel), with failures entering the p90 as infinitely late.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

RATE = 16_000
CONDITIONS = ("plain", "elderly", "bargein_interrupt", "bargein_backchannel")


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = q * (len(ordered) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi or math.isinf(ordered[hi]):
        return ordered[hi]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def stats(values: list[float]) -> dict[str, Any]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"n": 0}
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p90": pct(finite, 0.9),
        "p95": pct(finite, 0.95),
        "max": max(finite),
    }


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def map_turns(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    plan = receipt["plan"]
    turns = sorted((t for t in receipt["turns"] if t.get("end_sample") is not None), key=lambda t: t["end_sample"])
    rows = []
    for index, planned in enumerate(plan):
        window_end = plan[index + 1]["start_s"] if index + 1 < len(plan) else math.inf
        mine = [
            t for t in turns
            if planned["start_s"] - 0.3 <= t["end_sample"] / RATE < window_end - 0.3
            and not t.get("speculation_cancelled")
        ]
        heard = [t for t in mine if t.get("client_t_first_audio_ms") is not None]
        first = heard[0] if heard else None
        events = [e for t in mine for e in (t.get("bargein_events") or [])]
        rows.append(
            {
                "id": planned["id"],
                "server_turns": len(mine),
                "t_first_audio_ms": float(first["client_t_first_audio_ms"]) if first else math.inf,
                "t_first_audible_ms": float(first["client_t_first_audible_ms"]) if first and first.get("client_t_first_audible_ms") is not None else math.inf,
                "t_duck_ms": next((float(t["client_t_duck_ms"]) for t in mine if t.get("client_t_duck_ms") is not None), None),
                "t_stop_ms": next((float(t["client_t_stop_ms"]) for t in mine if t.get("client_t_stop_ms") is not None), None),
                "stopped": any(e.get("kind") == "stop" for e in events),
                "ducked": any(e.get("kind") == "duck" for e in events),
                "outcome": first.get("outcome") if first else None,
                "errors": [t.get("asr_error") or t.get("responder_error") for t in mine if t.get("asr_error") or t.get("responder_error")],
            }
        )
    return rows


def memory(csv_path: Path, pids: dict[str, Any]) -> dict[str, Any]:
    nemo = str(pids["nemo_pid"])
    vllm = set(str(pids["vllm_pids"]).split())
    peak: dict[str, float] = defaultdict(float)
    per_sample: dict[str, dict[str, float]] = defaultdict(dict)
    with csv_path.open() as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            stamp, pid, used = row[0].strip(), row[1].strip(), row[2].strip().split()[0]
            if pid == nemo or pid in vllm:
                role = "asr_nemo" if pid == nemo else "llm_vllm"
                per_sample[stamp][role] = per_sample[stamp].get(role, 0.0) + float(used)
                peak[role] = max(peak[role], per_sample[stamp][role])
    totals = [sum(v.values()) for v in per_sample.values() if len(v) == 2]
    return {
        "samples": len(per_sample),
        "peak_mib": dict(peak),
        "peak_stack_mib": max(totals) if totals else None,
        "cpu_components": ["Mana Piper TTS (ONNX Runtime, 8 threads)", "Silero VAD (ONNX)"],
        "note": "vLLM reserves its whole --gpu-memory-utilization budget up front; the peak is that reservation, not the minimum need.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite write-once receipt {args.out}")

    result: dict[str, Any] = {
        "schema_version": 1,
        "evidence_class": "automated_browser_loopback",
        "official_e2e_eligible": False,
        "protocol": "docs/LIVE_LATENCY_LOOPBACK_PROTOCOL.md",
        "hardware": "shared NVIDIA H100 host (not a 12-24 GB device; memory budget emulated, compute not)",
        "server_identity": json.loads((args.dir / "health.json").read_text()),
        "conditions": {},
    }
    for condition in CONDITIONS:
        path = args.dir / f"{condition}.json"
        receipt = json.loads(path.read_text())
        rows = map_turns(receipt)
        firsts = [r["t_first_audio_ms"] for r in rows]
        failures = sum(1 for v in firsts if not math.isfinite(v))
        entry: dict[str, Any] = {
            "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fake_mic_sha256": receipt["fake_mic_sha256"],
            "planned_turns": len(rows),
            "server_turn_records": receipt["server_turn_records"],
            "turns_without_first_audio": failures,
            "t_first_audio_ms": stats(firsts),
            "t_first_audio_p90_failures_as_late_ms": finite_or_none(pct(firsts, 0.9)),
            "t_first_audible_ms": stats([r["t_first_audible_ms"] for r in rows]),
            "share_le_500ms_of_planned": sum(1 for v in firsts if v <= 500) / max(1, len(rows)),
            "console_errors": len(receipt.get("console_errors") or []),
        }
        if condition.startswith("bargein"):
            interrupt = condition == "bargein_interrupt"
            correct = [r["stopped"] if interrupt else not r["stopped"] for r in rows]
            ducks = [r["t_duck_ms"] for r in rows if r["t_duck_ms"] is not None]
            stops = [r["t_stop_ms"] for r in rows if r["t_stop_ms"] is not None]
            entry.update(
                decision_accuracy=sum(correct) / max(1, len(correct)),
                stopped_share=sum(r["stopped"] for r in rows) / max(1, len(rows)),
                ducked_share=sum(r["ducked"] for r in rows) / max(1, len(rows)),
                t_duck_ms=stats(ducks),
                t_duck_share_le_150ms=sum(1 for d in ducks if d <= 150) / max(1, len(ducks)),
                t_stop_ms=stats(stops),
            )
        result["conditions"][condition] = entry
    plain = result["conditions"]["plain"]
    mean = plain["t_first_audio_ms"].get("mean", math.inf)
    p90 = plain["t_first_audio_p90_failures_as_late_ms"]
    p90 = math.inf if p90 is None else p90
    interrupt = result["conditions"]["bargein_interrupt"]
    backchannel = result["conditions"]["bargein_backchannel"]
    tp = round(interrupt["decision_accuracy"] * interrupt["planned_turns"])
    tn = round(backchannel["decision_accuracy"] * backchannel["planned_turns"])
    fn = interrupt["planned_turns"] - tp
    fp = backchannel["planned_turns"] - tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result["bargein_pooled"] = {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
    result["acceptance"] = {
        "plain_mean_le_500ms": bool(mean <= 500),
        "plain_p90_le_500ms_failures_counted": bool(p90 <= 500),
        "duck_median_le_150ms": bool(interrupt["t_duck_ms"].get("median", math.inf) <= 150),
        "bargein_accuracy_and_f1_gt_0_8": bool(
            result["bargein_pooled"]["accuracy"] > 0.8 and result["bargein_pooled"]["f1"] > 0.8
        ),
    }
    pids_path = args.dir / "stack_pids.json"
    if pids_path.exists() and (args.dir / "gpu_memory.csv").exists():
        result["gpu_memory"] = memory(args.dir / "gpu_memory.csv", json.loads(pids_path.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"acceptance": result["acceptance"], "bargein_pooled": result["bargein_pooled"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
