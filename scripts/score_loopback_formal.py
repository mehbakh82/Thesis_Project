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
    """Assign server turns to planned turns, tracking the capture-clock offset.

    The browser's capture clock drifts against the fake-microphone timeline
    (several seconds over a 40-minute run), so a fixed window fails. Planned
    turns are about 13 s apart; each planned turn takes the server turns that
    end inside its window shifted by the current offset estimate, and the
    offset is updated from the assigned end of speech nearest the expected one.
    """

    plan = receipt["plan"]
    turns = sorted((t for t in receipt["turns"] if t.get("end_sample") is not None), key=lambda t: t["end_sample"])
    ends = [t["end_sample"] / RATE for t in turns]
    # initial offset: median distance of the first answered server turns to the nearest planned end
    first = [
        min((e - p["end_s"] for p in plan[:12]), key=abs)
        for e, t in zip(ends, turns, strict=True)
        if t.get("client_t_first_audio_ms") is not None
    ][:8]
    offset = statistics.median(first) if first else 0.0
    rows = []
    for index, planned in enumerate(plan):
        lo = planned["start_s"] + offset - 1.5
        hi = (plan[index + 1]["start_s"] if index + 1 < len(plan) else math.inf) + offset - 1.5
        mine = [t for t, e in zip(turns, ends, strict=True) if lo <= e < hi]
        if mine:
            # the end of the planned utterance itself (not of an overlaid interruption)
            expected = planned["end_s"] + offset
            nearest = min((t["end_sample"] / RATE for t in mine), key=lambda e: abs(e - expected))
            offset = 0.7 * offset + 0.3 * (nearest - planned["end_s"])
        mine = [t for t in mine if not t.get("speculation_cancelled")]
        heard = [t for t in mine if t.get("client_t_first_audio_ms") is not None]
        first_reply = heard[0] if heard else None
        events = [e for t in mine for e in (t.get("bargein_events") or [])]
        rows.append(
            {
                "id": planned["id"],
                "server_turns": len(mine),
                "clock_offset_s": round(offset, 3),
                "t_first_audio_ms": float(first_reply["client_t_first_audio_ms"]) if first_reply else math.inf,
                "t_first_audible_ms": float(first_reply["client_t_first_audible_ms"])
                if first_reply and first_reply.get("client_t_first_audible_ms") is not None
                else math.inf,
                "t_duck_ms": next((float(t["client_t_duck_ms"]) for t in mine if t.get("client_t_duck_ms") is not None), None),
                "t_stop_ms": next((float(t["client_t_stop_ms"]) for t in mine if t.get("client_t_stop_ms") is not None), None),
                "stopped": any(e.get("kind") == "stop" for e in events),
                "ducked": any(e.get("kind") == "duck" for e in events),
                "outcome": first_reply.get("outcome") if first_reply else None,
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
    parser.add_argument("--protocol", default="docs/LIVE_LATENCY_LOOPBACK_PROTOCOL.md")
    parser.add_argument("--note", default=None, help="free-text provenance note recorded in the receipt")
    parser.add_argument(
        "--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS,
        help="conditions to score (a latency-only run has plain and elderly)",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite write-once receipt {args.out}")

    result: dict[str, Any] = {
        "schema_version": 1,
        "evidence_class": "automated_browser_loopback",
        "official_e2e_eligible": False,
        "protocol": args.protocol,
        **({"note": args.note} if args.note else {}),
        "hardware": "shared NVIDIA H100 host (not a 12-24 GB device; memory budget emulated, compute not)",
        "server_identity": json.loads((args.dir / "health.json").read_text()),
        "conditions": {},
    }
    for condition in args.conditions:
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
    result["acceptance"] = {
        "plain_mean_le_500ms": bool(mean <= 500),
        "plain_p90_le_500ms_failures_counted": bool(p90 <= 500),
    }
    if "elderly" in result["conditions"]:
        elderly_p90 = result["conditions"]["elderly"]["t_first_audio_p90_failures_as_late_ms"]
        result["acceptance"]["elderly_p90_le_500ms_failures_counted"] = bool(
            elderly_p90 is not None and elderly_p90 <= 500
        )
    if not {"bargein_interrupt", "bargein_backchannel"} <= set(result["conditions"]):
        return _write(args, result)
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
    result["acceptance"].update(
        duck_median_le_150ms=bool(interrupt["t_duck_ms"].get("median", math.inf) <= 150),
        bargein_accuracy_and_f1_gt_0_8=bool(
            result["bargein_pooled"]["accuracy"] > 0.8 and result["bargein_pooled"]["f1"] > 0.8
        ),
    )
    return _write(args, result)


def _write(args: argparse.Namespace, result: dict[str, Any]) -> int:
    pids_path = args.dir / "stack_pids.json"
    if pids_path.exists() and (args.dir / "gpu_memory.csv").exists():
        result["gpu_memory"] = memory(args.dir / "gpu_memory.csv", json.loads(pids_path.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"acceptance": result["acceptance"], "bargein_pooled": result.get("bargein_pooled")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
