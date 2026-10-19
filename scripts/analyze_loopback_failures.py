#!/usr/bin/env python3
"""Failure-mode analysis of a formal browser-loopback run (error analysis).

Reads the four condition receipts of a formal loopback directory
(``plain``, ``elderly``, ``bargein_interrupt``, ``bargein_backchannel``), maps
server turns to planned turns with the formal scorer, and categorizes:

* **slow turns** (first audio later than 500 ms): which server stage was
  slowest relative to its median (ASR, first token, first clause, synthesis),
  or the client side when the server stages were typical;
* **missed interruptions**: no duck at all, or ducked but resumed;
* **false stops** on backchannels.

Writes a JSON receipt (no transcript or reply text).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_loopback_formal import map_turns  # noqa: E402

LIMIT_MS = 500.0
STAGES = (
    ("asr", "asr_ms", None),
    ("first_token", "first_token_ms", "asr_ms"),
    ("first_clause", "first_clause_ms", "first_token_ms"),
    ("synthesis", "first_audio_ready_ms", "first_clause_ms"),
)


def stage_durations(timings: dict[str, Any]) -> dict[str, float] | None:
    out = {}
    for name, end, start in STAGES:
        if timings.get(end) is None or (start and timings.get(start) is None):
            return None
        out[name] = float(timings[end]) - (float(timings[start]) if start else 0.0)
    return out


def slow_turns(receipt: dict[str, Any]) -> dict[str, Any]:
    rows = map_turns(receipt)
    turns = sorted((t for t in receipt["turns"] if t.get("client_t_first_audio_ms") is not None), key=lambda t: t["end_sample"])
    by_first = {round(float(t["client_t_first_audio_ms"]), 3): t for t in turns}
    stages = [d for t in turns if (d := stage_durations(t.get("server_timings_ms") or {}))]
    medians = {name: statistics.median(d[name] for d in stages) for name, _, _ in STAGES}
    slow = [r for r in rows if math.isfinite(r["t_first_audio_ms"]) and r["t_first_audio_ms"] > LIMIT_MS]
    causes: dict[str, int] = {}
    for row in slow:
        turn = by_first.get(round(row["t_first_audio_ms"], 3))
        durations = stage_durations((turn or {}).get("server_timings_ms") or {})
        if durations is None:
            cause = "unknown"
        else:
            excess = {name: durations[name] - medians[name] for name in durations}
            name, value = max(excess.items(), key=lambda kv: kv[1])
            cause = name if value > 20.0 else "client_or_endpoint"
        causes[cause] = causes.get(cause, 0) + 1
    return {
        "planned_turns": len(rows),
        "missing_turns": sum(not math.isfinite(r["t_first_audio_ms"]) for r in rows),
        "turns_over_500_ms": len(slow),
        "slowest_stage_relative_to_median": dict(sorted(causes.items(), key=lambda kv: -kv[1])),
        "stage_medians_ms": medians,
    }


def bargein(receipt: dict[str, Any], positive: bool) -> dict[str, Any]:
    rows = map_turns(receipt)
    if positive:
        missed = [r for r in rows if not r["stopped"]]
        return {
            "events": len(rows),
            "stopped": len(rows) - len(missed),
            "missed_without_duck": sum(not r["ducked"] for r in missed),
            "missed_after_duck_resumed": sum(r["ducked"] for r in missed),
        }
    stopped = [r for r in rows if r["stopped"]]
    return {"events": len(rows), "false_stops": len(stopped), "ducked_and_resumed": sum(r["ducked"] and not r["stopped"] for r in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    load = lambda name: json.loads((args.dir / f"{name}.json").read_text(encoding="utf-8"))  # noqa: E731
    receipt = {
        "schema_version": 1,
        "experiment": "loopback_failure_modes",
        "source_dir_name": args.dir.name,
        "latency": {name: slow_turns(load(name)) for name in ("plain", "elderly")},
        "missed_interruptions": bargein(load("bargein_interrupt"), positive=True),
        "backchannel_false_stops": bargein(load("bargein_backchannel"), positive=False),
        "text_retained": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
