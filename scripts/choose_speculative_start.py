#!/usr/bin/env python3
"""Choose the speculative-start delay from the live loopback v2 development runs.

Protocol: ``docs/LIVE_LOOPBACK_V2_PROTOCOL.md``. Reads ``<dev>/<ms>/plain.json``
(harness receipts) for every candidate, maps server turns to planned turns with
the formal scorer, and applies the pre-registered rule: lowest p90 of
T_first_audible (missing turns count as infinitely late); a larger candidate
within 10 ms of the best is preferred; candidates with a missing turn are
excluded. Writes ``--out`` and prints the chosen value.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_loopback_formal import map_turns, pct, stats  # noqa: E402

CANDIDATES = (64, 96, 128, 160)
TIE_MS = 10.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for ms in CANDIDATES:
        receipt = json.loads((args.dev / str(ms) / "plain.json").read_text(encoding="utf-8"))
        rows = map_turns(receipt)
        audible = [r["t_first_audible_ms"] for r in rows]
        turns = receipt["turns"]
        speculative = [t for t in turns if t.get("speculative_start")]
        results[ms] = {
            "planned_turns": len(rows),
            "missing_turns": sum(math.isinf(r["t_first_audio_ms"]) for r in rows),
            "t_first_audio_ms": stats([r["t_first_audio_ms"] for r in rows]),
            "t_first_audible_ms": stats(audible),
            "t_first_audible_p90_ms": pct(audible, 0.9),
            "speculative_starts": len(speculative),
            "speculative_cancelled": sum(bool(t.get("speculation_cancelled")) for t in speculative),
            "speculative_starts_per_planned_turn": len(speculative) / max(1, len(rows)),
        }
    eligible = [ms for ms in CANDIDATES if results[ms]["missing_turns"] == 0]
    if not eligible:
        raise SystemExit("every candidate missed a turn")
    best = min(results[ms]["t_first_audible_p90_ms"] for ms in eligible)
    chosen = max(ms for ms in eligible if results[ms]["t_first_audible_p90_ms"] <= best + TIE_MS)
    payload = {
        "schema_version": 1,
        "protocol": "docs/LIVE_LOOPBACK_V2_PROTOCOL.md",
        "rule": "lowest development p90 of T_first_audible; a larger candidate within 10 ms is preferred; "
        "candidates with a missing turn are excluded",
        "candidates": {str(ms): results[ms] for ms in CANDIDATES},
        "chosen_speculative_ms": chosen,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
