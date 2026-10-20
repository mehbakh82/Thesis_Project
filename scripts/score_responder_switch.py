#!/usr/bin/env python3
"""Part 2 of the responder switch rule: the deployment qualification receipt.

Protocol: ``docs/RESPONDER_SWITCH_PROTOCOL.md``. Combines the loopback receipt
(``scripts/score_loopback_formal.py``, plain and elderly panels) and the
coherence receipt (``scripts/run_interruption_coherence.py score``) of the
Qwen3.5-4B elderly-prompt configuration, and checks every reply in the run
telemetry for Latin letters. Written before the listening-study codes are
ingested; the human part is computed by ``scripts/ingest_listening_study.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

MEMORY_LIMIT_MIB = 12_288
LATIN = re.compile(r"[A-Za-z]")


def qualification(loopback: dict[str, Any], coherence: dict[str, Any], replies: list[str]) -> dict[str, Any]:
    plain = loopback["conditions"]["plain"]
    acceptance = loopback["acceptance"]
    peak = (loopback.get("gpu_memory") or {}).get("peak_stack_mib")
    latin = [r for r in replies if LATIN.search(r)]
    checks = {
        "q1_memory": {"peak_stack_mib": peak, "limit_mib": MEMORY_LIMIT_MIB,
                      "passed": peak is not None and peak <= MEMORY_LIMIT_MIB},
        "q2_latency": {
            "plain_missing": plain["turns_without_first_audio"],
            "plain_mean_ms": plain["t_first_audio_ms"].get("mean"),
            "plain_p90_failures_as_late_ms": plain["t_first_audio_p90_failures_as_late_ms"],
            "elderly_p90_failures_as_late_ms": loopback["conditions"]["elderly"]["t_first_audio_p90_failures_as_late_ms"],
            "passed": bool(
                plain["turns_without_first_audio"] == 0
                and acceptance["plain_mean_le_500ms"]
                and acceptance["plain_p90_le_500ms_failures_counted"]
                and acceptance["elderly_p90_le_500ms_failures_counted"]
            ),
        },
        "q3_coherence": {"per_judge": coherence["criterion"]["per_judge"], "passed": bool(coherence["criterion"]["met"])},
        "q4_script": {"replies_checked": len(replies), "replies_with_latin": len(latin), "passed": not latin},
    }
    return {"checks": checks, "part2_passed": all(c["passed"] for c in checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loopback", type=Path, required=True)
    parser.add_argument("--coherence", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite the write-once receipt {args.out}")
    replies = [
        record["reply_text"]
        for path in args.telemetry
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        if record.get("reply_text")
    ]
    loopback = json.loads(args.loopback.read_text(encoding="utf-8"))
    coherence = json.loads(args.coherence.read_text(encoding="utf-8"))
    receipt = {
        "schema_version": 1,
        "experiment": "responder_switch_qualification",
        "protocol": "docs/RESPONDER_SWITCH_PROTOCOL.md",
        "configuration": {k: loopback["server_identity"].get(k) for k in
                          ("responder_model", "responder_revision", "responder_preset", "system_prompt_sha256",
                           "max_new_tokens")},
        "inputs_sha256": {
            "loopback": hashlib.sha256(args.loopback.read_bytes()).hexdigest(),
            "coherence": hashlib.sha256(args.coherence.read_bytes()).hexdigest(),
        },
        **qualification(loopback, coherence, replies),
        "human_part": "computed from the listening study by scripts/ingest_listening_study.py",
        "text_retained": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"part2_passed": receipt["part2_passed"],
                      "checks": {k: v["passed"] for k, v in receipt["checks"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
