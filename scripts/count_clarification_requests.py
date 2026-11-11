#!/usr/bin/env python3
"""Exploratory: how often does a responder ask the user to repeat in the interruption-coherence runs?

The responder v3 deployment qualification (docs/RESPONDER_V3_RESULT.md) failed its coherence-after-interruption
check: the replies after a late interruption were judged less relevant than the controls. Reading the runs
suggested that Gemma 4 12B asks the user to repeat when the recognizer garbles the follow-up, where Qwen3.5-4B
guesses. This script counts such replies with a fixed Persian pattern, per run, condition and interruption timing,
using the same follow-up turn as ``scripts/run_interruption_coherence.py score``, and reports the follow-up
transcript CER of the replies that ask and of those that do not. The receipt keeps counts only (no text) and is
exploratory: it was written after the qualification outcome was known and decides nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from thesis_s2s.eval.asr_normalize import NORMALIZER_VERSION, error_counts  # noqa: E402

# "I did not understand", "say it again", "repeat": the phrasings the responders use to ask for a repeat.
REPEAT_REQUEST = re.compile(r"متوجه[^.؟!]*نشدم|دوباره[^.؟!]*(?:بگویید|بگید|بفرمایید|تکرار)|تکرار\s*(?:کنید|کن)")


def follow_ups(run_dir: Path, scenarios: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    runs = [json.loads(line) for line in (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    raw = (run_dir / "telemetry.jsonl").read_bytes()
    out = []
    for run in runs:
        turns = [json.loads(line) for line in raw[run["offset_before"]:run["offset_after"]].decode("utf-8").splitlines()
                 if line]
        turns = sorted((t for t in turns if not t.get("speculation_cancelled")), key=lambda t: t["turn"])
        follow = turns[1] if len(turns) > 1 else {}
        sc = scenarios[run["scenario"]]
        transcript = follow.get("transcript") or ""
        counts = error_counts(sc["follow_up"], transcript) if transcript else None
        out.append({"scenario": run["scenario"], "condition": run["condition"], "timing": sc["timing"],
                    "asks_to_repeat": bool(REPEAT_REQUEST.search(follow.get("reply_text") or "")),
                    "follow_up_cer": counts["char_errors"] / max(1, counts["reference_chars"]) if counts else None})
    return out


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for condition in ("interrupt", "control"):
        part = [r for r in rows if r["condition"] == condition]
        table[condition] = {"replies": len(part), "asks_to_repeat": sum(r["asks_to_repeat"] for r in part),
                            "by_timing": {t: {"replies": sum(r["timing"] == t for r in part),
                                              "asks_to_repeat": sum(r["asks_to_repeat"] for r in part if r["timing"] == t)}
                                          for t in ("early", "mid", "late")}}
    for flag, name in ((True, "follow_up_cer_median_when_asking"), (False, "follow_up_cer_median_otherwise")):
        cers = [r["follow_up_cer"] for r in rows if r["asks_to_repeat"] is flag and r["follow_up_cer"] is not None]
        table[name] = round(statistics.median(cers), 3) if cers else None
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIR"), required=True,
                        help="a coherence run directory holding runs.jsonl and telemetry.jsonl")
    parser.add_argument("--scenarios", type=Path, default=ROOT / "configs/interruption_coherence_scenarios.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    scenarios = {r["scenario"]: r for r in (json.loads(line) for line in
                                             args.scenarios.read_text(encoding="utf-8").splitlines() if line)}
    runs = {}
    for label, directory in args.run:
        path = Path(directory)
        runs[label] = {"runs_sha256": hashlib.sha256((path / "runs.jsonl").read_bytes()).hexdigest(),
                       "telemetry_sha256": hashlib.sha256((path / "telemetry.jsonl").read_bytes()).hexdigest(),
                       **summary(follow_ups(path, scenarios))}
    receipt = {"schema_version": 1, "experiment": "coherence_repeat_requests", "label": "exploratory, post hoc",
               "protocol": "docs/RESPONDER_V3_RESULT.md", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scenarios_sha256": hashlib.sha256(args.scenarios.read_bytes()).hexdigest(),
               "pattern": REPEAT_REQUEST.pattern, "normalizer": NORMALIZER_VERSION, "runs": runs, "text_retained": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runs, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
