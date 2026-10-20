#!/usr/bin/env python3
"""Compare split turns between two loopback runs of the same panels.

A planned utterance is *split* when the server answered it as more than one
turn (a pause inside it was taken as the end of the turn). Planned turns are
mapped to server turns exactly as in ``scripts/score_loopback_formal.py``. The
two runs are compared per utterance with an exact McNemar test on the
discordant pairs (``docs/END_OF_TURN_PROTOCOL.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("score_loopback_formal", ROOT / "scripts/score_loopback_formal.py")
assert _SPEC and _SPEC.loader
_SCORER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCORER)


def split_status(receipt_path: Path) -> dict[str, bool]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {row["id"]: row["server_turns"] > 1 for row in _SCORER.map_turns(receipt)}


def mcnemar_exact(before_only: int, after_only: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs."""

    n = before_only + after_only
    if n == 0:
        return 1.0
    k = min(before_only, after_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(before: dict[str, bool], after: dict[str, bool]) -> dict[str, Any]:
    ids = sorted(set(before) & set(after))
    before_only = sum(before[i] and not after[i] for i in ids)
    after_only = sum(after[i] and not before[i] for i in ids)
    return {
        "planned": len(ids),
        "split_before": sum(before[i] for i in ids),
        "split_after": sum(after[i] for i in ids),
        "split_only_before": before_only,
        "split_only_after": after_only,
        "mcnemar_exact_p": mcnemar_exact(before_only, after_only),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="directory with plain.json and elderly.json")
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=["plain", "elderly"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result: dict[str, Any] = {"conditions": {}}
    pooled_before: dict[str, bool] = {}
    pooled_after: dict[str, bool] = {}
    for condition in args.conditions:
        before = split_status(args.before / f"{condition}.json")
        after = split_status(args.after / f"{condition}.json")
        result["conditions"][condition] = compare(before, after)
        pooled_before.update({f"{condition}/{k}": v for k, v in before.items()})
        pooled_after.update({f"{condition}/{k}": v for k, v in after.items()})
        result["conditions"][condition]["receipt_sha256"] = {
            "before": hashlib.sha256((args.before / f"{condition}.json").read_bytes()).hexdigest(),
            "after": hashlib.sha256((args.after / f"{condition}.json").read_bytes()).hexdigest(),
        }
    result["pooled"] = compare(pooled_before, pooled_after)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite the write-once receipt {args.out}")
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
