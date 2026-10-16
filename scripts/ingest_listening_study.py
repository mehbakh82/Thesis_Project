#!/usr/bin/env python3
"""Decode listening-study result codes into an aggregate evidence receipt.

Each line of ``--codes`` is one participant's code from the study page. Codes
are validated (consent, age group, integer 1-5 ratings, study id); a participant
failing the attention check is excluded and counted. The receipt reports, per
clip condition, the naturalness MOS with a participant-bootstrap 95% interval,
the same by age group, the interaction ratings, willingness to use, and the
number of participants aged 60+. Free-text notes are kept only in the private
work file, never in the receipt.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def decode(code: str) -> dict[str, Any]:
    return json.loads(base64.b64decode(code.strip()).decode("utf-8"))


def boot_ci(per_participant: list[float], draws: int = 5000) -> list[float] | None:
    if len(per_participant) < 2:
        return None
    rng = random.Random(20261016)
    means = sorted(
        statistics.fmean(rng.choice(per_participant) for _ in per_participant) for _ in range(draws)
    )
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    conditions = {c["id"]: c.get("condition") for c in spec["clips"]}
    attention = {c["id"]: int(c["expected"]) for c in spec["clips"] if c.get("attention")}
    accepted, excluded = [], defaultdict(int)
    for line in args.codes.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            answer = decode(line)
        except (ValueError, UnicodeDecodeError):
            excluded["undecodable"] += 1
            continue
        if answer.get("study") != spec["study_id"] or answer.get("consent") is not True:
            excluded["wrong_study_or_no_consent"] += 1
            continue
        ratings = answer.get("ratings") or {}
        if any(type(v) is not int or not 1 <= v <= 5 for v in ratings.values()) or set(ratings) != set(conditions):
            excluded["incomplete_or_invalid_ratings"] += 1
            continue
        if any(ratings[cid] != expected for cid, expected in attention.items()):
            excluded["failed_attention_check"] += 1
            continue
        accepted.append(answer)
    args.private_out.write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")

    by_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted({c for c in conditions.values() if c}):
        clip_ids = [cid for cid, c in conditions.items() if c == condition]
        per_participant = [statistics.fmean(a["ratings"][cid] for cid in clip_ids) for a in accepted]
        by_age: dict[str, float] = {}
        for age in ("under_40", "40_59", "60plus"):
            group = [statistics.fmean(a["ratings"][cid] for cid in clip_ids) for a in accepted if a.get("age") == age]
            if group:
                by_age[age] = statistics.fmean(group)
        by_condition[condition] = {
            "clips": len(clip_ids),
            "naturalness_mos": statistics.fmean(per_participant) if per_participant else None,
            "naturalness_mos_ci95": boot_ci(per_participant),
            "by_age_group": by_age,
        }
    interactions: dict[str, Any] = {}
    for item in spec["interactions"]:
        rows = [a["interactions"].get(item["id"]) for a in accepted if a.get("interactions", {}).get(item["id"])]
        summary: dict[str, Any] = {"responses": len(rows)}
        for key in ("responsiveness", "interruption", "satisfaction"):
            values = [r[key] for r in rows if isinstance(r.get(key), int)]
            if values:
                summary[f"{key}_mean"] = statistics.fmean(values)
        uses = [r.get("would_use") for r in rows]
        if uses:
            summary["would_use_yes_share"] = sum(u == "yes" for u in uses) / len(uses)
        interactions[item["id"]] = summary
    ages = [a.get("age") for a in accepted]
    receipt = {
        "schema_version": 1,
        "evidence_class": "human_remote_listening_study",
        "study_id": spec["study_id"],
        "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "participants_accepted": len(accepted),
        "participants_aged_60_plus": sum(age == "60plus" for age in ages),
        "age_groups": {g: ages.count(g) for g in ("under_40", "40_59", "60plus")},
        "native_persian": sum(a.get("native") == "yes" for a in accepted),
        "excluded": dict(excluded),
        "naturalness_by_condition": by_condition,
        "interactions": interactions,
        "free_text_retained_in_receipt": False,
        "definition_requirement": {
            "participants_5_to_10": 5 <= len(accepted) <= 10,
            "at_least_two_aged_60_plus": sum(age == "60plus" for age in ages) >= 2,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("participants_accepted", "participants_aged_60_plus", "excluded")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
