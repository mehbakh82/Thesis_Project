#!/usr/bin/env python3
"""Responder size comparison (docs/RESPONDER_SIZE_COMPARISON_PROTOCOL.md).

Stages:
  generate  stream one reply per v2 panel row from a vLLM server, exactly as the
            deployed path requests it (greedy, Persian-only structured output,
            thinking off), recording time to first and last token;
  judge     score every reply with one v2 judge, blind and in random order;
  score     per-judge means, paired comparisons against the deployed responder
            and against the best 4B, and the frozen decision rules.

Reply text stays in the private work directory; the receipt keeps scores,
timings and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@cache
def _v2() -> Any:
    """The v2 evaluator (panel I/O, prompts, judges); loaded on first use."""

    spec = importlib.util.spec_from_file_location("responder_v2", ROOT / "scripts/evaluate_responder_candidates_v2.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


SEED = "20261028-responder-size"
INCUMBENT = "qwen3_4b_prompt_v2"
REFERENCE = "qwen35_4b_elderly_v1"
V2_PANEL_SHA256 = "c97c1e1761e6706976e9ad2afa681e1a9203843cbe135f8e3380584977619993"
MAX_NEW_TOKENS = 96
LARGER = (
    "qwen35_9b_elderly_v1",
    "gemma4_12b_elderly_v1",
    "gemma4_26b_a4b_elderly_v1",
    "qwen35_35b_a3b_elderly_v1",
    "gemma4_31b_elderly_v1",
)
ARMS = (INCUMBENT, REFERENCE, "gemma4_e4b_elderly_v1", *LARGER)


def prompt_for(arm: str) -> str:
    return _v2().system_prompt("qwen4b_v2" if arm == INCUMBENT else "elderly_v1")


def stream_reply(endpoint: str, model: str, system: str, user: str) -> tuple[str, float | None, float]:
    from thesis_s2s.runtime.streaming import PERSIAN_ONLY_REGEX

    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "stream": True,
        "structured_outputs": {"regex": PERSIAN_ONLY_REGEX},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first = None
    parts: list[str] = []
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - local server
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            choices = json.loads(line[5:]).get("choices") or []
            delta = (choices[0].get("delta") or {}).get("content") if choices else None
            if delta:
                if first is None:
                    first = time.perf_counter() - started
                parts.append(delta)
    return "".join(parts).strip(), first, time.perf_counter() - started


def generate(work: Path, arm: str, endpoint: str, model: str) -> dict[str, Any]:
    panel = _v2().read_jsonl(work / "panel.jsonl")
    system = prompt_for(arm)
    stream_reply(endpoint, model, system, panel[0]["text"])  # warm-up, not recorded
    out = []
    for row in panel:
        reply, first, total = stream_reply(endpoint, model, system, row["text"])
        out.append({"row_id": row["row_id"], "arm": arm, "reply": reply,
                    "first_token_s": round(first, 4) if first is not None else None, "total_s": round(total, 4)})
    _v2().write_jsonl(work / f"replies_{arm}.jsonl", out)
    return {"arm": arm, "rows": len(out), "empty_replies": sum(not r["reply"] for r in out)}


def judge(work: Path, judge_name: str, arms: list[str], endpoint: str, cache: Path) -> dict[str, Any]:
    panel = {r["row_id"]: r for r in _v2().read_jsonl(work / "panel.jsonl")}
    jobs = [(arm, row) for arm in arms for row in _v2().read_jsonl(work / f"replies_{arm}.jsonl")]
    random.Random(f"{SEED}:{judge_name}").shuffle(jobs)
    out_path = work / f"judgments_{judge_name}.jsonl"
    done = {(r["arm"], r["row_id"]) for r in _v2().read_jsonl(out_path)} if out_path.is_file() else set()
    failures = 0
    with out_path.open("a", encoding="utf-8") as sink:
        for arm, row in jobs:
            if (arm, row["row_id"]) in done:
                continue
            user = panel[row["row_id"]]["text"]
            try:
                raw = (_v2().judge_qwen38(user, row["reply"], endpoint) if judge_name == "qwen38"
                       else _v2().judge_gemini(user, row["reply"], cache))
                scores = {k: int(raw[k]) for k in _v2().SCORE_KEYS}
                if not all(0 <= v <= 4 for v in scores.values()):
                    raise ValueError("score out of range")
            except Exception as exc:  # noqa: BLE001 - counted, retried on resume
                failures += 1
                print(f"judge failure {arm}/{row['row_id']}: {exc}", file=sys.stderr)
                continue
            sink.write(json.dumps({"arm": arm, "row_id": row["row_id"], "judge": judge_name, **scores}) + "\n")
            sink.flush()
    return {"judge": judge_name, "failures": failures}


def paired(table: dict[tuple[str, str], dict[str, int]], arm: str, base: str, rows: list[str], rng: random.Random) -> dict[str, Any]:
    ids = [r for r in rows if (arm, r) in table and (base, r) in table]
    gains = [table[(arm, r)]["relevance"] - table[(base, r)]["relevance"] for r in ids]
    boots = sorted(statistics.fmean(rng.choice(gains) for _ in gains) for _ in range(5000)) if gains else []
    return {
        "rows": len(ids),
        "relevance_gain": statistics.fmean(gains) if gains else None,
        "relevance_gain_ci95": [boots[124], boots[4874]] if boots else None,
        "coherence_gain": statistics.fmean(table[(arm, r)]["coherence"] - table[(base, r)]["coherence"] for r in ids) if ids else None,
        "appropriateness_gain": statistics.fmean(
            table[(arm, r)]["elderly_appropriateness"] - table[(base, r)]["elderly_appropriateness"] for r in ids
        ) if ids else None,
        "row_win_rate": sum(g > 0 for g in gains) / len(gains) if gains else None,
        "row_loss_rate": sum(g < 0 for g in gains) / len(gains) if gains else None,
    }


def meets(p: dict[str, Any]) -> bool:
    return (
        (p.get("relevance_gain") or 0) >= 0.25
        and (p.get("coherence_gain") or 0) >= 0
        and (p.get("appropriateness_gain") or 0) >= 0
        and (p.get("row_win_rate") or 0) >= 0.55 - 1e-9
    )


def timing(work: Path, arm: str) -> dict[str, Any]:
    rows = _v2().read_jsonl(work / f"replies_{arm}.jsonl")
    firsts = sorted(r["first_token_s"] for r in rows if r["first_token_s"] is not None)
    totals = sorted(r["total_s"] for r in rows)
    return {
        "median_first_token_ms": round(1000 * statistics.median(firsts), 1) if firsts else None,
        "p90_first_token_ms": round(1000 * firsts[int(0.9 * (len(firsts) - 1))], 1) if firsts else None,
        "median_total_ms": round(1000 * statistics.median(totals), 1),
        "empty_replies": sum(not r["reply"] for r in rows),
        "latin_letter_replies": sum(any("a" <= c.lower() <= "z" for c in r["reply"]) for r in rows),
    }


def score(work: Path, judges: list[str], fit: dict[str, Any]) -> dict[str, Any]:
    panel = {r["row_id"]: r for r in _v2().read_jsonl(work / "panel.jsonl")}
    arms = [a for a in ARMS if (work / f"replies_{a}.jsonl").is_file()]
    report: dict[str, Any] = {"arms": arms, "fit_24gb": fit, "judges": {}, "timing": {a: timing(work, a) for a in arms}}
    tables: dict[str, dict[tuple[str, str], dict[str, int]]] = {}
    for judge_name in judges:
        path = work / f"judgments_{judge_name}.jsonl"
        if not path.is_file():
            continue
        table = {(r["arm"], r["row_id"]): r for r in _v2().read_jsonl(path)}
        tables[judge_name] = table
        entry: dict[str, Any] = {}
        for arm in arms:
            means: dict[str, Any] = {}
            for name in ("elderly_requests", "natural_turns", "all"):
                ids = [rid for rid, r in panel.items() if name == "all" or r["panel"] == name]
                rows = [table[(arm, rid)] for rid in ids if (arm, rid) in table]
                means[name] = {k: statistics.fmean(r[k] for r in rows) if rows else None for k in _v2().SCORE_KEYS}
                means[name]["n"] = len(rows)
            rng = random.Random(f"{SEED}:{judge_name}:{arm}")
            if arm != INCUMBENT:
                means["vs_deployed"] = paired(table, arm, INCUMBENT, list(panel), rng)
            if arm in LARGER:
                means["vs_best_4b"] = paired(table, arm, REFERENCE, list(panel), rng)
            entry[arm] = means
        report["judges"][judge_name] = entry
    if len(tables) == 2:
        a, b = tables.values()
        keys = sorted(set(a) & set(b))
        report["inter_judge_relevance_exact_agreement"] = sum(a[k]["relevance"] == b[k]["relevance"] for k in keys) / len(keys)
    both = len(tables) == 2
    decisions: dict[str, Any] = {}
    for arm in LARGER:
        if arm not in arms:
            decisions[arm] = {"judged": False, "fits_24gb": bool(fit.get(arm, {}).get("fits")), "beats_best_4b": False}
            continue
        per = {j: report["judges"][j][arm] for j in tables}
        decisions[arm] = {
            "judged": True,
            "fits_24gb": bool(fit.get(arm, {}).get("fits")),
            "beats_best_4b": both and all(meets(p["vs_best_4b"]) for p in per.values()),
            "beats_deployed": both and all(meets(p["vs_deployed"]) for p in per.values()),
            "mean_relevance_gain_vs_best_4b": statistics.fmean(p["vs_best_4b"]["relevance_gain"] for p in per.values()),
        }
    winners = [a for a, d in decisions.items() if d["beats_best_4b"] and d["fits_24gb"]]
    report["decisions"] = decisions
    report["profile_24gb_candidate"] = max(winners, key=lambda a: decisions[a]["mean_relevance_gain_vs_best_4b"]) if winners else None
    report["size_buys_quality"] = bool(winners)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "judge", "score"), required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8031")
    parser.add_argument("--served-model", default="candidate")
    parser.add_argument("--judge", choices=("qwen38", "gemini"))
    parser.add_argument("--judge-endpoint", default="http://127.0.0.1:8003")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--fit", type=Path, help="JSON of per-arm 24 GB fit measurements")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    panel_sha = _v2().sha256_text((args.work / "panel.jsonl").read_text(encoding="utf-8"))
    if panel_sha != V2_PANEL_SHA256:
        raise SystemExit(f"panel hash {panel_sha} differs from the frozen v2 panel")
    if args.stage == "generate":
        result = generate(args.work, args.arm, args.endpoint, args.served_model)
    elif args.stage == "judge":
        arms = [a for a in ARMS if (args.work / f"replies_{a}.jsonl").is_file()]
        result = judge(args.work, args.judge, arms, args.judge_endpoint, args.cache or args.work / "gemini_cache")
    else:
        if args.out and args.out.exists():
            raise SystemExit(f"refusing to overwrite the write-once receipt {args.out}")
        fit = json.loads(args.fit.read_text(encoding="utf-8")) if args.fit else {}
        result = score(args.work, ["qwen38", "gemini"], fit)
        result.update(
            schema_version=1,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            protocol="docs/RESPONDER_SIZE_COMPARISON_PROTOCOL.md",
            evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            panel_sha256=panel_sha,
            elderly_prompt_sha256=_v2().sha256_text(_v2().ELDERLY_PROMPT_V1),
            text_retained=False,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
