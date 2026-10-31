#!/usr/bin/env python3
"""ASR candidate comparison v3 (docs/ASR_CANDIDATE_COMPARISON_V3_PROTOCOL.md).

Scores the v2 and v3 prediction files together with the unchanged v2 scorer
(same normalizer, bootstrap and strata), checks that the same-session rerun of
the incumbent reproduces the v2 transcripts, and applies the v3 replacement
rule. The receipt keeps metrics and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INCUMBENT = "nemo_v4_ctc"
RERUN = "nemo_v4_ctc_rerun"
V3_ARMS = (
    "rade_ctc_3b",
    "nvidia_fa_fastconformer_ctc",
    "neuraspeech_900h",
    "buzzasr_persian",
    "whisper_persian_v4",
    "whisper_large_v3_turbo_fa",
    "c1tech_whisper_base_fa",
    "shenava_koochik",
    RERUN,
)
LATENCY_BUDGET_MS = 150.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replacement(arms: dict[str, Any]) -> dict[str, Any]:
    """The v3 rule: lowest-CER eligible arm, significantly better than NeMo v4 and not worse on seniors."""

    eligible = sorted(
        (a for a, r in arms.items()
         if a != RERUN and r["missing_predictions"] == 0
         and (r["latency_ms"]["short_clip_median"] or float("inf")) <= LATENCY_BUDGET_MS),
        key=lambda a: arms[a]["micro_cer"],
    )
    best = eligible[0] if eligible else None
    verdict: dict[str, Any] = {"latency_eligible_arms": eligible, "lowest_cer_eligible_arm": best}
    if best is None or best == INCUMBENT:
        verdict.update(replaces_nemo_v4=False, reason="the incumbent is the lowest-CER eligible arm")
        return verdict
    paired = arms[best]["paired_vs_incumbent"]["cer"]
    significant = paired["ci95_high"] < 0
    seniors_ok = arms[best]["strata"]["senior"]["micro_cer"] <= arms[INCUMBENT]["strata"]["senior"]["micro_cer"]
    verdict.update(
        paired_cer_difference=paired,
        significantly_better=significant,
        senior_cer_not_higher=seniors_ok,
        replaces_nemo_v4=bool(significant and seniors_ok),
    )
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--v2-pred-dir", type=Path, required=True)
    parser.add_argument("--v3-pred-dir", type=Path, required=True)
    parser.add_argument("--v2-receipt", type=Path, default=ROOT / "results/eval/asr_candidates_v2_psrb_human.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite the write-once receipt {args.out}")
    v2_receipt = json.loads(args.v2_receipt.read_text(encoding="utf-8"))
    v2_arms = list(v2_receipt["arms"])
    for arm in v2_arms:
        recorded = v2_receipt["arms"][arm]["predictions_sha256"]
        if sha256_file(args.v2_pred_dir / f"{arm}.jsonl") != recorded:
            raise SystemExit(f"v2 predictions of {arm} differ from the v2 receipt")
    with tempfile.TemporaryDirectory() as tmp:
        combined = Path(tmp)
        for arm in v2_arms:
            for suffix in (".jsonl", ".meta.json"):
                source = args.v2_pred_dir / f"{arm}{suffix}"
                if source.exists():
                    (combined / source.name).symlink_to(source)
        for arm in V3_ARMS:
            for suffix in (".jsonl", ".meta.json"):
                source = args.v3_pred_dir / f"{arm}{suffix}"
                if not source.exists() and suffix == ".jsonl":
                    raise SystemExit(f"missing v3 predictions for {arm}")
                if source.exists():
                    (combined / source.name).symlink_to(source)
        scored = combined / "scored.json"
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/evaluate_asr_candidates_v2.py"), "--panel", str(args.panel),
             "--panel-name", "psrb_human", "--pred-dir", str(combined), "--arms", *v2_arms, *V3_ARMS,
             "--out", str(scored)],
            check=True, stdout=subprocess.DEVNULL,
        )
        result = json.loads(scored.read_text(encoding="utf-8"))
    arms = result["arms"]
    read = lambda d, a: {r["id"]: r["hypothesis"] for r in map(json.loads, (d / f"{a}.jsonl").read_text(encoding="utf-8").splitlines())}  # noqa: E731
    rerun, original = read(args.v3_pred_dir, RERUN), read(args.v2_pred_dir, INCUMBENT)
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "docs/ASR_CANDIDATE_COMPARISON_V3_PROTOCOL.md",
        "protocol_sha256": sha256_file(ROOT / "docs/ASR_CANDIDATE_COMPARISON_V3_PROTOCOL.md"),
        "scorer": "scripts/evaluate_asr_candidates_v2.py (unchanged), wrapped by this script",
        "scorer_sha256": result["scorer_sha256"],
        "wrapper_sha256": sha256_file(Path(__file__)),
        "panel_name": "psrb_human",
        "panel_sha256": result["panel_sha256"],
        "panel_clips": result["panel_clips"],
        "normalizer": result["normalizer"],
        "incumbent": INCUMBENT,
        "v3_arms": list(V3_ARMS),
        "incumbent_rerun_identical_transcripts": sum(rerun.get(k) == v for k, v in original.items()),
        "incumbent_rerun_short_clip_median_ms": arms[RERUN]["latency_ms"]["short_clip_median"],
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "arms": arms,
        "selection": {
            **replacement(arms),
            "accuracy_reference_asr": min((a for a in arms if a != RERUN), key=lambda a: arms[a]["micro_cer"]),
        },
        "text_retained": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = {a: (round(arms[a]["micro_cer"], 4), round(arms[a]["strata"]["senior"]["micro_cer"], 4),
                 arms[a]["latency_ms"]["short_clip_median"]) for a in arms}
    print(json.dumps({"cer, senior_cer, short_median_ms": table, "selection": receipt["selection"],
                      "rerun_identical": receipt["incumbent_rerun_identical_transcripts"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
