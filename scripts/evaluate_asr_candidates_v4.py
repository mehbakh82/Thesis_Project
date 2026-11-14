#!/usr/bin/env python3
"""ASR candidate comparison v4 (docs/ASR_CANDIDATE_COMPARISON_V4_PROTOCOL.md).

Scores the v2, v3 and v4 prediction files on PSRB-human and the v4 FLEURS fa
predictions with the unchanged v2 scorer (same normalizer, bootstrap and
strata), checks the v2 and v3 prediction files against their receipts, and
applies the v4 replacement rule. The receipt keeps metrics and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/ASR_CANDIDATE_COMPARISON_V4_PROTOCOL.md"
INCUMBENT = "nemo_v4_ctc"
V3_RERUN = "nemo_v4_ctc_rerun"
V4_RERUN = "nemo_v4_ctc_rerun_v4"  # stored as nemo_v4_ctc_rerun.jsonl in the v4 prediction directory
RERUNS = (V3_RERUN, V4_RERUN)
V4_PSRB_ARMS = ("nemotron_fa_la1120", "nemotron_fa_la320", "nemo_v8_ep00_ctc")
FLEURS_ARMS = (
    INCUMBENT,
    "nemotron_fa_la1120",
    "nemotron_fa_la320",
    "nemo_v8_ep00_ctc",
    "nemo_v6_ctc",
    "shenava_koochik",
    "seamless_v5ft_direct",
)
LATENCY_BUDGET_MS = 150.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replacement(psrb: dict[str, Any], fleurs: dict[str, Any]) -> dict[str, Any]:
    """The v4 rule: the v3 rule on PSRB plus no significant FLEURS regression."""

    eligible = sorted(
        (
            a
            for a, r in psrb.items()
            if a not in RERUNS
            and r["missing_predictions"] == 0
            and (r["latency_ms"]["short_clip_median"] or float("inf")) <= LATENCY_BUDGET_MS
        ),
        key=lambda a: psrb[a]["micro_cer"],
    )
    best = eligible[0] if eligible else None
    verdict: dict[str, Any] = {"latency_eligible_arms": eligible, "lowest_cer_eligible_arm": best}
    if best is None or best == INCUMBENT:
        verdict.update(replaces_nemo_v4=False, reason="the incumbent is the lowest-CER eligible arm")
        return verdict
    paired = psrb[best]["paired_vs_incumbent"]["cer"]
    significant = paired["ci95_high"] < 0
    seniors_ok = psrb[best]["strata"]["senior"]["micro_cer"] <= psrb[INCUMBENT]["strata"]["senior"]["micro_cer"]
    fleurs_paired = (fleurs.get(best) or {}).get("paired_vs_incumbent", {}).get("cer")
    fleurs_ok = fleurs_paired is not None and fleurs_paired["ci95_low"] <= 0
    verdict.update(
        paired_cer_difference=paired,
        significantly_better=significant,
        senior_cer_not_higher=seniors_ok,
        fleurs_paired_cer_difference=fleurs_paired,
        fleurs_not_significantly_worse=fleurs_ok,
        replaces_nemo_v4=bool(significant and seniors_ok and fleurs_ok),
    )
    return verdict


def score(panel: Path, panel_name: str, files: dict[str, Path], tmp: Path) -> dict[str, Any]:
    """Run the unchanged v2 scorer over ``files`` (arm name -> prediction JSONL)."""

    combined = tmp / panel_name
    combined.mkdir()
    for arm, source in files.items():
        (combined / f"{arm}.jsonl").symlink_to(source)
        meta = source.with_suffix(".meta.json")
        if meta.exists():
            (combined / f"{arm}.meta.json").symlink_to(meta)
    out = combined / "scored.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/evaluate_asr_candidates_v2.py"), "--panel", str(panel),
         "--panel-name", panel_name, "--pred-dir", str(combined), "--arms", *files, "--out", str(out)],
        check=True, stdout=subprocess.DEVNULL,
    )
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psrb-panel", type=Path, required=True)
    parser.add_argument("--fleurs-panel", type=Path, required=True)
    parser.add_argument("--v2-pred-dir", type=Path, required=True)
    parser.add_argument("--v3-pred-dir", type=Path, required=True)
    parser.add_argument("--v4-pred-dir", type=Path, required=True)
    parser.add_argument("--v4-fleurs-pred-dir", type=Path, required=True)
    parser.add_argument("--v3-receipt", type=Path, default=ROOT / "results/eval/asr_candidates_v3_psrb_human.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite the write-once receipt {args.out}")
    v3 = json.loads(args.v3_receipt.read_text(encoding="utf-8"))
    earlier = {arm: (args.v3_pred_dir if arm in v3["v3_arms"] else args.v2_pred_dir) / f"{arm}.jsonl"
               for arm in v3["arms"]}
    for arm, path in earlier.items():
        if sha256_file(path) != v3["arms"][arm]["predictions_sha256"]:
            raise SystemExit(f"predictions of {arm} differ from the v3 receipt")
    psrb_files = dict(earlier)
    for arm in V4_PSRB_ARMS:
        psrb_files[arm] = args.v4_pred_dir / f"{arm}.jsonl"
    psrb_files[V4_RERUN] = args.v4_pred_dir / f"{V3_RERUN}.jsonl"
    fleurs_files = {arm: args.v4_fleurs_pred_dir / f"{arm}.jsonl" for arm in FLEURS_ARMS}
    for path in (*psrb_files.values(), *fleurs_files.values()):
        if not path.is_file():
            raise SystemExit(f"missing predictions: {path}")
    with tempfile.TemporaryDirectory() as tmp:
        psrb = score(args.psrb_panel, "psrb_human", psrb_files, Path(tmp))
        fleurs = score(args.fleurs_panel, "fleurs_fa_test", fleurs_files, Path(tmp))
    read = lambda p: {r["id"]: r["hypothesis"] for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}  # noqa: E731
    rerun, original = read(psrb_files[V4_RERUN]), read(psrb_files[INCUMBENT])
    fleurs_v6 = fleurs["arms"]["nemo_v6_ctc"]["paired_vs_incumbent"]["cer"]
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": "docs/ASR_CANDIDATE_COMPARISON_V4_PROTOCOL.md",
        "protocol_sha256": sha256_file(PROTOCOL),
        "scorer": "scripts/evaluate_asr_candidates_v2.py (unchanged), wrapped by this script",
        "scorer_sha256": psrb["scorer_sha256"],
        "wrapper_sha256": sha256_file(Path(__file__)),
        "normalizer": psrb["normalizer"],
        "incumbent": INCUMBENT,
        "v4_arms": [*V4_PSRB_ARMS, V4_RERUN],
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "psrb": {
            "panel_sha256": psrb["panel_sha256"],
            "panel_clips": psrb["panel_clips"],
            "incumbent_rerun_identical_transcripts": sum(rerun.get(k) == v for k, v in original.items()),
            "incumbent_rerun_short_clip_median_ms": psrb["arms"][V4_RERUN]["latency_ms"]["short_clip_median"],
            "arms": psrb["arms"],
        },
        "fleurs": {
            "panel_sha256": fleurs["panel_sha256"],
            "panel_clips": fleurs["panel_clips"],
            "arms": fleurs["arms"],
            "nemo_v6_vs_v4_cer": fleurs_v6,
            "nemo_v6_significantly_better": fleurs_v6["ci95_high"] < 0,
        },
        "selection": replacement(psrb["arms"], fleurs["arms"]),
        "text_retained": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = {
        a: {
            "psrb_cer": round(psrb["arms"][a]["micro_cer"], 4) if a in psrb["arms"] else None,
            "fleurs_cer": round(fleurs["arms"][a]["micro_cer"], 4) if a in fleurs["arms"] else None,
            "short_median_ms": psrb["arms"][a]["latency_ms"]["short_clip_median"] if a in psrb["arms"] else None,
        }
        for a in sorted({*psrb["arms"], *fleurs["arms"]})
    }
    print(json.dumps({"arms": table, "selection": receipt["selection"], "fleurs_v6_vs_v4": fleurs_v6}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
