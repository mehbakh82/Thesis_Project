#!/usr/bin/env python3
"""Score TTS candidate comparison v2: round-trip intelligibility receipt.

Reads the synthesis receipt of ``scripts/evaluate_tts_candidates_v2.py``, the
per-arm panels it wrote, and the transcripts of every synthesized clip by two
independent ASR listeners (NeMo v4 CTC and Seamless v5ft, from
``scripts/asr_predict.py``). Writes one receipt with, per voice: corpus CER and
WER per listener (punctuation-free ``fa-asr-score-v1`` normalization), the
two-listener mean CER, a paired bootstrap interval of the mean per-clip CER
difference against the incumbent voice, and the synthesis timing and DNSMOS
proxy numbers. No reply text or transcript is copied into the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.eval.asr_normalize import NORMALIZER_VERSION, error_counts  # noqa: E402

INCUMBENT = "piper_mana"
LISTENERS = {"nemo_v4_ctc": "asr_nemo", "seamless_v5ft_direct": "asr_seamless"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def per_clip_cer(panel: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    hyps = {p["id"]: p for p in predictions}
    out = {}
    for row in panel:
        prediction = hyps[row["id"]]
        if prediction.get("error"):
            raise SystemExit(f"listener failed on {row['id']}")
        out[row["id"].split(":", 1)[1]] = error_counts(row["text"], prediction["hypothesis"])
    return out


def paired_bootstrap(diffs: list[float], draws: int = 10_000) -> list[float]:
    rng = random.Random(20261016)
    means = sorted(statistics.fmean(rng.choice(diffs) for _ in diffs) for _ in range(draws))
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="directory with panel_*.jsonl and asr_*/")
    parser.add_argument("--synthesis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite write-once receipt {args.out}")

    synthesis = json.loads(args.synthesis.read_text(encoding="utf-8"))
    clips: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    hashes: dict[str, str] = {}
    for arm in synthesis["arms"]:
        panel_path = args.work / f"panel_{arm}.jsonl"
        panel = load_jsonl(panel_path)
        hashes[f"panel_{arm}"] = hashlib.sha256(panel_path.read_bytes()).hexdigest()
        clips[arm] = {}
        for listener, folder in LISTENERS.items():
            prediction_path = args.work / folder / f"{arm}.jsonl"
            hashes[f"{folder}/{arm}"] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
            clips[arm][listener] = per_clip_cer(panel, load_jsonl(prediction_path))

    def clip_mean_cer(arm: str, row: str) -> float:
        return statistics.fmean(
            clips[arm][listener][row]["char_errors"] / max(1, clips[arm][listener][row]["reference_chars"])
            for listener in LISTENERS
        )

    rows = sorted(clips[INCUMBENT][next(iter(LISTENERS))])
    arms: dict[str, Any] = {}
    for arm, numbers in synthesis["arms"].items():
        entry: dict[str, Any] = dict(numbers)
        for listener in LISTENERS:
            counts = clips[arm][listener].values()
            entry[f"cer_{listener}"] = sum(c["char_errors"] for c in counts) / sum(c["reference_chars"] for c in counts)
            entry[f"wer_{listener}"] = sum(c["word_errors"] for c in counts) / sum(c["reference_words"] for c in counts)
        entry["cer_two_listener_mean"] = statistics.fmean(entry[f"cer_{listener}"] for listener in LISTENERS)
        if arm != INCUMBENT:
            diffs = [clip_mean_cer(arm, row) - clip_mean_cer(INCUMBENT, row) for row in rows]
            entry["clip_cer_minus_incumbent_mean"] = statistics.fmean(diffs)
            entry["clip_cer_minus_incumbent_ci95"] = paired_bootstrap(diffs)
        arms[arm] = entry
    ranking = sorted(arms, key=lambda a: arms[a]["cer_two_listener_mean"])
    receipt = {
        "schema_version": 1,
        "experiment": "tts_candidates_v2",
        "incumbent": INCUMBENT,
        "listeners": list(LISTENERS),
        "normalizer": NORMALIZER_VERSION,
        "texts": len(rows),
        "texts_sha256": synthesis["texts_sha256"],
        "dnsmos_model_sha256": synthesis["dnsmos_model_sha256"],
        "dnsmos_note": synthesis["dnsmos_note"],
        "input_sha256": hashes,
        "arms": arms,
        "ranking_by_two_listener_cer": ranking,
        "ranking_by_dnsmos_ovrl": sorted(arms, key=lambda a: -arms[a]["dnsmos_ovrl"]),
        "text_retained": False,
        "claim_boundary": (
            "Round-trip ASR intelligibility and DNSMOS are automatic proxies; Persian naturalness "
            "is measured only by the listening study."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({a: round(arms[a]["cer_two_listener_mean"], 4) for a in ranking}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
