#!/usr/bin/env python3
"""TTS v3: DNSMOS for the new arms and one synthesis receipt for all arms.

The two Piper arms reuse the TTS v2 renderings (deterministic Piper with the
same voice files and the same texts, so the audio is identical); their panels,
ASR-listener outputs, timing, and DNSMOS are linked from the v2 work directory.
New arms (``synthesis_<arm>.json`` + ``panel_<arm>.jsonl`` written by their
synthesis script) get DNSMOS here with the same model as v2. The result is a
``synthesis.json`` in the v3 work directory that
``scripts/score_tts_candidates_v2.py`` scores like v2 (two-listener CER against
``piper_mana``). Run with an interpreter that has onnxruntime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_tts_candidates_v2 import Dnsmos  # noqa: E402

PIPER_ARMS = ("piper_mana", "piper_ganji_adabi")


def read(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-work", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--new-arms", nargs="+", required=True)
    parser.add_argument("--dnsmos", type=Path, required=True)
    args = parser.parse_args()

    v2 = json.loads((args.v2_work / "synthesis.json").read_text(encoding="utf-8"))
    arms: dict[str, dict] = {}
    for arm in PIPER_ARMS:
        arms[arm] = dict(v2["arms"][arm], reused_from="TTS v2 (deterministic, identical inputs)", device="CPU")
        for name in (f"panel_{arm}.jsonl",):
            link = args.work / name
            if not link.exists():
                link.symlink_to(args.v2_work / name)
        for folder in ("asr_nemo", "asr_seamless"):
            (args.work / folder).mkdir(exist_ok=True)
            link = args.work / folder / f"{arm}.jsonl"
            if not link.exists():
                link.symlink_to(args.v2_work / folder / f"{arm}.jsonl")
    dnsmos = Dnsmos(args.dnsmos)
    for arm in args.new_arms:
        summary = json.loads((args.work / f"synthesis_{arm}.json").read_text(encoding="utf-8"))
        panel = [json.loads(line) for line in (args.work / f"panel_{arm}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        quality = [dnsmos(read(row["audio_filepath"])) for row in panel]
        arms[arm] = {
            **summary,
            "dnsmos_sig": statistics.fmean(q["sig"] for q in quality),
            "dnsmos_bak": statistics.fmean(q["bak"] for q in quality),
            "dnsmos_ovrl": statistics.fmean(q["ovrl"] for q in quality),
        }
        print(arm, round(arms[arm]["dnsmos_ovrl"], 3))
    receipt = {
        "schema_version": 1,
        "experiment": "tts_candidates_v3",
        "protocol": "docs/TTS_CANDIDATE_COMPARISON_V3_PROTOCOL.md",
        "texts_sha256": v2["texts_sha256"],
        "dnsmos_model_sha256": hashlib.sha256(args.dnsmos.read_bytes()).hexdigest(),
        "dnsmos_note": v2["dnsmos_note"],
        "arms": arms,
    }
    (args.work / "synthesis.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
