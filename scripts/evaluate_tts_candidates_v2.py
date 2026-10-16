#!/usr/bin/env python3
"""TTS candidate comparison v2: synthesis, timing, and DNSMOS quality proxy.

Synthesizes the frozen reply texts with every Piper voice candidate (run with an
interpreter that has ``piper-tts``), records render time and first-clause
latency, scores each file with DNSMOS P.835 (SIG/BAK/OVRL; an automatic
signal-quality proxy, not a Persian naturalness rating), and writes per-arm
ASR panels so round-trip intelligibility can be measured with
``scripts/asr_predict.py`` using two independent listeners (NeMo, Seamless).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import wave
from pathlib import Path

import numpy as np

RATE = 16_000
DNSMOS_SAMPLES = 144_160  # 9.01 s at 16 kHz, as in the official DNSMOS evaluator
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return resample_poly(audio, dst // g, src // g).astype(np.float32)


def write(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


class Dnsmos:
    def __init__(self, model: Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    def __call__(self, audio: np.ndarray) -> dict[str, float]:
        if audio.size < DNSMOS_SAMPLES:
            audio = np.tile(audio, int(np.ceil(DNSMOS_SAMPLES / max(1, audio.size))))
        scores = []
        hop = RATE
        for start in range(0, max(1, audio.size - DNSMOS_SAMPLES + 1), hop):
            segment = audio[start : start + DNSMOS_SAMPLES][None].astype(np.float32)
            sig, bak, ovr = self.session.run(None, {"input_1": segment})[0][0]
            scores.append((P_SIG(sig), P_BAK(bak), P_OVR(ovr)))
        arr = np.asarray(scores)
        return {"sig": float(arr[:, 0].mean()), "bak": float(arr[:, 1].mean()), "ovrl": float(arr[:, 2].mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, required=True, help="JSONL with row_id and reply")
    parser.add_argument("--voices", type=Path, nargs="+", required=True)
    parser.add_argument("--dnsmos", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from piper import PiperVoice
    from piper.config import SynthesisConfig

    texts = [json.loads(line) for line in args.texts.read_text(encoding="utf-8").splitlines() if line]
    texts = [t for t in texts if t["reply"].strip()]
    dnsmos = Dnsmos(args.dnsmos)
    config = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)
    arms = {}
    for voice_path in args.voices:
        arm = "piper_" + voice_path.name.replace("fa_IR-", "").replace("-medium.onnx", "")
        voice = PiperVoice.load(str(voice_path))
        list(voice.synthesize("سلام", syn_config=config))  # warm-up
        panel, render, first, quality = [], [], [], []
        for row in texts:
            text = row["reply"]
            started = time.perf_counter()
            chunks = list(voice.synthesize(text, syn_config=config))
            elapsed = time.perf_counter() - started
            audio = resample(
                np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32),
                chunks[0].sample_rate,
                RATE,
            )
            first_words = " ".join(text.split()[:4])
            started = time.perf_counter()
            list(voice.synthesize(first_words, syn_config=config))
            first.append(1000 * (time.perf_counter() - started))
            path = args.work / arm / f"{row['row_id']}.wav"
            write(path, audio)
            render.append(elapsed / max(1e-6, audio.size / RATE))
            quality.append(dnsmos(audio))
            panel.append({"id": f"{arm}:{row['row_id']}", "audio_filepath": str(path), "text": text, "duration": round(audio.size / RATE, 3)})
        (args.work / f"panel_{arm}.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in panel), encoding="utf-8"
        )
        arms[arm] = {
            "voice_sha256": hashlib.sha256(voice_path.read_bytes()).hexdigest(),
            "texts": len(panel),
            "render_rtf_median": statistics.median(render),
            "first_four_words_ms_median": statistics.median(first),
            "first_four_words_ms_p90": sorted(first)[int(0.9 * (len(first) - 1))],
            "dnsmos_sig": statistics.fmean(q["sig"] for q in quality),
            "dnsmos_bak": statistics.fmean(q["bak"] for q in quality),
            "dnsmos_ovrl": statistics.fmean(q["ovrl"] for q in quality),
        }
        print(arm, json.dumps(arms[arm]))
    receipt = {
        "schema_version": 1,
        "texts_sha256": hashlib.sha256(args.texts.read_bytes()).hexdigest(),
        "dnsmos_model_sha256": hashlib.sha256(args.dnsmos.read_bytes()).hexdigest(),
        "dnsmos_note": "DNSMOS P.835 is an automatic signal-quality proxy trained on noisy-speech ratings; it is not a Persian naturalness MOS.",
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
