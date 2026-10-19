#!/usr/bin/env python3
"""TTS v3: synthesize the frozen reply texts with the lab VITS model (star TTS).

Run with an interpreter where the lab repository's Coqui fork (``TTS``), its
Persian front-end dependencies (hazm, kenlm, the ezafe checkpoint) and the
espeak-ng binary are available. The text pipeline replicates the lab's Triton
service (``triton_models/tts_model/1/model.py``): digits mapped to Persian,
``eng2fa.map_characters``, the service's sentence splitting, one
``Synthesizer.tts`` call per sentence, then 22.05 → 16 kHz resampling
(``kaiser_best``) and silence trimming. Runs on CPU, as the service does.

With ``--plus`` (the directory containing ``star_tts_plus``) and ``--options``
(a JSON object of ``star_tts_plus.engine.Options`` fields, without ``cuda``) it
synthesizes with the optimized configuration instead, seeding every text with
the same per-row seed as the development runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

RATE = 16_000


def split_sentences(text: str) -> list[str]:
    """The lab service's ``_split_sentences``."""

    sents = [s.strip() for s in re.split(r"\.|،", text) if len(s.strip()) > 0]
    normal: list[str] = []
    for s in sents:
        if len(s.split()) < 10:
            normal.append(s)
        else:
            tokens = s.split()
            for i in range(0, len(tokens), 10):
                normal.append(" ".join(tokens[i : i + 10]))
    for i in range(len(normal) - 2, -1, -1):
        if len(normal[i].split() + normal[i + 1].split()) < 10:
            normal[i] = normal[i] + " " + normal.pop(i + 1)
    return normal


def write(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--arm", default="star_vits")
    parser.add_argument("--cuda", action="store_true", help="run the VITS model on the GPU")
    parser.add_argument("--timing-only", action="store_true", help="measure latency only; write no audio or panel")
    parser.add_argument("--plus", type=Path, help="directory containing star_tts_plus (optimized configuration)")
    parser.add_argument("--options", type=json.loads, default=None, help="star_tts_plus Options as JSON")
    args = parser.parse_args()
    if (args.plus is None) != (args.options is None):
        parser.error("--plus and --options go together")

    sys.path.insert(0, str(args.repo))
    os.chdir(args.repo)  # the heteronym module opens its data files by relative path
    import librosa
    import torch
    from eng2fa import map_characters
    from TTS.utils.synthesizer import Synthesizer

    torch.manual_seed(0)
    if args.plus is not None:
        sys.path.insert(0, str(args.plus))
        from star_tts_plus.engine import Options, StarTTS

        engine = StarTTS(args.repo, Options(cuda=args.cuda, **args.options))
        synth = engine.tts
    else:
        synthesizer = Synthesizer(
            tts_checkpoint=str(args.repo / "models/model.pth"),
            tts_config_path=str(args.repo / "models/config.json"),
            use_cuda=args.cuda,
        )
        digits = str.maketrans("12345678901234567890", "۱۲۳۴۵۶۷۸۹۰١٢٣٤٥٦٧٨٩٠")

        def synth(text: str) -> np.ndarray:
            processed = map_characters(text.translate(digits))
            wavs = [np.asarray(synthesizer.tts(s), dtype=np.float32) for s in split_sentences(processed)]
            wav = np.concatenate(wavs) if wavs else np.zeros(0, np.float32)
            wav = librosa.resample(wav, orig_sr=22050, target_sr=RATE, res_type="kaiser_best")
            wav, _ = librosa.effects.trim(wav)
            return wav.astype(np.float32)

    texts = [json.loads(line) for line in args.texts.read_text(encoding="utf-8").splitlines() if line]
    texts = [t for t in texts if t["reply"].strip()]
    synth("سلام، بفرمایید.")  # warm-up (loads the front-end models)
    panel, render, first = [], [], []

    def sync() -> None:
        if args.cuda:
            torch.cuda.synchronize()

    for row in texts:
        text = row["reply"]
        if args.plus is not None:  # the development runs' per-row seed
            torch.manual_seed(int(hashlib.sha256(row["row_id"].encode()).hexdigest()[:8], 16))
        sync()
        started = time.perf_counter()
        audio = synth(text)
        sync()
        elapsed = time.perf_counter() - started
        render.append(elapsed / max(1e-6, audio.size / RATE))
        words = " ".join(text.split()[:4])
        sync()
        started = time.perf_counter()
        synth(words)
        sync()
        first.append(1000 * (time.perf_counter() - started))
        if args.timing_only:
            continue
        path = args.work / args.arm / f"{row['row_id']}.wav"
        write(path, audio)
        panel.append({"id": f"{args.arm}:{row['row_id']}", "audio_filepath": str(path), "text": text, "duration": round(audio.size / RATE, 3)})
    if not args.timing_only:
        (args.work / f"panel_{args.arm}.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in panel), encoding="utf-8"
        )
    summary = {
        "arm": args.arm,
        "model_sha256": hashlib.sha256((args.repo / "models/model.pth").read_bytes()).hexdigest(),
        "ezafe_checkpoint_sha256": hashlib.sha256(
            (args.repo / "best_global_time_5_val_accuracy=0.9862.pt").read_bytes()
        ).hexdigest(),
        "device": torch.cuda.get_device_name(0) if args.cuda else "CPU",
        "vits_on_gpu": args.cuda,
        "torch_threads": torch.get_num_threads(),
        "texts": len(texts),
        "render_rtf_median": statistics.median(render),
        "first_four_words_ms_median": statistics.median(first),
        "first_four_words_ms_p90": sorted(first)[int(0.9 * (len(first) - 1))],
        "text_pipeline": (
            f"star_tts_plus: {json.dumps(args.options, sort_keys=True)}"
            if args.plus is not None
            else "lab Triton service: Persian digits, eng2fa, sentence split, per-sentence VITS, kaiser_best 16 kHz, trim"
        ),
    }
    suffix = ("_gpu_timing" if args.cuda else "_cpu_timing") if args.timing_only else ""
    name = f"synthesis_{args.arm}{suffix}.json"
    (args.work / name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
