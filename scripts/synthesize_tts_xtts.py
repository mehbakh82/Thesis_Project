#!/usr/bin/env python3
"""TTS v3: synthesize the frozen reply texts with ParsVoice-XTTS (zero-shot clone).

Run with an interpreter that has ``coqui-tts`` (set ``USE_TF=0`` if TensorFlow
is installed alongside transformers). Persian is not in upstream Coqui's XTTS
tokenizer; as in the fork ParsVoice-XTTS was trained with, ``fa`` text goes
through ``basic_cleaners`` with the default 300-character limit.

For every text: the full reply is synthesized (16 kHz WAV for the ASR
listeners), the first four words are synthesized separately (comparable to the
Piper first-clause timing of TTS v2), and the time to the first audio chunk of
streaming inference on the full reply is recorded. Decoding uses a fixed seed.
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


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--arm", default="parsvoice_xtts")
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()

    import torch
    import torchaudio.functional as F
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.layers.xtts import tokenizer as xtts_tokenizer
    from TTS.tts.models.xtts import Xtts

    original = xtts_tokenizer.VoiceBpeTokenizer.preprocess_text

    def preprocess_text(self, txt, lang):  # type: ignore[no-untyped-def]
        if lang.split("-")[0] == "fa":
            return xtts_tokenizer.basic_cleaners(txt)
        return original(self, txt, lang)

    xtts_tokenizer.VoiceBpeTokenizer.preprocess_text = preprocess_text

    config = XttsConfig()
    config.load_json(str(args.model_dir / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(args.model_dir / "model.pth"),
        vocab_path=str(args.model_dir / "vocab.json"),
        use_deepspeed=False,
    )
    model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    latents, speaker = model.get_conditioning_latents(audio_path=[str(p) for p in args.reference])
    out_rate = int(config.audio.output_sample_rate)

    def synth(text: str) -> np.ndarray:
        torch.manual_seed(args.seed)
        out = model.inference(text, "fa", latents, speaker, enable_text_splitting=True)
        return np.asarray(out["wav"], dtype=np.float32)

    texts = [json.loads(line) for line in args.texts.read_text(encoding="utf-8").splitlines() if line]
    texts = [t for t in texts if t["reply"].strip()]
    synth("سلام، بفرمایید.")  # warm-up
    panel, render, first, first_chunk = [], [], [], []
    for row in texts:
        text = row["reply"]
        torch.cuda.synchronize()
        started = time.perf_counter()
        audio = synth(text)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        render.append(elapsed / max(1e-6, audio.size / out_rate))
        words = " ".join(text.split()[:4])
        torch.cuda.synchronize()
        started = time.perf_counter()
        synth(words)
        torch.cuda.synchronize()
        first.append(1000 * (time.perf_counter() - started))
        torch.manual_seed(args.seed)
        started = time.perf_counter()
        for _chunk in model.inference_stream(text, "fa", latents, speaker, enable_text_splitting=True):
            torch.cuda.synchronize()
            first_chunk.append(1000 * (time.perf_counter() - started))
            break
        audio16 = F.resample(torch.from_numpy(audio)[None], out_rate, RATE)[0].numpy()
        path = args.work / args.arm / f"{row['row_id']}.wav"
        write(path, audio16, RATE)
        panel.append({"id": f"{args.arm}:{row['row_id']}", "audio_filepath": str(path), "text": text, "duration": round(audio16.size / RATE, 3)})
    (args.work / f"panel_{args.arm}.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in panel), encoding="utf-8"
    )
    summary = {
        "arm": args.arm,
        "model_sha256": hashlib.sha256((args.model_dir / "model.pth").read_bytes()).hexdigest(),
        "reference_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in args.reference],
        "device": torch.cuda.get_device_name(0),
        "texts": len(panel),
        "render_rtf_median": statistics.median(render),
        "first_four_words_ms_median": statistics.median(first),
        "first_four_words_ms_p90": sorted(first)[int(0.9 * (len(first) - 1))],
        "stream_first_chunk_ms_median": statistics.median(first_chunk),
        "stream_first_chunk_ms_p90": sorted(first_chunk)[int(0.9 * (len(first_chunk) - 1))],
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "seed": args.seed,
        "persian_text_preprocessing": "basic_cleaners (as in the ParsVoice training fork)",
    }
    (args.work / f"synthesis_{args.arm}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
