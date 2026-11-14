#!/usr/bin/env python3
"""TTS v4: synthesize the frozen reply texts with Chatterbox Persian.

Arm ``chatterbox_fa`` of ``docs/TTS_CANDIDATE_COMPARISON_V4_PROTOCOL.md``:
``Thomcles/Chatterbox-TTS-Persian-Farsi`` loaded from its ungated repackaging
``Reza2kn/Gooya-Bozorg-v1.5`` with that repository's own ``load_model`` and its
defaults (``language_id`` None, ``exaggeration`` 0.5, ``cfg_weight`` 0.5). The
weight files are checked against the repackaging's ``MODEL_MANIFEST.json``.
The reference voice is processed once (as XTTS latents were in v3); each reply
is split with the live server's ``ClauseChunker`` and the fragments are
concatenated. Chatterbox applies its Perth watermark to every output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATE = 16_000


def load_module(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ClauseChunker = load_module("_streaming", ROOT / "src/thesis_s2s/runtime/streaming.py").ClauseChunker


def clauses(text: str) -> list[str]:
    chunker = ClauseChunker()
    parts = chunker.push(text)
    rest = chunker.flush()
    return [*parts, *([rest] if rest else [])]


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--arm", default="chatterbox_fa")
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()

    manifest = json.loads((args.model_dir / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    weights = {name: sha256(args.model_dir / name) for name in manifest["files"] if (args.model_dir / name).exists()}
    mismatched = [name for name, digest in weights.items() if digest != manifest["files"][name]]
    if mismatched:
        raise SystemExit(f"files differ from MODEL_MANIFEST.json: {mismatched}")

    import torch
    import torchaudio.functional as F

    model = load_module("gooya_bozorg_inference", args.model_dir / "inference.py").load_model(args.model_dir, "cuda")
    torch.cuda.reset_peak_memory_stats()
    model.prepare_conditionals(str(args.reference), exaggeration=0.5)
    out_rate = int(model.sr)

    def synth_fragment(text: str) -> np.ndarray:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        wav = model.generate(text=text, language_id=None, exaggeration=0.5, cfg_weight=0.5)
        return wav.squeeze().float().cpu().numpy()

    def synth(text: str) -> np.ndarray:
        return np.concatenate([synth_fragment(fragment) for fragment in clauses(text)])

    texts = [json.loads(line) for line in args.texts.read_text(encoding="utf-8").splitlines() if line]
    texts = [t for t in texts if t["reply"].strip()]
    synth("سلام، بفرمایید.")  # warm-up
    panel, render, first, failures = [], [], [], []
    for row in texts:
        text = row["reply"]
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            audio = synth(text)
        except Exception as exc:  # noqa: BLE001 - a failed reply is scored as an empty clip
            failures.append({"row_id": row["row_id"], "error": f"{type(exc).__name__}: {exc}"[:300]})
            audio = np.zeros(out_rate // 10, dtype=np.float32)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        render.append(elapsed / max(1e-6, audio.size / out_rate))
        words = " ".join(text.split()[:4])
        torch.cuda.synchronize()
        started = time.perf_counter()
        synth_fragment(words)
        torch.cuda.synchronize()
        first.append(1000 * (time.perf_counter() - started))
        audio16 = F.resample(torch.from_numpy(audio)[None], out_rate, RATE)[0].numpy()
        path = args.work / args.arm / f"{row['row_id']}.wav"
        write(path, audio16, RATE)
        panel.append({"id": f"{args.arm}:{row['row_id']}", "audio_filepath": str(path), "text": text,
                      "duration": round(audio16.size / RATE, 3)})
    (args.work / f"panel_{args.arm}.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in panel), encoding="utf-8"
    )
    summary = {
        "arm": args.arm,
        "model_sha256": weights,
        "manifest_upstream": manifest.get("upstream"),
        "reference_sha256": [sha256(args.reference)],
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "texts": len(panel),
        "failed_texts": failures,
        "clause_chunking": "thesis_s2s.runtime.streaming.ClauseChunker defaults",
        "settings": {"language_id": None, "exaggeration": 0.5, "cfg_weight": 0.5, "watermark": "Perth (built in)"},
        "render_rtf_median": statistics.median(render),
        "first_four_words_ms_median": statistics.median(first),
        "first_four_words_ms_p90": sorted(first)[int(0.9 * (len(first) - 1))],
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "seed": args.seed,
    }
    (args.work / f"synthesis_{args.arm}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                          encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
