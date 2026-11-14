#!/usr/bin/env python3
"""TTS v4: synthesize the frozen reply texts with a MOSS-TTS-Nano Persian model.

Two arms of ``docs/TTS_CANDIDATE_COMPARISON_V4_PROTOCOL.md`` share this script:

* ``moss_tts_nano_fa``: ``nimaaaAI/MOSS-TTS-Nano-Persian`` on Persian script,
  with the effective defaults of the upstream ``infer.py`` (sampling on; audio
  temperature 0.8, top-p 0.95, top-k 25, repetition penalty 1.2; text
  temperature 1.0, top-p 1.0, top-k 50; at most 375 frames).
* ``gooya_v1``: ``Reza2kn/gooya-v1`` on phonemes from ``Reza2kn/negara-g2p-clean-v7``
  in the char-spaced form of the model's own ``matrix/manifest.json`` (word
  spaces removed, one space between characters), with the card's audio
  temperature 0.6 and top-p 0.9.

Each reply is split with the live server's ``ClauseChunker``; each fragment is
synthesized in voice-clone mode from one CC0 ManaTTS reference and the
fragments are concatenated. Run with an interpreter that has transformers 4.57
and torchaudio, with ``HF_HUB_OFFLINE=1`` so that no remote code is fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Load only the live server's chunker module (the runtime package imports the web server).
_SPEC = importlib.util.spec_from_file_location("_streaming", ROOT / "src/thesis_s2s/runtime/streaming.py")
assert _SPEC and _SPEC.loader
_STREAMING = importlib.util.module_from_spec(_SPEC)
sys.modules["_streaming"] = _STREAMING
_SPEC.loader.exec_module(_STREAMING)
ClauseChunker = _STREAMING.ClauseChunker

RATE = 16_000
MOSS_SAMPLING = {
    "text_temperature": 1.0, "text_top_p": 1.0, "text_top_k": 50,
    "audio_temperature": 0.8, "audio_top_p": 0.95, "audio_top_k": 25, "audio_repetition_penalty": 1.2,
}


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


class G2P:
    """Persian graphemes to the char-spaced phonemes Gooya v1 was trained on."""

    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = T5ForConditionalGeneration.from_pretrained(str(model_dir)).cuda().eval()

    def words(self, text: str) -> str:
        batch = self.tokenizer([text], return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            out = self.model.generate(**batch, max_new_tokens=512, num_beams=1, do_sample=False)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def __call__(self, text: str) -> str:
        return " ".join(self.words(text).replace(" ", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("moss_tts_nano_fa", "gooya_v1"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--g2p-dir", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--check-g2p-only", action="store_true")
    args = parser.parse_args()

    import soundfile as sf
    import torch
    import torchaudio
    import torchaudio.functional as F
    from transformers import AutoModelForCausalLM

    # torchaudio >= 2.9 reads and writes files only through torchcodec, which this
    # environment lacks; plain WAV I/O through soundfile is equivalent here.
    def _load(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), rate

    def _save(path, tensor, rate, *args, **kwargs):  # type: ignore[no-untyped-def]
        sf.write(str(path), tensor.detach().float().cpu().numpy().T, int(rate))

    torchaudio.load, torchaudio.save = _load, _save

    g2p = G2P(args.g2p_dir) if args.arm == "gooya_v1" else None
    if g2p is not None:
        manifest = json.loads((args.model_dir / "matrix" / "manifest.json").read_text(encoding="utf-8")) \
            if (args.model_dir / "matrix" / "manifest.json").exists() else []
        rows = {r["grapheme"]: r["moss"] for r in manifest if r.get("g2p") == "v7"}
        matches = sum(g2p(g) == m for g, m in rows.items())
        g2p_check = {"manifest_v7_rows": len(rows), "exact_matches": matches}
        print(json.dumps({"g2p_check": g2p_check}), flush=True)
        if args.check_g2p_only:
            return 0
    else:
        g2p_check = None

    sampling = dict(MOSS_SAMPLING)
    if args.arm == "gooya_v1":
        sampling.update(audio_temperature=0.6, audio_top_p=0.9)
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model.to(device="cuda", dtype=torch.bfloat16)
    model._set_attention_implementation("sdpa")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    scratch = Path(tempfile.mkdtemp(prefix="moss_tts_"))

    def synth_fragment(text: str) -> tuple[np.ndarray, int]:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model_text = g2p(text) if g2p is not None else text
        result = model.inference(
            text=model_text, output_audio_path=scratch / "out.wav", mode="voice_clone",
            prompt_audio_path=str(args.reference), text_tokenizer_path=str(args.model_dir),
            audio_tokenizer_type="moss-audio-tokenizer-nano",
            audio_tokenizer_pretrained_name_or_path=str(args.audio_tokenizer_dir),
            device="cuda", max_new_frames=375, do_sample=True, use_kv_cache=True, **sampling,
        )
        wav = result["waveform"].detach().float().cpu()
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        return wav.numpy(), int(result["sample_rate"])

    def synth(text: str) -> tuple[np.ndarray, int]:
        pieces, rate = [], RATE
        for fragment in clauses(text):
            audio, rate = synth_fragment(fragment)
            pieces.append(audio)
        return np.concatenate(pieces), rate

    texts = [json.loads(line) for line in args.texts.read_text(encoding="utf-8").splitlines() if line]
    texts = [t for t in texts if t["reply"].strip()]
    synth("سلام، بفرمایید.")  # warm-up
    panel, render, first, failures = [], [], [], []
    for row in texts:
        text = row["reply"]
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            audio, rate = synth(text)
        except Exception as exc:  # noqa: BLE001 - a failed reply is scored as an empty clip
            failures.append({"row_id": row["row_id"], "error": f"{type(exc).__name__}: {exc}"[:300]})
            audio, rate = np.zeros(RATE // 10, dtype=np.float32), RATE
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        render.append(elapsed / max(1e-6, audio.size / rate))
        words = " ".join(text.split()[:4])
        torch.cuda.synchronize()
        started = time.perf_counter()
        synth_fragment(words)
        torch.cuda.synchronize()
        first.append(1000 * (time.perf_counter() - started))
        audio16 = F.resample(torch.from_numpy(audio)[None], rate, RATE)[0].numpy() if rate != RATE else audio
        path = args.work / args.arm / f"{row['row_id']}.wav"
        write(path, audio16, RATE)
        panel.append({"id": f"{args.arm}:{row['row_id']}", "audio_filepath": str(path), "text": text,
                      "duration": round(audio16.size / RATE, 3)})
    (args.work / f"panel_{args.arm}.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in panel), encoding="utf-8"
    )
    weights = next(p for p in (args.model_dir / "pytorch_model.bin", args.model_dir / "model.safetensors") if p.exists())
    summary = {
        "arm": args.arm,
        "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "reference_sha256": [hashlib.sha256(args.reference.read_bytes()).hexdigest()],
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "texts": len(panel),
        "failed_texts": failures,
        "clause_chunking": "thesis_s2s.runtime.streaming.ClauseChunker defaults",
        "sampling": sampling,
        "g2p_check": g2p_check,
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
