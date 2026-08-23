"""Codec/component survey; it does not compare complete speech-language models."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from thesis_s2s.audio import read_wav, write_wav
from thesis_s2s.bakeoff.codecs import roundtrip_all
from thesis_s2s.config import project_root
from thesis_s2s.metrics import LatencySample, cuda_memory_cap_gb, summarize_latency, write_json
from thesis_s2s.runtime.tts import FormantTalker

PERSIAN_PROMPTS = [
    "سلام، هوا امروز چطوره؟",
    "لطفاً یک داستان کوتاه بگو.",
    "ساعت چند است؟",
]


def _load_or_synth_clips(sample_dir: Path, n_persian: int = 8) -> list[Path]:
    wavs = sorted(sample_dir.glob("*.wav"))
    if wavs:
        preferred = [p for p in wavs if "shakoori" in p.name.lower()] + [p for p in wavs if "shakoori" not in p.name.lower()]
        return preferred[:n_persian]
    out = sample_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    from thesis_s2s.runtime.tts import formant_synthesize

    for i, prompt in enumerate(PERSIAN_PROMPTS):
        audio = formant_synthesize(prompt)
        path = out / f"synth_fa_{i}.wav"
        write_wav(path, audio.astype(np.float32), 16000)
        paths.append(path)
    return paths


def try_whisper_probe(path: Path) -> dict:
    try:
        import torch
        from transformers import pipeline
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    try:
        asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=0 if torch.cuda.is_available() else -1,
        )
        out = asr(str(path), generate_kwargs={"language": "persian", "task": "transcribe"})
        text = str(out.get("text") or "")
        has_fa = any("\u0600" <= ch <= "\u06FF" for ch in text)
        return {"available": True, "text": text[:200], "contains_arabic_script": has_fa}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def measure_talker_latency(audio: np.ndarray, gpu_info: dict) -> LatencySample:
    talker = FormantTalker()
    t0 = time.perf_counter()
    chunk = talker.first_chunk(audio)
    t_first = 1000.0 * (time.perf_counter() - t0)
    t1 = time.perf_counter()
    from thesis_s2s.bargein.detector import EnergyVadBaseline

    EnergyVadBaseline().predict_binary(audio)
    t_barge = 1000.0 * (time.perf_counter() - t1)
    vram = gpu_info.get("total_gb")
    capped = bool(gpu_info.get("memory_capped"))
    return LatencySample(
        t_first_audio_ms=t_first,
        t_barge_in_ms=t_barge,
        gpu_name=str(gpu_info.get("device") or "cpu"),
        vram_gb=24.0 if capped else vram,
        memory_capped=capped,
        notes=f"FormantTalker first_chunk n={len(chunk)} backend={talker.backend}",
    )


def run_bakeoff(out_dir: Path | None = None, allow_hf: bool = True) -> dict:
    root = project_root()
    out_dir = Path(out_dir or root / "results" / "bakeoff")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = root / "data" / "samples"
    clips = _load_or_synth_clips(sample_dir)
    audio0, _ = read_wav(clips[0])
    codecs = roundtrip_all(clips[0])
    gpu_info = cuda_memory_cap_gb(24.0)
    lat_samples = [measure_talker_latency(audio0, gpu_info) for _ in range(6)]
    latency = summarize_latency(lat_samples, official_gpu=False)
    whisper = try_whisper_probe(clips[0]) if allow_hf else {"available": False, "notes": "skipped"}
    intel = codecs.get("whisper_intelligibility") or {}
    enc = codecs.get("encodec_24k") or {}
    mimi = codecs.get("moshi_mimi") or {}
    mimi_ok = bool(mimi.get("available")) and bool(mimi.get("persian_phones_preserved"))
    enc_ok = bool(enc.get("available")) and bool(intel.get("persian_phones_preserved"))
    models = {
        "llama-omni2-0.5b": {
            "weights_present": (root / "checkpoints" / "llama_omni2_fa" / "persian_omni2.pt").is_file(),
            "zero_shot_persian_out": False,
            "fits_24gb": True,
            "native_full_duplex": False,
            "trainable": True,
            "codec_proxy": "encodec_24k" if enc.get("available") else "mel_griffin",
            "notes": "Architectural reference only. The local 7 MB artifact is legacy/untyped and not a deployable S2S checkpoint.",
        },
        "llama-omni2-1.5b-bilingual": {
            "weights_present": False,
            "zero_shot_persian_out": False,
            "fits_24gb": True,
            "native_full_duplex": False,
            "trainable": True,
            "notes": "Scale-up only if 0.5B stays under 500 ms.",
        },
        "personaplex-7b": {
            "weights_present": False,
            "zero_shot_persian_out": False,
            "fits_24gb": True,
            "native_full_duplex": True,
            "trainable": True,
            "notes": "English-only card. Promote if Mimi round-trips Persian. 7B not downloaded on shared H100.",
        },
        "mini-omni": {
            "weights_present": False,
            "zero_shot_persian_out": False,
            "fits_24gb": True,
            "native_full_duplex": False,
            "trainable": False,
            "notes": "Named in the definition. English SNAC out. Baseline only.",
        },
        "qwen2.5-omni-3b": {
            "weights_present": False,
            "zero_shot_persian_out": False,
            "fits_24gb": True,
            "native_full_duplex": False,
            "trainable": True,
            "notes": "Near-duplex EN/ZH talker. Bake-off row, not default. Not downloaded.",
        },
    }
    for candidate in models.values():
        candidate["empirically_evaluated"] = False
        candidate["fits_24gb"] = None
        candidate["trainable"] = None
    decision = {
        "status": "no_end_to_end_model_winner",
        "primary": None,
        "deployable_baseline": "nemo_qwen_piper_cascade",
        "reason": (
            "This command measures codec reconstruction, dependency availability, and formant component timing; "
            "it does not fine-tune or compare complete Persian speech-language models."
        ),
        "encodec_persian_ok": enc_ok,
        "mimi_persian_ok": mimi_ok,
        "cosyvoice2_installed": bool(codecs.get("cosyvoice2", {}).get("available")),
        "official_latency": False,
    }
    payload = {
        "evidence_class": "component_survey",
        "end_to_end_models_compared": False,
        "official": False,
        "clips": [str(p) for p in clips],
        "codecs": codecs,
        "whisper_probe": whisper,
        "gpu_profile": gpu_info,
        "latency": latency.to_dict(),
        "models": models,
        "decision": decision,
    }
    payload["latency"]["samples"] = [s.__dict__ for s in latency.samples]
    payload["latency"]["measurement_scope"] = "cached_first_packet_and_detector_component"
    payload["latency"]["official_e2e_eligible"] = False
    payload["latency"]["component_proxy_gates"] = {
        "cached_packet_p50_under_500_ms": payload["latency"]["t_first_audio_gate_ok"],
        "detector_call_p95_under_300_ms": payload["latency"]["t_barge_in_gate_ok"],
    }
    payload["latency"]["t_first_audio_gate_ok"] = False
    payload["latency"]["t_barge_in_gate_ok"] = False
    write_json(out_dir / "bakeoff_report.json", payload)
    (out_dir / "DECISION.md").write_text(
        "# Bake-off result\n\n"
        "**Evidence status: component survey; no end-to-end model winner.**\n\n"
        "This command did not fine-tune or compare the candidate speech-language models.\n\n"
        f"Encodec proxy: {enc_ok}. Mimi proxy: {mimi_ok}. "
        f"CosyVoice2 available: {decision['cosyvoice2_installed']}.\n\n"
        "Current deployable baseline: NeMo ASR → local Qwen/rules → Piper. Select a direct model only after controlled Persian end-to-end evaluation.\n",
        encoding="utf-8",
    )
    return payload
