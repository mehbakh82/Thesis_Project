"""Automatic latency / interrupt benches on a 24 GB profile."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from thesis_s2s.bargein.detector import BargeinDetector
from thesis_s2s.bargein.synthetic import make_dataset
from thesis_s2s.bargein.train import train_and_eval
from thesis_s2s.config import project_root
from thesis_s2s.eval.wer import wer
from thesis_s2s.metrics import (
    LatencySample,
    cuda_memory_cap_gb,
    gpu_inventory,
    summarize_latency,
    write_json,
)
from thesis_s2s.runtime.cascade import cascade_first_audio
from thesis_s2s.runtime.duplex import DuplexSession, default_talker
from thesis_s2s.runtime.tts import piper_available, synthesize


def run_latency_bench(out_dir: Path | None = None, path: str = "A") -> dict:
    """Automated server-component benchmark; live browser telemetry is the official path."""

    out_dir = Path(out_dir or project_root() / "results" / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = gpu_inventory()
    if path.upper() == "B":
        gpu = {
            **inv,
            "memory_capped": False,
            "applied": False,
            "requested_cap_gb": None,
            "path": "B",
        }
        official = bool(inv.get("official_size"))
    else:
        gpu = cuda_memory_cap_gb(24.0)
        gpu["path"] = "A"
        official = False
    model_path = project_root() / "results" / "bargein" / "bargein_gbdt.pkl"
    if model_path.is_file():
        detector = BargeinDetector.load(model_path)
    else:
        train_and_eval(n_per_class=16, out_dir=project_root() / "results" / "bargein")
        detector = BargeinDetector.load(model_path)
    hardware_eligible = official
    official = False  # synthetic/server-only timing can never be an official E2E measurement
    talker = default_talker()
    samples = []
    clips = make_dataset(n_per_class=4, seed=7)
    for clip in clips:
        session = DuplexSession(detector, talker)
        t0 = time.perf_counter()
        session.on_user_end(clip.audio)
        t_first = session.log.server_generation_ms or (1000 * (time.perf_counter() - t0))
        session.on_mic_while_playing(clip.audio, interrupt_onset=clip.kind == "interrupt")
        vram = (
            gpu.get("total_gb")
            if path.upper() == "B"
            else (24.0 if gpu.get("memory_capped") else gpu.get("total_gb"))
        )
        samples.append(
            LatencySample(
                t_first_audio_ms=t_first,
                t_barge_in_ms=session.log.t_barge_in_ms if clip.kind == "interrupt" else None,
                gpu_name=str(gpu.get("device") or "cpu"),
                vram_gb=vram,
                memory_capped=bool(gpu.get("memory_capped")),
                notes=f"{clip.kind}:{type(talker).__name__}:{getattr(talker, 'backend', '')}",
            )
        )
    report = summarize_latency(samples, official_gpu=official)
    payload = report.to_dict()
    payload["samples"] = [s.__dict__ for s in samples]
    payload["gpu_profile"] = gpu
    payload["gpu_inventory"] = inv
    payload["official_gpu"] = bool(official)
    payload["hardware_eligible"] = bool(hardware_eligible)
    payload["measurement_scope"] = "server_generation_and_synthetic_detector_only"
    payload["official_e2e_eligible"] = False
    payload["talker"] = type(talker).__name__
    payload["tts_backend"] = getattr(talker, "backend", "")
    payload["piper_installed"] = piper_available()
    payload["cascade"] = cascade_first_audio(clips[0].audio)
    if path.upper() == "B":
        payload["path"] = "B"
        payload["path_b_hardware_present"] = bool(inv.get("path_b_possible"))
        payload["note"] = (
            "Path B hardware profile; still unofficial until measured from live browser events."
        )
    else:
        payload["path"] = "A_h100_memory_capped"
        payload["note"] = (
            "Path A: H100 component proxy with 24 GB cap; never an official E2E table."
        )
    name = "latency_bench_path_b.json" if path.upper() == "B" else "latency_bench.json"
    write_json(out_dir / name, payload)
    if path.upper() != "B":
        write_json(out_dir / "latency_bench.json", payload)
    return payload


def run_interrupt_bench(out_dir: Path | None = None, n_per_class: int = 40) -> dict:
    out_dir = Path(out_dir or project_root() / "results" / "eval")
    report = train_and_eval(
        n_per_class=n_per_class, seed=3, out_dir=project_root() / "results" / "bargein"
    )
    report["measurement_scope"] = (
        "recorded_group_heldout" if report.get("recorded_eval") else "synthetic_proxy"
    )
    report["official_detector_eligible"] = bool(report.get("recorded_eval"))
    report["note"] = (
        report.get("note") or "Synthetic evidence cannot satisfy the thesis detector gate."
    )
    write_json(out_dir / "interrupt_bench.json", report)
    return report


def run_reply_wer(out_dir: Path | None = None) -> dict:
    """WER of Whisper on formant/Omni reply vs intended Persian text (intent, not GPT-4o QA)."""

    out_dir = Path(out_dir or project_root() / "results" / "eval")
    prompts = ["سلام", "خداحافظ", "حالت چطوره"]
    rows = []
    try:
        import tempfile

        import torch
        from transformers import pipeline

        from thesis_s2s.audio import write_wav

        asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=0 if torch.cuda.is_available() else -1,
        )
        for text in prompts:
            audio, backend = synthesize(text)
            tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
            write_wav(tmp, audio)
            asr_result = asr(
                str(tmp), generate_kwargs={"language": "persian", "task": "transcribe"}
            )
            asr_payload: dict[str, Any] = asr_result if isinstance(asr_result, dict) else {}
            hyp = str(asr_payload.get("text") or "")
            tmp.unlink(missing_ok=True)
            rows.append({"ref": text, "hyp": hyp, "wer": wer(text, hyp), "tts_backend": backend})
    except Exception as exc:
        rows = [{"error": str(exc)[:300]}]
    payload = {
        "measurement_scope": "diagnostic_asr_on_tts_proxy",
        "official_quality_eligible": False,
        "n": len(rows),
        "rows": rows,
        "mean_wer": float(
            np.mean([r["wer"] for r in rows if "wer" in r])
            if any("wer" in r for r in rows)
            else float("nan")
        ),
        "note": "MOS is not measured on formant. Piper/CosyVoice is the talker for the study sheet.",
    }
    write_json(out_dir / "reply_wer.json", payload)
    return payload


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def write_eval_summary(latency: dict, interrupt: dict, wer_report: dict | None = None) -> Path:
    root = project_root()
    path = root / "results" / "eval" / "SUMMARY.md"
    barge = interrupt.get("proposed") or {}
    base = interrupt.get("energy_vad_baseline") or {}
    hop = interrupt.get("hop_cpu_ms")
    lines = [
        "# Evaluation evidence snapshot",
        "",
        "**Automated timings are component/synthetic proxies and cannot satisfy an end-to-end thesis gate.**",
        "",
        "## Barge-in detector",
        "",
        f"- Proposed GBDT: accuracy **{barge.get('accuracy')}**, F1 {barge.get('interrupt_f1')}, FAR {barge.get('far')}, FRR {barge.get('frr')} (n={barge.get('n')})",
        f"- Energy-VAD baseline: accuracy {base.get('accuracy')}, FAR {base.get('far')}",
        f"- Synthetic proxy result: **{'pass' if barge.get('target_ok') else 'fail'}**. "
        f"Recorded speaker/session-held-out eval: {interrupt.get('recorded_eval')} "
        f"(recorded_target_ok={interrupt.get('recorded_target_ok')}); thesis >80% gate remains pending without it.",
        f"- CPU hop: {hop} ms (budget ≪ 20 ms)",
        "",
        "## Latency component proxies",
        "",
        "Path A (H100 24 GB software cap): server-only and unofficial.",
        "Path B: still server-only; official values require client acknowledgements on a physical 12–24 GB GPU.",
        "",
        f"- Talker: `{latency.get('talker')}` backend `{latency.get('tts_backend')}`",
        f"- Server-generation proxy p50 = {latency.get('t_first_audio_p50_ms')} ms (not E2E)",
        f"- Synthetic detector-stop proxy p95 = {latency.get('t_barge_in_p95_ms')} ms (not live playback)",
        f"- official_gpu: **{latency.get('official_gpu')}** (path `{latency.get('path')}`). H100 24 GB cap is not a physical 12–24 GB card.",
        f"- Cascade: {latency.get('cascade')}",
        "",
        "## Diagnostic ASR-on-TTS proxy",
        "",
        f"- mean_wer={wer_report.get('mean_wer') if isinstance(wer_report, dict) else wer_report} "
        f"(not a MOS or naturalness result; formant output is never study-eligible)",
        "",
        "## Human study",
        "",
        "Session software: `.venv/bin/python -m thesis_s2s.cli serve --study --retention features`. "
        "Status remains incomplete until 5–10 participants / ≥2 aged 60+ have complete ratings and client timing. See `docs/HUMAN_STUDY.md`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
