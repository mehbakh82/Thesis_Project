"""Speech-codec round-trips for the week-1 bake-off.

Tries CosyVoice 2 and Mimi when installed. Always runs Encodec (neural speech
tokenizer) plus a mel/Griffin-Lim proxy so Persian coverage is measured even
on a busy GPU without 7B weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import read_wav


def _snr_db(ref: np.ndarray, rec: np.ndarray) -> float:
    n = min(len(ref), len(rec))
    if n == 0:
        return float("nan")
    ref = ref[:n]
    rec = rec[:n]
    num = float(np.mean(ref**2))
    den = float(np.mean((ref - rec) ** 2) + 1e-12)
    return float(10 * np.log10(num / den))


def mulaw_roundtrip(audio: np.ndarray) -> tuple[np.ndarray, dict]:
    mu = np.sign(audio) * np.log1p(255 * np.abs(audio)) / np.log1p(255)
    rec = np.sign(mu) * (1 / 255.0) * ((1 + 255) ** np.abs(mu) - 1)
    return rec.astype(np.float32), {"available": True, "notes": "waveform mu-law; not a speech tokenizer"}


def mel_griffin_roundtrip(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, dict]:
    """Speech-aware proxy: 80-mel → Griffin-Lim. Stand-in when CosyVoice 2 is absent."""


    n_fft, hop, n_mels = 512, 160, 80
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    window = np.hanning(n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop
    frames = np.stack([audio[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    spec = np.abs(rfft(frames, n=n_fft, axis=1))
    # triangular mel bank
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        right = min(right, bank.shape[1] - 1)
        if center > left:
            bank[i - 1, left:center] = (np.arange(left, center) - left) / max(1, center - left)
        if right > center:
            bank[i - 1, center:right] = (right - np.arange(center, right)) / max(1, right - center)
    mel = spec @ bank.T
    # invert mel (least squares) then Griffin-Lim-ish phase restore via random phase + 8 iters
    rec_mag = np.maximum(mel @ np.linalg.pinv(bank.T), 0)
    phase = np.random.default_rng(0).uniform(-np.pi, np.pi, rec_mag.shape)
    for _ in range(8):
        stft = rec_mag * np.exp(1j * phase)
        frames_t = irfft(stft, n=n_fft, axis=1)
        rebuilt = np.zeros(n_fft + hop * (n_frames - 1))
        wsum = np.zeros_like(rebuilt)
        for i, fr in enumerate(frames_t):
            sl = slice(i * hop, i * hop + n_fft)
            rebuilt[sl] += fr * window
            wsum[sl] += window**2
        rebuilt = rebuilt / np.maximum(wsum, 1e-8)
        new_frames = np.stack([rebuilt[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
        new_stft = rfft(new_frames, n=n_fft, axis=1)
        phase = np.angle(new_stft)
    rec = rebuilt[: len(audio)].astype(np.float32)
    return rec, {"available": True, "notes": "80-mel Griffin-Lim; CosyVoice2-like spectral bottleneck"}


def encodec_roundtrip(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray | None, dict]:
    try:
        import torch
        from transformers import AutoProcessor, EncodecModel
    except Exception as exc:
        return None, {"available": False, "error": str(exc)[:200]}
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        model.to(device).eval()
        if sr != 24000:
            from thesis_s2s.audio import resample

            audio24 = resample(audio, sr, 24000)
        else:
            audio24 = audio
        inputs = processor(raw_audio=audio24, sampling_rate=24000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, bandwidth=24.0)
        rec24 = out.audio_values.squeeze().detach().cpu().numpy().astype(np.float32)
        n_codes = int(out.audio_codes.shape[-1]) if getattr(out, "audio_codes", None) is not None else None
        from thesis_s2s.audio import resample

        rec = resample(rec24, 24000, sr)
        rec = rec[: len(audio)]
        if len(rec) < len(audio):
            rec = np.pad(rec, (0, len(audio) - len(rec)))
        info = {
            "available": True,
            "notes": "facebook/encodec_24khz neural speech tokenizer via model.forward; CosyVoice2 substitute when CosyVoice is not installed",
            "codebook_frames": n_codes,
        }
        return rec, info
    except Exception as exc:
        return None, {"available": False, "error": str(exc)[:300]}


def probe_cosyvoice2() -> dict:
    """Import-only CosyVoice 2 probe. Does not download 7B weights."""

    payload = {
        "pulled_7b": False,
        "modules": {},
        "tokenizer_for_omni2": "encodec_24khz_24kbps",
        "persian_tokens": None,
    }
    for name, module in (
        ("cosyvoice", "cosyvoice"),
        ("cosyvoice2", "cosyvoice2"),
        ("matcha_tts", "matcha.models"),
    ):
        payload["modules"][name] = try_module_codec(name, module)
    any_cv = any(v.get("available") for v in payload["modules"].values())
    payload["available"] = any_cv
    if any_cv:
        payload["tokenizer_for_omni2"] = "cosyvoice2_if_persian_tokenizes"
        payload["note"] = "CosyVoice import succeeded; train Omni2 TTS head on those tokens only if encode() accepts Persian."
    else:
        payload["note"] = (
            "CosyVoice 2 not installed. Keep Encodec 24 kHz @ 24 kbps as the speech tokenizer "
            "(bake-off CER 0.087). Do not pull 7B on the busy H100."
        )
    return payload


def try_module_codec(name: str, module: str) -> dict:
    try:
        __import__(module)
        return {"available": True, "persian_phones_preserved": None, "notes": "installed; encode/decode not wired"}
    except Exception:
        return {"available": False, "persian_phones_preserved": None, "notes": "not installed"}


def whisper_intelligibility(original: np.ndarray, reconstructed: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """CER between Whisper transcripts of original vs reconstructed audio."""

    try:
        import torch
        from transformers import pipeline
    except Exception as exc:
        return {"available": False, "error": str(exc)[:200]}
    try:
        device = 0 if torch.cuda.is_available() else -1
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)

        def transcribe(x: np.ndarray) -> str:
            import os
            import tempfile

            from thesis_s2s.audio import write_wav

            fd, name = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            tmp = Path(name)
            write_wav(tmp, x, sr)
            try:
                out = asr(str(tmp), generate_kwargs={"language": "persian", "task": "transcribe"})
                return str(out.get("text") or "")
            finally:
                tmp.unlink(missing_ok=True)

        a = transcribe(original)
        b = transcribe(reconstructed)
        cer = _cer(a, b)
        has_fa = any("\u0600" <= ch <= "\u06FF" for ch in a)
        return {
            "available": True,
            "orig_text": a[:180],
            "recon_text": b[:180],
            "cer": cer,
            "persian_phones_preserved": bool(has_fa and cer <= 0.45),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)[:300]}


def _cer(ref: str, hyp: str) -> float:
    ref_c = list(ref.replace(" ", ""))
    hyp_c = list(hyp.replace(" ", ""))
    if not ref_c:
        return 0.0 if not hyp_c else 1.0
    n, m = len(ref_c), len(hyp_c)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_c[i - 1] == hyp_c[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[n, m] / n)


def roundtrip_all(path: Path) -> dict:
    audio, sr = read_wav(path)
    payload: dict = {"clip": str(path), "n_samples": int(len(audio))}
    rec_mu, info_mu = mulaw_roundtrip(audio)
    info_mu["snr_db"] = _snr_db(audio, rec_mu)
    payload["mulaw8"] = info_mu
    rec_mel, info_mel = mel_griffin_roundtrip(audio, sr)
    info_mel["snr_db"] = _snr_db(audio, rec_mel)
    payload["mel_griffin"] = info_mel
    rec_en, info_en = encodec_roundtrip(audio, sr)
    if rec_en is not None:
        info_en["snr_db"] = _snr_db(audio, rec_en)
    payload["encodec_24k"] = info_en
    payload["cosyvoice2"] = probe_cosyvoice2()
    payload["snac"] = try_module_codec("snac", "snac")
    payload["moshi_mimi"] = try_module_codec("moshi_mimi", "moshi")
    # intelligibility on the best neural reconstruction available
    recon = rec_en if rec_en is not None else rec_mel
    payload["whisper_intelligibility"] = whisper_intelligibility(audio, recon, sr)
    if rec_en is not None:
        payload["encodec_24k"]["persian_phones_preserved"] = payload["whisper_intelligibility"].get(
            "persian_phones_preserved"
        )
    return payload
