"""Experimental Whisper/projector reconstruction adapters; not a deployable S2S model.

Stages
1. ASR adapter: speech -> text
2. TTS tokens: speech/text -> mel (CosyVoice/Piper/Encodec teacher when present)
3. Spoken QA: speech in, text+mel out

DummySpeechLM remains for unit tests that must not download weights.
``train_s2s`` is retained only as an explicitly gated ablation path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import read_wav
from thesis_s2s.config import project_root
from thesis_s2s.data.verbatim import s2s_text


class DummySpeechLM(nn.Module):
    """Tiny stand-in used when Qwen/Whisper weights are not on disk."""

    def __init__(self, dim: int = 64, n_tokens: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, dim, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
        )
        self.asr_head = nn.Linear(dim, n_tokens)
        self.tts_head = nn.Linear(dim, n_tokens)

    def forward(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(audio.unsqueeze(1))
        pooled = h.mean(dim=-1)
        return {
            "asr_logits": self.asr_head(pooled),
            "tts_logits": self.tts_head(pooled),
            "hidden": pooled,
        }


def log_mel(
    audio: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = 80, n_fft: int = 400, hop: int = 160
) -> np.ndarray:
    from scipy.fft import rfft

    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    window = np.hanning(n_fft)
    n = 1 + (len(audio) - n_fft) // hop
    frames = np.stack([audio[i * hop : i * hop + n_fft] * window for i in range(max(1, n))])
    spec = np.abs(rfft(frames, n=n_fft, axis=1)) ** 2

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10 ** (np.asarray(mel) / 2595.0) - 1.0)

    n_fft_bins = spec.shape[1]
    mels = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((2 * (n_fft_bins - 1) + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft_bins - 1)
    bank = np.zeros((n_mels, n_fft_bins))
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center > left:
            bank[i - 1, left:center] = (np.arange(left, center) - left) / max(1, center - left)
        if right > center:
            bank[i - 1, center:right] = (right - np.arange(center, right)) / max(1, right - center)
    mel = spec @ bank.T
    return np.log(mel + 1e-6).astype(np.float32)


class PersianOmni2(nn.Module):
    """Whisper-small encoder (frozen) + projector + ASR/TTS heads + optional Qwen LoRA prefix."""

    def __init__(
        self,
        whisper_name: str = "openai/whisper-small",
        llm_name: str | None = "Qwen/Qwen2.5-0.5B-Instruct",
        llm_dim: int = 896,
        n_mels: int = 80,
        vocab: int = 256,
        load_llm: bool = True,
    ):
        super().__init__()
        from transformers import WhisperModel

        self.whisper_name = whisper_name
        self.llm_name = llm_name
        self.whisper = WhisperModel.from_pretrained(whisper_name)
        for p in self.whisper.parameters():
            p.requires_grad = False
        hid = self.whisper.config.d_model
        self.proj = nn.Sequential(
            nn.Linear(hid, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        self.asr_head = nn.Linear(llm_dim, vocab)
        self.tts_head = nn.Linear(llm_dim, n_mels)
        self.vocab = vocab
        self.n_mels = n_mels
        self.llm = None
        self.tokenizer = None
        if load_llm and llm_name:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                llm: Any = AutoModelForCausalLM.from_pretrained(
                    llm_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
                try:
                    from peft import LoraConfig, get_peft_model

                    lora = LoraConfig(
                        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]
                    )
                    llm = get_peft_model(llm, lora)
                except Exception:
                    for p in llm.parameters():
                        p.requires_grad = False
                self.llm = llm
            except Exception:
                self.llm = None

    def encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.whisper.encoder(input_features=input_features).last_hidden_state
        return self.proj(hidden.float())

    def forward(
        self, input_features: torch.Tensor, mel_target: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        speech = self.encode_audio(input_features)
        pooled = speech.mean(dim=1)
        asr_logits = self.asr_head(pooled)
        tts_mel = self.tts_head(speech)
        out = {"asr_logits": asr_logits, "tts_mel": tts_mel, "speech": speech}
        if mel_target is not None:
            t = min(tts_mel.size(1), mel_target.size(1))
            out["tts_loss"] = nn.functional.l1_loss(tts_mel[:, :t], mel_target[:, :t])
        return out


class JsonlSpeechDataset(Dataset):
    def __init__(
        self, jsonl: Path, max_seconds: float = 8.0, n_mels: int = 80, split: str | None = None
    ):
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        if not isinstance(n_mels, int) or isinstance(n_mels, bool) or n_mels <= 0:
            raise ValueError("n_mels must be a positive integer")
        rows = [
            json.loads(line)
            for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("every manifest row must be a JSON object")
        if split is not None and any(row.get("split") for row in rows):
            rows = [row for row in rows if row.get("split") == split]
        if not rows:
            raise ValueError(f"no rows for split={split!r} in {jsonl}")
        self.rows = rows
        self.max_samples = int(max_seconds * SAMPLE_RATE)
        self.n_mels = n_mels
        self.max_mel = int(max_seconds * 100)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx % len(self.rows)]
        path = row.get("audio_filepath") or row.get("audio_path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("manifest row requires a non-empty audio path")
        if not Path(path).is_file():
            raise FileNotFoundError(f"manifest audio does not exist: {path}")
        audio, _ = read_wav(path)
        audio = audio[: self.max_samples]
        if len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))
        text = s2s_text(row)
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        token = int.from_bytes(digest, "big") % 64
        mel = log_mel(audio, n_mels=self.n_mels)
        if len(mel) < self.max_mel:
            mel = np.pad(mel, ((0, self.max_mel - len(mel)), (0, 0)))
        else:
            mel = mel[: self.max_mel]
        return {
            "audio": torch.from_numpy(audio.astype(np.float32)),
            "token": torch.tensor(token, dtype=torch.long),
            "mel": torch.from_numpy(mel.astype(np.float32)),
            "text": text,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "audio": torch.stack([b["audio"] for b in batch]),
        "token": torch.stack([b["token"] for b in batch]),
        "mel": torch.stack([b["mel"] for b in batch]),
    }


def write_tiny_manifest(path: Path, n: int = 8) -> Path:
    from thesis_s2s.audio import write_wav
    from thesis_s2s.runtime.tts import formant_synthesize

    path.parent.mkdir(parents=True, exist_ok=True)
    wav_dir = path.parent / "tiny_wavs"
    wav_dir.mkdir(exist_ok=True)
    texts = ["سلام", "خداحافظ", "حالت چطوره", "یک قصه بگو"]
    with path.open("w", encoding="utf-8") as handle:
        for i in range(n):
            audio = formant_synthesize(texts[i % len(texts)])
            wav = wav_dir / f"utt_{i}.wav"
            write_wav(wav, audio[:SAMPLE_RATE])
            handle.write(
                json.dumps(
                    {
                        "audio_filepath": str(wav),
                        "text": texts[i % len(texts)],
                        "transcript_caption": texts[i % len(texts)],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def whisper_features(audio: torch.Tensor, processor, device: str) -> torch.Tensor:
    waves = audio.detach().cpu().numpy()
    feats = []
    for w in waves:
        out = processor(w, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        feats.append(out.input_features)
    return torch.cat(feats, dim=0).to(device)


def train_smoke(
    steps: int = 5,
    device: str | None = None,
    out_dir: Path | None = None,
    jsonl: Path | None = None,
) -> dict:
    out_dir = Path(out_dir or project_root() / "checkpoints" / "llama_omni2_fa")
    out_dir.mkdir(parents=True, exist_ok=True)
    if jsonl is None:
        jsonl = write_tiny_manifest(out_dir / "tiny_train.jsonl")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = JsonlSpeechDataset(jsonl)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate)
    model = DummySpeechLM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    history = []
    it = iter(loader)
    for _step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        audio = batch["audio"].to(device)
        token = batch["token"].to(device) % 64
        out = model(audio)
        loss = loss_fn(out["asr_logits"], token) + 0.5 * loss_fn(out["tts_logits"], token)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))
    ckpt = out_dir / "dummy_adapter.pt"
    torch.save({"state_dict": model.state_dict(), "history": history}, ckpt)
    report = {
        "backbone": "DummySpeechLM (tests only)",
        "device": device,
        "steps": steps,
        "loss_last": history[-1] if history else None,
        "checkpoint": str(ckpt),
        "stages": ["asr_adapter", "tts_tokens", "spoken_qa"],
    }
    (out_dir / "train_smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _pick_jsonl(explicit: Path | None) -> Path:
    root = project_root()
    if explicit is not None:
        selected = Path(explicit)
        if not selected.is_file() or selected.stat().st_size <= 0:
            raise FileNotFoundError(f"explicit training manifest is missing or empty: {selected}")
        return selected
    for cand in (
        root / "data" / "processed" / "manifests" / "filtered.jsonl",
        root / "data" / "processed" / "manifests" / "filtered_caption.jsonl",
        root / "data" / "processed" / "manifests" / "youtube_all.jsonl",
        root / "data" / "processed" / "manifests" / "test_manifest_youtube.jsonl",
    ):
        if cand.is_file() and cand.stat().st_size > 0:
            return cand
    raise FileNotFoundError(
        "no non-empty training manifest found; use train-s2s-smoke for synthetic smoke data"
    )


def train_s2s(
    steps: int | None = 40,
    device: str | None = None,
    out_dir: Path | None = None,
    jsonl: Path | None = None,
    whisper_name: str = "openai/whisper-small",
    load_llm: bool = True,
    memory_cap_gb: float = 24.0,
    epochs: float | None = None,
    max_steps: int | None = None,
    use_encodec_tokens: bool = False,
    allow_experimental: bool = False,
) -> dict:
    from transformers import WhisperProcessor

    from thesis_s2s.metrics import cuda_memory_cap_gb

    if not allow_experimental:
        raise RuntimeError(
            "train_s2s is an experimental encoder/reconstruction objective, not a deployable "
            "speech-to-speech LLM. Pass allow_experimental=True only for ablation research."
        )
    if use_encodec_tokens:
        raise ValueError(
            "Encodec targets are not implemented in this experimental path; log-mel reconstruction only"
        )
    if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0):
        raise ValueError("steps must be a positive integer or None")
    if max_steps is not None and (
        not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer or None")
    if epochs is not None and (not math.isfinite(epochs) or epochs <= 0):
        raise ValueError("epochs must be positive and finite")

    out_dir = Path(out_dir or project_root() / "checkpoints" / "llama_omni2_fa")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = _pick_jsonl(jsonl)
    gpu = cuda_memory_cap_gb(memory_cap_gb)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = WhisperProcessor.from_pretrained(whisper_name)
    model = PersianOmni2(whisper_name=whisper_name, load_llm=load_llm)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=2e-4)
    ds = JsonlSpeechDataset(jsonl, max_seconds=6.0, split="train")
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate)
    history: list[float] = []
    model.train()
    n_rows = max(1, len(ds))
    if epochs is not None:
        steps = int(max(1, round(float(epochs) * n_rows)))
    elif steps is None:
        steps = 40
    if max_steps is not None:
        steps = min(int(steps), int(max_steps))
    it = iter(loader)
    ce = nn.CrossEntropyLoss()
    stages = ["asr_adapter", "tts_tokens", "spoken_qa"]
    per = max(1, int(steps) // len(stages))
    step = 0
    tokenizer_note = "log_mel_reconstruction"
    for stage in stages:
        for _ in range(per):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            feats = whisper_features(batch["audio"], processor, device)
            mel = batch["mel"].to(device)
            out = model(feats, mel_target=mel)
            loss = out["tts_loss"] + ce(out["asr_logits"], batch["token"].to(device))
            if stage == "spoken_qa" and model.llm is None:
                loss = loss + 0.1 * out["speech"].pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            history.append(float(loss.detach().cpu()))
            step += 1
    ckpt = out_dir / "persian_omni2.pt"
    torch.save(
        {
            "format_version": 2,
            "artifact_kind": "experimental_encoder_reconstruction",
            "runtime_ready": False,
            "proj": model.proj.state_dict(),
            "asr_head": model.asr_head.state_dict(),
            "tts_head": model.tts_head.state_dict(),
            "whisper_name": whisper_name,
            "history": history,
            "llm_attached": model.llm is not None,
            "speech_tokenizer": tokenizer_note,
        },
        ckpt,
    )
    report = {
        "backbone": "Whisper adapter reconstruction experiment",
        "artifact_status": "experimental_not_end_to_end",
        "runtime_ready": False,
        "whisper": whisper_name,
        "llm": model.llm_name if model.llm is not None else None,
        "device": device,
        "steps": step,
        "epochs": epochs,
        "max_steps": max_steps,
        "loss_last": history[-1] if history else None,
        "loss_first": history[0] if history else None,
        "checkpoint": str(ckpt),
        "jsonl": str(jsonl),
        "gpu_profile": gpu,
        "stages": stages,
        "n_train_rows": len(ds),
        "training_split": "train",
        "speech_tokenizer": tokenizer_note,
        "s2s_text": "transcript_caption",
        "published_model": None,
        "limitations": [
            "Qwen is not in forward()",
            "target mel reconstructs input speech",
            "no response-speech supervision",
        ],
    }
    (out_dir / "train_s2s.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def checkpoint_runtime_status(path: str | Path) -> dict:
    """Return a fail-closed deployability verdict for a speech-model artifact."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        return {"exists": False, "runtime_ready": False, "reason": "checkpoint_missing"}
    try:
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        return {
            "exists": True,
            "runtime_ready": False,
            "reason": f"checkpoint_unreadable:{type(exc).__name__}",
        }
    if not isinstance(bundle, dict):
        return {"exists": True, "runtime_ready": False, "reason": "checkpoint_not_a_mapping"}
    kind = bundle.get("artifact_kind", "legacy_untyped")
    declared_ready = bundle.get("runtime_ready") is True and kind == "deployable_s2s_v1"
    return {
        "exists": True,
        "runtime_ready": False,
        "declared_runtime_ready": declared_ready,
        "artifact_kind": kind,
        "format_version": bundle.get("format_version"),
        "reason": (
            "direct_runtime_not_implemented"
            if declared_ready
            else "not_a_deployable_s2s_checkpoint"
        ),
    }


class OmniTalker:
    """Decode first-packet audio. Piper/ManaTTS is the audible first packet when installed."""

    def __init__(self, ckpt: Path | None = None):
        from thesis_s2s.runtime.tts import FormantTalker

        self.fallback = FormantTalker()
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.backend = self.fallback.backend
        path = Path(ckpt or project_root() / "checkpoints" / "llama_omni2_fa" / "persian_omni2.pt")
        self.checkpoint_status = checkpoint_runtime_status(path)
        if not self.checkpoint_status["runtime_ready"]:
            return
        try:
            bundle = torch.load(path, map_location="cpu", weights_only=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = PersianOmni2(
                whisper_name=bundle.get("whisper_name", "openai/whisper-small"), load_llm=False
            )
            model.proj.load_state_dict(bundle["proj"])
            model.tts_head.load_state_dict(bundle["tts_head"])
            model.eval()
            self.model = model.to(device)
            self.device = device
            from transformers import WhisperProcessor

            self.processor = WhisperProcessor.from_pretrained(
                bundle.get("whisper_name", "openai/whisper-small")
            )
            from thesis_s2s.runtime.tts import warm_piper

            if warm_piper():
                self.backend = "piper"
        except Exception:
            self.model = None

    def first_chunk(self, user_audio: np.ndarray, text: str | None = None) -> np.ndarray:
        from thesis_s2s.runtime.tts import first_packet, piper_synthesize

        phrase = text or "سلام"
        piper = piper_synthesize(phrase if len(phrase) < 24 else "سلام")
        if piper is not None and len(piper) > 0:
            self.backend = "piper"
            return first_packet(piper)
        if self.model is None or self.processor is None:
            return self.fallback.first_chunk(user_audio, text)
        wav: np.ndarray = user_audio.astype(np.float32)
        if len(wav) < 400:
            wav = np.pad(wav, (0, 400 - len(wav)))
        feats = self.processor(
            wav, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device)
        with torch.no_grad():
            out = self.model(feats)
            mel = out["tts_mel"][0].detach().cpu().numpy()
        audio = _mel_to_wave(mel)
        n = min(len(audio), int(0.3 * SAMPLE_RATE))
        if n < 100:
            return self.fallback.first_chunk(user_audio, text)
        self.backend = "omni2_mel"
        return audio[:n]


def _mel_to_wave(mel: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cheap inverse: treat mel bands as harmonic amplitudes (first packet, not MOS-grade)."""

    (n_mels,) = (mel.shape[1],) if mel.ndim == 2 else (80,)
    hop = 160
    t = np.arange(mel.shape[0] * hop) / sr
    audio = np.zeros_like(t, dtype=np.float64)
    for k in range(min(n_mels, mel.shape[1])):
        freq = 80 + k * 40
        env: np.ndarray = np.repeat(np.exp(np.clip(mel[:, k], -5, 5)), hop)[: len(t)]
        audio += env * np.sin(2 * np.pi * freq * t)
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio)) + 1e-9
    return (0.8 * audio / peak).astype(np.float32)


def attach_lora_qwen(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    return tok, get_peft_model(model, lora)
