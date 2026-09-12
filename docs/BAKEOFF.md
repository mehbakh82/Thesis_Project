# Model and codec investigation

**Status: component survey only.** Run `thesis-s2s bakeoff` to regenerate
`results/bakeoff/bakeoff_report.json`. The current source-backed candidate
catalog and fail-closed requirements audit are separate:

```bash
thesis-s2s audit-model-selection
```

See `docs/MODEL_AND_DATA_SELECTION_AUDIT.md` and
`results/model_selection_audit.json`.

## What actually ran

- Encodec, mel/Griffin-Lim, and correctly quantized 8-bit mu-law reconstruction probes.
- Import/availability checks for CosyVoice2, SNAC, and Mimi.
- A Whisper-small transcription proxy on reconstructed Persian audio where dependencies were available.
- No controlled fine-tuning or end-to-end comparison of LLaMA-Omni2, PersonaPlex, Mini-Omni2, and Qwen3-Omni.

## Current architecture decision

The deployable prototype uses the modular cascade because it is the only path in this repository with real speech input, generated Persian response text, and complete synthesized reply audio. It is a baseline, not the thesis's desired direct S2S model.

The selected direct-training path is now **Moshi + the official Moshi-Finetune LoRA trainer**. Kyutai's trainer has a real stereo-conversation objective with text and assistant-audio token losses and produces adapters loadable by its streaming runtime. The local exporter converts each non-reused Persian user/response pair into that exact schema. This is an engineering selection based on trainability, duplex behavior, license, and hardware feasibility—not yet an empirical Persian winner.

The earlier local LLaMA-Omni2 encoder/reconstruction ablation remains runtime-ineligible. A final model claim still requires the same Persian conversational train/eval set, held-out quality metrics, true client-observed latency, and interruption tests.

## Interpretation of the candidates

- LLaMA-Omni2 is a useful architectural reference, but its public repository does not provide the complete training recipe reproduced here.
- Moshi demonstrates true full duplex, has a released Apache-2.0 LoRA trainer, CC-BY-4.0 weights, and a documented 24 GB inference path. Its base is English and Persian remains an adaptation risk.
- PersonaPlex demonstrates true full duplex, but its published release is English-oriented and its documented tested hardware is outside the thesis target.
- Mini-Omni2 is compact and end-to-end, but its released speech output is English-only.
- Qwen3-Omni supports many languages but not Persian speech input/output in its published language lists and is much larger than the intended base.

Therefore Moshi is the implementation path, while the cascade remains the deployable baseline until a Persian adapter passes the stated evidence gates. See `results/bakeoff/DECISION.md`, `docs/MOSHI_H100_RUNBOOK.md`, and `docs/REVIEW.md`.

## Post-release candidate refresh (2026-09-10)

Controlled comparison remains missing for the direct track, but
the documentary shortlist now also
includes MiniCPM-o 4.5, Covo-Audio-Chat-FD, BayLing-Duplex, DuplexOmni,
Qwen3-ASR, Shenava-Koochik, Rade-ASR-CTC-3B-fa, Omnilingual-ASR,
Qwen3.5-4B/0.8B, Qwen3-TTS, and MOSS-TTS-Nano-Persian. The machine-readable
audit fails closed on Persian quality, adaptation, or 24 GB fields that have not
been verified. A newly frozen, equal-channel, session-disjoint 40-row responder
development comparison retained Qwen3-4B: Qwen3.5-4B had zero relevance gain
and a 15% row win rate, while Qwen3.5-0.8B regressed. The separately locked final
panel remains sealed. This is a catalog-bounded automatic-proxy result, not a
global-best claim. A separate frozen 40-row Digiato/Zoomit ASR development
comparison also retained the fine-tuned NeMo service: CER/WER was
0.230119/0.371593, versus 0.383788/0.651363 for Qwen3-ASR-1.7B and
0.492509/0.807747 for Qwen3-ASR-0.6B. Its separately locked final panel remains
sealed because neither challenger passed the predeclared development gate.
An explicitly post-hoc Shenava screen reused only that open development panel:
it was faster (median CPU RTF 0.016045) but worse on CER/WER
(0.241644/0.403156), so NeMo remained selected and the final panel stayed
sealed.
The TTS comparison also retained Mana-Piper on 40 identical Persian response
texts: CER/WER was 0.174401/0.342520, versus 0.250892/0.494094 for Meta
MMS-TTS Persian, with no failures in either arm. Its final panel likewise
remains sealed. This is an automatic ASR round-trip result, not human listening.

The 2026-09-10 official-source recheck makes two conservative corrections:
Qwen3-Omni is now an explicit Persian speech-output failure because its ten
published output languages exclude Persian; MiniCPM-o 4.5 now passes only the
documentary public-adaptation field because the official repository lists
LLaMA-Factory/SWIFT support. MiniCPM-o still has no verified Persian speech
output or documented Persian/full-duplex audio-token adaptation, so no direct
candidate becomes eligible and no new model run is justified by this refresh.

The component-survey implementation was subsequently hardened so the historical
`mulaw8` label now denotes an actual 256-level quantization bottleneck (rather
than a floating-point compand/expand identity), undefined silent-input SNR is
serialized as `null`, temporary ASR files are removed even on write failure,
and report paths are made repository-relative before persistence. This
correction affects only the diagnostic mu-law SNR; it does not change the
Encodec intelligibility observation or any model-selection conclusion.
The Whisper CER field is also explicitly an automatic transcript-content proxy;
phonetic and perceptual preservation remain unknown without human listening.
