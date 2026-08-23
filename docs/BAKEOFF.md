# Model and codec investigation

**Status: component survey only.** Run `python3 -m thesis_s2s.cli bakeoff` to regenerate `results/bakeoff/bakeoff_report.json`.

## What actually ran

- Encodec, mel/Griffin-Lim, and mu-law reconstruction probes.
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

