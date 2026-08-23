# Model and codec investigation

**Status: component survey only.** Run `python3 -m thesis_s2s.cli bakeoff` to regenerate `results/bakeoff/bakeoff_report.json`.

## What actually ran

- Encodec, mel/Griffin-Lim, and mu-law reconstruction probes.
- Import/availability checks for CosyVoice2, SNAC, and Mimi.
- A Whisper-small transcription proxy on reconstructed Persian audio where dependencies were available.
- No controlled fine-tuning or end-to-end comparison of LLaMA-Omni2, PersonaPlex, Mini-Omni2, and Qwen3-Omni.

## Current architecture decision

The deployable prototype uses the modular cascade because it is the only path in this repository with real speech input, generated Persian response text, and complete synthesized reply audio. It is a baseline, not the thesis's desired direct S2S model.

The direct-model work remains an experimental encoder/reconstruction ablation and cannot be selected by the runtime. A future decision requires the same Persian conversational train/eval set, identical hardware, quality metrics, true client-observed latency, and interruption tests for every candidate.

## Interpretation of the candidates

- LLaMA-Omni2 is a useful architectural reference, but its public repository does not provide the complete training recipe reproduced here.
- PersonaPlex demonstrates true full duplex, but its published release is English-oriented and its documented tested hardware is outside the thesis target.
- Mini-Omni2 is compact and end-to-end, but its released speech output is English-only.
- Qwen3-Omni supports many languages but not Persian speech input/output in its published language lists and is much larger than the intended base.

Therefore no model is “locked” by the historical codec report. See `results/bakeoff/DECISION.md` and `docs/REVIEW.md`.

