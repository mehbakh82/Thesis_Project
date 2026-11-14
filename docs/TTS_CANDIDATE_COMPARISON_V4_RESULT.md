# TTS candidate comparison v4: result

Protocol: `docs/TTS_CANDIDATE_COMPARISON_V4_PROTOCOL.md` (frozen 2026-11-14,
before any v4 model was downloaded). Receipt: `results/eval/tts_candidates_v4.json`
(scored by the unchanged `scripts/score_tts_candidates_v2.py` together with the
two Piper arms of v2). **Outcome: Mana stays the most intelligible voice, and
no new voice would have qualified for the listening study.** As fixed in the
protocol, no v4 result could change the deployed voice, because the listening
study was not conducted.

## Results

The same 60 assistant replies as TTS v2 and v3. CER is the mean of two ASR
listeners (NeMo v4 CTC and Seamless v5ft); the difference against Mana is the
paired clip mean with a 95% bootstrap interval (percentage points). DNSMOS
P.835 OVRL is an automatic signal-quality proxy, not a Persian naturalness MOS.
The new arms were synthesized clause by clause with the live server's
`ClauseChunker`, cloning one CC0 ManaTTS reference.

| Voice | Two-listener CER | vs Mana | DNSMOS OVRL | First four words (median) | Real-time factor |
|---|---:|---|---:|---:|---:|
| **Mana, Piper (deployed)** | 5.82% | | 3.13 | 84 ms (CPU) | 0.06 |
| **Chatterbox Persian** (500M) | 6.47% | +0.4 (−2.2 to +3.3) | 3.12 | 1,981 ms (H100) | 0.97 |
| ganji_adabi, Piper (v2) | 6.93% | +1.3 (−1.1 to +4.9) | 3.40 | 92 ms (CPU) | 0.04 |
| **Gooya v1** (MOSS-Nano LoRA, phonemes) | 18.22% | +13.1 (+2.3 to +27.5) | 3.06 | 2,015 ms (H100) | 0.81 |
| **MOSS-TTS-Nano-Persian** (117M) | 21.42% | +14.9 (+4.4 to +29.3) | 3.01 | 3,298 ms (H100) | 0.75 |

New v4 arms in bold. For comparison, the v3 arms: ParsVoice-XTTS 13.55% (+7.7,
significantly worse), the lab VITS 9.89% as served and 9.19% optimized (both
significantly worse).

## Rules

1. **Would qualify for the listening study** (CER not significantly worse
   than Mana and DNSMOS OVRL at least Mana's):
   - Chatterbox Persian is as intelligible as Mana (the interval includes
     zero) but its DNSMOS is 0.011 below Mana's (3.116 against 3.127), so it
     does not qualify. Its output carries Chatterbox's built-in Perth
     watermark.
   - Gooya v1 and MOSS-TTS-Nano-Persian are significantly less intelligible.
     Seamless, the second listener, finds them much worse than NeMo does
     (25% and 30% CER against 11% and 12%).
2. **Latency:** no new voice is near the 100 ms bound. All three are
   autoregressive and take about 2 to 3.3 s on the H100 to synthesize four
   words, 24 to 39 times Mana's time on a CPU.
3. **Deployment:** none (no listening study).

The answer to "is Mana the best available voice" is therefore: among the 12
voices and configurations tested in v2 to v4, Mana is the most intelligible to
two automatic listeners and, like the other Piper voices, within the latency
budget; Chatterbox Persian matches its intelligibility but not its speed.
Naturalness was never measured by people.

## Deviations

- MOSS and Gooya ran in the NeMo environment (transformers 4.57.6, torch
  2.13). Its torchaudio reads and writes files only through torchcodec, which
  is not installed; the synthesis script routes `torchaudio.load`/`save` to
  soundfile for these WAV files.
- Chatterbox ran with torch 2.13 and torchaudio 2.11 instead of the pinned
  2.6, with transformers 5.2.0 and diffusers 0.29.0 as pinned, and
  setuptools < 81 for its watermark package. All five weight files matched
  the repackaging's `MODEL_MANIFEST.json`.
- Gooya v1: one reply (`eq12`) failed because the G2P returned no phonemes for
  one fragment; it is scored as an empty clip (counts against Gooya). The G2P
  reproduced all six v7 rows of the model's own `matrix/manifest.json`.
- Revisions resolved on download: `Reza2kn/negara-g2p-clean-v7` @ `b6b36919…`,
  `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` @ `6aa02b01…`.
- The Piper arms are the deterministic v2 renderings (whole replies, CPU
  timing); the new arms were synthesized clause by clause on the H100.

## Claim boundary

Automatic listeners and an automatic quality proxy on 60 texts, one cloned
reference voice, H100 timing for the new arms; no human rating.
