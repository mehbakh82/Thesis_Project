# ASR candidate comparison v2 — protocol

Status: frozen before any v2 prediction was generated.

## Why a v2

1. The v1 development panel (Digiato/Zoomit YouTube captions) is no longer a
   fair test: the newer NeMo v5/v6 checkpoints were trained with Digiato,
   Zoomit, and Mehran YouTube replay, and v1's references are automatic
   captions.
2. v1 scored with the conservative `fa-verbatim-2` text normalizer, which keeps
   punctuation, so systems that punctuate (Qwen3-ASR, Seamless, Whisper) were
   penalized for punctuation that the unpunctuated references lack. v1 kept only
   hash/count receipts, so it cannot be re-scored; v2 fixes the policy.
3. New candidates exist: the user's NeMo v6 and Seamless v6 (checkpoint 55,000)
   fine-tunes, and Gemma 4 E4B/12B with native audio input.

## Panels

- **PSRB-human (primary human-reference panel):** all 344 clips (53.4 min) of
  the public PSRB 1-hour sample (PartAI), transcribed manually by native
  speakers with two-pass QA; research/education terms. It includes 47
  senior-speaker clips (6.7 min), which form the elderly stratum. Audio is
  converted to 16 kHz mono; panel file SHA-256
  `bc80e48ead162f48323617389852b6a01a90c54eada6b11226696c78283d2e13`.
  Caveat: PSRB was used as an evaluation (never training) set in the user's
  earlier noisy-speech study, which may have influenced which NeMo/Seamless
  checkpoints were kept; this is disclosed, not corrected.
- **Fresh-podcast panel (Gemini references):** episode-disjoint clips from
  newly crawled podcast channels never used to train any candidate, with
  Gemini 3.8 Flash verbatim references (independent automatic labels). Added
  under this same protocol when the crawl is available; reported separately.

## Arms

| Arm | Model | Decoding | Environment |
|---|---|---|---|
| `nemo_v4_ctc` / `nemo_v4_rnnt` | NeMo Soroosh v4 (thesis incumbent) | greedy | NeMo 2.7.3 |
| `nemo_v6_ctc` / `nemo_v6_rnnt` | NeMo v6 epoch-2 export | greedy | NeMo 2.7.3 |
| `seamless_v6_55000` | SeamlessM4T v2 fine-tune, step 55,000 | beam 1 | Seamless env |
| `seamless_v5ft` | SeamlessM4T v2 fine-tune v5 | beam 1 | Seamless env |
| `gemma4_e4b` / `gemma4_12b` | Gemma 4 QAT w4a16, zero-shot audio prompt | greedy | project env |
| `qwen3_asr_1p7b` | Qwen3-ASR-1.7B, Persian | default | NeMo env + vendor |
| `omniasr_ctc_1b` | Omnilingual ASR CTC-1B v2 (HF conversion) | greedy | NeMo env |
| `whisper_large_v3` | Whisper large-v3, language fa | default | NeMo env |

Predictions come from `scripts/asr_predict.py`, one clip at a time after one
warm-up clip, so `latency_ms` is per-utterance processing time on the shared
H100. Errors and empty outputs count as full deletions.

## Scoring

`src/thesis_s2s/eval/asr_normalize.py` (`fa-asr-score-v1`): Arabic-script
variants to Persian letters, diacritics and tatweel removed, digits unified,
ZWNJ/ZWJ removed, punctuation stripped. Micro CER (primary) and micro WER over
the whole panel; strata by age (senior vs. other), acoustic environment, and
spontaneity; 10,000-draw clip bootstrap 95% intervals and paired bootstrap
differences against the incumbent.

## Selection rule

- **Deployment ASR:** the arm with the lowest PSRB-human micro CER among arms
  whose median per-utterance latency on clips of at most 8 s is at most 150 ms
  (the share of the 500 ms first-audio budget left after a 256 ms endpoint
  hangover, first-clause LLM decoding, and first-chunk TTS). If the fresh
  podcast panel is available, the winner must also not be worse than the
  incumbent on it (paired bootstrap upper bound ≤ 0).
- **Accuracy reference:** the lowest-CER arm regardless of latency is reported
  as the offline/reference ASR and as a candidate front end for the
  speech-input LLM comparison.
- The full latency–accuracy Pareto front is reported (research question 3).
