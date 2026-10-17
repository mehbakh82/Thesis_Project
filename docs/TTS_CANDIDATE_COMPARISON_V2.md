# TTS candidate comparison v2 — result

Receipt: `results/eval/tts_candidates_v2.json` (scored by
`scripts/score_tts_candidates_v2.py` from the synthesis receipt of
`scripts/evaluate_tts_candidates_v2.py` and two ASR listeners).

## Pre-registration status

This comparison was run before a separate protocol file was committed. The
selection rule below is therefore documented after the run, and it is kept
deliberately narrow: the automatic measures only choose which **two** voices go
to the human listening study, and naturalness is decided there, not here.

## Setup

- Texts: 60 Persian assistant replies to the elderly-request panel (Qwen3-4B
  with the elderly prompt, from responder v2), synthesized once per voice with deterministic
  Piper settings (`noise_scale = noise_w = 0`).
- Candidates: all six Persian Piper voices (`mana` = incumbent, `amir`,
  `ganji`, `ganji_adabi`, `gyro`, `reza_ibrahim`).
- Intelligibility: round-trip transcription by two independent ASR listeners
  (NeMo v4 CTC and Seamless v5ft), scored with the punctuation-free
  `fa-asr-score-v1` normalizer; the two-listener mean CER and a paired bootstrap
  interval per clip against the incumbent.
- Quality proxy: DNSMOS P.835 (SIG/BAK/OVRL) — a noise-suppression quality
  model, not a Persian naturalness MOS.
- Speed: render real-time factor and time to synthesize the first four words
  (the live server's first clause) on CPU.

## Results

| Voice | CER NeMo | CER Seamless | Mean CER | Δ vs Mana (95% CI) | DNSMOS OVRL | First 4 words (ms, median) |
|---|---|---|---|---|---|---|
| mana (incumbent) | 6.34% | 5.31% | **5.82%** | — | 3.13 | 84 |
| ganji_adabi | 5.59% | 8.27% | 6.93% | [−1.1, +4.9] pp | **3.40** | 92 |
| gyro | 6.29% | 8.32% | 7.30% | [+0.1, +4.0] pp | 3.30 | 61 |
| reza_ibrahim | 8.22% | 8.55% | 8.39% | [+0.6, +5.2] pp | 3.26 | 102 |
| ganji | 4.54% | 13.49% | 9.02% | [−1.3, +11.2] pp | 3.29 | 61 |
| amir | 12.28% | 11.18% | 11.73% | [+4.5, +7.6] pp | 3.32 | 63 |

## Decision

- **Mana is retained** as the deployed voice: it has the lowest two-listener
  CER; three voices are significantly less intelligible, and none is
  significantly more intelligible.
- **ganji_adabi** is statistically tied with Mana on intelligibility and has the
  best DNSMOS score, so Mana and ganji_adabi are the two voices compared for
  naturalness in the listening study (`scripts/build_listening_study.py`).
- All voices synthesize the first clause in about 60–100 ms on CPU, well inside
  the latency budget, so speed does not separate them.

## Claim boundary

Automatic proxies only. ASR round-trip error mixes TTS clarity with the
listeners' own errors (the two listeners disagree on `ganji`, for example).
DNSMOS was not trained on Persian or on synthetic speech. Neither is a
substitute for the human naturalness rating.
